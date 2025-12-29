import os

import torch
import torch.nn as nn
import torchvision.transforms.v2 as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def set_requires_grad(module: nn.Module, flag: bool):
    """
    Enable or disable gradient computation for all parameters in a module.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be modified.
    flag : bool
        If ``True``, sets ``requires_grad=True`` for all parameters in
        ``module``. If ``False``, sets ``requires_grad=False`` (freezes the
        parameters)
    """
    for p in module.parameters():
        p.requires_grad = flag

def apply_freeze_preset(resnet_model: nn.Module, preset: dict, freeze_bn_stats: bool = True):
    """
    Freeze/unfreeze ResNet components according to a preset specification.

    The preset controls which parts of a torchvision ResNet should be frozen
    (i.e., not updated during training) by setting ``requires_grad=False`` on
    their parameters. Parts are grouped as:

    - ``stem``: ``conv1`` + ``bn1`` (ReLU/maxpool have no parameters)
    - ``layer1``..``layer4``: residual stages
    - ``fc``: final classification layer

    Parameters
    ----------
    resnet_model : torch.nn.Module
        A torchvision ResNet model (or a compatible module) exposing attributes
        ``conv1``, ``bn1``, ``layer1``, ``layer2``, ``layer3``, ``layer4``, and
        ``fc``.
    preset : dict
        Mapping from part name to freeze flag. Each key must be one of
        ``{"stem", "layer1", "layer2", "layer3", "layer4", "fc"}``.

        Values are booleans where:

        - ``True``  means freeze that part (set ``requires_grad=False``)
        - ``False`` means train that part (set ``requires_grad=True``)
    """
    b = resnet_model

    parts = {
        "stem": nn.Sequential(b.conv1, b.bn1),  # relu/maxpool have no params
        "layer1": b.layer1,
        "layer2": b.layer2,
        "layer3": b.layer3,
        "layer4": b.layer4,
        "fc": b.fc,
    }

    # Freeze/train per preset
    for name, module in parts.items():
        freeze = preset[name]
        set_requires_grad(module, not freeze)

    # BatchNorm running stats handling (important)
    if freeze_bn_stats:
        def bn_eval_if_frozen(m):
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                params = list(m.parameters())
                if params and all(not p.requires_grad for p in params):
                    m.eval()
        b.apply(bn_eval_if_frozen)

def make_optimizer(model: nn.Module, optimizer_type: str, lr: float, wd: float, momentum: float = 0.9, nesterov: bool = True):
    """
    Create a PyTorch optimizer for the trainable parameters of a model.

    Only parameters with ``requires_grad=True`` are passed to the optimizer.
    This is especially useful when fine-tuning models with frozen backbones.

    Parameters
    ----------
    model : torch.nn.Module
        Model containing parameters to optimize.
    optimizer_type : str
        Optimizer name (case-insensitive). Supported values are:

        - ``"sgd"``
        - ``"adam"``
        - ``"adamw"``
    lr : float
        Learning rate.
    wd : float
        Weight decay (L2 regularization).
    momentum : float, optional
        Momentum factor for SGD. Ignored for Adam/AdamW. Default is ``0.9``.
    nesterov : bool, optional
        Whether to use Nesterov momentum for SGD. Ignored for Adam/AdamW.
        Default is ``True``.

    Returns
    -------
    torch.optim.Optimizer
        Instantiated optimizer.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
    
def make_scheduler(optimizer: torch.optim.Optimizer,
                   scheduler_type: str,
                   num_epochs: int,
                   steps_per_epoch: int | None = None):
    """
    Create a learning-rate scheduler for a given optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose learning rate will be scheduled.
    scheduler_type : str
        Scheduler name (case-insensitive). Supported values are:

        - ``"none"`` or ``""``: no scheduler (returns ``None``)
        - ``"steplr"``: StepLR with ``step_size=10`` and ``gamma=0.1``
        - ``"multisteplr"``: MultiStepLR with milestones ``[10, 15]`` and ``gamma=0.1``
        - ``"cosine"``: CosineAnnealingLR with ``T_max=num_epochs``
        - ``"onecycle"``: OneCycleLR (requires ``steps_per_epoch``)
        - ``"plateau"``: ReduceLROnPlateau (mode ``"min"``)
    num_epochs : int
        Total number of epochs (used by cosine annealing and one-cycle).
    steps_per_epoch : int or None, optional
        Number of training steps (batches) per epoch. Required when
        ``scheduler_type="onecycle"``. Default is ``None``.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.ReduceLROnPlateau or None
        Instantiated scheduler, or ``None`` if ``scheduler_type`` indicates no scheduler.
    """
    st = (scheduler_type or "").lower()

    if st in ("", "none", None):
        return None

    if st == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if st == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    if st == "cosine":
        # Steps once per epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    if st == "onecycle":
        # Steps once per *batch* (needs steps_per_epoch)
        if steps_per_epoch is None:
            raise ValueError("OneCycleLR needs steps_per_epoch")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
        )

    if st == "plateau":
        # Steps after validation using a metric (e.g., val loss)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )

    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

def make_data_loaders(dataset_path, augmentation, batch_size):
    """
    Create train and test DataLoaders for an ImageFolder-style dataset.

    The dataset directory is expected to contain ``train`` and ``test`` folders,
    each with subdirectories per class:

    ``dataset_path/train/<class_name>/*.jpg``
    ``dataset_path/test/<class_name>/*.jpg``

    Parameters
    ----------
    dataset_path : str or pathlib.Path
        Root directory of the dataset containing ``train`` and ``test`` folders.
    augmentation : bool
        If ``True``, applies data augmentation to the training split using
        random resized crops and horizontal flips. Test preprocessing remains
        deterministic. If ``False``, both splits use deterministic transforms.
    batch_size : Sequence[int]
        Two-element sequence ``[train_batch_size, test_batch_size]``.

    Returns
    -------
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        ``(train_loader, test_loader)`` DataLoaders for the train and test splits.
    """
    if augmentation:
        train_transform = F.Compose([
            F.RandomResizedCrop(224, scale=(0.8, 1.0)),
            F.RandomHorizontalFlip(0.5),
            F.ToTensor(),
            F.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        test_transform = F.Compose([
            F.Resize(256),
            F.CenterCrop(224),
            F.ToTensor(),
            F.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = test_transform = F.Compose([
            F.Resize(256),
            F.CenterCrop(224),
            F.ToTensor(),
            F.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        
    data_train = ImageFolder(os.path.join(dataset_path, "train"), transform=train_transform)
    data_test = ImageFolder(os.path.join(dataset_path, "test"), transform=test_transform) 

    train_loader = DataLoader(data_train, batch_size=batch_size[0], pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=batch_size[1], pin_memory=True, shuffle=False, num_workers=8)

    return train_loader, test_loader