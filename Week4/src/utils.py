"""
This file contains utils for data processing and evaluation.
"""
import os
import copy
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as F
from pathlib import Path

from src.model import switch_model_to_deploy



# -------------
# Dataloaders
# -------------
def make_data_loaders(dataset_path, batch_size, train_pipeline="basic"):
    dataset_path = os.path.expanduser(dataset_path)
    imagenet_norm = F.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    if train_pipeline == "plain":
        train_ops = [
            F.Resize(256),
            F.CenterCrop(224),
        ]

    elif train_pipeline == "basic":
        train_ops = [
            F.RandomResizedCrop(224, scale=(0.8, 1.0)),
            F.RandomHorizontalFlip(0.5),
            F.ColorJitter(0.2, 0.2, 0.2, 0.05),
            F.RandomGrayscale(p=0.05),
        ]

    elif train_pipeline == "basic_erasing":
        train_ops = [
            F.RandomResizedCrop(224, scale=(0.8, 1.0)),
            F.RandomHorizontalFlip(0.5),
            F.ColorJitter(0.2, 0.2, 0.2, 0.05),
            F.RandomGrayscale(p=0.05),
        ]

    elif train_pipeline == "randaug":
        train_ops = [
            F.RandomResizedCrop(224, scale=(0.8, 1.0)),
            F.RandomHorizontalFlip(0.5),
            F.RandAugment(),
        ]

    elif train_pipeline == "randaug_erasing":
        train_ops = [
            F.RandomResizedCrop(224, scale=(0.8, 1.0)),
            F.RandomHorizontalFlip(0.5),
            F.RandAugment(),
        ]

    else:
        raise ValueError(f"Unknown train_pipeline: {train_pipeline}")

    train_ops += [
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
    ]

    if train_pipeline in {"basic_erasing", "randaug_erasing"}:
        train_ops += [F.RandomErasing(p=0.25)]

    train_ops += [imagenet_norm]
    train_transform = F.Compose(train_ops)

    test_transform = F.Compose([
        F.Resize(256),
        F.CenterCrop(224),
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        imagenet_norm,
    ])

    data_train = ImageFolder(os.path.join(dataset_path, "train"), transform=train_transform)
    data_test  = ImageFolder(os.path.join(dataset_path, "test"),  transform=test_transform)

    train_loader = DataLoader(data_train, batch_size=batch_size[0], shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(data_test,  batch_size=batch_size[1], shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# -------------
# Evaluation utils
# -------------
@torch.no_grad()
def check_equivalence(model, device, input_shape=(8, 3, 224, 224)):
    model.eval()
    x = torch.randn(*input_shape).to(device)

    y_before = model(x)

    m_deploy = copy.deepcopy(model).to(device)
    m_deploy.eval()
    switch_model_to_deploy(m_deploy)

    y_after = m_deploy(x)

    max_diff = (y_before - y_after).abs().max().item()
    mean_diff = (y_before - y_after).abs().mean().item()
    print(f"max abs diff:  {max_diff:.6e}")
    print(f"mean abs diff: {mean_diff:.6e}")

    return max_diff, mean_diff


# -------------
# Optimization
# -------------
def build_optimizer(name, params, lr, weight_decay, momentum=0.9):
    name = name.lower()
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(name, optimizer, epochs):
    name = name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs//3,1), gamma=0.1)
    raise ValueError(f"Unknown scheduler: {name}")


