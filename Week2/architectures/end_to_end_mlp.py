import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as F
import wandb
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import Week2.models as models

from ..main import test, train

PROJECT_ROOT = Path.cwd()
WEEK_2_ROOT = PROJECT_ROOT / "Week2"

torch.manual_seed(42)

def save_results(results, results_path, timestamp):


    os.makedirs(results_path, exist_ok=True)
    output_path = os.path.join(results_path, timestamp+'.json')
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        print("Saved results in file grid_results.json")

def get_optimizer(optimizer, model_params, lr,
                  weight_decay=0.0, momentum=0.0, nesterov=False,
                  betas=(0.9, 0.999)):
    """
    Returns:
        optimizer_obj: torch.optim.Optimizer
        optimizer_cfg: dict (JSON/YAML serializable)
    """

    opt_class = getattr(torch.optim, optimizer, None)
    if opt_class is None:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    lr = float(lr)
    weight_decay = float(weight_decay)
    momentum = float(momentum)
    kwargs = {"lr": lr}

    optimizer_cfg = {
        "name": optimizer,
        "lr": lr
    }

    if optimizer == "SGD":
        kwargs.update({
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov
        })
        optimizer_cfg.update({
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov
        })

    elif optimizer in {"Adam", "AdamW"}:
        kwargs.update({
            "weight_decay": weight_decay,
            "betas": betas
        })
        optimizer_cfg.update({
            "weight_decay": weight_decay,
            "betas": betas
        })

    elif optimizer == "Adagrad":
        kwargs.update({
            "weight_decay": weight_decay
        })
        optimizer_cfg.update({
            "weight_decay": weight_decay
        })

    optimizer_obj = opt_class(model_params, **kwargs)

    return optimizer_obj, optimizer_cfg


def run_experiment(cfg, shape, train_loader, test_loader, device, optimizer_type, lr, wd, m, nesterov, activation, dropout, num_epochs, timestamp):
    wandb_run = wandb.init(
        project="week2-mlp",
        name=timestamp,
        group="grid-search",
        reinit=True,
        config={
            "layers": cfg["layers"],
            "activation": activation,
            "dropout": dropout,
            "epochs": num_epochs,
            "optimizer": optimizer_type,
            "lr": float(lr),
            "weight_decay": float(wd),
            "momentum": float(m),
            "nesterov": nesterov,
            "batch_size": train_loader.batch_size,
            "input_shape": shape,
        }
    )

    layers = cfg["layers"].copy()
    C, H, W = shape

    # Insert input layer
    first_layer = [C*H*W, layers[0][0]]
    layers.insert(0, first_layer)

    # Build model
    model = models.DynamicMLP(
        layer_sizes=[tuple(x) for x in layers],
        activation=activation,
        dropout=dropout
    ).to(device)
    print("Model device:", next(model.parameters()).device)
    criterion = nn.CrossEntropyLoss()
    optimizer, optimizer_cfg = get_optimizer(
            optimizer=optimizer_type,
            model_params=model.parameters(),
            lr=lr,
            momentum=m,
            weight_decay=wd,
            nesterov=nesterov
        )

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=15, gamma=0.5
    )
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "test/loss": test_loss,
            "test/accuracy": test_acc
        })
        scheduler.step()
    metrics = {
        "final_train_acc": train_accuracies[-1],
        "final_test_acc": test_accuracies[-1],
        "train_curve": train_accuracies,
        "test_curve": test_accuracies,
    }
    wandb.finish()

    return metrics, model, optimizer_cfg


def grid_search(experiments_models, data_train, data_test, resize_sizes, augmentation):
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    models = []

    for i, exp_cfg in enumerate(experiments_models):
        print(f"Running experiment {i+1}/{len(experiments_models)}")
        
        model_params = exp_cfg['model']
        pipeline_params = exp_cfg['pipeline']
        optimizer_params = exp_cfg['optimizer']
        
        for batch_size in pipeline_params['batch_size']:
            train_loader = DataLoader(data_train, batch_size=batch_size[0], pin_memory=True, shuffle=True)
            images, _ = next(iter(train_loader))
            shape =  images[0].shape
            test_loader = DataLoader(data_test, batch_size=batch_size[1], pin_memory=True, shuffle=False)
            for epochs in pipeline_params['num_epochs']:

                for activation in model_params['activation']:
                    for dropout in model_params['dropout']:
                        
                        for optimizer in optimizer_params['type']:
                            for lr in optimizer_params['lr']:
                                for wd in optimizer_params['weight_decay']:
                                    for m in optimizer_params['momentum']:
                                        for nesterov in optimizer_params['nesterov']:

                                            
                                            result, model, optimizer_cfg = run_experiment(model_params, shape, train_loader, test_loader, device, 
                                                                    optimizer, lr, wd, m, nesterov,
                                                                    activation, dropout, epochs, timestamp)
                                            


                                            result["config"] = {
                                                "layers": model_params["layers"],
                                                "batch_size": batch_size,
                                                "epochs": epochs,
                                                "activation": activation,
                                                "dropout": dropout,
                                                "optimizer": optimizer_cfg,
                                                "img_size": resize_sizes,
                                                "data_agmentation": augmentation
                                            }  # store what config generated this result

                                            results.append(result)
                                            models.append(model)

    best_id, best_result = max( enumerate(results), key=lambda x: x[1]["final_test_acc"])
    best_model = models[best_id]
    best_config = best_result["config"]
    cfg["best"] = best_config

    print("Best model:")
    print(best_result["config"])
    print("Accuracy:", best_result["final_test_acc"])

    save_results(results, os.path.join(WEEK_2_ROOT,"results"), timestamp)
    # save best model's weights and config
    save_dir = WEEK_2_ROOT / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), save_dir / f"{timestamp}_best_model.pth")

if __name__=="__main__":
    with open(WEEK_2_ROOT / "configs" / "NN1.yaml", "r") as f:
        print(WEEK_2_ROOT / "configs" / "NN1.yaml",)
        cfg = yaml.safe_load(f)
        experiments_models = cfg["experiments"]
        resize_sizes = cfg['resize']
        augmentation = cfg['data_augm']
    for resize in resize_sizes:
        if augmentation:
            train_transform = F.Compose([
                F.ToImage(),
                F.ToDtype(torch.float32, scale=True),
                F.Resize((resize[0], resize[1])),
                F.RandomHorizontalFlip(p=0.5),
                F.ColorJitter(brightness=0.2, contrast=0.2),
            ])

            test_transform = F.Compose([
                F.ToImage(),
                F.ToDtype(torch.float32, scale=True),
                F.Resize((resize[0], resize[1])),
            ])
            data_train = ImageFolder(r'c:\Users\maiol\Desktop\Master\C3\places_reduced\train', transform=train_transform)
            data_test = ImageFolder(r'c:\Users\maiol\Desktop\Master\C3\places_reduced\val', transform=test_transform)
        else:
            transformation  = F.Compose([
                                        F.ToImage(),
                                        F.ToDtype(torch.float32, scale=True),
                                        F.Resize(size=(resize[0], resize[1])),
                                    ])
        
            data_train = ImageFolder(r'c:\Users\maiol\Desktop\Master\C3\places_reduced\train', transform=transformation)
            data_test = ImageFolder(r'c:\Users\maiol\Desktop\Master\C3\places_reduced\val', transform=transformation) 

        grid_search(experiments_models, data_train, data_test, resize_sizes, augmentation)

