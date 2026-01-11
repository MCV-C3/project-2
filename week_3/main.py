import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as F
import tqdm
import yaml
from itertools import product
from models import ResNet50, SimpleModel, WraperModel
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    RandomResizedCrop, ToTensor)
from torchviz import make_dot
from utils import (apply_freeze_preset, make_data_loaders, make_optimizer,
                   make_scheduler)

import wandb

FREEZE_PRESETS = {
"fc_only":       {"stem": True,  "layer1": True,  "layer2": True,  "layer3": True,  "layer4": True,  "fc": False},
"l4_fc":         {"stem": True,  "layer1": True,  "layer2": True,  "layer3": True,  "layer4": False, "fc": False},
"l3_l4_fc":      {"stem": True,  "layer1": True,  "layer2": True,  "layer3": False, "layer4": False, "fc": False},
"l2_l3_l4_fc":   {"stem": True,  "layer1": True,  "layer2": False, "layer3": False, "layer4": False, "fc": False},
"l1_l2_l3_l4_fc": {"stem": True,  "layer1": False,  "layer2": False, "layer3": False, "layer4": False, "fc": False},
"none_frozen":   {"stem": False, "layer1": False, "layer2": False, "layer3": False, "layer4": False, "fc": False},
}

def ensure_list(x):
    return x if isinstance(x, list) else [x]

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

def denormalize(tensor):
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    return torch.clamp(tensor * std + mean, 0, 1)

# Train function
def train(model, dataloader, criterion, optimizer, device, epoch, adv_training=True, epsilon=0.04):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0
    logged_images = False 

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        original_inputs = inputs.clone().detach()

        if adv_training:
            inputs.requires_grad = True 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            
            data_grad = inputs.grad.data
            inputs = fgsm_attack(inputs, epsilon, data_grad).detach()

            if not logged_images and epoch % 5 == 0:
                noise = inputs - original_inputs
                noise_visual = torch.clamp(noise * 20 + 0.5, 0, 1)
                
                wandb.log({
                    "adversarial/example": [wandb.Image(denormalize(inputs[0]), caption=f"Adv Epoch {epoch}")],
                    "adversarial/noise_amplified": [wandb.Image(noise_visual[0], caption="FGSM Noise (x20)")]
                }, commit=False)
                logged_images = True

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return train_loss / total, correct / total



def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str, file_str: str, plots_dir: str):
    """
    Plots and saves metrics for training and testing.

    Args:
        train_metrics (Dict): Dictionary containing training metrics.
        test_metrics (Dict): Dictionary containing testing metrics.
        metric_name (str): The name of the metric to plot (e.g., "loss", "accuracy").

    Saves:
        - loss.png for loss plots
        - metrics.png for other metrics plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot with the appropriate name
    out_dir = Path(plots_dir) / ("losses" if metric_name.lower() == "loss" else "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{file_str}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory

def plot_computational_graph(model: torch.nn.Module, input_size: tuple, filename: str = "computational_graph"):
    """
    Generates and saves a plot of the computational graph of the model.

    Args:
        model (torch.nn.Module): The PyTorch model to visualize.
        input_size (tuple): The size of the dummy input tensor (e.g., (batch_size, input_dim)).
        filename (str): Name of the file to save the graph image.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Generate a dummy input based on the specified input size
    dummy_input = torch.randn(*input_size)

    # Create a graph from the model
    graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True).render(filename, format="png")

    print(f"Computational graph saved as {filename}")

def parse_args():
    """
    Parse command-line arguments for the neural network training script.

    This function defines and parses command-line options required to run
    the training pipeline. Currently, it supports specifying the path to
    a YAML configuration file.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments. The returned namespace contains:

        config_path : str
            Path to the YAML configuration file used to configure the training
            run. Defaults to ``configs/config_NN.yaml``.
    """
    parser = argparse.ArgumentParser(description="Train ResNet50 with YAML config.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(Path("configs") / "config_NN.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (used for sweeps too).",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        run = None
        if args.wandb:
            run = wandb.init(
                project="week3-ResNet2",
                entity="xavipba-universitat-aut-noma-de-barcelona"
            )

    in_sweep = args.wandb and (os.getenv("WANDB_SWEEP_ID") is not None)
    if in_sweep:
        sweep_cfg = dict(wandb.config)
        if "freeze_preset" in sweep_cfg:
            cfg["preset"] = sweep_cfg["freeze_preset"]
        if "batch_size" in sweep_cfg:
            cfg["pipeline"]["batch_size"] = list(sweep_cfg["batch_size"])
        if "epochs" in sweep_cfg:
            cfg["pipeline"]["num_epochs"] = int(sweep_cfg["epochs"])
        if "lr" in sweep_cfg:
            cfg["optimizer"]["lr"] = float(sweep_cfg["lr"])
        if "weight_decay" in sweep_cfg:
            cfg["optimizer"]["weight_decay"] = float(sweep_cfg["weight_decay"])
        if "momentum" in sweep_cfg:
            cfg["optimizer"]["momentum"] = float(sweep_cfg["momentum"])
        if "nesterov" in sweep_cfg:
            cfg["optimizer"]["nesterov"] = bool(sweep_cfg["nesterov"])
        if "optimizer" in sweep_cfg:
            cfg["optimizer"]["type"] = str(sweep_cfg["optimizer"])
        if "data_augm" in sweep_cfg:
            cfg["data_augm"] = bool(sweep_cfg["data_augm"])
        if "scheduler" in sweep_cfg:
            cfg["scheduler"]["type"] = str(sweep_cfg["scheduler"])

    optimizer_cfg = cfg["optimizer"]
    pipeline_cfg = cfg["pipeline"]
    scheduler_cfg = cfg["scheduler"]
    paths = cfg["paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_data_loaders(
        paths["dataset"],
        cfg["data_augm"],
        pipeline_cfg["batch_size"],
    )

    model = ResNet50(num_classes=8, feature_extraction=False).to(device)
    
    preset_name = cfg["preset"]
    preset = FREEZE_PRESETS[preset_name]
    apply_freeze_preset(model.backbone, preset, freeze_bn_stats=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    optimizer = make_optimizer(
        model,
        optimizer_type=str(optimizer_cfg["type"]),
        lr=float(optimizer_cfg["lr"]),
        wd=float(optimizer_cfg["weight_decay"]),
        momentum=float(optimizer_cfg["momentum"]),
        nesterov=bool(optimizer_cfg["nesterov"]),
    )

    scheduler = make_scheduler(
        optimizer,
        scheduler_cfg["type"],
        num_epochs=int(pipeline_cfg["num_epochs"]),
        steps_per_epoch=len(train_loader)
    )

    for epoch in tqdm.tqdm(range(int(pipeline_cfg["num_epochs"])), desc="TRAINING"):
        train_loss, train_accuracy = train(
            model, train_loader, criterion, optimizer, device, epoch,
            adv_training=False
        )
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "test/loss": test_loss,
                "test/accuracy": test_accuracy,
                "lr": optimizer.param_groups[0]["lr"]
            })

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

    save_dir = Path(paths["models_out_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "best_layers.pth")
    if args.wandb:
        wandb.finish()