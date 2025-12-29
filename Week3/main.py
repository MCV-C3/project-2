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
"none_frozen":   {"stem": False, "layer1": False, "layer2": False, "layer3": False, "layer4": False, "fc": False},
}

# Train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


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
        help="Path to YAML config file (default: configs\\config1.yaml).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    #Parse config
    args = parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    #Organize parsed config
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        optimizer_cfg = cfg['optimizer']
        augmentation = cfg['data_augm']
        freeze_preset_name = cfg["preset"]
        pipeline_cfg = cfg["pipeline"]
        scheduler_cfg = cfg["scheduler"]
        paths = cfg['paths']

    torch.manual_seed(42)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M")

    #Create loaders
    train_loader, test_loader = make_data_loaders(paths['dataset'], augmentation, pipeline_cfg["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device used', device)

    num_epochs = pipeline_cfg["num_epochs"]

    #Create model and freeze
    model = ResNet50(num_classes=8, feature_extraction=False)
    preset = FREEZE_PRESETS[freeze_preset_name]
    model = model.to(device)
    apply_freeze_preset(model.backbone, preset, freeze_bn_stats=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(
        model,
        optimizer_type=str(optimizer_cfg["type"]),
        lr=float(optimizer_cfg["lr"]),
        wd=float(optimizer_cfg["weight_decay"]),
        momentum=float(optimizer_cfg["momentum"]),
        nesterov=bool(optimizer_cfg["nesterov"]),
    )
    scheduler = make_scheduler(optimizer, scheduler_cfg["type"], num_epochs=num_epochs)
    

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    
    wandb_run = wandb.init(
        project="week3-ResNet",
        name=f"{timestamp}-{freeze_preset_name}",
        group="grid-search",
        reinit=True,
        config={
            "freeze_preset": freeze_preset_name,
            **preset,
            "lr": float(optimizer_cfg['lr']),
            "weight_decay": float(optimizer_cfg['weight_decay']),
            "momentum": float(optimizer_cfg['momentum']),
            "optimizer": optimizer_cfg['type'],
            "batch_size": train_loader.batch_size,
            "epochs": num_epochs,
            "data_augm": augmentation,
            "scheduler": scheduler_cfg["type"]
        }
    )

    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "test/loss": test_loss,
            "test/accuracy": test_accuracy
        })
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        if scheduler is not None:
            scheduler.step()
    wandb.finish()

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, 
                 "loss", f"{timestamp}_{freeze_preset_name}", paths["plots_out_path"])
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, 
                 "accuracy", f"{timestamp}_{freeze_preset_name}", paths["plots_out_path"])

    save_dir = Path(paths["models_out_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / f"{timestamp}_{freeze_preset_name}.pth")