import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
import models


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.shape[0]

        # Process full images through CNN+MLP
        outputs = model(inputs)  # (batch_size, num_classes)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.shape[0]

            # Process full images through CNN+MLP
            outputs = model(inputs)  # (batch_size, num_classes)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            val_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def reconstruct_model(config: Dict, train_loader: DataLoader, device: torch.device, img_size: Tuple[int, int]):
    model_cfg = config['model']

    images, _ = next(iter(train_loader))
    C = images[0].shape[0]  
    H, W = img_size 

    cnn_layers = model_cfg.get("cnn_layers", [])

    cnn_h, cnn_w = H, W
    out_channels = C
    for layer_cfg in cnn_layers:
        out_channels = layer_cfg['out_channels']
        pool_stride = layer_cfg.get('pool_stride', 2)
        cnn_h = cnn_h // pool_stride
        cnn_w = cnn_w // pool_stride

    cnn_output_size = out_channels * cnn_h * cnn_w

    mlp_layers = model_cfg["layers"].copy()
    first_mlp_layer = [cnn_output_size, mlp_layers[0][0]]
    mlp_layers.insert(0, first_mlp_layer)
    mlp_layer_sizes = [tuple(l) for l in mlp_layers]

    model = models.DynamicMLPCNN(
        cnn_layers=cnn_layers,
        mlp_layer_sizes=mlp_layer_sizes,
        activation=model_cfg["activation"][0] if isinstance(model_cfg["activation"], list) else model_cfg["activation"],
        dropout=model_cfg["dropout"][0] if isinstance(model_cfg["dropout"], list) else model_cfg.get("dropout", 0.0),
        input_channels=C
    ).to(device)

    print(f"Model architecture:")
    print(f"  Input: {C}x{H}x{W} full images")
    print(f"  CNN Layers: {cnn_layers}")
    print(f"  CNN Output: {out_channels}x{cnn_h}x{cnn_w} = {cnn_output_size} features")
    print(f"  MLP Layers: {mlp_layer_sizes}")

    model.eval()
    return model, config
