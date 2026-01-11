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


def train_epoch(model, dataloader, criterion, optimizer, device, patch_size, stride=None, aggregation='mean'):

    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    if stride is None:
        stride = patch_size

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size, channels, height, width = inputs.shape

        patches = []
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch = inputs[:, :, i:i+patch_size, j:j+patch_size]
                patches.append(patch)

        patches = torch.stack(patches, dim=1)  # (batch_size, num_patches, channels, patch_size, patch_size)
        num_patches = patches.shape[1]
        patches = patches.view(-1, channels, patch_size, patch_size)  # (batch_size * num_patches, channels, patch_size, patch_size)

        outputs = model(patches)  # (batch_size * num_patches, num_classes)

        num_classes = outputs.shape[1]
        outputs = outputs.view(batch_size, num_patches, num_classes)

        if aggregation == 'mean':
            aggregated_outputs = outputs.mean(dim=1)  # (batch_size, num_classes)
        elif aggregation == 'max':
            aggregated_outputs = outputs.max(dim=1)[0]  # (batch_size, num_classes)
        elif aggregation in ['vote', 'voting']:
            aggregated_outputs = torch.softmax(outputs, dim=2).mean(dim=1)  # (batch_size, num_classes)
        else:
            raise ValueError(f"Unknown or non-differentiable aggregation method for training: {aggregation}")

        loss = criterion(aggregated_outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size
        _, predicted = aggregated_outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, patch_size, stride=None, aggregation='mean'):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    all_predictions = []
    all_labels = []

    if stride is None:
        stride = patch_size

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size, channels, height, width = inputs.shape

            patches = []
            for i in range(0, height - patch_size + 1, stride):
                for j in range(0, width - patch_size + 1, stride):
                    patch = inputs[:, :, i:i+patch_size, j:j+patch_size]
                    patches.append(patch)

            patches = torch.stack(patches, dim=1)
            num_patches = patches.shape[1]
            patches = patches.view(-1, channels, patch_size, patch_size)

            outputs = model(patches)  # (batch_size * num_patches, num_classes)

            num_classes = outputs.shape[1]
            outputs = outputs.view(batch_size, num_patches, num_classes)

            if aggregation == 'mean':
                aggregated_outputs = outputs.mean(dim=1)  # (batch_size, num_classes)
            elif aggregation in ['vote', 'voting']:
                patch_predictions = outputs.argmax(dim=2)  # (batch_size, num_patches)
                aggregated_outputs = torch.zeros(batch_size, num_classes, device=device)
                for b in range(batch_size):
                    for pred in patch_predictions[b]:
                        aggregated_outputs[b, pred] += 1
            elif aggregation == 'max':
                aggregated_outputs = outputs.max(dim=1)[0]  # (batch_size, num_classes)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            loss = criterion(aggregated_outputs, labels)

            val_loss += loss.item() * batch_size
            _, predicted = aggregated_outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def reconstruct_model(config: Dict, train_loader: DataLoader, device: torch.device, patch_size: int):
    import yaml

    WEEK_2_ROOT = Path(__file__).parent.parent
    PROJECT_ROOT = WEEK_2_ROOT.parent

    nn_config_path = WEEK_2_ROOT / "configs" / config['model']['config_file']
    with open(nn_config_path, 'r') as f:
        nn_cfg = yaml.safe_load(f)

    exp_idx = config['model'].get('experiment_idx', 0)
    exp_cfg = nn_cfg['experiments'][exp_idx]
    model_cfg = exp_cfg['model']

    images, _ = next(iter(train_loader))
    C = images[0].shape[0]  # Number of channels
    H, W = patch_size, patch_size

    layers = model_cfg["layers"].copy()
    first_layer = [C*H*W, layers[0][0]]
    layers.insert(0, first_layer)
    layer_sizes = [tuple(l) for l in layers]

    model = models.DynamicMLP(
        layer_sizes=layer_sizes,
        activation=model_cfg["activation"][0] if isinstance(model_cfg["activation"], list) else model_cfg["activation"],
        dropout=model_cfg["dropout"][0] if isinstance(model_cfg["dropout"], list) else model_cfg.get("dropout", 0.0)
    ).to(device)

    print(f"Model architecture:")
    print(f"  Input: {C}x{H}x{W} = {C*H*W} features")
    print(f"  Layers: {layer_sizes}")

    model.eval()
    return model, exp_cfg
