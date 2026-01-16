"""
This file contains utils for data processing and evaluation.
"""
import os
import copy
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as F
from pathlib import Path

from src.model import switch_model_to_deploy



# -------------
# Dataloaders
# -------------
def make_data_loaders(dataset_path, augmentation, batch_size):
    """
    Create train and test DataLoaders with an extended augmentation pipeline.
    """
    dataset_path = os.path.expanduser(dataset_path)
    # Define ImageNet stats
    imagenet_norm = F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augmentation:
        train_transform = F.Compose([
            F.RandomResizedCrop(224, scale=(0.8, 1.0)),
            F.RandomHorizontalFlip(0.5),
            F.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            F.RandomGrayscale(p=0.05),
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            imagenet_norm
        ])
    else:
        train_transform = F.Compose([
            F.Resize(256),
            F.CenterCrop(224),
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            imagenet_norm,
        ])

    test_transform = F.Compose([
        F.Resize(256),
        F.CenterCrop(224),
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        imagenet_norm,
    ])
    
    data_train = ImageFolder(os.path.join(dataset_path, "train"), transform=train_transform)
    data_test = ImageFolder(os.path.join(dataset_path, "test"), transform=test_transform) 

    train_loader = DataLoader(data_train, batch_size=batch_size[0], pin_memory=True, shuffle=True, num_workers=4)
    test_loader = DataLoader(data_test, batch_size=batch_size[1], pin_memory=True, shuffle=False, num_workers=4)

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


