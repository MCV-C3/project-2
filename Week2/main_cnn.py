from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as F
import tqdm
import wandb
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import SimpleCNN

PROJECT_ROOT = Path.cwd()
WEEK_2_ROOT = PROJECT_ROOT / "Week2"

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy





if __name__ == "__main__":

    torch.manual_seed(42)

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])

    data_train = ImageFolder(PROJECT_ROOT / "Week1" / "places_reduced" / "train", transform=transformation)
    data_test = ImageFolder(PROJECT_ROOT / "Week1" / "places_reduced" / "val", transform=transformation)

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(input_channels=C, num_classes=11)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 40

    wandb.init(
        project="week2-simple-cnn",
        config={
            "model": "SimpleCNN",
            "learning_rate": 0.001,
            "epochs": num_epochs,
            "batch_size": 256,
            "optimizer": "Adam",
            "input_size": (C, H, W),
        }
    )

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "test/loss": test_loss,
            "test/accuracy": test_accuracy
        })

    torch.save(model.state_dict(), "simple_cnn_model.pth")
    print("Model saved as simple_cnn_model.pth")

    wandb.finish()
