import yaml
import json
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms.v2  as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from Week2.main import train, test
import Week2.models as models


PROJECT_ROOT = Path.cwd()
WEEK_2_ROOT = PROJECT_ROOT / "Week2"

torch.manual_seed(42)


def run_experiment(cfg, shape, train_loader, test_loader, device, num_epochs=5):
    layers = cfg["layers"].copy()
    C, H, W = shape

    # Insert input layer
    first_layer = [C*H*W, layers[0][0]]
    layers.insert(0, first_layer)

    # Build model
    model = models.DynamicMLP(
        layer_sizes=[tuple(x) for x in layers],
        activation=cfg["activation"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

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

    return {
        "final_train_acc": train_accuracies[-1],
        "final_test_acc": test_accuracies[-1],
        "train_curve": train_accuracies,
        "test_curve": test_accuracies,
        "model": model
    }


def grid_search(experiments, train_loader, test_loader, num_epochs):
    images, _ = next(iter(train_loader))
    shape =  images[0].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for i, exp_cfg in enumerate(experiments):
        print(f"Running experiment {i+1}/{len(experiments)}")
        result = run_experiment(exp_cfg, shape, train_loader, test_loader, device, num_epochs)
        result["config"] = exp_cfg  # store what config generated this result
        results.append(result)
      
    best_result = max(results, key=lambda x: x["final_test_acc"])
    best_model = best_result["model"]
    best_config = best_result["config"]
    cfg["best"] = best_config

    print("Best model:")
    print(best_result["config"])
    print("Accuracy:", best_result["final_test_acc"])

    with open("grid_results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("Saved results in file grid_results.json")

    # save best model's weights and config
    save_dir = WEEK_2_ROOT / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), save_dir / "best_model.pth")

    with open(WEEK_2_ROOT / "configs" / "NN1.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

if __name__=="__main__":
    with open(WEEK_2_ROOT / "configs" / "NN1.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        experiments = cfg["experiments"]

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    data_train = ImageFolder(PROJECT_ROOT / "data" / "places_reduced" / "train", transform=transformation)
    data_test = ImageFolder(PROJECT_ROOT / "data" / "places_reduced" / "val", transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

    grid_search(experiments, train_loader, test_loader, num_epochs=10)

