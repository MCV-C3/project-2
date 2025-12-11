import numpy as np
import yaml
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torchvision.transforms.v2  as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import Week2.models as models

PROJECT_ROOT = Path.cwd()
WEEK_2_ROOT = PROJECT_ROOT / "Week2"


def reconstruct_best_model(train_loader):
    """
    Reconstruct the best model.
    """
    with open(WEEK_2_ROOT / "configs" / "NN1.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    best_cfg = cfg.get("best", None)
    if best_cfg is None:
        print("No 'best' configuration found in YAML. Run grid search first.")
        return None
    
    images, _ = next(iter(train_loader))
    C, H, W = images[0].shape

    layers = best_cfg["layers"].copy()

    # Ensure first layer exactly as it was in best modelx
    first_layer = [C*H*W, layers[0][0]]
    layers.insert(0, first_layer)
    layer_sizes = [tuple(l) for l in layers]

    model = models.DynamicMLP(
        layer_sizes=layer_sizes,
        activation=best_cfg["activation"]
    ).to(device)

    weights_path = WEEK_2_ROOT / "models" / "best_model.pth"
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded best model from: {weights_path}")
    except FileNotFoundError:
        print("No saved model found.")
        print("You must run grid_search(...) to train and save a best model first.")
        return None
    except Exception as e:
        print("Error loading model weights:", e)
        return None
    model.eval()

    return model


def extract_embeddings(model, dataloader, layer_idx, device):
    """
    Extract embeddings from the specified layer of DynamicMLP.
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)

            # Full list of layer outputs
            layer_outputs = model(imgs, return_features=True)

            # Select the layer we want
            feats = layer_outputs[layer_idx]      # shape (batch, dim)

            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    return X, y


def mlp_svm_experiment(model, train_loader, test_loader, device, layer_idx=-2):
    """
    Train an SVM on embeddings extracted from a given MLP layer.
    """
    print("Extracting embeddings")
    X_train, y_train = extract_embeddings(model, train_loader, layer_idx, device)
    X_test, y_test = extract_embeddings(model, test_loader, layer_idx, device)

    print("Training SVM")
    svm = SVC(kernel="linear", C=1.0)
    svm.fit(X_train, y_train)

    print("Evaluating SVM")
    preds = svm.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"SVM accuracy using layer {layer_idx}: {acc:.4f}")
    return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.Resize(size=(224, 224)),
                                ])
    
    data_train = ImageFolder(PROJECT_ROOT / "data" / "places_reduced" / "train", transform=transformation)
    data_test = ImageFolder(PROJECT_ROOT / "data" / "places_reduced" / "val", transform=transformation) 

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)
    

    model = reconstruct_best_model()
    layer_to_use = -2 # use last hidden layer of the MLP

    acc = mlp_svm_experiment(model, train_loader, test_loader, device, layer_idx=layer_to_use)
