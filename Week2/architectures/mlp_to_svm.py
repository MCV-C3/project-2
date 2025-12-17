import numpy as np
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torchvision.transforms.v2  as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import Week2.models as models
from sklearn.metrics import accuracy_score, recall_score
import csv
import os
PROJECT_ROOT = Path.cwd()
WEEK_2_ROOT = PROJECT_ROOT / "Week2"


def reconstruct_best_model(train_loader, model_name, cfg):
    """
    Reconstruct the best model.
    """
    cfg = cfg[0]
    images, _ = next(iter(train_loader))
    C, H, W = images[0].shape

    layers = cfg['config']["layers"].copy()

    # Ensure first layer exactly as it was in best modelx
    first_layer = [C*H*W, layers[0][0]]
    layers.insert(0, first_layer)
    layer_sizes = [tuple(l) for l in layers]

    model = models.DynamicMLP(
        layer_sizes=layer_sizes,
        activation=cfg['config']["activation"],
        dropout=cfg['config']['dropout']
    ).to(device)

    weights_path = WEEK_2_ROOT / "models" / f"{model_name}_best_model.pth"
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
        for imgs, labels in tqdm(dataloader, total=len(dataloader), desc='Extract Embeddings'):
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


def mlp_svm_experiment(model, train_loader, test_loader, device, model_name):
    """
    Train an SVM on embeddings extracted from a given MLP layer.
    Prints training and test accuracy + recall.
    """
    print("Extracting embeddings")
    #for i in range(0,5,1):
    i=4
    X_train, y_train = extract_embeddings(model, train_loader, i, device)
    X_test, y_test = extract_embeddings(model, test_loader, i, device)
    kernels = ["linear", "rbf"]
    cs = [1e-2, 1e-1, 1, 10, 100]
    gs = [1e-3, 1e-2, 1e-1, 1]
    result_file = r"C:\Users\maiol\Desktop\Master\C3\project-2\Week2\results\svm\results_parameters.csv"
    file_exists = os.path.isfile(result_file)
    
    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "model_name",
                "layer_idx",
                "kernel",
                "C",
                "gamma",
                "train_accuracy",
                "train_recall",
                "test_accuracy",
                "test_recall",
            ])
        for kernel in kernels:
            for c in cs:
                
                if kernel == "linear":

                    print(f"Training SVM | kernel={kernel}, C={c}")

                    svm = SVC(kernel="linear", C=c)
                    svm.fit(X_train, y_train)

                    # Train metrics
                    train_preds = svm.predict(X_train)
                    train_acc = accuracy_score(y_train, train_preds)
                    train_recall = recall_score(y_train, train_preds, average="macro")

                    # Test metrics
                    test_preds = svm.predict(X_test)
                    test_acc = accuracy_score(y_test, test_preds)
                    test_recall = recall_score(y_test, test_preds, average="macro")

                    writer.writerow([
                        model_name,
                        i,
                        kernel,
                        c,
                        None,
                        train_acc,
                        train_recall,
                        test_acc,
                        test_recall,
                    ])

                # -------------------------
                # RBF kernel
                # -------------------------
                else:
                    for g in gs:

                        print(f"Training SVM | kernel={kernel}, C={c}, gamma={g}")

                        svm = SVC(kernel="rbf", C=c, gamma=g)
                        svm.fit(X_train, y_train)

                        # Train metrics
                        train_preds = svm.predict(X_train)
                        train_acc = accuracy_score(y_train, train_preds)
                        train_recall = recall_score(y_train, train_preds, average="macro")

                        # Test metrics
                        test_preds = svm.predict(X_test)
                        test_acc = accuracy_score(y_test, test_preds)
                        test_recall = recall_score(y_test, test_preds, average="macro")

                        writer.writerow([
                            model_name,
                            i,
                            kernel,
                            c,
                            g,
                            train_acc,
                            train_recall,
                            test_acc,
                            test_recall,
                        ])
            print(f"SVM results using layer {i}:")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Train Recall:   {train_recall:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  Test Recall:    {test_recall:.4f}")

    return {
        "train_acc": train_acc,
        "train_recall": train_recall,
        "test_acc": test_acc,
        "test_recall": test_recall
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_names = ["14122025_2152","14122025_2153","14122025_2219","14122025_2117","15122025_1133","15122025_1142","15122025_1208"]
    for model_name in model_names:
        with open(WEEK_2_ROOT / "results" / f"{model_name}.json", "r") as f:
            cfg = json.load(f)
        augmentation = cfg[0]['config']['data_agmentation']

        if augmentation:
                train_transform = F.Compose([
                    F.ToImage(),
                    F.ToDtype(torch.float32, scale=True),
                    F.Resize((cfg[0]['config']['img_size'][0][0], cfg[0]['config']['img_size'][0][1])),
                    F.RandomHorizontalFlip(p=0.5),
                    F.ColorJitter(brightness=0.2, contrast=0.2),
                ])

                test_transform = F.Compose([
                    F.ToImage(),
                    F.ToDtype(torch.float32, scale=True),
                    F.Resize((cfg[0]['config']['img_size'][0][0], cfg[0]['config']['img_size'][0][1])),
                ])
                data_train = ImageFolder(r'c:\Users\maiol\Desktop\Master\C3\places_reduced\train', transform=train_transform)
                data_test = ImageFolder(r'c:\Users\maiol\Desktop\Master\C3\places_reduced\val', transform=test_transform)
        else:
                transformation  = F.Compose([
                                            F.ToImage(),
                                            F.ToDtype(torch.float32, scale=True),
                                            F.Resize(size=(cfg[0]['config']['img_size'][0][0], cfg[0]['config']['img_size'][0][1])),
                                        ])
            
                data_train = ImageFolder(r'c:\Users\maiol\Desktop\Master\C3\places_reduced\train', transform=transformation)
                data_test = ImageFolder(r'c:\Users\maiol\Desktop\Master\C3\places_reduced\val', transform=transformation)    

        train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
        test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)
        

        model = reconstruct_best_model(train_loader, model_name, cfg)
        #layer_to_use = -2 # use last hidden layer of the MLP

        acc = mlp_svm_experiment(model, train_loader, test_loader, device, model_name)
