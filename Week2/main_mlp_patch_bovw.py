import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchvision.transforms.v2 as F
import tqdm
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from architectures.mlp_patch import reconstruct_model

WEEK_2_ROOT = Path(__file__).parent
PROJECT_ROOT = WEEK_2_ROOT.parent


def load_config(config_path: Path) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_patch_features(model, dataloader, device, layer_id, patch_size, stride=None):
    model.eval()
    feats, labels_list = [], []

    if stride is None:
        stride = patch_size

    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(dataloader, desc="Extracting patch features"):
            imgs = imgs.to(device)
            batch_size, channels, height, width = imgs.shape

            patches = []
            for i in range(0, height - patch_size + 1, stride):
                for j in range(0, width - patch_size + 1, stride):
                    patch = imgs[:, :, i:i+patch_size, j:j+patch_size]
                    patches.append(patch)

            patches = torch.stack(patches, dim=1)
            num_patches = patches.shape[1]
            patches = patches.view(-1, channels, patch_size, patch_size)

            all_features = model(patches, return_features=True)

            relu_outputs = []
            for i, layer in enumerate(model.model):
                if isinstance(layer, torch.nn.ReLU):
                    relu_outputs.append(i)

            if layer_id >= len(relu_outputs):
                raise ValueError(f"layer_id {layer_id} is out of range. Model has {len(relu_outputs)} hidden layers.")

            feature_idx = relu_outputs[layer_id]
            patch_features = all_features[feature_idx]

            feature_dim = patch_features.shape[1]
            patch_features = patch_features.view(batch_size, num_patches, feature_dim)

            feats.append(patch_features.cpu().numpy())
            labels_list.append(labels.numpy())

    feats = np.vstack(feats)
    labels = np.hstack(labels_list)
    return feats, labels


def build_bovw_codebook(train_features, codebook_size, batch_size=1000, max_iter=100):
    num_images, num_patches, feature_dim = train_features.shape

    all_patches = train_features.reshape(-1, feature_dim)
    print(f"Building codebook from {all_patches.shape[0]} patches with {feature_dim} dimensions")

    print(f"Running K-means with {codebook_size} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=codebook_size,
        random_state=42,
        batch_size=batch_size,
        max_iter=max_iter,
        verbose=1
    )
    kmeans.fit(all_patches)

    print("Codebook built successfully!")
    return kmeans


def encode_bovw(features, kmeans, codebook_size):
    num_images = features.shape[0]
    bovw_features = np.zeros((num_images, codebook_size))

    for i in tqdm.tqdm(range(num_images), desc="Encoding BoVW"):
        patches = features[i]

        assignments = kmeans.predict(patches)

        hist, _ = np.histogram(assignments, bins=np.arange(codebook_size + 1))

        bovw_features[i] = hist / (hist.sum() + 1e-6)

    return bovw_features


def save_results_to_csv(csv_path: Path, config: Dict, accuracy: float, codebook_size: int):
    row_data = {
        'experiment_name': config.get('name', 'unnamed'),
        'model_path': config['model']['path'],
        'patch_size': config['patches']['size'],
        'stride': config['patches']['stride'],
        'layer_id': config['bovw']['layer_id'],
        'codebook_size': codebook_size,
        'svm_C': config['svm']['C'],
        'test_accuracy': accuracy,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

    print(f"Results appended to CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='MLP Patch-Based BoVW')
    parser.add_argument(
        '--config',
        type=str,
        default='patch_bovw.yaml',
        help='Path to config file (default: patch_bovw.yaml)'
    )

    args = parser.parse_args()

    config_path = WEEK_2_ROOT / "configs" / args.config
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    img_size = config['data']['image_size']
    transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=(img_size, img_size)),
    ])

    data_dir = PROJECT_ROOT / "data" / "places_reduced"
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / "Week1" / "places_reduced"

    print(f"Loading data from: {data_dir}")
    data_train = ImageFolder(data_dir / "train", transform=transformation)
    data_test = ImageFolder(data_dir / "val", transform=transformation)

    train_loader = DataLoader(
        data_train,
        batch_size=config['data']['batch_size'],
        pin_memory=True,
        shuffle=False,  # No need to shuffle for feature extraction
        num_workers=config['data'].get('num_workers', 8)
    )
    test_loader = DataLoader(
        data_test,
        batch_size=config['data']['batch_size'],
        pin_memory=True,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 8)
    )

    print(f"Training images: {len(data_train)}")
    print(f"Test images: {len(data_test)}")
    print(f"Image size: {img_size}x{img_size}")

    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)

    model_path = WEEK_2_ROOT / config['model']['path']
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    patch_size = config['patches']['size']
    dummy_loader = DataLoader(data_train, batch_size=1, shuffle=False)
    model, model_cfg = reconstruct_model(config, dummy_loader, device, patch_size)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from: {model_path}")

    print("\n" + "="*60)
    print("EXTRACTING PATCH FEATURES (DENSE DESCRIPTORS)")
    print("="*60)

    layer_id = config['bovw']['layer_id']
    stride = config['patches']['stride']

    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Stride: {stride}")
    print(f"Layer ID for features: {layer_id}")

    num_patches_h = (img_size - patch_size) // stride + 1
    num_patches_w = (img_size - patch_size) // stride + 1
    total_patches = num_patches_h * num_patches_w
    print(f"Patches per image: {total_patches} ({num_patches_h}x{num_patches_w})")

    print("\nExtracting features from training set...")
    train_feats, train_labels = extract_patch_features(
        model, train_loader, device, layer_id, patch_size, stride
    )

    print("\nExtracting features from test set...")
    test_feats, test_labels = extract_patch_features(
        model, test_loader, device, layer_id, patch_size, stride
    )

    print(f"\nTrain features shape: {train_feats.shape}")
    print(f"Test features shape: {test_feats.shape}")

    print("\n" + "="*60)
    print("BUILDING BAG OF VISUAL WORDS")
    print("="*60)

    codebook_size = config['bovw']['codebook_size']
    kmeans_batch_size = config['bovw'].get('kmeans_batch_size', 1000)
    kmeans_max_iter = config['bovw'].get('kmeans_max_iter', 100)

    print(f"Codebook size: {codebook_size}")

    kmeans = build_bovw_codebook(
        train_feats,
        codebook_size=codebook_size,
        batch_size=kmeans_batch_size,
        max_iter=kmeans_max_iter
    )

    print("\nEncoding training images as BoVW...")
    train_bovw = encode_bovw(train_feats, kmeans, codebook_size)

    print("\nEncoding test images as BoVW...")
    test_bovw = encode_bovw(test_feats, kmeans, codebook_size)

    print(f"\nTrain BoVW shape: {train_bovw.shape}")
    print(f"Test BoVW shape: {test_bovw.shape}")

    print("\n" + "="*60)
    print("TRAINING SVM ON BOVW REPRESENTATIONS")
    print("="*60)

    print("Standardizing features...")
    scaler = StandardScaler()
    train_bovw_scaled = scaler.fit_transform(train_bovw)
    test_bovw_scaled = scaler.transform(test_bovw)

    svm_C = config['svm']['C']
    svm_max_iter = config['svm'].get('max_iter', 10000)

    print(f"Training LinearSVC (C={svm_C}, max_iter={svm_max_iter})...")
    svm = LinearSVC(C=1.0, max_iter=svm_max_iter, random_state=42, verbose=1, kernel="rbf", gamma="0.01")
    svm.fit(train_bovw_scaled, train_labels)

    train_acc = svm.score(train_bovw_scaled, train_labels)
    test_acc = svm.score(test_bovw_scaled, test_labels)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("="*60)

    csv_filename = config.get('output', {}).get('results_csv', None)
    if csv_filename:
        results_dir = WEEK_2_ROOT / config['output'].get('results_dir', 'results')
        results_dir.mkdir(exist_ok=True)
        csv_path = results_dir / csv_filename
        save_results_to_csv(csv_path, config, test_acc, codebook_size)

    print("\nBoVW evaluation completed!")


if __name__ == "__main__":
    main()
