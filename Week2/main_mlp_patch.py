import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torchvision.transforms.v2 as F
import wandb
import yaml
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from architectures.mlp_patch import (
    reconstruct_model,
    train_epoch,
    validate_epoch,
)

WEEK_2_ROOT = Path(__file__).parent
PROJECT_ROOT = WEEK_2_ROOT.parent


def load_config(config_path: Path) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_datasets(config: Dict):
    nn_config_path = WEEK_2_ROOT / "configs" / config['model']['config_file']
    with open(nn_config_path, 'r') as f:
        nn_cfg = yaml.safe_load(f)

    img_size = tuple(nn_cfg.get('resize', [[128,128]])[0])
    augmentation = nn_cfg.get('data_augm', True)

    if augmentation:
        train_transform = F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize((img_size[0], img_size[1])),
            F.RandomHorizontalFlip(p=0.5),
            F.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        val_transform = F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize((img_size[0], img_size[1])),
        ])
    else:
        train_transform = val_transform = F.Compose([
            F.ToImage(),
            F.ToDtype(torch.float32, scale=True),
            F.Resize(size=(img_size[0], img_size[1])),
        ])

    data_dir = PROJECT_ROOT / "data" / "places_reduced"
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / "Week1" / "places_reduced"

    print(f"Loading data from: {data_dir}")
    data_train = ImageFolder(data_dir / "train", transform=train_transform)
    data_val = ImageFolder(data_dir / "val", transform=val_transform)

    return data_train, data_val, img_size


def evaluate_predictions(predictions, labels, prefix=""):
    return {
        f'{prefix}accuracy': accuracy_score(labels, predictions),
        f'{prefix}recall': recall_score(labels, predictions, average='macro'),
        f'{prefix}precision': precision_score(labels, predictions, average='macro'),
        f'{prefix}f1': f1_score(labels, predictions, average='macro'),
    }


def save_results_to_csv(csv_path: Path, config: Dict, final_train_acc: float, final_val_acc: float):
    row_data = {
        'experiment_name': config.get('name', 'unnamed'),
        'patch_size': config['patches']['size'],
        'stride': config['patches']['stride'],
        'num_epochs': config['training']['num_epochs'],
        'batch_size': config['training'].get('batch_size', 32),
        'lr': config['training']['lr'],
        'weight_decay': config['training'].get('weight_decay', 0.0),
        'train_aggregation': config['training'].get('aggregation', 'mean'),
        'val_aggregation': config['validation'].get('aggregation', 'mean'),
        'nn_config_file': config['model']['config_file'],
        'experiment_idx': config['model'].get('experiment_idx', 0),
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    file_exists = csv_path.exists()

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

    print(f"Results appended to CSV: {csv_path}")


def train(config: Dict, device: torch.device):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n" + "="*60)
    print("PREPARING DATASETS")
    print("="*60)
    data_train, data_val, img_size = prepare_datasets(config)

    patch_size = config['patches']['size']
    stride = config['patches']['stride']

    print(f"Image size: {img_size[0]}x{img_size[1]}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Stride: {stride}")

    num_patches_h = (img_size[0] - patch_size) // stride + 1
    num_patches_w = (img_size[1] - patch_size) // stride + 1
    total_patches = num_patches_h * num_patches_w
    print(f"Patches per image: {total_patches} ({num_patches_h}x{num_patches_w})")

    print("\n" + "="*60)
    print("CREATING DATALOADERS")
    print("="*60)
    train_batch_size = config['training'].get('batch_size', 32)
    val_batch_size = config['validation'].get('batch_size', 64)

    train_loader = DataLoader(
        data_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        data_val,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )

    print(f"Training: {len(data_train)} images, {len(train_loader)} batches")
    print(f"Validation: {len(data_val)} images, {len(val_loader)} batches")
    print(f"Train batch size: {train_batch_size} images (= ~{train_batch_size * total_patches} patches per batch)")
    print(f"Val batch size: {val_batch_size} images")

    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)

    dummy_loader = DataLoader(data_train, batch_size=1, shuffle=False)
    model, model_cfg = reconstruct_model(config, dummy_loader, device, patch_size)

    num_epochs = config['training']['num_epochs']
    lr = config['training']['lr']
    weight_decay = config['training'].get('weight_decay', 0.0)
    train_aggregation = config['training'].get('aggregation', 'mean')
    val_aggregation = config['validation'].get('aggregation', 'mean')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
    print(f"Criterion: CrossEntropyLoss")
    print(f"Epochs: {num_epochs}")
    print(f"Training aggregation: {train_aggregation}")
    print(f"Validation aggregation: {val_aggregation}")

    use_wandb = config.get('wandb', {}).get('enabled', True)
    if use_wandb:
        exp_name = config.get('name', f"patch_training_{timestamp}")
        wandb.init(
            project=config.get('wandb', {}).get('project', 'week2-patch-training'),
            name=exp_name,
            config={
                'experiment_name': exp_name,
                'patch_size': patch_size,
                'stride': stride,
                'img_size': img_size,
                'total_patches_per_image': total_patches,
                'num_epochs': num_epochs,
                'train_batch_size': train_batch_size,
                'val_batch_size': val_batch_size,
                'effective_patch_batch_size': train_batch_size * total_patches,
                'lr': lr,
                'weight_decay': weight_decay,
                'train_aggregation': train_aggregation,
                'val_aggregation': val_aggregation,
                'nn_config_file': config['model']['config_file'],
                'experiment_idx': config['model'].get('experiment_idx', 0),
            }
        )

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            patch_size=patch_size,
            stride=stride,
            aggregation=train_aggregation
        )

        val_loss, val_acc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device,
            patch_size=patch_size,
            stride=stride,
            aggregation=val_aggregation
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  → New best validation accuracy: {val_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    val_loss, val_acc, val_preds, val_labels = validate_epoch(
        model, val_loader, criterion, device,
        patch_size=patch_size,
        stride=stride,
        aggregation=val_aggregation
    )

    metrics = evaluate_predictions(val_preds, val_labels, 'val_')

    print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Validation Recall: {metrics['val_recall']:.4f}")
    print(f"Validation Precision: {metrics['val_precision']:.4f}")
    print(f"Validation F1: {metrics['val_f1']:.4f}")

    if use_wandb:
        wandb.log({
            'final/val_accuracy': metrics['val_accuracy'],
            'final/val_recall': metrics['val_recall'],
            'final/val_precision': metrics['val_precision'],
            'final/val_f1': metrics['val_f1'],
        })
        wandb.finish()

    csv_filename = config.get('output', {}).get('results_csv', None)
    if csv_filename:
        results_dir = WEEK_2_ROOT / config['output'].get('results_dir', 'results')
        results_dir.mkdir(exist_ok=True)
        csv_path = results_dir / csv_filename

        final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0.0
        final_val_acc = metrics['val_accuracy']

        save_results_to_csv(csv_path, config, final_train_acc, final_val_acc)

    if config['output'].get('save_results', True):
        results_dir = WEEK_2_ROOT / config['output'].get('results_dir', 'results')
        results_dir.mkdir(exist_ok=True)

        results = {
            'config': {
                'experiment_name': config.get('name', 'unnamed'),
                'patch_size': patch_size,
                'stride': stride,
                'img_size': img_size,
                'train_aggregation': train_aggregation,
                'val_aggregation': val_aggregation,
                'num_epochs': num_epochs,
                'train_batch_size': train_batch_size,
                'val_batch_size': val_batch_size,
                'lr': lr,
                'weight_decay': weight_decay,
            },
            'metrics': metrics,
            'history': history,
        }

        exp_name_safe = config.get('name', 'unnamed').replace(' ', '_')
        output_file = results_dir / f'{exp_name_safe}_results_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {output_file}")

    if config['output'].get('save_model', True):
        models_dir = WEEK_2_ROOT / "models"
        models_dir.mkdir(exist_ok=True)

        exp_name_safe = config.get('name', 'unnamed').replace(' ', '_')
        model_file = models_dir / f"{exp_name_safe}_model_{timestamp}.pth"
        torch.save(model.state_dict(), model_file)
        print(f"Model saved to: {model_file}")

    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='MLP Patch-Based Training')
    parser.add_argument(
        '--config',
        type=str,
        default='patch_training.yaml',
    )
    parser.add_argument(
        '--experiment',
        type=int,
        default=None,
    )

    args = parser.parse_args()

    config_path = WEEK_2_ROOT / "configs" / args.config
    full_config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if 'experiments' in full_config:
        experiments = full_config['experiments']

        if args.experiment is not None:
            if args.experiment >= len(experiments):
                print(f"Error: Experiment {args.experiment} not found (only {len(experiments)} experiments defined)")
                return

            exp_config = experiments[args.experiment].copy()
            exp_config['data'] = full_config.get('data', {})
            exp_config['wandb'] = full_config.get('wandb', {})
            exp_config['output'] = full_config.get('output', {})

            print(f"Running experiment {args.experiment}: {exp_config.get('name', 'unnamed')}")
            train(exp_config, device)
        else:
            print(f"Found {len(experiments)} experiments")

            for i, exp in enumerate(experiments):
                exp_config = exp.copy()
                exp_config['data'] = full_config.get('data', {})
                exp_config['wandb'] = full_config.get('wandb', {})
                exp_config['output'] = full_config.get('output', {})

                print(f"Running experiment {i}/{len(experiments)}: {exp_config.get('name', 'unnamed')}")
                train(exp_config, device)
    else:
        train(full_config, device)


if __name__ == "__main__":
    main()
