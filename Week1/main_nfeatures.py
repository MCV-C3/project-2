#!/usr/bin/env python3
"""
OPTIONAL 1: Feature Count Experiment

Tests how the number of extracted local features affects classification accuracy
for SIFT and AKAZE descriptors.

20 configurations total:
- 10 SIFT configurations with different nfeatures
- 10 AKAZE configurations with different thresholds

Usage:
    python3 main_nfeatures.py                    # Run all
    python3 main_nfeatures.py --run 0-9          # SIFT only
    python3 main_nfeatures.py --run 10-19        # AKAZE only
    python3 main_nfeatures.py --count-configs    # Count configs
"""

import os

# Limit threads per process BEFORE importing numpy/sklearn/opencv
# This prevents each parallel process from spawning too many threads
# With 4 parallel processes and 2 threads each = 8 cores total
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OPENCV_NUM_THREADS'] = '2'

import sys
import argparse
from typing import *
from PIL import Image
import glob

# Load custom parameters
import parameters_nfeatures as params
sys.modules['__params__'] = params

from grid_search import gridsearch


def Dataset(ImageFolder:str = "data/MIT_split/train") -> List[Tuple[Type[Image.Image], int]]:
    """
    Load dataset from folder structure.

    Expected Structure:
        ImageFolder/<cls label>/xxx1.png
        ImageFolder/<cls label>/xxx2.png
        ...

    Example:
        ImageFolder/cat/123.png
        ImageFolder/cat/nsdf3.png
    """
    map_classes = {clsi: idx for idx, clsi  in enumerate(os.listdir(ImageFolder))}
    dataset :List[Tuple] = []

    for _, cls_folder in enumerate(os.listdir(ImageFolder)):
        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")
            dataset.append((img_pil, map_classes[cls_folder]))

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='OPTIONAL 1: Feature Count Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 20 configurations
  python3 main_nfeatures.py

  # Run only SIFT experiments (configs 0-9)
  python3 main_nfeatures.py --run 0-9

  # Run only AKAZE experiments (configs 10-19)
  python3 main_nfeatures.py --run 10-19

  # Run specific configs
  python3 main_nfeatures.py --run 0,5,10,15

  # Count total configurations
  python3 main_nfeatures.py --count-configs

  # Run with 3-fold CV (faster)
  python3 main_nfeatures.py --n-folds 3
        """
    )

    parser.add_argument(
        '--run',
        type=str,
        default=None,
        help='Comma-separated list of configuration indices (e.g., "0,2,3" or "1-4")'
    )

    parser.add_argument(
        '--count-configs',
        action='store_true',
        help='Count total number of configurations and exit'
    )

    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    args = parser.parse_args()

    # Parse configuration indices if provided
    config_indices = None
    if args.run:
        config_indices = []
        for part in args.run.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                config_indices.extend(range(start, end + 1))
            else:
                config_indices.append(int(part))

    # Print experiment info
    if not args.count_configs:
        print("=" * 70)
        print("OPTIONAL 1: Feature Count Experiment")
        print("=" * 70)
        print(f"W&B Project: {params.WANDB_PROJECT}")
        print(f"Configurations: {len(params.SELECTED_DETECTOR)}")
        print(f"  - SIFT: {len(params.SIFT_NFEATURES_RANGE)} configs")
        print(f"  - AKAZE: {len(params.AKAZE_THRESHOLD_RANGE)} configs")
        print(f"Cross-validation folds: {args.n_folds}")
        if config_indices:
            print(f"Running configs: {config_indices}")
        else:
            print("Running all configurations")
        print("=" * 70)
        print()

    # Load datasets
    data_train = Dataset(ImageFolder="./places_reduced/train")
    data_test = Dataset(ImageFolder="./places_reduced/val")

    # Run grid search with k-fold cross-validation
    best_config = gridsearch(
        data_train,
        data_test,
        n_folds=args.n_folds,
        config_indices=config_indices,
        count_only=args.count_configs
    )

    if not args.count_configs:
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION FOUND:")
        print("=" * 70)
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        print("=" * 70)
