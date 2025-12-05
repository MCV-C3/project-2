import os

# Limit threads per process BEFORE importing numpy/sklearn/opencv
# This prevents each parallel process from spawning too many threads
# With 4 parallel processes and 2 threads each = 8 cores total
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OPENCV_NUM_THREADS'] = '2'

import argparse
from typing import *
from PIL import Image
import glob
from grid_search import gridsearch


def Dataset(ImageFolder:str = "data/MIT_split/train") -> List[Tuple[Type[Image.Image], int]]:

    """
    Expected Structure:

        ImageFolder/<cls label>/xxx1.png
        ImageFolder/<cls label>/xxx2.png
        ImageFolder/<cls label>/xxx3.png
        ...

    Example:
        ImageFolder/cat/123.png
        ImageFolder/cat/nsdf3.png
        ImageFolder/cat/[...]/asd932_.png

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
    parser = argparse.ArgumentParser(description='Run BoVW grid search experiments')
    parser.add_argument('--run', type=str, default=None,
                        help='Comma-separated list of configuration indices to run (e.g., "0,2,3" or "1-4")')
    parser.add_argument('--count-configs', action='store_true',
                        help='Count total number of configurations and exit')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')

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
        print("\nBEST CONFIGURATION FOUND:")
        print(best_config)



