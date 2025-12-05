import os
import pickle
import itertools
import wandb
import numpy as np
from pathlib import Path

from typing import *
from PIL import Image
import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import importlib
import sys

# Allow specifying which parameters file to use
PARAMS_MODULE = sys.modules.get('__params__', None)
if PARAMS_MODULE is None:
    from parameters import (
        CLASSIFIER_PARAMETERS, CODEBOOK_SIZE,
        SELECTED_DETECTOR, DETECTOR_PARAMETERS,
        SELECTED_CLASSIFIER,
        SPATIAL_PYRAMID_TYPES, PYRAMID_LEVELS,
        USE_DENSE_SIFT, DENSE_STEP_SIZES, DENSE_SCALES,
    )
    DETECTOR_TYPE_MAP = None
    WANDB_PROJECT = "BoVW-GridSearch"
else:
    CLASSIFIER_PARAMETERS = PARAMS_MODULE.CLASSIFIER_PARAMETERS
    CODEBOOK_SIZE = PARAMS_MODULE.CODEBOOK_SIZE
    SELECTED_DETECTOR = PARAMS_MODULE.SELECTED_DETECTOR
    DETECTOR_PARAMETERS = PARAMS_MODULE.DETECTOR_PARAMETERS
    SELECTED_CLASSIFIER = PARAMS_MODULE.SELECTED_CLASSIFIER
    SPATIAL_PYRAMID_TYPES = PARAMS_MODULE.SPATIAL_PYRAMID_TYPES
    PYRAMID_LEVELS = PARAMS_MODULE.PYRAMID_LEVELS
    USE_DENSE_SIFT = PARAMS_MODULE.USE_DENSE_SIFT
    DENSE_STEP_SIZES = PARAMS_MODULE.DENSE_STEP_SIZES
    DENSE_SCALES = PARAMS_MODULE.DENSE_SCALES
    DETECTOR_TYPE_MAP = getattr(PARAMS_MODULE, 'DETECTOR_TYPE_MAP', None)
    WANDB_PROJECT = getattr(PARAMS_MODULE, 'WANDB_PROJECT', 'BoVW-GridSearch')

from classifiers import create_classifier
from bovw import BOVW
from utils import compute_hash


root = Path.cwd() / "Week1"


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"],
                            keypoints_list: list = None, images: list = None):
    """
    Extract BOVW histograms, with optional spatial pyramid support.

    Args:
        bovw: BOVW object
        descriptors: List of descriptors for each image
        keypoints_list: List of keypoints for each image (needed for spatial pyramids)
        images: List of images (needed for spatial pyramids)

    Returns:
        Array of histograms
    """
    if bovw.spatial_pyramid and keypoints_list is not None and images is not None:
        # Use spatial pyramid
        histograms = []
        for img, kps, descs in zip(images, keypoints_list, descriptors):
            hist = bovw._compute_spatial_pyramid_descriptor(
                image=np.array(img),
                keypoints=kps,
                descriptors=descs,
                kmeans=bovw.codebook_algo
            )
            histograms.append(hist)
        return np.array(histograms)
    else:
        # Regular BOVW without spatial pyramid
        return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])

def _get_cache_file(dataset, bovw):
    dataset_id = len(dataset)
    cache_key = compute_hash({
        "dataset_id": dataset_id,
        "descriptor": str(bovw.detector_type),
        "spatial_pyramid": str(bovw.spatial_pyramid),
        "pyramid_levels": str(bovw.pyramid_levels) if hasattr(bovw, 'pyramid_levels') else "1",
        "dense_sift": str(bovw.dense_sift) if hasattr(bovw, 'dense_sift') else "False",
        "dense_step": str(bovw.dense_step) if hasattr(bovw, 'dense_step') else "8",
        "dense_scales": str(bovw.dense_scales) if hasattr(bovw, 'dense_scales') else "[16]"
    })
    return os.path.join(root / "cache", f"descriptors_{cache_key}.pkl")

def load_cache(dataset, bovw):
    cache_file = _get_cache_file(dataset, bovw)

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        # Support both old format (tuple) and new format (dict)
        if isinstance(data, tuple):
            all_descriptors, all_labels = data
            all_keypoints = None
        else:
            all_descriptors = data['descriptors']
            all_labels = data['labels']
            all_keypoints = data.get('keypoints', None)
        return all_descriptors, all_labels, all_keypoints
    else:
        return None

def test(dataset: List[Tuple[Type[Image.Image], int]],
         bovw: Type[BOVW],
         classifier:Type[object]):

    use_spatial = bovw.spatial_pyramid is not None
    test_descriptors = []
    descriptors_labels = []
    test_keypoints = [] if use_spatial else None
    test_images = []

    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Eval]: Extracting the descriptors"):
        image, label = dataset[idx]
        keypoints, descriptors = bovw._extract_features(image=np.array(image))

        if descriptors is not None:
            test_descriptors.append(descriptors)
            descriptors_labels.append(label)
            if use_spatial:
                test_keypoints.append(keypoints)
                test_images.append(image)


    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(
        descriptors=test_descriptors,
        bovw=bovw,
        keypoints_list=test_keypoints,
        images=test_images if use_spatial else None
    )

    print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)

    print("Accuracy on Phase[Test]:", accuracy_score(y_true=descriptors_labels, y_pred=y_pred))
    
def train(dataset: List[Tuple[Image.Image, int]],
          bovw: Type[BOVW],
          classifier_name: str = "logreg",
          classifier_kwargs: dict | None = None):

    if classifier_kwargs is None:
        classifier_kwargs = {}

    use_spatial = bovw.spatial_pyramid is not None
    train_images = []

    cached = load_cache(dataset, bovw)

    if cached is not None:
        print("Load cached descriptors.")
        all_descriptors, all_labels, all_keypoints = cached
        if use_spatial and all_keypoints is None:
            # Need to re-extract if we need keypoints but don't have them
            print("Keypoints not in cache, re-extracting...")
            cached = None

    if cached is None:
        all_descriptors = []
        all_labels = []
        all_keypoints = [] if use_spatial else None

        for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):

            image, label = dataset[idx]
            keypoints, descriptors = bovw._extract_features(image=np.array(image))

            if descriptors is not None:
                all_descriptors.append(descriptors)
                all_labels.append(label)
                if use_spatial:
                    all_keypoints.append(keypoints)
                    train_images.append(image)

        os.makedirs(root / "cache", exist_ok=True)
        cache_file = _get_cache_file(dataset, bovw)
        cache_data = {
            'descriptors': all_descriptors,
            'labels': all_labels,
            'keypoints': all_keypoints
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
    else:
        # If we loaded from cache and need images for spatial pyramid
        if use_spatial:
            train_images = [img for img, _ in dataset]

    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(
        descriptors=all_descriptors,
        bovw=bovw,
        keypoints_list=all_keypoints,
        images=train_images if use_spatial else None
    ) 
    
    print(f"Creating classifier: {classifier_name}")
    classifier = create_classifier(classifier_name, **classifier_kwargs)

    print("Fitting the classifier")
    classifier.fit(bovw_histograms, all_labels)

    print("Accuracy on Phase[Train]:", accuracy_score(y_true=all_labels, y_pred=classifier.predict(bovw_histograms)))
    
    return bovw, classifier


def extract_test_accuracy(bovw, classifier, dataset):
    """
    Compute BoVW histograms and accuracy on test set
    """
    use_spatial = bovw.spatial_pyramid is not None
    descriptors_list = []
    keypoints_list = [] if use_spatial else None
    images_list = []
    labels = []

    for img, label in dataset:
        keypoints, descriptors = bovw._extract_features(np.array(img))
        if descriptors is not None:
            descriptors_list.append(descriptors)
            labels.append(label)
            if use_spatial:
                keypoints_list.append(keypoints)
                images_list.append(img)

    histograms = extract_bovw_histograms(
        bovw=bovw,
        descriptors=descriptors_list,
        keypoints_list=keypoints_list,
        images=images_list if use_spatial else None
    )

    preds = classifier.predict(histograms)
    acc = accuracy_score(labels, preds)
    return acc

def gridsearch(train_data, test_data, n_folds=5, config_indices=None, count_only=False):
    """
    Full grid search over:
    - detector types (SIFT, Dense SIFT, AKAZE, ORB)
    - classifier types
    - classifier hyperparameters
    - codebook sizes
    - spatial pyramid configurations
    - dense SIFT configurations
    Uses k-fold cross-validation on the training data.
    And logs everything to W&B.

    Args:
        train_data: Training dataset
        test_data: Test dataset for final evaluation
        n_folds: Number of folds for cross-validation (default: 5)
        config_indices: List of configuration indices to run (None = run all)
        count_only: If True, only count and print total configurations, then exit
    """

    best_acc = -1
    best_config = None
    run_index = -1  # Will be incremented before use

    # Ensure SELECTED_DETECTOR and USE_DENSE_SIFT are lists
    detector_list = SELECTED_DETECTOR if isinstance(SELECTED_DETECTOR, list) else [SELECTED_DETECTOR]
    dense_sift_list = USE_DENSE_SIFT if isinstance(USE_DENSE_SIFT, list) else [USE_DENSE_SIFT]

    # They should have the same length
    assert len(detector_list) == len(dense_sift_list), \
        "SELECTED_DETECTOR and USE_DENSE_SIFT must have the same length"

    for classifier_name in SELECTED_CLASSIFIER:
        param_grid = CLASSIFIER_PARAMETERS[classifier_name]
        param_names = list(param_grid.keys())
        param_value_lists = list(param_grid.values())
        combinations = list(itertools.product(*param_value_lists))

        for codebook_size in CODEBOOK_SIZE:
            # Iterate over detector/dense_sift pairs
            for detector_type, use_dense in zip(detector_list, dense_sift_list):
                # Only iterate dense parameters if using dense SIFT with SIFT detector
                if use_dense and detector_type == "SIFT":
                    dense_step_range = DENSE_STEP_SIZES
                    dense_scales_range = DENSE_SCALES
                else:
                    dense_step_range = [8]  # Default value (won't be used)
                    dense_scales_range = [[16]]  # Default value (won't be used)

                for dense_step in dense_step_range:
                    for dense_scales in dense_scales_range:
                        # Skip dense iterations if not using dense SIFT
                        if not use_dense and (dense_step != dense_step_range[0] or dense_scales != dense_scales_range[0]):
                            continue

                        for spatial_pyramid_type in SPATIAL_PYRAMID_TYPES:
                            # Only iterate over pyramid levels if we have a spatial pyramid type
                            pyramid_level_range = PYRAMID_LEVELS if spatial_pyramid_type else [1]

                            for pyramid_level in pyramid_level_range:
                                for param_tuple in combinations:
                                    clf_params = dict(zip(param_names, param_tuple))
                                    run_index += 1

                                    # If counting only, just increment and continue
                                    if count_only:
                                        continue

                                    # Skip if config_indices specified and this index not in list
                                    if config_indices is not None and run_index not in config_indices:
                                        continue

                                    # Create descriptive run name
                                    detector_name = f"Dense-SIFT" if use_dense and detector_type == "SIFT" else detector_type
                                    pyramid_str = f"_pyr-{spatial_pyramid_type}-L{pyramid_level}" if spatial_pyramid_type else ""
                                    run_name = f"{detector_name}_k{codebook_size}_{classifier_name}{pyramid_str}"

                                    print("\n------------")
                                    print(f" RUN {run_index}: {run_name}")
                                    print(f" Detector={detector_type}, Dense SIFT: {use_dense}, step: {dense_step}, scales: {dense_scales}")
                                    print(f" Spatial pyramid: {spatial_pyramid_type}, levels: {pyramid_level}")
                                    print(f" params={clf_params}")
                                    print("-------------")

                                    # Get actual detector type (SIFT/AKAZE/ORB)
                                    actual_detector_type = DETECTOR_TYPE_MAP.get(detector_type, detector_type) if DETECTOR_TYPE_MAP else detector_type

                                    wandb.init(
                                        project=WANDB_PROJECT,
                                        name=run_name,
                                        config={
                                            "run_id": run_index,
                                            "classifier": classifier_name,
                                            "clf_params": clf_params,
                                            "codebook_size": int(codebook_size),
                                            "detector": detector_type,
                                            "actual_detector_type": actual_detector_type,
                                            "detector_params": DETECTOR_PARAMETERS[detector_type],
                                            "dense_sift": bool(use_dense),
                                            "dense_step": int(dense_step),
                                            "dense_scales": dense_scales,
                                            "spatial_pyramid": str(spatial_pyramid_type),
                                            "pyramid_levels": int(pyramid_level),
                                            "n_folds": n_folds,
                                        }
                                    )

                                    # Perform k-fold cross-validation
                                    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                                    fold_accuracies = []

                                    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_data)):
                                        print(f"\n  Fold {fold_idx + 1}/{n_folds}")

                                        # Split data into train and validation folds
                                        fold_train = [train_data[i] for i in train_idx]
                                        fold_val = [train_data[i] for i in val_idx]

                                        # Build BoVW for this fold
                                        bovw = BOVW(
                                            detector_type=actual_detector_type,
                                            codebook_size=int(codebook_size),
                                            detector_kwargs=DETECTOR_PARAMETERS[detector_type],
                                            spatial_pyramid=spatial_pyramid_type,
                                            pyramid_levels=int(pyramid_level),
                                            dense_sift=bool(use_dense),
                                            dense_step=int(dense_step),
                                            dense_scales=dense_scales
                                        )

                                        # Train on fold
                                        bovw, classifier = train(
                                            dataset=fold_train,
                                            bovw=bovw,
                                            classifier_name=classifier_name,
                                            classifier_kwargs=clf_params,
                                        )

                                        # Evaluate on validation fold
                                        fold_acc = extract_test_accuracy(bovw, classifier, fold_val)
                                        fold_accuracies.append(fold_acc)
                                        print(f"  Fold {fold_idx + 1} validation accuracy: {fold_acc:.4f}")

                                        wandb.log({
                                            f"fold_{fold_idx + 1}_val_accuracy": fold_acc
                                        })

                                    # Calculate mean CV accuracy
                                    mean_cv_acc = np.mean(fold_accuracies)
                                    std_cv_acc = np.std(fold_accuracies)

                                    print(f"\nCross-validation accuracy: {mean_cv_acc:.4f} (+/- {std_cv_acc:.4f})")

                                    # Train on full training set and evaluate on test set
                                    print("\nTraining on full training set...")
                                    bovw_final = BOVW(
                                        detector_type=actual_detector_type,
                                        codebook_size=int(codebook_size),
                                        detector_kwargs=DETECTOR_PARAMETERS[detector_type],
                                        spatial_pyramid=spatial_pyramid_type,
                                        pyramid_levels=int(pyramid_level),
                                        dense_sift=bool(use_dense),
                                        dense_step=int(dense_step),
                                        dense_scales=dense_scales
                                    )

                                    bovw_final, classifier_final = train(
                                        dataset=train_data,
                                        bovw=bovw_final,
                                        classifier_name=classifier_name,
                                        classifier_kwargs=clf_params,
                                    )

                                    test_acc = extract_test_accuracy(bovw_final, classifier_final, test_data)

                                    wandb.log({
                                        "mean_cv_accuracy": mean_cv_acc,
                                        "std_cv_accuracy": std_cv_acc,
                                        "test_accuracy": test_acc
                                    })

                                    print(f"Test set accuracy: {test_acc:.4f}")

                                    # Track best based on CV accuracy
                                    if mean_cv_acc > best_acc:
                                        best_acc = mean_cv_acc
                                        best_config = {
                                            "classifier": classifier_name,
                                            "detector": detector_type,
                                            "codebook_size": codebook_size,
                                            "dense_sift": use_dense,
                                            "dense_step": dense_step,
                                            "dense_scales": dense_scales,
                                            "spatial_pyramid": spatial_pyramid_type,
                                            "pyramid_levels": pyramid_level,
                                            "params": clf_params,
                                            "cv_accuracy": mean_cv_acc,
                                            "cv_std": std_cv_acc,
                                            "test_accuracy": test_acc
                                        }

                                    wandb.finish()

    # If count_only mode, print total and return
    if count_only:
        total_configs = run_index + 1
        print(f"\nTotal number of configurations: {total_configs}")
        return None

    return best_config
