import itertools
import os
import pickle
from pathlib import Path
import csv

from typing import *

import numpy as np
import tqdm
import wandb
from bovw import BOVW
from classifiers import create_classifier
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler
import cv2

import importlib
import sys
import hashlib


def _get_descriptor_cache_dir(bovw):
    detector_kwargs = getattr(bovw, 'detector_kwargs', {})
    detector_kwargs_str = '_'.join(f"{k}={v}" for k, v in sorted(detector_kwargs.items()))

    config_str = (
        f"{bovw.detector_type}_"
        f"dkwargs{detector_kwargs_str}_"
        f"ds{bovw.dense_sift if hasattr(bovw, 'dense_sift') else False}_"
        f"step{bovw.dense_step if hasattr(bovw, 'dense_step') else 8}_"
        f"scales{bovw.dense_scales if hasattr(bovw, 'dense_scales') else [16]}"
    )
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    cache_dir = root / "cache" / f"descriptors_{config_hash}"
    return cache_dir


def _serialize_keypoints(keypoints):
    if keypoints is None:
        return None
    return [{
        'pt': kp.pt,
        'size': kp.size,
        'angle': kp.angle,
        'response': kp.response,
        'octave': kp.octave,
        'class_id': kp.class_id
    } for kp in keypoints]


def _deserialize_keypoints(kp_data_list):
    if kp_data_list is None:
        return None
    return [cv2.KeyPoint(
        x=kp['pt'][0],
        y=kp['pt'][1],
        size=kp['size'],
        angle=kp['angle'],
        response=kp['response'],
        octave=kp['octave'],
        class_id=kp['class_id']
    ) for kp in kp_data_list]


def precompute_descriptors(dataset, bovw):
    cache_dir = _get_descriptor_cache_dir(bovw)
    os.makedirs(cache_dir, exist_ok=True)

    use_spatial = bovw.spatial_pyramid is not None

    print(f"Pre-computing descriptors in: {cache_dir}")

    for idx in tqdm.tqdm(range(len(dataset)), desc="Pre-computing descriptors"):
        cache_file = cache_dir / f"image_{idx}.pkl"

        if cache_file.exists():
            continue

        image, label = dataset[idx]
        keypoints, descriptors = bovw._extract_features(image=np.array(image))

        serialized_keypoints = _serialize_keypoints(keypoints)

        cache_data = {
            'descriptors': descriptors,
            'keypoints': serialized_keypoints,
            'label': label,
            'image': image if use_spatial else None 
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    return cache_dir


def load_descriptors_from_cache(dataset, bovw, indices=None):
    cache_dir = _get_descriptor_cache_dir(bovw)
    use_spatial = bovw.spatial_pyramid is not None

    if indices is None:
        indices = range(len(dataset))

    all_descriptors = []
    all_labels = []
    all_keypoints = [] if use_spatial else None
    train_images = []

    for idx in indices:
        cache_file = cache_dir / f"image_{idx}.pkl"

        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}. Run precompute_descriptors first.")

        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        descriptors = cache_data['descriptors']

        if descriptors is not None:
            all_descriptors.append(descriptors)
            all_labels.append(cache_data['label'])

            if use_spatial:
                serialized_kp = cache_data['keypoints']
                keypoints = _deserialize_keypoints(serialized_kp)
                all_keypoints.append(keypoints)

                if cache_data['image'] is not None:
                    train_images.append(cache_data['image'])
                else:
                    train_images.append(dataset[idx][0])

    return all_descriptors, all_labels, all_keypoints, train_images


PARAMS_MODULE = sys.modules.get('__params__', None)
if PARAMS_MODULE is None:
    from parameters import (
        CLASSIFIER_PARAMETERS, CODEBOOK_SIZE,
        SELECTED_DETECTOR, DETECTOR_PARAMETERS,
        SELECTED_CLASSIFIER,
        SPATIAL_PYRAMID_TYPES, PYRAMID_LEVELS,
        USE_DENSE_SIFT, DENSE_STEP_SIZES, DENSE_SCALES,
    )
    try:
        from parameters import DETECTOR_TYPE_MAP
    except ImportError:
        DETECTOR_TYPE_MAP = None
    try:
        from parameters import CSV_FILENAME
    except ImportError:
        CSV_FILENAME = 'grid_search_results.csv'
    try:
        from parameters import PCA_DIMENSIONS
    except ImportError:
        PCA_DIMENSIONS = [None]  # No PCA by default
    try:
        from parameters import SCALER_TYPES
    except ImportError:
        SCALER_TYPES = [None]
    try:
        from parameters import FEATURE_ENCODINGS
    except ImportError:
        FEATURE_ENCODINGS = ["bovw"]
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
    CSV_FILENAME = getattr(PARAMS_MODULE, 'CSV_FILENAME', 'grid_search_results.csv')
    PCA_DIMENSIONS = getattr(PARAMS_MODULE, 'PCA_DIMENSIONS', [None])
    SCALER_TYPES = getattr(PARAMS_MODULE, 'SCALER_TYPES', [None])
    FEATURE_ENCODINGS = getattr(PARAMS_MODULE, 'FEATURE_ENCODINGS', ["bovw"])

from classifiers import create_classifier
from bovw import BOVW
from utils import compute_hash
from fisher_vector import FisherEncoder
root = Path.cwd() / "Week1"


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"],
                            keypoints_list: list = None, images: list = None):
    if bovw.spatial_pyramid and keypoints_list is not None and images is not None:
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
        return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])


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
          classifier_kwargs: dict | None = None,
          dataset_indices: List[int] = None,
          pca_dim: int = None,
          encoding_type: str = "bovw",
          scaler_type: str = None):

    if classifier_kwargs is None:
        classifier_kwargs = {}

    use_spatial = bovw.spatial_pyramid is not None

    all_descriptors, all_labels, all_keypoints, train_images = load_descriptors_from_cache(
        dataset=dataset,
        bovw=bovw,
        indices=dataset_indices
    )

    if encoding_type == "bovw":
        print("Fitting BoVW codebook")
        bovw._update_fit_codebook(all_descriptors)

        print("Computing features (BoVW)")
        fv_encoder = None
        features = extract_bovw_histograms(
            descriptors=all_descriptors,
            bovw=bovw,
            keypoints_list=all_keypoints,
            images=train_images if use_spatial else None
        )
    elif encoding_type == "fisher":
        print("Fitting Fisher Vector encoder")
        fv_encoder = FisherEncoder(n_components=bovw.codebook_size)
        fv_encoder.fit(all_descriptors)
        features = fv_encoder.transform(all_descriptors)
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}")

    pca = None
    if pca_dim is not None:
        print(f"Applying PCA: {features.shape[1]} -> {pca_dim} dimensions")
        pca = PCA(n_components=pca_dim, random_state=42)
        features = pca.fit_transform(features)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

    if scaler_type == "standard":
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif scaler_type == "l2":
        scaler = Normalizer(norm="l2")
        features = scaler.fit_transform(features)
    elif scaler_type == "l1":
        scaler = Normalizer(norm="l1")
        features = scaler.fit_transform(features)
    else:
        scaler = None

    print(f"Creating classifier: {classifier_name}")
    classifier = create_classifier(classifier_name, **classifier_kwargs)
    classifier.fit(features, all_labels)

    train_pred = classifier.predict(features)
    train_acc = accuracy_score(all_labels, train_pred)
    print(f"Accuracy on Phase[Train]: {train_acc}")

    return bovw, classifier, train_acc, pca, scaler, fv_encoder


def extract_test_accuracy(bovw, classifier, dataset, dataset_indices=None, pca=None,
                          scaler=None, fv_encoder=None, encoding_type="bovw"):
    if dataset_indices is not None:
        try:
            descriptors_list, labels, keypoints_list, images_list = load_descriptors_from_cache(
                dataset=dataset,
                bovw=bovw,
                indices=dataset_indices
            )
        except FileNotFoundError:
            dataset_indices = None

    if dataset_indices is None:
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

    if encoding_type == "bovw":
        histograms = extract_bovw_histograms(
            bovw=bovw,
            descriptors=descriptors_list,
            keypoints_list=keypoints_list,
            images=images_list if keypoints_list else None
        )
    elif encoding_type == "fisher":
        histograms = fv_encoder.transform(descriptors_list)
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}")

    if pca is not None:
        histograms = pca.transform(histograms)

    if scaler is not None:
        histograms = scaler.transform(histograms)

    preds = classifier.predict(histograms)
    acc = accuracy_score(labels, preds)
    return acc

def gridsearch(train_data, test_data, n_folds=5, config_indices=None, count_only=False):

    best_acc = -1
    best_config = None
    run_index = -1

    csv_file = root / "results" / CSV_FILENAME
    os.makedirs(root / "results", exist_ok=True)

    csv_headers = [
        "run_id", "classifier", "detector", "codebook_size", "spatial_pyramid", "pyramid_levels",
        "dense_sift", "dense_step", "dense_scales", "pca_dim", "encoding_type", "scaler_type"
    ]
    for fold_idx in range(1, n_folds + 1):
        csv_headers.extend([f"fold_{fold_idx}_train_acc", f"fold_{fold_idx}_val_acc"])
    csv_headers.extend([
        "mean_train_acc", "std_train_acc", "mean_val_acc", "std_val_acc",
        "final_train_acc", "test_acc", "pca_explained_variance"
    ])
    csv_headers.append("clf_params")

    file_exists = csv_file.exists()
    if not file_exists:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()

    detector_list = SELECTED_DETECTOR if isinstance(SELECTED_DETECTOR, list) else [SELECTED_DETECTOR]
    dense_sift_list = USE_DENSE_SIFT if isinstance(USE_DENSE_SIFT, list) else [USE_DENSE_SIFT]

    assert len(detector_list) == len(dense_sift_list), \
        "SELECTED_DETECTOR and USE_DENSE_SIFT must have the same length"

    for classifier_name in SELECTED_CLASSIFIER:
        param_grid = CLASSIFIER_PARAMETERS[classifier_name]
        param_names = list(param_grid.keys())
        param_value_lists = list(param_grid.values())
        combinations = list(itertools.product(*param_value_lists))

        for encoding_type in FEATURE_ENCODINGS:
            for codebook_size in CODEBOOK_SIZE:
                for detector_type, use_dense in zip(detector_list, dense_sift_list):
                    actual_detector_type = DETECTOR_TYPE_MAP.get(detector_type, detector_type) if DETECTOR_TYPE_MAP else detector_type

                    dense_step_range = DENSE_STEP_SIZES
                    dense_scales_range = DENSE_SCALES

                    for dense_step in dense_step_range:
                        for dense_scales in dense_scales_range:
                            if not use_dense and (dense_step != dense_step_range[0] or dense_scales != dense_scales_range[0]):
                                continue

                            for spatial_pyramid_type in SPATIAL_PYRAMID_TYPES:
                                pyramid_level_range = PYRAMID_LEVELS if spatial_pyramid_type else [1]

                                for pyramid_level in pyramid_level_range:
                                    for scaler_type in SCALER_TYPES:
                                        for pca_dim in PCA_DIMENSIONS:
                                            for param_tuple in combinations:
                                                clf_params = dict(zip(param_names, param_tuple))
                                                run_index += 1

                                                if count_only:
                                                    continue

                                                if config_indices is not None and run_index not in config_indices:
                                                    continue

                                                detector_name = f"Dense-SIFT" if use_dense and actual_detector_type == "SIFT" else actual_detector_type
                                                pyramid_str = f"_pyr-{spatial_pyramid_type}-L{pyramid_level}" if spatial_pyramid_type else ""
                                                pca_str = f"_pca{pca_dim}" if pca_dim else ""
                                                scaler_str = f"_scaler-{scaler_type}" if scaler_type else ""
                                                encoding_str = f"_{encoding_type}"
                                                run_name = f"{detector_name}_k{codebook_size}_{classifier_name}{pyramid_str}{pca_str}{scaler_str}{encoding_str}"

                                                print(f" RUN {run_index}: {run_name}")
                                                print(f" Detector={detector_type}, Dense SIFT: {use_dense}, step: {dense_step}, scales: {dense_scales}")
                                                print(f" Spatial pyramid: {spatial_pyramid_type}, levels: {pyramid_level}")
                                                print(f" Encoding: {encoding_type}, Scaler: {scaler_type}")
                                                print(f" PCA dimensions: {pca_dim if pca_dim else 'None'}")
                                                print(f" params={clf_params}")

                                                bovw_for_descriptors = BOVW(
                                                    detector_type=actual_detector_type,
                                                    codebook_size=int(codebook_size),
                                                    detector_kwargs=DETECTOR_PARAMETERS[detector_type],
                                                    spatial_pyramid=spatial_pyramid_type,
                                                    pyramid_levels=int(pyramid_level),
                                                    dense_sift=bool(use_dense),
                                                    dense_step=int(dense_step),
                                                    dense_scales=dense_scales
                                                )

                                                print("\nPre-computing descriptors for all training images...")
                                                precompute_descriptors(train_data, bovw_for_descriptors)

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
                                                        "pca_dim": pca_dim if pca_dim else "None",
                                                        "encoding_type": encoding_type,
                                                        "scaler_type": scaler_type if scaler_type else "None",
                                                        "n_folds": n_folds,
                                                    }
                                                )

                                                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                                                fold_accuracies = []
                                                fold_train_accuracies = []
                                                fold_pca_variances = []

                                                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_data)):
                                                    print(f"\n  Fold {fold_idx + 1}/{n_folds}")

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

                                                    bovw, classifier, train_acc, fold_pca, fold_scaler, fold_fv = train(
                                                        dataset=train_data,  # Pass full dataset
                                                        bovw=bovw,
                                                        classifier_name=classifier_name,
                                                        classifier_kwargs=clf_params,
                                                        dataset_indices=train_idx,  # Pass indices for this fold
                                                        pca_dim=pca_dim,
                                                        encoding_type=encoding_type,
                                                        scaler_type=scaler_type
                                                    )

                                                    if fold_pca is not None:
                                                        fold_pca_variances.append(fold_pca.explained_variance_ratio_.sum())

                                                    fold_acc = extract_test_accuracy(
                                                        bovw=bovw,
                                                        classifier=classifier,
                                                        dataset=train_data,  # Pass full dataset
                                                        dataset_indices=val_idx,  # Pass validation indices
                                                        pca=fold_pca,
                                                        scaler=fold_scaler,
                                                        fv_encoder=fold_fv,
                                                        encoding_type=encoding_type
                                                    )
                                                    fold_accuracies.append(fold_acc)
                                                    fold_train_accuracies.append(train_acc)
                                                    print(f"  Fold {fold_idx + 1} validation accuracy: {fold_acc:.4f}")

                                                    wandb.log({
                                                        f"fold_{fold_idx + 1}_train_accuracy": train_acc,
                                                        f"fold_{fold_idx + 1}_val_accuracy": fold_acc
                                                    })

                                                mean_cv_acc = np.mean(fold_accuracies)
                                                std_cv_acc = np.std(fold_accuracies)
                                                mean_train_acc = np.mean(fold_train_accuracies)
                                                std_train_acc = np.std(fold_train_accuracies)
                                                mean_pca_variance = np.mean(fold_pca_variances) if fold_pca_variances else None

                                                print(f"\nCross-validation train accuracy: {mean_train_acc:.4f} (+/- {std_train_acc:.4f})")
                                                print(f"Cross-validation val accuracy: {mean_cv_acc:.4f} (+/- {std_cv_acc:.4f})")
                                                if mean_pca_variance:
                                                    print(f"Mean PCA explained variance: {mean_pca_variance:.4f}")

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

                                                bovw_final, classifier_final, final_train_acc, final_pca, final_scaler, final_fv = train(
                                                    dataset=train_data,
                                                    bovw=bovw_final,
                                                    classifier_name=classifier_name,
                                                    classifier_kwargs=clf_params,
                                                    pca_dim=pca_dim,
                                                    encoding_type=encoding_type,
                                                    scaler_type=scaler_type
                                                )

                                                test_acc = extract_test_accuracy(
                                                    bovw_final, classifier_final, test_data,
                                                    pca=final_pca, scaler=final_scaler,
                                                    fv_encoder=final_fv, encoding_type=encoding_type
                                                )

                                                wandb_log_dict = {
                                                    "mean_train_accuracy": mean_train_acc,
                                                    "std_train_accuracy": std_train_acc,
                                                    "mean_cv_accuracy": mean_cv_acc,
                                                    "std_cv_accuracy": std_cv_acc,
                                                    "final_train_accuracy": final_train_acc,
                                                    "test_accuracy": test_acc
                                                }
                                                if mean_pca_variance:
                                                    wandb_log_dict["pca_explained_variance"] = mean_pca_variance
                                                wandb.log(wandb_log_dict)

                                                print(f"Test set accuracy: {test_acc:.4f}")

                                                csv_row = {
                                                    "run_id": run_index,
                                                    "classifier": classifier_name,
                                                    "detector": detector_type,
                                                    "codebook_size": codebook_size,
                                                    "spatial_pyramid": spatial_pyramid_type if spatial_pyramid_type else "None",
                                                    "pyramid_levels": pyramid_level,
                                                    "dense_sift": use_dense,
                                                    "dense_step": dense_step,
                                                    "dense_scales": str(dense_scales),
                                                    "pca_dim": pca_dim if pca_dim else "None",
                                                    "encoding_type": encoding_type,
                                                    "scaler_type": scaler_type if scaler_type else "None",
                                                    "mean_train_acc": mean_train_acc,
                                                    "std_train_acc": std_train_acc,
                                                    "mean_val_acc": mean_cv_acc,
                                                    "std_val_acc": std_cv_acc,
                                                    "final_train_acc": final_train_acc,
                                                    "test_acc": test_acc,
                                                    "pca_explained_variance": mean_pca_variance if mean_pca_variance else "N/A",
                                                    "clf_params": str(clf_params)
                                                }
                                                for fold_idx in range(n_folds):
                                                    csv_row[f"fold_{fold_idx + 1}_train_acc"] = fold_train_accuracies[fold_idx]
                                                    csv_row[f"fold_{fold_idx + 1}_val_acc"] = fold_accuracies[fold_idx]

                                                with open(csv_file, 'a', newline='') as f:
                                                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                                                    writer.writerow(csv_row)

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
                                                        "pca_dim": pca_dim,
                                                        "encoding_type": encoding_type,
                                                        "scaler_type": scaler_type,
                                                        "params": clf_params,
                                                        "cv_accuracy": mean_cv_acc,
                                                        "cv_std": std_cv_acc,
                                                        "test_accuracy": test_acc
                                                    }

                                                wandb.finish()

    if count_only:
        total_configs = run_index + 1
        print(f"\nTotal number of configurations: {total_configs}")
        return None

    return best_config
