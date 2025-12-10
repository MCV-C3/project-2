import numpy as np

"""
Experiment: Testing Different Numbers of Local Features

This experiment tests how the number of extracted features affects performance
for SIFT and AKAZE descriptors.

Goal: Find the optimal trade-off between:
- Number of features (computational cost)
- Classification accuracy

We test 10 configurations for each descriptor (20 total).
"""

# Classifier parameters - Keep simple for fair comparison
CLASSIFIER_PARAMETERS = {
    "LogisticRegression": {
        "penalty": ["l2"],
        "C": [1.0],
        "class_weight": ["balanced"],
        "max_iter": [2000],
        "solver": ["lbfgs"]
    }
}

SELECTED_CLASSIFIER = ["LogisticRegression"]


# Feature count experiments
# Range from very few features (50) to many features (2000)
SIFT_NFEATURES_RANGE = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 0]  # 0 = unlimited
AKAZE_THRESHOLD_RANGE = [
    0.01,    # Very few features (~20-50)
    0.005,   # Few features (~50-100)
    0.002,   # (~100-200)
    0.001,   # Standard (~200-400)
    0.0005,  # More features (~400-800)
    0.0002,  # Many features (~800-1500)
    0.0001,  # Very many features (~1500-2500)
    0.00005, # (~2500-4000)
    0.00002, # (~4000-6000)
    0.00001  # Maximum features (~6000+)
]

# Build detector configurations
DETECTOR_PARAMETERS = {}
SELECTED_DETECTOR = []

# SIFT configurations (10 configs)
for i, nfeatures in enumerate(SIFT_NFEATURES_RANGE):
    detector_name = f"SIFT_n{nfeatures if nfeatures > 0 else 'unlimited'}"
    DETECTOR_PARAMETERS[detector_name] = {
        "nfeatures": nfeatures,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10
    }
    SELECTED_DETECTOR.append(detector_name)

# AKAZE configurations (10 configs)
for i, threshold in enumerate(AKAZE_THRESHOLD_RANGE):
    detector_name = f"AKAZE_t{threshold}"
    DETECTOR_PARAMETERS[detector_name] = {
        "descriptor_type": 5,  # MLDB
        "descriptor_size": 0,
        "descriptor_channels": 3,
        "threshold": threshold,
        "nOctaves": 4,
        "nOctaveLayers": 4
    }
    SELECTED_DETECTOR.append(detector_name)


# Map to actual detector types for BOVW
DETECTOR_TYPE_MAP = {}
for detector_name in SELECTED_DETECTOR:
    if detector_name.startswith("SIFT"):
        DETECTOR_TYPE_MAP[detector_name] = "SIFT"
    elif detector_name.startswith("AKAZE"):
        DETECTOR_TYPE_MAP[detector_name] = "AKAZE"


# Codebook parameters - moderate size for all
CODEBOOK_SIZE = [512]


# No spatial pyramid for this experiment
SPATIAL_PYRAMID_TYPES = [None]
PYRAMID_LEVELS = [1]


# No dense SIFT for this experiment
USE_DENSE_SIFT = [False] * len(SELECTED_DETECTOR)
DENSE_STEP_SIZES = [8]
DENSE_SCALES = [[16]]


# No cache for this experiment
USE_DENSE_CACHE = False
DENSE_CACHE_DIR = "cache/dense_sift"

# W&B Project
WANDB_PROJECT = "OPTIONAL1"


# Print summary
if __name__ == "__main__":
    print("=" * 70)
    print("Feature Count Experiment Configuration")
    print("=" * 70)
    print(f"\nTotal configurations: {len(SELECTED_DETECTOR)}")
    print(f"SIFT configurations: {len(SIFT_NFEATURES_RANGE)}")
    print(f"AKAZE configurations: {len(AKAZE_THRESHOLD_RANGE)}")
    print(f"\nClassifier: LogisticRegression")
    print(f"Codebook size: {CODEBOOK_SIZE[0]}")
    print(f"W&B Project: {WANDB_PROJECT}")

    print("\n" + "=" * 70)
    print("SIFT Configurations:")
    print("=" * 70)
    for i, (name, params) in enumerate(list(DETECTOR_PARAMETERS.items())[:len(SIFT_NFEATURES_RANGE)]):
        nf = params['nfeatures']
        print(f"  {i:2d}. {name:20s} - nfeatures: {nf if nf > 0 else 'unlimited'}")

    print("\n" + "=" * 70)
    print("AKAZE Configurations:")
    print("=" * 70)
    for i, (name, params) in enumerate(list(DETECTOR_PARAMETERS.items())[len(SIFT_NFEATURES_RANGE):]):
        print(f"  {i+10:2d}. {name:20s} - threshold: {params['threshold']}")

    print("\n" + "=" * 70)
    print("Expected Feature Counts (approximate):")
    print("=" * 70)
    print("\nSIFT:")
    for nf in SIFT_NFEATURES_RANGE:
        if nf == 0:
            print(f"  nfeatures={nf:5d} -> ~1000-3000 features")
        else:
            print(f"  nfeatures={nf:5d} -> max {nf} features")

    print("\nAKAZE (approximate, image-dependent):")
    for t in AKAZE_THRESHOLD_RANGE:
        if t >= 0.005:
            count = "~20-100"
        elif t >= 0.001:
            count = "~100-400"
        elif t >= 0.0001:
            count = "~400-2500"
        else:
            count = "~2500-6000+"
        print(f"  threshold={t:8.5f} -> {count} features")
