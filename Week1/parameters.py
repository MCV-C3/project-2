import numpy as np

# Classifier parameters
CLASSIFIER_PARAMETERS = {
    "LogisticRegression": {
        "class_weight": ["balanced"],  # Handle class imbalance
    },

    "SVM": {
        "kernel": ["linear"],
        "C": [0.01, 0.1, 1.0],
        "class_weight": ["balanced"]
    },

    "HistIntersectionSVM": {
        "C": np.linspace(0.001, 1.0, 50)
    }
}

SELECTED_CLASSIFIER = ["LogisticRegression"] # Selected Classifier


# Detector parameters
DETECTOR_PARAMETERS = {
    "SIFT": {
        "nfeatures": 0,  # 0 = unlimited features (standard SIFT default)
    },

    "AKAZE": {
        "descriptor_type": 5,  # MLDB (Modified-Local Difference Binary) - most robust
        "descriptor_size": 0,  # Full size descriptor (best quality)
        "descriptor_channels": 3,  # 3 channels for better discrimination
        "threshold": 0.001,  # Sensitive threshold for more features
        "nOctaves": 4,  # Standard pyramid octaves
        "nOctaveLayers": 4  # Layers per octave
    },

    "ORB": {
        "nfeatures": 1000,  # Good number of features for ORB
        "scaleFactor": 1.2,  # Standard scale between pyramid levels
        "nlevels": 8,  # Standard pyramid levels
        "edgeThreshold": 31,  # Standard border size
        "firstLevel": 0,  # Start from original image
        "WTA_K": 2,  # 2 points for BRIEF descriptor (standard)
        "patchSize": 31  # Standard patch size
    }
}

# Run all four descriptor configurations for comparison
SELECTED_DETECTOR = ["SIFT", ]


# Codebook parameters
CODEBOOK_SIZE = [300]  # Good vocabulary size for all descriptors


# Spatial Pyramid parameters
SPATIAL_PYRAMID_TYPES = ["square"]
PYRAMID_LEVELS = [5]  # No spatial pyramid for fair comparison


# Dense SIFT parameters
# First SIFT will use regular SIFT (False), second will use Dense SIFT (True)
USE_DENSE_SIFT = [False, ]
DENSE_STEP_SIZES = [8]  # Good step size for dense sampling
DENSE_SCALES = [
    [8, 16, 24, 32],  # Multi-scale: captures features at different scales
]

# Dense SIFT Cache parameters
USE_DENSE_CACHE = False  # Set to True to use pre-computed cache
DENSE_CACHE_DIR = "cache/dense_sift"  # Cache directory

# Cache building parameters (for build_dense_cache.py)
# Extract at high density, then subsample for experiments
CACHE_STEP = 4  # Dense grid for caching
CACHE_SCALES = [8, 12, 16, 24, 32]  # All scales you might want to test
