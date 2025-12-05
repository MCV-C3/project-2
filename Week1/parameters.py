import numpy as np

# Classifier parameters
CLASSIFIER_PARAMETERS = {
    "LogisticRegression": {
        "penalty": ["l2"],
        "C": [1.0],  # Good default regularization
        "class_weight": ["balanced"],  # Handle class imbalance
        "max_iter": [2000],  # Sufficient iterations for convergence
        "solver": ["lbfgs"]  # Efficient solver for multiclass problems
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
        "contrastThreshold": 0.04,  # Good balance for feature detection
        "edgeThreshold": 10  # Standard default
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
SELECTED_DETECTOR = ["SIFT", "SIFT", "AKAZE", "ORB"]


# Codebook parameters
CODEBOOK_SIZE = [512]  # Good vocabulary size for all descriptors


# Spatial Pyramid parameters
SPATIAL_PYRAMID_TYPES = [None]
PYRAMID_LEVELS = [1]  # No spatial pyramid for fair comparison


# Dense SIFT parameters
# First SIFT will use regular SIFT (False), second will use Dense SIFT (True)
USE_DENSE_SIFT = [False, True, False, False]
DENSE_STEP_SIZES = [8]  # Good step size for dense sampling
DENSE_SCALES = [
    [8, 16, 24, 32],  # Multi-scale: captures features at different scales
]
