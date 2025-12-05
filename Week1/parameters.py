import numpy as np

# Classifier parameters
CLASSIFIER_PARAMETERS = {
    "LogisticRegression": {
        "penalty": ["l2"],
        "C": [1.0],  # Moderate regularization - best for dense features
        "class_weight": ["balanced"],
        "max_iter": [1000]
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
# TODO: Add parameters for each detector if necessary
DETECTOR_PARAMETERS = {
    "SIFT": {
        "nfeatures":500
    },

    "AKAZE": {
    },

    "ORB": {
    }
}
SELECTED_DETECTOR = "SIFT"


# Codebook parameters
CODEBOOK_SIZE = [300]


# Spatial Pyramid parameters
SPATIAL_PYRAMID_TYPES = [None]
PYRAMID_LEVELS = [1]  # Number of divisions (1 = no pyramid)


# Dense SIFT parameters
USE_DENSE_SIFT = [True]  # Dense SIFT for comprehensive scene coverage
DENSE_STEP_SIZES = [8]  # Optimal balance: dense coverage without excessive computation
DENSE_SCALES = [
    [8, 16, 24, 32],  # Multiple scales: captures textures (8) to structures (32)
]

SCALER_TYPES = [None, "l1", "l2", "standard"]
FEATURE_ENCODINGS = ["fisher"]