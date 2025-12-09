import numpy as np

# Test Configuration: Dense SIFT Step Size Test
# Research Question: How does the step size affect dense SIFT performance with fixed scale?

# Baseline configuration
CODEBOOK_SIZE_BASELINE = 512

# No PCA
PCA_DIMENSIONS = [None]

# Logistic Regression classifier
CLASSIFIER_PARAMETERS = {
    "LogisticRegression": {
        "C": [1.0],
        "max_iter": [1000],
        "solver": ["lbfgs"],
    }
}

SELECTED_CLASSIFIER = ["LogisticRegression"]

# Fixed SIFT detector (dense SIFT will be enabled)
DETECTOR_TYPE_MAP = {"SIFT": "SIFT"}
SELECTED_DETECTOR = ["SIFT"]
DETECTOR_PARAMETERS = {
    "SIFT": {
        "nfeatures": 0,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6
    }
}

# Enable dense SIFT
USE_DENSE_SIFT = [True]

# Fixed codebook size
CODEBOOK_SIZE = [CODEBOOK_SIZE_BASELINE]

# No spatial pyramid
SPATIAL_PYRAMID_TYPES = [None]
PYRAMID_LEVELS = [1]

# TEST: Dense SIFT step sizes with fixed scale
DENSE_STEP_SIZES = [2, 4, 6, 10, 16, 24, 32]
DENSE_SCALES = [[16]]  # Fixed scale

# Single encoding and scaler for consistent comparison
FEATURE_ENCODINGS = ["bovw"]
SCALER_TYPES = ["none"]

# This creates 10 step sizes × 1 encoding × 1 scaler × 1 classifier = 10 configurations total

# W&B project name
WANDB_PROJECT = "Dense-SIFT-Step-Test"

# CSV output filename for this experiment
CSV_FILENAME = "dense_sift_step_test_results.csv"
