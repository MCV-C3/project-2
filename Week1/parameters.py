import numpy as np

# AdaBoost Best Single Configuration
# Based on ensemble learning best practices for BoVW
# Using SIFT with spatial pyramid (best configuration from previous experiments)
# Codebook size: 1000, Spatial pyramid: square 2x2

# Research Question: Can a well-tuned AdaBoost ensemble improve over single classifiers?

# Baseline configuration
CODEBOOK_SIZE_BASELINE = 1000

# No PCA - use full feature representation
PCA_DIMENSIONS = [None]

# Single best AdaBoost configuration
# Using:
# - 200 estimators: enough to build strong ensemble without overfitting
# - learning_rate 1.0: standard rate, balances speed and performance
# - max_depth 3: shallow trees are good weak learners (not too weak, not too strong)
#   depth=1 (stumps) might be too weak for complex BoVW features
#   depth=3 gives trees enough capacity to learn useful patterns
CLASSIFIER_PARAMETERS = {
    "AdaBoost": {
        "n_estimators": [200],
        "learning_rate": [1.0],
        "base_estimator": ["DecisionTree"],
        "base_estimator_params": [{"max_depth": 3, "random_state": 42}],
    }
}

SELECTED_CLASSIFIER = ["AdaBoost"]

# Detector parameters - SIFT with standard configuration
DETECTOR_PARAMETERS = {
    "SIFT": {
        "nfeatures": 0,  # Unlimited features (best from nfeatures experiment)
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6
    }
}

DETECTOR_TYPE_MAP = {
    "SIFT": "SIFT",
}

# Single detector configuration - Standard SIFT
SELECTED_DETECTOR = ["SIFT"]
USE_DENSE_SIFT = [False]

# Fixed codebook size
CODEBOOK_SIZE = [CODEBOOK_SIZE_BASELINE]

# Use spatial pyramid - square grid with 2 levels (2x2 = 4 cells)
SPATIAL_PYRAMID_TYPES = ["square"]
PYRAMID_LEVELS = [2]

# Dense SIFT parameters - Not used
DENSE_STEP_SIZES = [8]
DENSE_SCALES = [[16]]

# W&B project name
WANDB_PROJECT = "BoVW-AdaBoost-Best"

# CSV output filename for this experiment
CSV_FILENAME = "adaboost_best_results.csv"

# Feature encoding and scaling options
SCALER_TYPES = [None]  # Can add: "l1", "l2", "standard"
FEATURE_ENCODINGS = ["bovw"]  # Can add: "fisher"
