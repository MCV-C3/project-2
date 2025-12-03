import numpy as np

# Classifier parameters
CLASSIFIER_PARAMETERS = {
    "LogisticRegression": {
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1.0],
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
    },

    "AKAZE": {
    },

    "ORB": {
    }
}
SELECTED_DETECTOR = "SIFT"


# Codebook parameters
CODEBOOK_SIZE = [50]
