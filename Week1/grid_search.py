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
from sklearn.model_selection import cross_val_score

from parameters import (
    CLASSIFIER_PARAMETERS, CODEBOOK_SIZE,
    SELECTED_DETECTOR, DETECTOR_PARAMETERS,
    SELECTED_CLASSIFIER,
)
from classifiers import create_classifier
from bovw import BOVW
from utils import compute_hash


root = Path.cwd() / "Week1"


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])

def _get_cache_file(dataset, bovw):
    dataset_id = len(dataset)
    cache_key = compute_hash({
        "dataset_id": dataset_id,
        "descriptor": str(bovw.detector_type)
    })
    return os.path.join(root / "cache", f"descriptors_{cache_key}.pkl")

def load_cache(dataset, bovw):
    cache_file = _get_cache_file(dataset, bovw)

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            all_descriptors, all_labels = pickle.load(f)
        return all_descriptors, all_labels
    else:
        return None
    
def cross_validation(classifier: Type[object], X, Y, cv=5):
    """
    Performs cross-validation and prints results.
    """

    print(f"--- Performing {cv}-Fold Cross-Validation ---")
    scores = cross_val_score(classifier, X, Y, cv=cv, scoring='accuracy')

    print(f"CV Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print("----------------------------------------------------")

    if wandb.run is not None:
        wandb.log({"cv_mean_accuracy": scores.mean()})

    return scores.mean()


def test(dataset: List[Tuple[Type[Image.Image], int]], 
         bovw: Type[BOVW], 
         classifier:Type[object]):
    
    test_descriptors = []
    descriptors_labels = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Eval]: Extracting the descriptors"):
        image, label = dataset[idx]
        _, descriptors = bovw._extract_features(image=np.array(image))
        
        if descriptors is not None:
            test_descriptors.append(descriptors)
            descriptors_labels.append(label)
            
    
    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=test_descriptors, bovw=bovw)
    
    print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    
    print("Accuracy on Phase[Test]:", accuracy_score(y_true=descriptors_labels, y_pred=y_pred))
    
def train(dataset: List[Tuple[Image.Image, int]],
          bovw: Type[BOVW],
          classifier_name: str = "logreg",
          classifier_kwargs: dict | None = None):
    
    if classifier_kwargs is None:
        classifier_kwargs = {}

    cached = load_cache(dataset, bovw)

    if cached is not None:
        print("Load cached descriptors.")
        all_descriptors, all_labels = cached
    else:
        all_descriptors = []
        all_labels = []
    
        for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):
            
            image, label = dataset[idx]
            _, descriptors = bovw._extract_features(image=np.array(image))
            
            if descriptors  is not None:
                all_descriptors.append(descriptors)
                all_labels.append(label)
        
        os.makedirs(root / "cache", exist_ok=True)
        cache_file = _get_cache_file(dataset, bovw)
        with open(cache_file, "wb") as f:
            pickle.dump((all_descriptors, all_labels), f)
            
    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=all_descriptors, bovw=bovw) 
    
    print(f"Creating classifier: {classifier_name}")
    classifier = create_classifier(classifier_name, **classifier_kwargs)

    print("Fitting the classifier")
    cross_validation(classifier, bovw_histograms, all_labels)
    classifier.fit(bovw_histograms, all_labels)

    train_accuracy = accuracy_score(y_true=all_labels, y_pred=classifier.predict(bovw_histograms))
    print("Accuracy on Phase[Train]:", train_accuracy)
    wandb.log({"Train Accuracy": train_accuracy})
    return bovw, classifier


def extract_test_accuracy(bovw, classifier, dataset):
    """
    Compute BoVW histograms and accuracy on test set
    """
    descriptors_list = []
    labels = []

    for img, label in dataset:
        _, descriptors = bovw._extract_features(np.array(img))
        if descriptors is not None:
            descriptors_list.append(descriptors)
            labels.append(label)

    histograms = np.array([
        bovw._compute_codebook_descriptor(d, bovw.codebook_algo)
        for d in descriptors_list
    ])

    preds = classifier.predict(histograms)
    acc = accuracy_score(labels, preds)
    return acc

def gridsearch(train_data, test_data):
    """
    Full grid search over:
    - classifier types
    - classifier hyperparameters
    - codebook sizes
    And logs everything to W&B.
    """

    best_acc = -1
    best_config = None
    run_index = 0

    for classifier_name in SELECTED_CLASSIFIER:
        param_grid = CLASSIFIER_PARAMETERS[classifier_name]
        param_names = list(param_grid.keys())
        param_value_lists = list(param_grid.values())
        combinations = list(itertools.product(*param_value_lists))

        for codebook_size in CODEBOOK_SIZE:
            for param_tuple in combinations:
                clf_params = dict(zip(param_names, param_tuple))
                run_index += 1

                print("\n------------")
                print(f" RUN {run_index}: Clf={classifier_name}, codebook={codebook_size}")
                print(f" params={clf_params}")
                print("-------------")

                wandb.init(
                    project="BoVW-GridSearch",
                    config={
                        "run_id": run_index,
                        "classifier": classifier_name,
                        "clf_params": clf_params,
                        "codebook_size": int(codebook_size),
                        "detector": SELECTED_DETECTOR,
                        "detector_params": DETECTOR_PARAMETERS[SELECTED_DETECTOR],
                    }
                )

                # Build BoVW
                bovw = BOVW(
                    detector_type=SELECTED_DETECTOR,
                    codebook_size=int(codebook_size),
                    detector_kwargs=DETECTOR_PARAMETERS[SELECTED_DETECTOR]
                )

                bovw, classifier = train(
                    dataset=train_data,
                    bovw=bovw,
                    classifier_name=classifier_name,
                    classifier_kwargs=clf_params,
                )

                acc = extract_test_accuracy(bovw, classifier, test_data)
                wandb.log({"Test Accuracy": acc})

                if acc > best_acc:
                    best_acc = acc
                    best_config = {
                        "classifier": classifier_name,
                        "codebook_size": codebook_size,
                        "params": clf_params,
                        "accuracy": acc
                    }

                wandb.finish()
    return best_config
