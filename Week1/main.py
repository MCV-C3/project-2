import os
import pickle
from pathlib import Path

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm

from sklearn.metrics import accuracy_score
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
    classifier.fit(bovw_histograms, all_labels)

    print("Accuracy on Phase[Train]:", accuracy_score(y_true=all_labels, y_pred=classifier.predict(bovw_histograms)))
    
    return bovw, classifier


def Dataset(ImageFolder:str = "data/MIT_split/train") -> List[Tuple[Type[Image.Image], int]]:

    """
    Expected Structure:

        ImageFolder/<cls label>/xxx1.png
        ImageFolder/<cls label>/xxx2.png
        ImageFolder/<cls label>/xxx3.png
        ...

        Example:
            ImageFolder/cat/123.png
            ImageFolder/cat/nsdf3.png
            ImageFolder/cat/[...]/asd932_.png
    
    """

    map_classes = {clsi: idx for idx, clsi  in enumerate(os.listdir(ImageFolder))}
    dataset :List[Tuple] = []

    for _, cls_folder in enumerate(os.listdir(ImageFolder)):
        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")
            dataset.append((img_pil, map_classes[cls_folder]))

    return dataset


if __name__ == "__main__":
    data_train = Dataset(ImageFolder="./data/MIT_split/train")
    data_test = Dataset(ImageFolder="./data/MIT_split/test") 

    bovw = BOVW() # default AKAZE
    classifier_kwargs = {"C" : 0.1}
    bovw, classifier = train(dataset=data_train, bovw=bovw,
                             classifier_name="hist", 
                             classifier_kwargs = classifier_kwargs)
    test(dataset=data_test, bovw=bovw, classifier=classifier)