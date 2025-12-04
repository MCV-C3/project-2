from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"],
                            keypoints_list: list = None, images: list = None):
    
    if bovw.spatial_pyramid and keypoints_list is not None and images is not None:
        # Use spatial pyramid
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
        # Regular BOVW without spatial pyramid
        return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])


def test(dataset: List[Tuple[Type[Image.Image], int]]
         , bovw: Type[BOVW],
         classifier:Type[object]):

    use_spatial = bovw.spatial_pyramid is not None
    test_images = []
    test_descriptors = []
    descriptors_labels = []
    test_keypoints = [] if use_spatial else None

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
    

def train(dataset: List[Tuple[Type[Image.Image], int]],
           bovw:Type[BOVW]):

    use_spatial = bovw.spatial_pyramid is not None
    train_images = []
    all_descriptors = []
    all_labels = []
    all_keypoints = [] if use_spatial else None

    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):
        image, label = dataset[idx]
        keypoints, descriptors = bovw._extract_features(image=np.array(image))

        if descriptors is not None:
            all_descriptors.append(descriptors)
            all_labels.append(label)
            if use_spatial:
                all_keypoints.append(keypoints)
                train_images.append(image)

    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(
        descriptors=all_descriptors,
        bovw=bovw,
        keypoints_list=all_keypoints,
        images=train_images if use_spatial else None
    )

    print("Fitting the classifier")
    classifier = LogisticRegression(class_weight="balanced").fit(bovw_histograms, all_labels)

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

    for idx, cls_folder in enumerate(os.listdir(ImageFolder)):

        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")

            dataset.append((img_pil, map_classes[cls_folder]))


    return dataset


    


if __name__ == "__main__":
     #/home/cboned/data/Master/MIT_split
    data_train = Dataset(ImageFolder="../places_reduced/train")
    data_test = Dataset(ImageFolder="../places_reduced/val")

    detector_type = "SIFT"
    detector_kwargs = {"nfeatures": 1000}
    codebook_size = 300

    spatial_pyramid = "horizontal"  #  none, horizontal o vertical
    pyramid_levels = 2

    # Create BOVW with spatial pyramid configuration
    bovw = BOVW(
        detector_type=detector_type,
        detector_kwargs=detector_kwargs,
        codebook_size=codebook_size,
        spatial_pyramid=spatial_pyramid,
        pyramid_levels=pyramid_levels
    )

    # Train and test
    bovw, classifier = train(dataset=data_train, bovw=bovw)
    test(dataset=data_test, bovw=bovw, classifier=classifier)