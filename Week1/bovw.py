import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import os
import glob


from typing import *

class BOVW():

    def __init__(self, detector_type="AKAZE", codebook_size:int=50, detector_kwargs:dict={}, codebook_kwargs:dict={},
                 spatial_pyramid:str=None, pyramid_levels:int=1):
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create(**detector_kwargs)
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(**detector_kwargs)
        else:
            raise ValueError("Detector type must be 'SIFT', 'SURF', or 'ORB'")

        self.codebook_size = codebook_size
        self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, **codebook_kwargs)
        self.spatial_pyramid = spatial_pyramid
        self.pyramid_levels = pyramid_levels
        
               
    ## Modify this function in order to be able to create a dense sift
    def _extract_features(self, image: Literal["H", "W", "C"]) -> Tuple:

        return self.detector.detectAndCompute(image, None)
    
    
    def _update_fit_codebook(self, descriptors: Literal["N", "T", "d"])-> Tuple[Type[MiniBatchKMeans],
                                                                               Literal["codebook_size", "d"]]:
        
        all_descriptors = np.vstack(descriptors)

        self.codebook_algo = self.codebook_algo.partial_fit(X=all_descriptors)

        return self.codebook_algo, self.codebook_algo.cluster_centers_
    
    def _compute_codebook_descriptor(self, descriptors: Literal["1 T d"], kmeans: Type[KMeans]) -> np.ndarray:

        visual_words = kmeans.predict(descriptors)


        # Create a histogram of visual words
        codebook_descriptor = np.zeros(kmeans.n_clusters)
        for label in visual_words:
            codebook_descriptor[label] += 1

        # Normalize the histogram (optional)
        if np.linalg.norm(codebook_descriptor) > 0:
            codebook_descriptor = codebook_descriptor / np.linalg.norm(codebook_descriptor)

        return codebook_descriptor

    def _extract_features_with_spatial_pyramid(self, image: Literal["H", "W", "C"]) -> Tuple[List, List]:
        keypoints, descriptors = self._extract_features(image)
        return keypoints, descriptors

    def _get_spatial_regions(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        h, w = image_shape
        regions = []

        if self.spatial_pyramid is None:
            return [(0, 0, w, h)]

        elif self.spatial_pyramid == 'horizontal':
            # Divide image into horizontal strips
            strip_height = h // self.pyramid_levels
            for i in range(self.pyramid_levels):
                y1 = i * strip_height
                y2 = h if i == self.pyramid_levels - 1 else (i + 1) * strip_height
                regions.append((0, y1, w, y2))

        elif self.spatial_pyramid == 'vertical':
            # Divide image into vertical strips
            strip_width = w // self.pyramid_levels
            for i in range(self.pyramid_levels):
                x1 = i * strip_width
                x2 = w if i == self.pyramid_levels - 1 else (i + 1) * strip_width
                regions.append((x1, 0, x2, h))

        elif self.spatial_pyramid == 'square':
            # Divide image into square grid
            strip_height = h // self.pyramid_levels
            strip_width = w // self.pyramid_levels
            for i in range(self.pyramid_levels):
                for j in range(self.pyramid_levels):
                    y1 = i * strip_height
                    y2 = h if i == self.pyramid_levels - 1 else (i + 1) * strip_height
                    x1 = j * strip_width
                    x2 = w if j == self.pyramid_levels - 1 else (j + 1) * strip_width
                    regions.append((x1, y1, x2, y2))

        return regions

    def _compute_spatial_pyramid_descriptor(self, image: Literal["H", "W", "C"], keypoints, descriptors, kmeans: Type[KMeans]) -> np.ndarray:
        if descriptors is None or len(descriptors) == 0:
            # Return zero vector if no descriptors
            regions = self._get_spatial_regions(image.shape[:2])
            return np.zeros(kmeans.n_clusters * len(regions))

        h, w = image.shape[:2]
        regions = self._get_spatial_regions((h, w))

        pyramid_histogram = []

        for (x1, y1, x2, y2) in regions:
            # Find keypoints within this region
            region_descriptors = []
            for kp, desc in zip(keypoints, descriptors):
                kp_x, kp_y = kp.pt
                if x1 <= kp_x < x2 and y1 <= kp_y < y2:
                    region_descriptors.append(desc)

            # Compute histogram for this region
            if len(region_descriptors) > 0:
                region_descriptors = np.array(region_descriptors)
                region_histogram = self._compute_codebook_descriptor(region_descriptors, kmeans)
            else:
                region_histogram = np.zeros(kmeans.n_clusters)

            pyramid_histogram.append(region_histogram)

        # Concatenate all region histograms
        pyramid_histogram = np.concatenate(pyramid_histogram)

        # Normalize the final concatenated histogram
        if np.linalg.norm(pyramid_histogram) > 0:
            pyramid_histogram = pyramid_histogram / np.linalg.norm(pyramid_histogram)

        return pyramid_histogram       
    




def visualize_bow_histogram(histogram, image_index, output_folder="./test_example.jpg"):
    """
    Visualizes the Bag of Visual Words histogram for a specific image and saves the plot to the output folder.
    
    Args:
        histogram (np.array): BoVW histogram.
        cluster_centers (np.array): Cluster centers (visual words).
        image_index (int): Index of the image for reference.
        output_folder (str): Folder where the plot will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(histogram)), histogram)
    plt.title(f"BoVW Histogram for Image {image_index}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.xticks(range(len(histogram)))
    
    # Save the plot to the output folder
    plot_path = os.path.join(output_folder, f"bovw_histogram_image_{image_index}.png")
    plt.savefig(plot_path)
    
    # Optionally, close the plot to free up memory
    plt.close()

    print(f"Plot saved to: {plot_path}")

