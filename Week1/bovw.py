import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import os
import glob


from typing import *

class BOVW():

    def __init__(self, detector_type="AKAZE", codebook_size:int=50, detector_kwargs:dict={}, codebook_kwargs:dict={},
                 spatial_pyramid=None, pyramid_levels=1, dense_sift=False, dense_step=8, dense_scales=[8, 16, 24, 32],
                 use_dense_cache=False, dense_cache_dir="cache/dense_sift"):
        self.detector_type = detector_type
        self.detector_kwargs = detector_kwargs  # Store detector_kwargs for cache key generation
        if self.detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)
        elif self.detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create(**detector_kwargs)
        elif self.detector_type == 'ORB':
            self.detector = cv2.ORB_create(**detector_kwargs)
        else:
            raise ValueError("Detector type must be 'SIFT', 'AKAZE', or 'ORB'")

        self.codebook_size = codebook_size
        self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, **codebook_kwargs)

        self.spatial_pyramid = spatial_pyramid
        self.pyramid_levels = pyramid_levels

        self.dense_sift = dense_sift
        self.dense_step = dense_step
        self.dense_scales = dense_scales if isinstance(dense_scales, list) else [dense_scales]

        self.use_dense_cache = use_dense_cache
        self.dense_cache_dir = dense_cache_dir
        self._dense_cache = None

        if self.use_dense_cache and self.dense_sift and self.detector_type == 'SIFT':
            from dense_cache import DenseSIFTCache
            self._dense_cache = DenseSIFTCache(self.dense_cache_dir)


    def _extract_features(self, image: Literal["H", "W", "C"], image_id: str = None) -> Tuple:
        if self.dense_sift and self.detector_type == 'SIFT':
            if self.use_dense_cache and self._dense_cache is not None and image_id is not None:
                try:
                    return self._extract_from_cache(image_id)
                except (FileNotFoundError, ValueError) as e:
                    print(f"Cache miss for {image_id}, extracting: {e}")
                    pass

            return self._extract_dense_sift(image)
        else:
            return self.detector.detectAndCompute(image, None)

    def _extract_from_cache(self, image_id: str) -> Tuple:
        if self._dense_cache is None:
            raise ValueError("Cache not initialized")

        keypoints, descriptors = self._dense_cache.load_and_subsample(
            image_id=image_id,
            target_step=self.dense_step,
            target_scales=self.dense_scales
        )

        return keypoints, descriptors

    def _extract_dense_sift(self, image: np.ndarray) -> Tuple:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        height, width = gray.shape
        keypoints = []

        for scale in self.dense_scales:
            for y in range(0, height - scale, self.dense_step):
                for x in range(0, width - scale, self.dense_step):
                    kp = cv2.KeyPoint(x=float(x + scale // 2),
                                     y=float(y + scale // 2),
                                     size=float(scale))
                    keypoints.append(kp)

        if len(keypoints) > 0:
            _, descriptors = self.detector.compute(gray, keypoints)
            if descriptors is not None:
                return keypoints, descriptors

        return [], None
    
    
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
        codebook_descriptor = codebook_descriptor / np.linalg.norm(codebook_descriptor)

        return codebook_descriptor

    def _compute_spatial_pyramid_descriptor(self, image: np.ndarray, keypoints: list,
                                           descriptors: np.ndarray, kmeans: Type[KMeans]) -> np.ndarray:
        height, width = image.shape[:2]
        all_histograms = []

        visual_words = kmeans.predict(descriptors)

        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

        if self.spatial_pyramid == 'horizontal':
            strip_height = height / self.pyramid_levels
            for i in range(self.pyramid_levels):
                y_start = i * strip_height
                y_end = (i + 1) * strip_height

                mask = (kp_coords[:, 1] >= y_start) & (kp_coords[:, 1] < y_end)
                strip_words = visual_words[mask]

                hist = np.zeros(kmeans.n_clusters)
                for word in strip_words:
                    hist[word] += 1

                if np.linalg.norm(hist) > 0:
                    hist = hist / np.linalg.norm(hist)

                all_histograms.append(hist)

        elif self.spatial_pyramid == 'vertical':
            strip_width = width / self.pyramid_levels
            for i in range(self.pyramid_levels):
                x_start = i * strip_width
                x_end = (i + 1) * strip_width

                mask = (kp_coords[:, 0] >= x_start) & (kp_coords[:, 0] < x_end)
                strip_words = visual_words[mask]

                hist = np.zeros(kmeans.n_clusters)
                for word in strip_words:
                    hist[word] += 1

                if np.linalg.norm(hist) > 0:
                    hist = hist / np.linalg.norm(hist)

                all_histograms.append(hist)

        elif self.spatial_pyramid == 'square':
            rows = cols = int(np.sqrt(self.pyramid_levels))
            if rows * cols < self.pyramid_levels:
                rows += 1

            cell_height = height / rows
            cell_width = width / cols

            for i in range(rows):
                for j in range(cols):
                    if len(all_histograms) >= self.pyramid_levels:
                        break

                    y_start = i * cell_height
                    y_end = (i + 1) * cell_height
                    x_start = j * cell_width
                    x_end = (j + 1) * cell_width

                    mask = ((kp_coords[:, 0] >= x_start) & (kp_coords[:, 0] < x_end) &
                           (kp_coords[:, 1] >= y_start) & (kp_coords[:, 1] < y_end))
                    cell_words = visual_words[mask]

                    hist = np.zeros(kmeans.n_clusters)
                    for word in cell_words:
                        hist[word] += 1

                    if np.linalg.norm(hist) > 0:
                        hist = hist / np.linalg.norm(hist)

                    all_histograms.append(hist)

        return np.concatenate(all_histograms)       
    




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

