import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import os
import glob


from typing import *

class BOVW():

    def __init__(self, detector_type="AKAZE", codebook_size:int=50, detector_kwargs:dict={}, codebook_kwargs:dict={},
                 spatial_pyramid=None, pyramid_levels=1, dense_sift=False, dense_step=8, dense_scales=[8, 16, 24, 32]):
        self.detector_type = detector_type
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

        # Spatial pyramid parameters
        self.spatial_pyramid = spatial_pyramid
        self.pyramid_levels = pyramid_levels

        # Dense SIFT parameters
        self.dense_sift = dense_sift
        self.dense_step = dense_step
        self.dense_scales = dense_scales if isinstance(dense_scales, list) else [dense_scales]
        
               
    ## Modify this function in order to be able to create a dense sift
    def _extract_features(self, image: Literal["H", "W", "C"]) -> Tuple:
        if self.dense_sift and self.detector_type == 'SIFT':
            return self._extract_dense_sift(image)
        else:
            return self.detector.detectAndCompute(image, None)

    def _extract_dense_sift(self, image: np.ndarray) -> Tuple:
        """
        Extract dense SIFT features at regular grid positions with multiple scales.

        Args:
            image: Input image (H, W, C)

        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: Array of SIFT descriptors
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        height, width = gray.shape
        keypoints = []

        # Create dense grid of keypoints at multiple scales
        for scale in self.dense_scales:
            # Generate grid positions
            for y in range(0, height - scale, self.dense_step):
                for x in range(0, width - scale, self.dense_step):
                    # Create keypoint at grid position with specific scale
                    kp = cv2.KeyPoint(x=float(x + scale // 2),
                                     y=float(y + scale // 2),
                                     size=float(scale))
                    keypoints.append(kp)

        # Compute SIFT descriptors at the dense keypoints
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
        """
        Compute spatial pyramid representation of the image.

        Args:
            image: Input image array
            keypoints: List of keypoints
            descriptors: Descriptors array
            kmeans: Trained K-means model

        Returns:
            Concatenated histogram from all pyramid levels
        """
        height, width = image.shape[:2]
        all_histograms = []

        # Get visual words for all descriptors
        visual_words = kmeans.predict(descriptors)

        # Create keypoint coordinates array
        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

        if self.spatial_pyramid == 'horizontal':
            # Divide image horizontally into strips
            strip_height = height / self.pyramid_levels
            for i in range(self.pyramid_levels):
                y_start = i * strip_height
                y_end = (i + 1) * strip_height

                # Find descriptors in this strip
                mask = (kp_coords[:, 1] >= y_start) & (kp_coords[:, 1] < y_end)
                strip_words = visual_words[mask]

                # Create histogram for this strip
                hist = np.zeros(kmeans.n_clusters)
                for word in strip_words:
                    hist[word] += 1

                # Normalize
                if np.linalg.norm(hist) > 0:
                    hist = hist / np.linalg.norm(hist)

                all_histograms.append(hist)

        elif self.spatial_pyramid == 'vertical':
            # Divide image vertically into strips
            strip_width = width / self.pyramid_levels
            for i in range(self.pyramid_levels):
                x_start = i * strip_width
                x_end = (i + 1) * strip_width

                # Find descriptors in this strip
                mask = (kp_coords[:, 0] >= x_start) & (kp_coords[:, 0] < x_end)
                strip_words = visual_words[mask]

                # Create histogram for this strip
                hist = np.zeros(kmeans.n_clusters)
                for word in strip_words:
                    hist[word] += 1

                # Normalize
                if np.linalg.norm(hist) > 0:
                    hist = hist / np.linalg.norm(hist)

                all_histograms.append(hist)

        elif self.spatial_pyramid == 'square':
            # Divide image into square grid
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

                    # Find descriptors in this cell
                    mask = ((kp_coords[:, 0] >= x_start) & (kp_coords[:, 0] < x_end) &
                           (kp_coords[:, 1] >= y_start) & (kp_coords[:, 1] < y_end))
                    cell_words = visual_words[mask]

                    # Create histogram for this cell
                    hist = np.zeros(kmeans.n_clusters)
                    for word in cell_words:
                        hist[word] += 1

                    # Normalize
                    if np.linalg.norm(hist) > 0:
                        hist = hist / np.linalg.norm(hist)

                    all_histograms.append(hist)

        # Concatenate all histograms
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

