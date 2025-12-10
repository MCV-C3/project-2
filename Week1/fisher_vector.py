import numpy as np
from sklearn.mixture import GaussianMixture


class FisherEncoder:
    def __init__(self, n_components=32):
        self.n_components = n_components
        self.gmm = None

    def fit(self, descriptors_list):
        # Concatenate all descriptors
        all_desc = np.concatenate(descriptors_list, axis=0)
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            max_iter=200
        ).fit(all_desc)

    def compute_fisher_vector(self, descriptors, gmm):
        """
        descriptors: (N, D)
        gmm: fitted sklearn GaussianMixture
        returns: Fisher Vector (1 x (2*D*K))
        """

        if descriptors is None or len(descriptors) == 0:
            return np.zeros(2 * gmm.n_components * gmm.means_.shape[1])

        # responsibilities (posterior probabilities)
        Q = gmm.predict_proba(descriptors)              # (N, K)
        N_k = np.sum(Q, axis=0)                         # (K)
        D = descriptors.shape[1]
        K = gmm.n_components

        # u: first order stats
        diff = descriptors[:, None, :] - gmm.means_[None, :, :]
        u = (Q[..., None] * diff).sum(axis=0) / np.sqrt(gmm.covariances_)

        # v: second order stats
        v = (Q[..., None] * (diff**2 - gmm.covariances_)).sum(axis=0) / (np.sqrt(2) * gmm.covariances_)

        fv = np.concatenate([u.flatten(), v.flatten()])

        # Normalization (critical!)
        fv = np.sign(fv) * np.sqrt(np.abs(fv))  # power norm
        fv /= np.linalg.norm(fv) + 1e-10        # L2 norm

        return fv

    def transform(self, descriptors_list):
        return np.array([self.compute_fisher_vector(desc, self.gmm) for desc in descriptors_list])
