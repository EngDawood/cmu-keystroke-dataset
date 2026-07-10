"""Benchmark anomaly detectors from Killourhy & Maxion (DSN 2009).

Each detector takes the genuine user's training matrix and a test matrix and
returns one anomaly score per test row (lower = more genuine).  The scaled
Manhattan detector delegates to keystroke_dynamics.model, which is the
module reused by the live verification backend.
"""

import numpy as np

from keystroke_dynamics.model import build_profile, score


def scaled_manhattan_scores(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    profile = build_profile(train)
    return np.array([score(profile, x) for x in test])


def manhattan_scores(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    mean = train.mean(axis=0)
    return np.abs(test - mean).sum(axis=1)


def euclidean_scores(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    mean = train.mean(axis=0)
    return np.sqrt(((test - mean) ** 2).sum(axis=1))


def nearest_neighbor_mahalanobis_scores(train: np.ndarray,
                                        test: np.ndarray) -> np.ndarray:
    # Score = Mahalanobis distance to the nearest training sample (k=1),
    # using the covariance of the training data (Killourhy & Maxion, 2009).
    cov_inv = np.linalg.pinv(np.cov(train, rowvar=False))
    diffs = test[:, None, :] - train[None, :, :]          # (n_test, n_train, d)
    d2 = np.einsum("ijk,kl,ijl->ij", diffs, cov_inv, diffs)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


DETECTORS = {
    "Scaled Manhattan": scaled_manhattan_scores,
    "Nearest Neighbor (Mahalanobis)": nearest_neighbor_mahalanobis_scores,
    "Manhattan": manhattan_scores,
    "Euclidean": euclidean_scores,
}
