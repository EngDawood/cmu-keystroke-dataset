"""Keystroke-dynamics matching algorithm (scaled Manhattan detector).

Implements the "Manhattan (scaled)" anomaly detector from:
    K. Killourhy and R. Maxion, "Comparing Anomaly-Detection Algorithms for
    Keystroke Dynamics", DSN 2009.

The module is dataset-agnostic: it only sees numeric feature vectors, so the
same code is used both by the offline CMU benchmark (train.py) and, later,
by the Flask backend for live enrollment/verification.
"""

import numpy as np

# Floor for the mean absolute deviation: a feature with zero variability in
# the training samples would otherwise cause a division by zero.
MAD_EPSILON = 1e-6


def build_profile(samples: np.ndarray) -> dict:
    """Build a user profile from enrollment samples.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, n_features)
        Timing feature vectors of the genuine user.

    Returns
    -------
    dict with keys:
        "mean" : per-feature mean, shape (n_features,)
        "mad"  : per-feature mean absolute deviation from the mean,
                 shape (n_features,), floored at MAD_EPSILON
    """
    samples = np.asarray(samples, dtype=float)
    mean = samples.mean(axis=0)
    mad = np.abs(samples - mean).mean(axis=0)
    mad = np.maximum(mad, MAD_EPSILON)
    return {"mean": mean, "mad": mad}


def score(profile: dict, sample: np.ndarray) -> float:
    """Scaled Manhattan distance of one sample to a profile.

    score = sum_i |x_i - mean_i| / mad_i

    Lower score = closer to the profile = more likely genuine.
    """
    sample = np.asarray(sample, dtype=float)
    return float(np.sum(np.abs(sample - profile["mean"]) / profile["mad"]))
