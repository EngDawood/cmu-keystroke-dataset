"""Equal-error-rate computation for the Killourhy & Maxion (2009) protocol."""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray):
    """EER of one subject, plus the score threshold at which it occurs.

    Scores are anomaly scores (lower = genuine), so the ROC is computed on
    negated scores with genuine as the positive class.  The EER is found by
    linear interpolation between the two ROC points where FAR - FRR changes
    sign.

    Returns (eer, threshold): a sample is accepted when score <= threshold.
    """
    y_true = np.concatenate([np.ones_like(genuine_scores),
                             np.zeros_like(impostor_scores)])
    y_score = -np.concatenate([genuine_scores, impostor_scores])
    far, tpr, thresholds = roc_curve(y_true, y_score)
    frr = 1.0 - tpr

    diff = far - frr
    idx = np.argmax(diff >= 0)  # first point where FAR >= FRR
    if diff[idx] == 0 or idx == 0:
        eer = far[idx]
        thr = -thresholds[idx]
    else:
        # Interpolate between points idx-1 (FAR < FRR) and idx (FAR > FRR).
        x0, x1 = diff[idx - 1], diff[idx]
        w = -x0 / (x1 - x0)
        eer = far[idx - 1] + w * (far[idx] - far[idx - 1])
        thr = -(thresholds[idx - 1] + w * (thresholds[idx] - thresholds[idx - 1]))
    return float(eer), float(thr)
