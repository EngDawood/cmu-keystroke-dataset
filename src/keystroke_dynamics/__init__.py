"""Keystroke-dynamics anomaly detection (Killourhy & Maxion, 2009).

Public API:
    build_profile / score  -- the scaled-Manhattan matcher (reusable by the
                              Flask backend for live enrollment/verification)
    DETECTORS              -- all benchmark detectors, name -> score function
    compute_eer            -- per-subject equal error rate
"""

from keystroke_dynamics.detectors import DETECTORS
from keystroke_dynamics.evaluation import compute_eer
from keystroke_dynamics.model import build_profile, score

__all__ = ["build_profile", "score", "DETECTORS", "compute_eer"]
