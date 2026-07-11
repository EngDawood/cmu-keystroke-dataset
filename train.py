"""Benchmark evaluation on the CMU Keystroke Dynamics dataset.

Reproduces the evaluation protocol of:
    K. Killourhy and R. Maxion, "Comparing Anomaly-Detection Algorithms for
    Keystroke Dynamics", DSN 2009.  (http://www.cs.cmu.edu/~keystroke/)

For each of the 51 subjects:
  - Train:    first 200 repetitions of the genuine user.
  - Genuine:  last 200 repetitions of the genuine user.
  - Impostor: first 5 repetitions of each of the other 50 subjects (250).
  - EER:      point where the false-accept rate equals the false-reject rate.

Run with:  uv run train.py

Outputs: results/results_summary.csv, results/roc_scaled_manhattan.png,
results/confusion_matrix.png, and a printed summary table.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from keystroke_dynamics import DETECTORS, compute_eer

CSV_PATH = os.path.join("data", "DSL-StrongPasswordData.csv")
RESULTS_DIR = "results"

N_TRAIN = 200        # first 200 repetitions -> enrollment
N_IMPOSTOR_REPS = 5  # first 5 repetitions of every other subject

# Published mean EERs from Killourhy & Maxion (2009), for the sanity check.
EXPECTED_EER = {
    "Scaled Manhattan": 0.096,
    "Nearest Neighbor (Mahalanobis)": 0.100,
    "Manhattan": 0.153,
    "Euclidean": 0.171,
}


def load_dataset(csv_path: str):
    """Return {subject: feature matrix ordered by (sessionIndex, rep)}."""
    df = pd.read_csv(csv_path)
    feature_cols = df.columns[3:]  # the 31 timing features
    assert len(feature_cols) == 31, f"expected 31 features, got {len(feature_cols)}"
    # Rows are already ordered by (sessionIndex, rep) within each subject;
    # sort defensively so "first 200" / "last 200" match the protocol.
    df = df.sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True)
    subjects = sorted(df["subject"].unique())
    data = {s: df.loc[df["subject"] == s, feature_cols].to_numpy(dtype=float)
            for s in subjects}
    print(f"Loaded {len(df)} rows, {len(subjects)} subjects, "
          f"{len(feature_cols)} timing features.\n")
    return subjects, data


def plot_roc(genuine: np.ndarray, impostor: np.ndarray, mean_eer: float,
             path: str):
    """Pooled ROC curve for the scaled Manhattan detector."""
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    far, tpr, _ = roc_curve(y_true, -np.concatenate([genuine, impostor]))

    plt.figure(figsize=(6, 6))
    plt.plot(far, tpr, color="tab:blue",
             label="Scaled Manhattan (pooled over 51 subjects)")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
    plt.xlabel("False Accept Rate (FAR)")
    plt.ylabel("Genuine Accept Rate (1 − FRR)")
    plt.title(f"ROC — Scaled Manhattan detector\n"
              f"mean per-subject EER = {mean_eer:.3f}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(confusion: np.ndarray, path: str):
    """Genuine/impostor confusion matrix at the per-subject EER thresholds."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], ["Accepted\n(genuine)", "Rejected\n(impostor)"])
    ax.set_yticks([0, 1], ["Genuine", "Impostor"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — Scaled Manhattan\n"
                 "(per-subject EER threshold, pooled over 51 subjects)")
    for r in range(2):
        for c in range(2):
            frac = confusion[r, c] / confusion[r].sum()
            ax.text(c, r, f"{confusion[r, c]}\n({frac:.1%})",
                    ha="center", va="center",
                    color="white" if confusion[r, c] > confusion.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    subjects, data = load_dataset(CSV_PATH)

    per_subject_eer = {name: [] for name in DETECTORS}
    # Extra collections for the scaled-Manhattan plots.
    pooled_genuine, pooled_impostor = [], []
    confusion = np.zeros((2, 2), dtype=int)  # [true genuine/impostor][pred]

    for subject in subjects:
        genuine_all = data[subject]
        train = genuine_all[:N_TRAIN]
        genuine_test = genuine_all[N_TRAIN:]
        impostor_test = np.vstack([data[o][:N_IMPOSTOR_REPS]
                                   for o in subjects if o != subject])

        for name, detector in DETECTORS.items():
            g = detector(train, genuine_test)
            i = detector(train, impostor_test)
            eer, thr = compute_eer(g, i)
            per_subject_eer[name].append(eer)

            if name == "Scaled Manhattan":
                pooled_genuine.append(g)
                pooled_impostor.append(i)
                # Confusion counts at this subject's EER threshold
                # (accept if score <= threshold).
                confusion[0, 0] += int((g <= thr).sum())   # genuine accepted
                confusion[0, 1] += int((g > thr).sum())    # genuine rejected
                confusion[1, 0] += int((i <= thr).sum())   # impostor accepted
                confusion[1, 1] += int((i > thr).sum())    # impostor rejected

    # ---------------- results CSV ----------------
    summary = pd.DataFrame({"subject": subjects})
    for name in DETECTORS:
        col = "eer_" + name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        summary[col] = per_subject_eer[name]
    summary.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)

    # ---------------- plots (scaled Manhattan) ----------------
    mean_eer_sm = float(np.mean(per_subject_eer["Scaled Manhattan"]))
    plot_roc(np.concatenate(pooled_genuine), np.concatenate(pooled_impostor),
             mean_eer_sm, os.path.join(RESULTS_DIR, "roc_scaled_manhattan.png"))
    plot_confusion(confusion, os.path.join(RESULTS_DIR, "confusion_matrix.png"))

    # ---------------- summary table ----------------
    print(f"{'Detector':<34}{'Mean EER':>10}{'Std':>8}{'Expected':>10}")
    print("-" * 62)
    for name in DETECTORS:
        eers = np.array(per_subject_eer[name])
        print(f"{name:<34}{eers.mean():>10.3f}{eers.std():>8.3f}"
              f"{EXPECTED_EER[name]:>10.3f}")
    print(f"\nOutputs written to '{RESULTS_DIR}/': results_summary.csv, "
          f"roc_scaled_manhattan.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
