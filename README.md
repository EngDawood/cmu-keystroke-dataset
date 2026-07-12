# Keystroke Dynamics — CMU Benchmark Pipeline

Behavioral biometrics-based intrusion detection using keystroke dynamics,
evaluated on the [CMU Keystroke Dynamics Benchmark](http://www.cs.cmu.edu/~keystroke/)
following the protocol of Killourhy & Maxion (DSN 2009).

## Project structure

```
data/                        CMU dataset (51 subjects × 400 reps of ".tie5Roanl")
src/keystroke_dynamics/      installable package
    model.py                 scaled-Manhattan matcher (build_profile / score) —
                             reusable by the Flask verification backend
    detectors.py             all four benchmark detectors
    evaluation.py            EER computation
train.py                     benchmark evaluation script
notebooks/                   interactive walkthrough covering the full pipeline
results/                     results_summary.csv + ROC + confusion matrix
```

## Setup & run

Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync          # create the environment and install dependencies
uv run train.py  # run the full benchmark
```

## Notebook walkthrough

`notebooks/keystroke_dynamics_walkthrough.ipynb` covers the whole pipeline interactively:
downloading the dataset, exploring the features, the scaled-Manhattan matcher, all four
detectors, EER computation, the full benchmark loop, results, and plots.

Run locally:

```bash
uv run jupyter notebook notebooks/keystroke_dynamics_walkthrough.ipynb
```

Or open directly in Google Colab (the first cell clones this repo and installs
dependencies automatically):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EngDawood/cmu-keystroke-dataset/blob/main/notebooks/keystroke_dynamics_walkthrough.ipynb)

## Results (mean EER ± std over 51 subjects)

| Detector | Mean EER | Published (K&M 2009) |
|---|---|---|
| **Scaled Manhattan** (primary) | **0.096 ± 0.069** | 0.096 |
| Nearest Neighbor (Mahalanobis) | 0.100 ± 0.064 | 0.100 |
| Manhattan | 0.153 ± 0.092 | 0.153 |
| Euclidean | 0.171 ± 0.094 | 0.171 |

## Reusing the matcher

```python
import numpy as np
from keystroke_dynamics import build_profile, score

profile = build_profile(enrollment_samples)   # (n_samples, n_features)
s = score(profile, new_sample)                # lower = more genuine
```
