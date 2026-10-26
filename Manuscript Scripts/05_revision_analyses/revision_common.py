"""Shared helpers for the reviewer-response analyses.

Every script in this folder reuses the *exact* scoring functions of the
main age-axis pipeline (`03_age_axis_analysis/analysis_utils.py`, imported
unchanged) and reads the participant-level 28-feature vectors that the
production run of that pipeline wrote to `results/robust_median/01_dataset/`.
Nothing in the main pipeline or its results is modified.

Paths
-----
results root : $FAPA_OUTPUT_ROOT, default <repo>/results
outputs      : <results root>/revision_analyses/<analysis>/
private data : $TCFC_DATA_DIR, default <repo>/data (not distributed)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PIPELINE_DIR = REPO_ROOT / "03_age_axis_analysis"
RESULTS_BASE = Path(os.environ.get("FAPA_OUTPUT_ROOT", REPO_ROOT / "results"))
METHOD_DIR = RESULTS_BASE / "robust_median"     # branch reported in the manuscript
MAHALANOBIS_DIR = METHOD_DIR / "mahalanobis_ledoit_wolf"
OUTPUT_DIR = RESULTS_BASE / "revision_analyses"
DATA_DIR = Path(os.environ.get("TCFC_DATA_DIR", REPO_ROOT / "data"))

# Configure the pipeline config *before* importing it.
os.environ["FAPA_ALT2_METHOD"] = "robust_median"
os.environ["FAPA_OUTPUT_ROOT"] = str(RESULTS_BASE)
sys.path.insert(0, str(PIPELINE_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis_config as cfg  # noqa: E402
from analysis_utils import fit_axis, score_axis, winsorize_axis_data  # noqa: E402

PRIMARY = "geometric_median"
ESTIMATORS = ("geometric_median", "coordinate_median", "mean")
YOUTH_METRICS = ("position", "relative_distance_youth", "rsa_fisher_difference")
METRIC_LABELS = {
    "position": "Phenotype position",
    "deviation": "Off-axis deviation",
    "relative_distance_youth": "Relative distance",
    "rsa_fisher_difference": "RSA difference",
}
ESTIMATOR_LABELS = {
    "geometric_median": "Geometric median (primary)",
    "coordinate_median": "Coordinate-wise median",
    "mean": "Arithmetic mean",
}

# Figure styling (neutral ink, one accent per role).
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NULL_FILL = "#b9b8b2"
ACCENT = "#2a78d6"     # observed statistic
ACCENT2 = "#eb6834"    # null centre
GROUP_COLORS = {"YAC": "#2a78d6", "OAC": "#eb6834", "TCOA": "#1baf7a"}


def out_dir(name: str) -> Path:
    path = OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_features() -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray]]:
    """Return the feature table, feature names and per-group arrays."""
    features = pd.read_csv(
        METHOD_DIR / "01_dataset" / "participant_features.csv",
        dtype={"participant": str},
    )
    names = pd.read_csv(METHOD_DIR / "01_dataset" / "feature_metadata.csv")[
        "feature_name"
    ].tolist()
    arrays = {
        group: features.loc[features["group"].eq(group), names].to_numpy(float)
        for group in ("YAC", "OAC", "TCOA")
    }
    return features, names, arrays


def score_groups(young, old, test, estimator=PRIMARY):
    """Identical to stage-3 `score_groups` for the robust_median branch.

    Radial winsorization is fitted on pooled young+old, applied to all three
    groups; the axis is fitted on (winsorized) young and old only; the test
    group is scored out-of-sample.
    """
    _, (adj_young, adj_old, adj_test) = winsorize_axis_data(young, old, test)
    model = fit_axis(adj_young, adj_old, estimator=estimator)
    return score_axis(adj_test, model), score_axis(adj_old, model), model


def median_difference(young, old, test, metric="position", estimator=PRIMARY):
    test_scores, old_scores, _ = score_groups(young, old, test, estimator)
    return float(np.median(test_scores[metric]) - np.median(old_scores[metric]))


def upper_tail_p(null: np.ndarray, observed: float) -> float:
    """The manuscript's one-sided rule: (1 + #{null >= observed}) / (N + 1)."""
    null = np.asarray(null, float)
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
