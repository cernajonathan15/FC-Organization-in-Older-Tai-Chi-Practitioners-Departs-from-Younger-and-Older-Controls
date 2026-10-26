"""Stage 5: descriptive miscellaneous analyses and figure generation.

This script currently creates a descriptive decomposition of the difference
between TCOA and OAC mean squared off-axis deviation. It does not perform
feature-level hypothesis tests.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis_config as cfg


ANALYSIS_METHOD = "robust_median"
FIGURE_FILENAME = "off_axis_feature_contributions.png"


def readable_feature_name(feature_name: str) -> str:
    """Convert a stored feature identifier into a publication-facing label."""
    network_names = {
        "Visual": "Visual",
        "Somatomotor": "Somatomotor",
        "DorsalAttention": "Dorsal attention",
        "VentralAttention": "Ventral attention",
        "Limbic": "Limbic",
        "Frontoparietal": "Frontoparietal",
        "Default": "Default",
    }
    if feature_name.startswith("within__"):
        network = feature_name.removeprefix("within__")
        return f"Within: {network_names.get(network, network)}"
    if feature_name.startswith("between__"):
        first, second = feature_name.removeprefix("between__").split("__")
        return (
            f"Between: {network_names.get(first, first)} – "
            f"{network_names.get(second, second)}"
        )
    return re.sub(r"(?<!^)(?=[A-Z])", " ", feature_name).replace("__", " – ")


def calculate_feature_contributions(
    phenotype_dir: Path,
) -> tuple[pd.DataFrame, float]:
    """Decompose each participant's squared off-axis deviation by feature."""
    standardized_path = phenotype_dir / "participant_standardized_features.csv"
    scores_path = phenotype_dir / "participant_phenotype_scores.csv"
    model_path = phenotype_dir / "axis_model.npz"
    for path in (standardized_path, scores_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required stage-2 output is missing: {path}")

    standardized = pd.read_csv(standardized_path, dtype={"participant": str})
    scores = pd.read_csv(scores_path, dtype={"participant": str})
    model = np.load(model_path, allow_pickle=False)
    feature_names = [
        column
        for column in standardized.columns
        if column not in {"participant", "group", "axis_estimator"}
    ]
    if len(feature_names) != 28:
        raise ValueError(f"Expected 28 connectivity features; found {len(feature_names)}")

    merged = standardized.merge(
        scores[["participant", "group", "analysis_position", "analysis_deviation"]],
        on=["participant", "group"],
        how="inner",
        validate="one_to_one",
    )
    standardized_values = merged[feature_names].to_numpy(float)
    positions = merged["analysis_position"].to_numpy(float)
    projections = (
        model["old_endpoint"] + positions[:, None] * model["direction"]
    )
    residuals = standardized_values - projections

    # Each row sums to that participant's squared RMS off-axis deviation.
    squared_contributions = residuals**2 / len(feature_names)
    reconstructed_deviation = np.sqrt(np.sum(squared_contributions, axis=1))
    saved_deviation = merged["analysis_deviation"].to_numpy(float)
    maximum_error = float(np.max(np.abs(reconstructed_deviation - saved_deviation)))
    if maximum_error > 1e-10:
        raise RuntimeError(
            "Off-axis decomposition did not reconstruct the saved deviation: "
            f"maximum absolute error={maximum_error:.3e}"
        )

    groups = merged["group"].to_numpy(str)
    if np.count_nonzero(groups == "TCOA") != 15 or np.count_nonzero(groups == "OAC") != 15:
        raise ValueError("Expected 15 TCOA and 15 OAC participants")
    tcoa_mean = np.mean(squared_contributions[groups == "TCOA"], axis=0)
    oac_mean = np.mean(squared_contributions[groups == "OAC"], axis=0)

    summary = pd.DataFrame(
        {
            "feature_name": feature_names,
            "feature_label": [readable_feature_name(name) for name in feature_names],
            "tcoa_mean_squared_contribution": tcoa_mean,
            "oac_mean_squared_contribution": oac_mean,
            "tcoa_minus_oac": tcoa_mean - oac_mean,
        }
    ).sort_values("tcoa_minus_oac", ascending=False, ignore_index=True)

    observed_difference = float(
        np.mean(saved_deviation[groups == "TCOA"] ** 2)
        - np.mean(saved_deviation[groups == "OAC"] ** 2)
    )
    if not np.isclose(summary["tcoa_minus_oac"].sum(), observed_difference):
        raise RuntimeError("Feature differences do not sum to the group difference")
    return summary, maximum_error


def create_off_axis_contribution_figure(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create the descriptive TCOA-minus-OAC contribution plot."""
    values = summary["tcoa_minus_oac"].to_numpy(float)
    labels = summary["feature_label"].tolist()
    positions = np.arange(len(summary))
    colors = np.where(values >= 0, "#3B82F6", "#F97316")
    limit = float(np.max(np.abs(values)) * 1.14)

    fig, ax = plt.subplots(figsize=(13.5, 12.0))
    ax.barh(positions, values, color=colors, height=0.78)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(-limit, limit)
    ax.axvline(0, color="#111827", linewidth=1.2)
    ax.xaxis.grid(True, color="#D1D5DB", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "Difference in mean squared off-axis contribution "
        "(TCOA − OAC; standardized units²)",
        fontsize=11.5,
        labelpad=12,
    )
    ax.tick_params(axis="x", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#374151")
        spine.set_linewidth(0.9)

    label_offset = limit * 0.018
    for y_position, value in zip(positions, values):
        ax.text(
            value + (label_offset if value >= 0 else -label_offset),
            y_position,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color="#111827",
        )

    fig.suptitle(
        "Feature contributions to greater off-axis deviation in TCOAs",
        x=0.5,
        y=0.982,
        ha="center",
        fontsize=18,
        fontweight="normal",
    )
    fig.text(
        0.5,
        0.951,
        "Positive = greater mean squared contribution in TCOA; "
        "negative = greater contribution in OAC",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.32, right=0.965, top=0.925, bottom=0.075)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=cfg.OUTPUT_BASE_ROOT,
        help="Alt-2 output root containing the robust_median directory",
    )
    args = parser.parse_args()

    analysis_root = args.output_root / ANALYSIS_METHOD
    phenotype_dir = analysis_root / "02_phenotype"
    output_path = analysis_root / "04_final_results" / FIGURE_FILENAME
    summary, maximum_error = calculate_feature_contributions(phenotype_dir)
    create_off_axis_contribution_figure(summary, output_path)
    print(f"Figure saved to: {output_path}")
    print(f"Deviation reconstruction maximum absolute error: {maximum_error:.3e}")


if __name__ == "__main__":
    main()
