"""Stage 4: create the final workbook and publication-oriented figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

import analysis_config as cfg
from analysis_utils import ensure_output_directories


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required result is missing: {path}")
    return pd.read_csv(path, **kwargs)


def create_figures() -> list[Path]:
    (cfg.FINAL_DIR / "phenotype_behavior_associations.png").unlink(
        missing_ok=True
    )
    if not HAS_MATPLOTLIB:
        print(
            "matplotlib is not installed; the final workbook will be created, "
            "but PNG figures will be skipped."
        )
        return []
    figures: list[Path] = []
    scores = read_csv(
        cfg.PHENOTYPE_DIR / "participant_phenotype_scores.csv",
        dtype={"participant": str},
    )
    group_order = ["YAC", "OAC", "TCOA"]
    colors = {"YAC": "#3B82F6", "OAC": "#F97316", "TCOA": "#10B981"}
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    data = [
        scores.loc[scores["group"].eq(group), "analysis_position"].to_numpy()
        for group in group_order
    ]
    
    # Matplotlib 3.9+ compatibility fix: handles both 'tick_labels' and legacy 'labels'
    try:
        boxes = ax.boxplot(data, tick_labels=group_order, widths=0.52, patch_artist=True, showfliers=False)
    except TypeError:
        boxes = ax.boxplot(data, labels=group_order, widths=0.52, patch_artist=True, showfliers=False)

    for patch, group in zip(boxes["boxes"], group_order):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.32)
    for position, group in enumerate(group_order, start=1):
        subset = scores[scores["group"].eq(group)].reset_index(drop=True)
        values = subset["analysis_position"].to_numpy()
        jittered = position + rng.uniform(-0.09, 0.09, len(values))
        ax.scatter(
            jittered,
            values,
            color=colors[group],
            edgecolor="white",
            linewidth=0.5,
            s=38,
            zorder=3,
        )
        flagged = subset["is_winsorized"].astype(str).str.lower().eq("true").to_numpy()
        ax.scatter(
            jittered[flagged],
            values[flagged],
            facecolors="none",
            edgecolors="black",
            linewidth=1.2,
            s=72,
            zorder=4,
        )
        for x_value, y_value, participant in zip(
            jittered[flagged],
            values[flagged],
            subset.loc[flagged, "participant"],
        ):
            ax.annotate(
                participant,
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
    for value, label in [
        (0, "OAC robust endpoint"),
        (0.5, "equal proximity"),
        (1, "YAC robust endpoint"),
    ]:
        ax.axhline(value, color="#4B5563", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.text(3.43, value, label, va="center", fontsize=8, color="#4B5563")
    ax.set_ylabel("FC age-phenotype position")
    ax.set_title(
        "Participant position after label-blind position winsorization"
        if cfg.ANALYSIS_METHOD == "position_winsor"
        else "Participant position with robust median inference"
    )
    ax.set_xlim(0.55, 3.75)
    fig.tight_layout()
    path = cfg.FINAL_DIR / "participant_phenotype_position.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for group in group_order:
        subset = scores[scores["group"].eq(group)]
        ax.scatter(
            subset["analysis_position"],
            subset["analysis_deviation"],
            color=colors[group],
            label=group,
            s=48,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
    flagged = scores["is_winsorized"].astype(str).str.lower().eq("true")
    ax.scatter(
        scores.loc[flagged, "analysis_position"],
        scores.loc[flagged, "analysis_deviation"],
        facecolors="none",
        edgecolors="black",
        linewidth=1.2,
        s=82,
        zorder=4,
    )
    for _, row in scores.loc[flagged].iterrows():
        ax.annotate(
            row["participant"],
            (row["analysis_position"], row["analysis_deviation"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axvline(0.5, color="#4B5563", linestyle="--", linewidth=0.9)
    ax.set_xlabel("FC age-phenotype position")
    ax.set_ylabel("Off-axis deviation (RMS standardized units)")
    ax.set_title(
        "Position-winsorized phenotype position and deviation"
        if cfg.ANALYSIS_METHOD == "position_winsor"
        else "Radially winsorized phenotype position and deviation"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    path = cfg.FINAL_DIR / "phenotype_position_and_deviation.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    weights = read_csv(cfg.PHENOTYPE_DIR / "axis_feature_weights.csv")
    weights = weights.sort_values("axis_unit_weight")
    fig, ax = plt.subplots(figsize=(9, 8.5))
    weight_colors = np.where(weights["axis_unit_weight"] >= 0, "#3B82F6", "#F97316")
    ax.barh(weights["feature_name"], weights["axis_unit_weight"], color=weight_colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Unit-vector weight (positive = YAC-directed)")
    ax.set_title("Features defining the YAC-OAC connectivity direction")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    path = cfg.FINAL_DIR / "axis_feature_weights.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    permutation = np.load(
        cfg.INFERENCE_DIR / "permutation_distributions.npz", allow_pickle=False
    )
    primary = read_csv(cfg.INFERENCE_DIR / "primary_phenotype_inference.csv")
    primary = primary[
        primary["is_primary_estimator"].astype(str).str.lower().eq("true")
    ].copy()
    observed = float(
        primary.loc[primary["metric"].eq("position"), "observed_tcoa_minus_oac"].iloc[0]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(permutation["position"], bins=45, color="#93C5FD", edgecolor="white")
    ax.axvline(observed, color="#DC2626", linewidth=2, label=f"Observed = {observed:.3f}")
    ax.set_xlabel("Permuted TCOA minus OAC phenotype-position difference")
    ax.set_ylabel("Frequency")
    ax.set_title("Older-group label-permutation distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = cfg.FINAL_DIR / "position_permutation_distribution.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)
    return figures


def analysis_settings() -> pd.DataFrame:
    if cfg.ANALYSIS_METHOD == "position_winsor":
        outlier_handling = (
            "label-blind symmetric position winsorization fitted from pooled OAC+TCOA; "
            f"median +/- {cfg.POSITION_WINSORIZATION_MAD_CUTOFF} normal-consistent MADs"
        )
        primary_location = "mean after pooled older-adult position winsorization"
    else:
        outlier_handling = (
            "multivariate radial winsorization fitted from pooled YAC+OAC; "
            "primary position/deviation inference uses medians"
        )
        primary_location = "median; mean retained as sensitivity analysis"
    return pd.DataFrame(
        [
            ("Analysis version", cfg.ANALYSIS_VERSION),
            ("Alt-2 analysis method", cfg.ANALYSIS_METHOD),
            ("Input root", str(cfg.INPUT_ROOT)),
            ("Output root", str(cfg.OUTPUT_ROOT)),
            ("Matrix filename", cfg.MATRIX_FILENAME_TEMPLATE),
            ("Configured matrix scale", cfg.MATRIX_SCALE),
            ("Analysis scale", "Fisher z"),
            ("Atlas resolution", "100 parcels summarized into 28 network features"),
            ("Within-mode averaging", "all available HMM-derived matrices, equal matrix weight"),
            ("Across-mode averaging", "EC and EO equal weight"),
            ("State/window matching", "not performed"),
            ("Reference definition", "YAC and OAC only; TCOA excluded"),
            ("Outlier handling", outlier_handling),
            ("Primary group-location statistic", primary_location),
            ("Primary axis estimator", cfg.PRIMARY_AXIS_ESTIMATOR),
            (
                "Primary feature standardization",
                "pooled YAC+OAC coordinate-wise median and normal-consistent MAD",
            ),
            (
                "Primary endpoints",
                "separate YAC and OAC Euclidean geometric medians",
            ),
            (
                "Geometric-median metric",
                "Euclidean distance in robustly standardized 28-feature space",
            ),
            (
                "Geometric-median convergence",
                f"modified Weiszfeld; tolerance={cfg.GEOMETRIC_MEDIAN_TOLERANCE}; "
                f"maximum iterations={cfg.GEOMETRIC_MEDIAN_MAX_ITERATIONS}",
            ),
            (
                "Sensitivity estimators",
                "coordinate-wise median and conventional arithmetic mean",
            ),
            ("Behavior sample", "TCOA participants only"),
            (
                "Behavior predictor",
                "primary geometric-median analysis_deviation (off-axis deviation)",
            ),
            (
                "Behavior outcomes",
                "RBANS, Mini-BESTest, and reverse-scored GDS only",
            ),
            (
                "Behavior inference",
                "Pearson correlations with one-sided parametric p-values; "
                "directional alternative is greater deviation associated with healthier function; "
                "no permutation, bootstrap, covariates, or multiplicity adjustment",
            ),
            (
                "Primary reference-group scores",
                "full robust axis with OAC endpoint=0 and YAC endpoint=1",
            ),
            ("TCOA scores", "full independent YAC-OAC reference"),
            ("Permutations", cfg.N_PERMUTATIONS),
            ("Bootstraps", cfg.N_BOOTSTRAPS),
            ("Random seed", cfg.RANDOM_SEED),
        ],
        columns=["setting", "value"],
    )


def build_workbook() -> Path:
    workbook_path = cfg.FINAL_DIR / "fc_age_phenotype_analysis_results.xlsx"
    sheets = {
        "Analysis_Settings": analysis_settings(),
        "Matrix_Counts": read_csv(cfg.DATASET_DIR / "matrix_counts.csv", dtype={"participant": str}),
        "Scale_Detection": read_csv(cfg.DATASET_DIR / "matrix_scale_detection.csv"),
        "Matrix_Validation": read_csv(cfg.DATASET_DIR / "matrix_validation.csv"),
        "Feature_Metadata": read_csv(cfg.DATASET_DIR / "feature_metadata.csv"),
        "Participant_Features_Raw": read_csv(cfg.DATASET_DIR / "participant_features.csv", dtype={"participant": str}),
        "Participant_Features_Analysis": read_csv(
            cfg.PHENOTYPE_DIR / "participant_features_analysis.csv",
            dtype={"participant": str},
        ),
        "Participant_Scores": read_csv(cfg.PHENOTYPE_DIR / "participant_phenotype_scores.csv", dtype={"participant": str}),
        "Estimator_Sensitivity_Scores": read_csv(
            cfg.PHENOTYPE_DIR / "participant_phenotype_sensitivity_scores.csv",
            dtype={"participant": str},
        ),
        "Group_Summary": read_csv(cfg.PHENOTYPE_DIR / "phenotype_group_summary.csv"),
        "Axis_Weights": read_csv(cfg.PHENOTYPE_DIR / "axis_feature_weights.csv"),
        "Estimator_Axis_Weights": read_csv(
            cfg.PHENOTYPE_DIR / "axis_feature_weights_by_estimator.csv"
        ),
        "Estimator_Metadata": read_csv(
            cfg.PHENOTYPE_DIR / "axis_estimator_metadata.csv"
        ),
        "Primary_Inference": read_csv(cfg.INFERENCE_DIR / "primary_phenotype_inference.csv"),
        "Robustness": read_csv(cfg.INFERENCE_DIR / "robustness_inference.csv"),
        "Bootstrap_Summary": read_csv(cfg.INFERENCE_DIR / "bootstrap_summary.csv"),
        "Behavior_Data": read_csv(
            cfg.INFERENCE_DIR / "behavior_analysis_data.csv",
            dtype={"participant": str},
        ),
        "Behavior_Correlations": read_csv(
            cfg.INFERENCE_DIR / "behavior_correlation_results.csv"
        ),
    }
    if cfg.ANALYSIS_METHOD == "position_winsor":
        sheets["Outlier_Diagnostics"] = read_csv(
            cfg.PHENOTYPE_DIR / "position_winsorization_diagnostics.csv",
            dtype={"participant": str},
        )
        sheets["Winsorization_Metadata"] = read_csv(
            cfg.PHENOTYPE_DIR / "position_winsorization_metadata.csv"
        )
    else:
        sheets["Outlier_Diagnostics"] = read_csv(
            cfg.PHENOTYPE_DIR / "multivariate_outlier_diagnostics.csv",
            dtype={"participant": str},
        )
        sheets["Winsorization_Metadata"] = read_csv(
            cfg.PHENOTYPE_DIR / "radial_winsorization_metadata.csv"
        )
        sheets["Winsorization_Parameters"] = read_csv(
            cfg.PHENOTYPE_DIR / "radial_winsorization_feature_parameters.csv"
        )
        sheets["Mean_Sensitivity"] = read_csv(
            cfg.INFERENCE_DIR / "mean_sensitivity_inference.csv"
        )
    readme = pd.DataFrame(
        {
            "topic": [
                "Primary question",
                "Phenotype position",
                "Midpoint",
                "Off-axis deviation",
                "Averaging",
                "Outlier handling",
                "Behavior correlations",
                "Behavior direction",
                "Robustness",
                "Interpretation boundary",
            ],
            "description": [
                "Are TCOAs shifted toward the YAC multivariate FC phenotype relative to OACs?",
                "On the primary robust axis, 0 is the OAC geometric-median endpoint and 1 is the YAC geometric-median endpoint; higher values are more youth-directed.",
                "A value above 0.5 is closer to the YAC than OAC robust endpoint along the primary axis.",
                "RMS distance from the nearest point on the YAC-OAC axis; larger values indicate a more distinct phenotype.",
                "Every signed thresholded matrix is Fisher transformed and summarized into 28 features; matrices are averaged within mode and EC/EO are averaged equally.",
                (
                    "Phenotype position is capped symmetrically using pooled older-adult median/MAD bounds without OAC/TCOA labels."
                    if cfg.ANALYSIS_METHOD == "position_winsor"
                    else "A pooled YAC+OAC median/MAD transformation defines a radial cutoff; primary group inference uses median differences."
                ),
                "Within TCOAs only, primary off-axis deviation is correlated separately with RBANS, Mini-BESTest, and reverse-scored GDS using Pearson correlations.",
                "One-sided parametric p-values test whether greater off-axis deviation is associated with healthier function; behavioral p-values are unadjusted and use no permutation or bootstrap inference.",
                "Coordinate-median and conventional-mean axes test estimator dependence; relative Euclidean distance and whole-vector RSA test metric dependence.",
                "The cross-sectional design supports association, not causal rejuvenation or reversal.",
            ],
        }
    )
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        for name, dataframe in sheets.items():
            dataframe.to_excel(writer, sheet_name=name[:31], index=False)
        workbook = writer.book
        header = workbook.add_format(
            {"bold": True, "bg_color": "#DCEAF7", "border": 1, "text_wrap": True}
        )
        wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
        for name, dataframe in {"README": readme, **sheets}.items():
            worksheet = writer.sheets[name[:31]]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(
                0, 0, max(len(dataframe), 1), max(len(dataframe.columns) - 1, 0)
            )
            for column_number, column in enumerate(dataframe.columns):
                worksheet.write(0, column_number, column, header)
                lengths = [len(str(column)), *[len(str(x)) for x in dataframe[column].head(200)]]
                width = min(max(lengths) + 2, 55)
                worksheet.set_column(
                    column_number,
                    column_number,
                    width,
                    wrap if width > 30 else None,
                )
    return workbook_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cfg.validate_config()
    ensure_output_directories()
    figures = create_figures()
    workbook = build_workbook()
    print(f"Final workbook: {workbook}")
    print(f"Created {len(figures)} figures in {cfg.FINAL_DIR}")


if __name__ == "__main__":
    main()
