"""Stage 2: fit the YAC-OAC axis and calculate participant phenotype scores."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import analysis_config as cfg
from analysis_utils import (
    apply_radial_winsorizer,
    ensure_output_directories,
    feature_metadata,
    fit_axis,
    fit_position_winsorizer,
    fit_radial_winsorizer,
    apply_position_winsorizer,
    score_axis,
)


SCORE_NAMES = (
    "position",
    "deviation",
    "distance_young",
    "distance_old",
    "relative_distance_youth",
    "rsa_r_young",
    "rsa_r_old",
    "rsa_fisher_difference",
)


def load_features() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    path = cfg.DATASET_DIR / "participant_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run stage 1 first: {path}")
    frame = pd.read_csv(path, dtype={"participant": str})
    metadata = pd.read_csv(cfg.DATASET_DIR / "feature_metadata.csv")
    feature_names = metadata["feature_name"].tolist()
    values = frame[feature_names].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Participant feature matrix contains non-finite values")
    return frame, feature_names, values


def calculate_estimator_scores(
    frame: pd.DataFrame,
    values: np.ndarray,
    groups: np.ndarray,
    estimator: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], object]:
    young = values[groups == "YAC"]
    old = values[groups == "OAC"]
    model = fit_axis(young, old, estimator=estimator)
    full = score_axis(values, model)

    result = frame[["participant", "group"]].copy()
    for column in (
        "original_radius",
        "adjusted_radius",
        "shrinkage_factor",
        "is_winsorized",
    ):
        if column in frame:
            result[column] = frame[column].to_numpy()
    if "is_winsorized" not in result:
        result["is_winsorized"] = False
    result["axis_estimator"] = estimator
    result["is_primary_estimator"] = estimator == cfg.PRIMARY_AXIS_ESTIMATOR
    for name in SCORE_NAMES:
        result[f"full_{name}"] = full[name]
        result[f"analysis_{name}"] = full[name]
    if cfg.ANALYSIS_METHOD == "position_winsor":
        older_mask = result["group"].isin(["OAC", "TCOA"]).to_numpy()
        position_winsorizer = fit_position_winsorizer(full["position"][older_mask])
        adjusted_position, flags = apply_position_winsorizer(
            full["position"][older_mask], position_winsorizer
        )
        result["position_before_winsorization"] = full["position"]
        result["position_lower_bound"] = position_winsorizer.lower_bound
        result["position_upper_bound"] = position_winsorizer.upper_bound
        result.loc[older_mask, "analysis_position"] = adjusted_position
        result["is_winsorized"] = False
        result.loc[older_mask, "is_winsorized"] = flags
        result["outlier_method"] = "pooled older-adult symmetric position winsorization"
        result["score_source"] = (
            f"raw features; position capped label-blind among older adults ({estimator})"
        )
    else:
        result["outlier_method"] = "multivariate radial winsorization"
        result["score_source"] = np.where(
            result["group"].eq("TCOA"),
            f"radially winsorized; independent full YAC-OAC reference ({estimator})",
            f"radially winsorized; full YAC-OAC reference ({estimator})",
        )
    return result, full, model


def model_feature_weights(
    feature_names: list[str],
    estimator: str,
    model: object,
) -> pd.DataFrame:
    metadata = feature_metadata()
    if metadata["feature_name"].tolist() != feature_names:
        raise ValueError("Feature metadata order changed between stages")
    metadata.insert(0, "axis_estimator", estimator)
    metadata.insert(
        1,
        "is_primary_estimator",
        estimator == cfg.PRIMARY_AXIS_ESTIMATOR,
    )
    metadata["reference_center_raw"] = model.center
    metadata["reference_scale_raw"] = model.scale
    metadata["yac_endpoint_standardized"] = model.young_endpoint
    metadata["oac_endpoint_standardized"] = model.old_endpoint
    metadata["axis_direction_yac_minus_oac"] = model.direction
    metadata["axis_unit_weight"] = (
        model.direction / np.sqrt(model.direction_norm_squared)
    )
    metadata["center_method"] = model.center_method
    metadata["scale_method"] = model.scale_method
    metadata["endpoint_method"] = model.endpoint_method
    metadata["distance_metric"] = model.distance_metric
    return metadata


def save_axis_model(path: object, model: object) -> None:
    np.savez_compressed(
        path,
        estimator=np.asarray(model.estimator),
        center_method=np.asarray(model.center_method),
        scale_method=np.asarray(model.scale_method),
        endpoint_method=np.asarray(model.endpoint_method),
        distance_metric=np.asarray(model.distance_metric),
        center=model.center,
        scale=model.scale,
        young_endpoint=model.young_endpoint,
        old_endpoint=model.old_endpoint,
        # Backward-compatible aliases for external code written for version 1.
        young_centroid=model.young_endpoint,
        old_centroid=model.old_endpoint,
        direction=model.direction,
        direction_norm_squared=model.direction_norm_squared,
        scale_floor=model.scale_floor,
        n_scaled_features_floored=model.n_scaled_features_floored,
        young_endpoint_iterations=model.young_endpoint_iterations,
        old_endpoint_iterations=model.old_endpoint_iterations,
        young_endpoint_converged=model.young_endpoint_converged,
        old_endpoint_converged=model.old_endpoint_converged,
    )


def calculate_scores() -> pd.DataFrame:
    frame, feature_names, values = load_features()
    groups = frame["group"].to_numpy(str)
    diagnostics = frame[["participant", "group"]].copy()
    if cfg.ANALYSIS_METHOD == "robust_median":
        reference_mask = np.isin(groups, ["YAC", "OAC"])
        winsorizer = fit_radial_winsorizer(values[reference_mask])
        adjusted_values, radial_diagnostics = apply_radial_winsorizer(
            values,
            winsorizer,
        )
        diagnostics["original_radius"] = radial_diagnostics["original_radius"]
        diagnostics["adjusted_radius"] = radial_diagnostics["adjusted_radius"]
        diagnostics["radius_cutoff"] = winsorizer.radius_cutoff
        diagnostics["shrinkage_factor"] = radial_diagnostics["shrinkage_factor"]
        diagnostics["is_winsorized"] = radial_diagnostics["is_winsorized"]
        diagnostics.to_csv(
            cfg.PHENOTYPE_DIR / "multivariate_outlier_diagnostics.csv",
            index=False,
        )
    else:
        adjusted_values = values.copy()
        diagnostics["original_radius"] = np.nan
        diagnostics["adjusted_radius"] = np.nan
        diagnostics["radius_cutoff"] = np.nan
        diagnostics["shrinkage_factor"] = np.nan
        diagnostics["is_winsorized"] = False

    adjusted_frame = pd.DataFrame(adjusted_values, columns=feature_names)
    adjusted_frame.insert(0, "group", frame["group"])
    adjusted_frame.insert(0, "participant", frame["participant"])
    adjusted_frame.to_csv(
        cfg.PHENOTYPE_DIR / "participant_features_analysis.csv",
        index=False,
    )
    if cfg.ANALYSIS_METHOD == "robust_median":
        adjusted_frame.to_csv(
            cfg.PHENOTYPE_DIR / "participant_features_radially_winsorized.csv",
            index=False,
        )
        parameter_frame = pd.DataFrame(
            {
                "feature_name": feature_names,
                "reference_center": winsorizer.center,
                "reference_scale": winsorizer.scale,
            }
        )
        parameter_frame.to_csv(
            cfg.PHENOTYPE_DIR / "radial_winsorization_feature_parameters.csv",
            index=False,
        )
        pd.DataFrame([{
            "method": "multivariate radial winsorization",
            "reference_groups": "pooled YAC+OAC",
            "distance": "RMS Euclidean radius in median/MAD-standardized feature space",
            "direction_preserved": True,
            "mad_cutoff": winsorizer.mad_cutoff,
            "reference_radius_median": winsorizer.radius_median,
            "reference_radius_normal_consistent_mad": winsorizer.radius_mad,
            "radius_cutoff": winsorizer.radius_cutoff,
            "feature_scale_floor": winsorizer.feature_scale_floor,
            "n_feature_scales_floored": winsorizer.n_feature_scales_floored,
            "n_participants_winsorized": int(np.count_nonzero(radial_diagnostics["is_winsorized"])),
        }]).to_csv(
            cfg.PHENOTYPE_DIR / "radial_winsorization_metadata.csv", index=False
        )

    analysis_frame = frame[["participant", "group"]].merge(
        diagnostics,
        on=["participant", "group"],
        how="left",
        validate="one_to_one",
    )
    estimator_results: list[pd.DataFrame] = []
    estimator_weights: list[pd.DataFrame] = []
    full_by_estimator: dict[str, dict[str, np.ndarray]] = {}
    model_by_estimator: dict[str, object] = {}

    for estimator in cfg.AXIS_ESTIMATORS:
        result, full, model = calculate_estimator_scores(
            analysis_frame,
            adjusted_values,
            groups,
            estimator,
        )
        estimator_results.append(result)
        estimator_weights.append(
            model_feature_weights(feature_names, estimator, model)
        )
        full_by_estimator[estimator] = full
        model_by_estimator[estimator] = model
        save_axis_model(
            cfg.PHENOTYPE_DIR / f"axis_model_{estimator}.npz",
            model,
        )

    sensitivity = pd.concat(estimator_results, ignore_index=True)
    sensitivity.to_csv(
        cfg.PHENOTYPE_DIR / "participant_phenotype_sensitivity_scores.csv",
        index=False,
    )
    primary = sensitivity[
        sensitivity["axis_estimator"].eq(cfg.PRIMARY_AXIS_ESTIMATOR)
    ].copy()
    primary.to_csv(
        cfg.PHENOTYPE_DIR / "participant_phenotype_scores.csv",
        index=False,
    )
    if cfg.ANALYSIS_METHOD == "position_winsor":
        position_columns = [
            "participant", "group", "axis_estimator",
            "position_before_winsorization", "analysis_position",
            "position_lower_bound", "position_upper_bound", "is_winsorized",
        ]
        sensitivity[position_columns].to_csv(
            cfg.PHENOTYPE_DIR / "position_winsorization_diagnostics.csv",
            index=False,
        )
        pd.DataFrame([{
            "method": "label-blind symmetric position winsorization",
            "fitting_sample": "pooled OAC+TCOA",
            "center": "median",
            "scale": "normal-consistent MAD",
            "mad_cutoff": cfg.POSITION_WINSORIZATION_MAD_CUTOFF,
            "tails": "symmetric two-sided",
            "applied_metric": "phenotype position only",
        }]).to_csv(
            cfg.PHENOTYPE_DIR / "position_winsorization_metadata.csv", index=False
        )

    primary_full = full_by_estimator[cfg.PRIMARY_AXIS_ESTIMATOR]
    standardized = pd.DataFrame(
        primary_full["standardized"],
        columns=feature_names,
    )
    standardized.insert(0, "axis_estimator", cfg.PRIMARY_AXIS_ESTIMATOR)
    standardized.insert(0, "group", frame["group"])
    standardized.insert(0, "participant", frame["participant"])
    standardized.to_csv(
        cfg.PHENOTYPE_DIR / "participant_standardized_features.csv",
        index=False,
    )

    contributions = pd.DataFrame(
        primary_full["contributions"],
        columns=feature_names,
    )
    contributions.insert(0, "axis_estimator", cfg.PRIMARY_AXIS_ESTIMATOR)
    contributions.insert(0, "group", frame["group"])
    contributions.insert(0, "participant", frame["participant"])
    contributions.to_csv(
        cfg.PHENOTYPE_DIR / "participant_feature_contributions.csv",
        index=False,
    )

    all_weights = pd.concat(estimator_weights, ignore_index=True)
    all_weights.to_csv(
        cfg.PHENOTYPE_DIR / "axis_feature_weights_by_estimator.csv",
        index=False,
    )
    primary_weights = all_weights[
        all_weights["axis_estimator"].eq(cfg.PRIMARY_AXIS_ESTIMATOR)
    ].copy()
    primary_weights.to_csv(
        cfg.PHENOTYPE_DIR / "axis_feature_weights.csv",
        index=False,
    )

    primary_model = model_by_estimator[cfg.PRIMARY_AXIS_ESTIMATOR]
    save_axis_model(cfg.PHENOTYPE_DIR / "axis_model.npz", primary_model)

    summary = (
        sensitivity.groupby(
            ["axis_estimator", "is_primary_estimator", "group"],
            observed=True,
        )
        .agg(
            n=("participant", "size"),
            mean_position=("analysis_position", "mean"),
            sd_position=("analysis_position", "std"),
            median_position=("analysis_position", "median"),
            mean_deviation=("analysis_deviation", "mean"),
            median_deviation=("analysis_deviation", "median"),
            mean_relative_distance=("analysis_relative_distance_youth", "mean"),
            mean_rsa_difference=("analysis_rsa_fisher_difference", "mean"),
        )
        .reset_index()
    )
    model_details = pd.DataFrame(
        [
            {
                "axis_estimator": estimator,
                "axis_scale_floor": model.scale_floor,
                "n_features_scale_floored": model.n_scaled_features_floored,
                "center_method": model.center_method,
                "scale_method": model.scale_method,
                "endpoint_method": model.endpoint_method,
                "distance_metric": model.distance_metric,
                "yac_endpoint_iterations": model.young_endpoint_iterations,
                "oac_endpoint_iterations": model.old_endpoint_iterations,
                "yac_endpoint_converged": model.young_endpoint_converged,
                "oac_endpoint_converged": model.old_endpoint_converged,
            }
            for estimator, model in model_by_estimator.items()
        ]
    )
    summary = summary.merge(
        model_details,
        on="axis_estimator",
        how="left",
        validate="many_to_one",
    )
    summary.to_csv(
        cfg.PHENOTYPE_DIR / "phenotype_group_summary.csv",
        index=False,
    )
    model_details.to_csv(
        cfg.PHENOTYPE_DIR / "axis_estimator_metadata.csv",
        index=False,
    )
    print(
        f"Primary {cfg.PRIMARY_AXIS_ESTIMATOR} phenotype scores and "
        f"{len(cfg.AXIS_ESTIMATORS) - 1} sensitivity estimators saved to "
        f"{cfg.PHENOTYPE_DIR}"
    )
    return primary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cfg.validate_config()
    ensure_output_directories()
    calculate_scores()


if __name__ == "__main__":
    main()
