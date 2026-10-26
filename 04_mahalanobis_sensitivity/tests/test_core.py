"""Focused synthetic tests for averaging and phenotype calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis_config as cfg
from analysis_utils import (
    apply_radial_winsorizer,
    average_participant_features,
    feature_metadata,
    fit_axis,
    fit_position_winsorizer,
    fit_radial_winsorizer,
    geometric_median,
    huber_regression,
    matrix_to_network_features,
    score_axis,
    apply_position_winsorizer,
    validate_and_transform_matrix,
    winsorize_axis_data,
)


class CoreTests(unittest.TestCase):
    def test_network_mapping_and_metadata(self) -> None:
        cfg.validate_config()
        metadata = feature_metadata()
        self.assertEqual(len(metadata), 28)
        self.assertEqual((metadata["feature_type"] == "within").sum(), 7)
        self.assertEqual((metadata["feature_type"] == "between").sum(), 21)

    def test_constant_offdiagonal_matrix_features(self) -> None:
        matrix = np.full((102, 102), 0.2, dtype=float)
        np.fill_diagonal(matrix, 1.0)
        transformed, diagnostics = validate_and_transform_matrix(
            matrix, "synthetic_raw.npz:0_0", "pearson"
        )
        features = matrix_to_network_features(transformed)
        self.assertTrue(np.allclose(features, np.arctanh(0.2)))
        self.assertEqual(diagnostics["analysis_scale"], "fisher_z")

    def test_equal_mode_average(self) -> None:
        names = feature_metadata()["feature_name"].tolist()
        rows = []
        for mode, values in [("EC", [1.0, 3.0]), ("EO", [10.0])]:
            for value in values:
                row = {"participant": "P1", "group": "YAC", "mode": mode}
                row.update({name: value for name in names})
                rows.append(row)
        mode, combined = average_participant_features(pd.DataFrame(rows), names)
        self.assertTrue(np.allclose(mode.loc[mode["mode"].eq("EC"), names], 2.0))
        self.assertTrue(np.allclose(combined[names], 6.0))

    def test_axis_endpoints_and_midpoint(self) -> None:
        young = np.array([[2.0, 0.0, 1.0], [2.2, 0.1, 1.2], [1.8, -0.1, 0.8]])
        old = np.array([[0.0, 1.0, -1.0], [0.2, 1.1, -0.8], [-0.2, 0.9, -1.2]])
        model = fit_axis(young, old, estimator="mean")
        raw_young_centroid = np.mean(young, axis=0)
        raw_old_centroid = np.mean(old, axis=0)
        midpoint = (raw_young_centroid + raw_old_centroid) / 2
        scores = score_axis(
            np.vstack([raw_old_centroid, midpoint, raw_young_centroid]), model
        )
        self.assertTrue(np.allclose(scores["position"], [0.0, 0.5, 1.0]))
        self.assertTrue(
            np.allclose(scores["contributions"].sum(axis=1), scores["position"])
        )

    def test_geometric_median_symmetric_points(self) -> None:
        points = np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        center, iterations, converged = geometric_median(points)
        self.assertTrue(converged)
        self.assertGreaterEqual(iterations, 1)
        self.assertTrue(np.allclose(center, [0.0, 0.0], atol=1e-10))

    def test_geometric_median_handles_bootstrap_duplicates(self) -> None:
        points = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [10.0, 10.0],
            ]
        )
        center, _, converged = geometric_median(points)
        self.assertTrue(converged)
        self.assertTrue(np.allclose(center, [0.0, 0.0], atol=1e-10))

    def test_geometric_axis_endpoint_anchors(self) -> None:
        young = np.array(
            [
                [2.0, 0.0, 1.0],
                [2.2, 0.1, 1.2],
                [1.8, -0.1, 0.8],
                [2.1, 0.2, 0.9],
            ]
        )
        old = np.array(
            [
                [0.0, 1.0, -1.0],
                [0.2, 1.1, -0.8],
                [-0.2, 0.9, -1.2],
                [0.1, 0.8, -0.9],
            ]
        )
        model = fit_axis(young, old, estimator="geometric_median")
        old_raw = model.old_endpoint_standardized * model.scale + model.center
        young_raw = model.young_endpoint_standardized * model.scale + model.center
        midpoint = (old_raw + young_raw) / 2
        scores = score_axis(
            np.vstack([old_raw, midpoint, young_raw]),
            model,
        )
        self.assertTrue(np.allclose(scores["position"], [0.0, 0.5, 1.0]))
        self.assertTrue(model.young_endpoint_converged)
        self.assertTrue(model.old_endpoint_converged)

    def test_ledoit_wolf_covariance_and_whitening_are_consistent(self) -> None:
        rng = np.random.default_rng(12)
        young = rng.normal(loc=0.5, size=(15, 6))
        old = rng.normal(loc=-0.5, size=(15, 6))
        model = fit_axis(young, old, estimator="geometric_median")
        identity = model.whitener @ model.covariance @ model.whitener
        self.assertTrue(np.allclose(identity, np.eye(6), atol=1e-9))
        self.assertGreaterEqual(model.ledoit_wolf_shrinkage, 0.0)
        self.assertLessEqual(model.ledoit_wolf_shrinkage, 1.0)
        self.assertIn("Mahalanobis", model.distance_metric)

    def test_geometric_axis_resists_single_extreme_reference(self) -> None:
        young = np.array(
            [
                [2.0, 2.0],
                [2.1, 1.9],
                [1.9, 2.1],
                [2.2, 2.0],
                [1.8, 2.0],
            ]
        )
        old = np.array(
            [
                [0.0, 0.0],
                [0.1, -0.1],
                [-0.1, 0.1],
                [0.2, 0.0],
                [-0.2, 0.0],
            ]
        )
        query = np.array([[1.0, 1.0]])
        robust_baseline = score_axis(
            query,
            fit_axis(young, old, estimator="geometric_median"),
        )["position"][0]
        mean_baseline = score_axis(
            query,
            fit_axis(young, old, estimator="mean"),
        )["position"][0]
        old_with_outlier = np.vstack([old, [100.0, -100.0]])
        robust_outlier = score_axis(
            query,
            fit_axis(
                young,
                old_with_outlier,
                estimator="geometric_median",
            ),
        )["position"][0]
        mean_outlier = score_axis(
            query,
            fit_axis(young, old_with_outlier, estimator="mean"),
        )["position"][0]
        self.assertLess(
            abs(robust_outlier - robust_baseline),
            abs(mean_outlier - mean_baseline),
        )

    def test_coordinate_median_axis_is_available(self) -> None:
        young = np.array([[2.0, 2.0], [2.1, 1.9], [1.9, 2.1]])
        old = np.array([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1]])
        model = fit_axis(young, old, estimator="coordinate_median")
        self.assertEqual(
            model.endpoint_method,
            "group coordinate-wise median in Ledoit-Wolf covariance-whitened space",
        )
        self.assertEqual(
            model.scale_method,
            "pooled reference normal-consistent MAD",
        )

    def test_radial_winsorization_caps_radius_and_preserves_direction(self) -> None:
        reference = np.array(
            [
                [-1.0, -0.8],
                [-0.7, -1.1],
                [-0.2, -0.1],
                [0.1, 0.2],
                [0.8, 0.9],
                [1.1, 0.7],
            ]
        )
        winsorizer = fit_radial_winsorizer(reference)
        values = np.vstack([reference[2], [100.0, -80.0]])
        adjusted, diagnostics = apply_radial_winsorizer(values, winsorizer)

        self.assertFalse(bool(diagnostics["is_winsorized"][0]))
        self.assertTrue(bool(diagnostics["is_winsorized"][1]))
        self.assertTrue(np.allclose(adjusted[0], values[0]))
        self.assertAlmostEqual(
            diagnostics["adjusted_radius"][1],
            winsorizer.radius_cutoff,
        )
        original_z = (values[1] - winsorizer.center) / winsorizer.scale
        adjusted_z = (adjusted[1] - winsorizer.center) / winsorizer.scale
        self.assertTrue(
            np.allclose(
                adjusted_z,
                original_z * diagnostics["shrinkage_factor"][1],
            )
        )

    def test_winsorization_is_fitted_only_from_reference_groups(self) -> None:
        young = np.array([[2.0, 2.1], [2.2, 1.9], [1.8, 2.0]])
        old = np.array([[0.0, 0.1], [0.2, -0.1], [-0.2, 0.0]])
        target = np.array([[500.0, -500.0]])
        winsorizer, adjusted = winsorize_axis_data(young, old, target)
        adjusted_target = adjusted[2]
        _, diagnostics = apply_radial_winsorizer(target, winsorizer)
        self.assertTrue(bool(diagnostics["is_winsorized"][0]))
        self.assertTrue(np.allclose(adjusted_target, apply_radial_winsorizer(target, winsorizer)[0]))

    def test_position_winsorization_is_symmetric_and_label_blind(self) -> None:
        pooled = np.array([-10.0, -0.3, -0.1, 0.0, 0.2, 0.4, 12.0])
        winsorizer = fit_position_winsorizer(pooled)
        adjusted, flags = apply_position_winsorizer(pooled, winsorizer)
        self.assertAlmostEqual(
            winsorizer.center - winsorizer.lower_bound,
            winsorizer.upper_bound - winsorizer.center,
        )
        self.assertTrue(bool(flags[0]))
        self.assertTrue(bool(flags[-1]))
        self.assertAlmostEqual(adjusted[0], winsorizer.lower_bound)
        self.assertAlmostEqual(adjusted[-1], winsorizer.upper_bound)

    def test_huber_regression_reduces_extreme_outcome_influence(self) -> None:
        x = np.arange(12, dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        outcome = 1.0 + 2.0 * x
        outcome[-1] += 100.0
        ols_beta = np.linalg.pinv(design) @ outcome
        huber_beta, _, weights, _, converged = huber_regression(outcome, design)
        self.assertTrue(converged)
        self.assertLess(abs(huber_beta[1] - 2.0), abs(ols_beta[1] - 2.0))
        self.assertLess(weights[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
