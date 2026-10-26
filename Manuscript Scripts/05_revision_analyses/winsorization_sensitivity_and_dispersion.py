"""Reviewer 1, 3b - covariance inversion, winsorization, and per-group dispersion.

Part A  Ledoit-Wolf details of the Mahalanobis sensitivity analysis (read from the
        saved model metadata): shrinkage intensity, condition number, inversion.
Part B  Which participants were radially winsorized (group, radius, shrinkage).
Part C  No-winsorization sensitivity analysis: the full primary analysis (all three
        estimators) rerun with radial winsorization switched off, with complete
        refitting in every permutation. Position one-sided (upper tail); deviation
        two-sided, as in the manuscript.
Part D  Per-group dispersion in the primary standardized feature space:
          - median across the 28 features of the per-feature MAD (normal-consistent);
          - mean RMS distance from each participant to its own group's geometric median;
          - split of mean squared off-axis deviation into a shared component
            (squared RMS length of the group's mean residual vector) and a spread
            component (mean squared distance of residuals from that mean).
        TCOA - OAC differences in dispersion are tested with OAC/TCOA label
        permutations, refitting winsorization, standardization and endpoints each time.

Usage:  python winsorization_sensitivity_and_dispersion.py [--n-perm 10000] [--quick]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import revision_common as rc
from analysis_utils import fit_axis, geometric_median, score_axis, winsorize_axis_data

OUT = rc.out_dir("winsorization_dispersion")
MAD_K = 1.4826


def score_nowins(young, old, test, estimator=rc.PRIMARY):
    model = fit_axis(young, old, estimator=estimator)
    return score_axis(test, model), score_axis(old, model), model


def diffs(A, scorer, estimator):
    t, o, _ = scorer(A["YAC"], A["OAC"], A["TCOA"], estimator)
    return (float(np.median(t["position"]) - np.median(o["position"])),
            float(np.median(t["deviation"]) - np.median(o["deviation"])), t, o)


def permute(A, n, rng, fn):
    pool = np.vstack([A["OAC"], A["TCOA"]]); nO = len(A["OAC"])
    out = []
    for _ in range(n):
        idx = rng.permutation(len(pool))
        out.append(fn({"YAC": A["YAC"], "OAC": pool[idx[:nO]], "TCOA": pool[idx[nO:]]}))
    return np.array(out)


def residuals(scores, model):
    z = scores["standardized"]
    proj = model.old_endpoint + scores["position"][:, None] * model.direction
    return z - proj


def dispersion(A):
    """Dispersion summaries for OAC and TCOA (and YAC) in the primary model space."""
    _, (y, o, t) = winsorize_axis_data(A["YAC"], A["OAC"], A["TCOA"])
    model = fit_axis(y, o, estimator=rc.PRIMARY)
    res = {}
    for g, X in (("YAC", y), ("OAC", o), ("TCOA", t)):
        s = score_axis(X, model)
        z = s["standardized"]
        mad = MAD_K * np.median(np.abs(z - np.median(z, 0)), 0)
        gm = geometric_median(z)[0]
        d_own = np.sqrt(np.mean(np.square(z - gm), 1))
        r = residuals(s, model)
        shared = float(np.mean(np.square(r.mean(0))))          # per-feature mean square of mean residual
        total = float(np.mean(np.square(r)))                   # mean squared deviation
        res[g] = {"median_feature_MAD": float(np.median(mad)),
                  "mean_feature_MAD": float(np.mean(mad)),
                  "mean_distance_to_own_GM": float(d_own.mean()),
                  "median_distance_to_own_GM": float(np.median(d_own)),
                  "mean_sq_deviation": total, "shared_component": shared,
                  "spread_component": total - shared,
                  "shared_fraction": shared / total}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n = 200 if a.quick else a.n_perm
    rng = np.random.default_rng(a.seed)
    F, names, A = rc.load_features()

    # ---------------- Part A
    mdir = rc.MAHALANOBIS_DIR / "02_phenotype"
    meta = pd.read_csv(mdir / "axis_estimator_metadata.csv")
    cov = pd.read_csv(mdir / "ledoit_wolf_covariance.csv", index_col=0).to_numpy(float)
    ev = np.linalg.eigvalsh(cov)
    offdiag = cov[~np.eye(len(cov), dtype=bool)]
    partA = {"per_estimator": meta[["axis_estimator", "ledoit_wolf_shrinkage", "covariance_condition_number",
                                    "covariance_eigenvalue_floor", "covariance_reference_n"]].to_dict("records"),
             "saved_cov_eigen_min": float(ev.min()), "saved_cov_eigen_max": float(ev.max()),
             "saved_cov_condition_number": float(ev.max() / ev.min()),
             "saved_cov_mean_abs_offdiag_over_mean_diag": float(np.mean(np.abs(offdiag)) / np.mean(np.diag(cov)))}
    (OUT / "A_ledoit_wolf.json").write_text(json.dumps(partA, indent=2))
    print(json.dumps(partA, indent=2))

    # ---------------- Part B
    diag = pd.read_csv(rc.METHOD_DIR / "02_phenotype" / "multivariate_outlier_diagnostics.csv", dtype={"participant": str})
    w = diag[diag.is_winsorized]
    w.to_csv(OUT / "B_winsorized_participants.csv", index=False)
    print(w.to_string(index=False))

    # ---------------- Part C
    rows = []
    for est in rc.ESTIMATORS:
        for lab, scorer in (("with radial winsorization (reported)", rc.score_groups),
                            ("no winsorization", score_nowins)):
            p_obs, d_obs, t, o = diffs(A, scorer, est)
            null = permute(A, n, rng, lambda B: diffs(B, scorer, est)[:2])
            rows.append({"estimator": est, "version": lab,
                         "OAC_median_position": float(np.median(o["position"])), "TCOA_median_position": float(np.median(t["position"])),
                         "position_diff": p_obs, "position_null_mean": float(null[:, 0].mean()),
                         "position_p_one_sided": float((1 + np.sum(null[:, 0] >= p_obs)) / (n + 1)),
                         "OAC_median_deviation": float(np.median(o["deviation"])), "TCOA_median_deviation": float(np.median(t["deviation"])),
                         "deviation_diff": d_obs, "deviation_null_mean": float(null[:, 1].mean()),
                         "deviation_p_two_sided": float((1 + np.sum(np.abs(null[:, 1]) >= abs(d_obs))) / (n + 1)),
                         "n_permutations": n})
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "C_no_winsorization_sensitivity.csv", index=False)

    # ---------------- Part D
    obs = dispersion(A)
    keys = ("median_feature_MAD", "mean_distance_to_own_GM", "shared_component", "spread_component", "shared_fraction")
    obs_d = {k: obs["TCOA"][k] - obs["OAC"][k] for k in keys}
    nullD = permute(A, n, rng, lambda B: [(lambda r: r["TCOA"][k] - r["OAC"][k])(dispersion(B)) for k in keys])
    tab = []
    for g in ("YAC", "OAC", "TCOA"):
        tab.append({"group": g, **obs[g]})
    pd.DataFrame(tab).to_csv(OUT / "D_dispersion_by_group.csv", index=False)
    tests = {k: {"TCOA_minus_OAC": obs_d[k], "null_mean": float(nullD[:, j].mean()),
                 "p_two_sided": float((1 + np.sum(np.abs(nullD[:, j] - nullD[:, j].mean()) >= abs(obs_d[k] - nullD[:, j].mean()))) / (n + 1)),
                 "p_upper": float((1 + np.sum(nullD[:, j] >= obs_d[k])) / (n + 1))}
             for j, k in enumerate(keys)}
    (OUT / "D_dispersion_tests.json").write_text(json.dumps(tests, indent=2))
    print(pd.DataFrame(tab).round(3).to_string(index=False))
    print(json.dumps(tests, indent=2))

    # per-feature MAD table (for minor point: median and MAD per feature per group)
    _, (y, o, t) = winsorize_axis_data(A["YAC"], A["OAC"], A["TCOA"])
    model = fit_axis(y, o, estimator=rc.PRIMARY)
    feat = {"feature": names}
    for g, X in (("YAC", y), ("OAC", o), ("TCOA", t)):
        z = score_axis(X, model)["standardized"]
        feat[f"{g}_median"] = np.median(z, 0)
        feat[f"{g}_MAD"] = MAD_K * np.median(np.abs(z - np.median(z, 0)), 0)
    ft = pd.DataFrame(feat)
    ft["TCOA_minus_OAC_median"] = ft.TCOA_median - ft.OAC_median
    ft.to_csv(OUT / "D_feature_median_MAD_by_group.csv", index=False)


if __name__ == "__main__":
    main()
