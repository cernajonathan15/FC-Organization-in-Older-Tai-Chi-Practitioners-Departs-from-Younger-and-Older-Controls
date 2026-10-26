"""Reviewer 1, comment 2.1 - permutation tail direction, null centring,
positive controls, and the observed-vs-bootstrap median-position gap.

Parts (run all by default, or choose with --parts):
  A  Reproduce the published observed statistics from the saved features.
  B  Summarise and plot the saved 10,000-permutation null distributions
     (mean, median, observed value, exact tail rule).
  C  Mechanism: in each permutation, record the permuted-OAC and permuted-TCOA
     median positions separately, to show where the positive null shift comes
     from (in-sample OAC sits at ~0 by construction; out-of-sample group does not).
  D  Positive control 1 - identical pipeline, YAC vs OAC. YAC is split at random
     into an axis-defining half (8) and a held-out "test" half (7) that takes
     the TCOA slot. Repeated over many random splits.
  E  Positive control 2 - known-effect injection. TCOA vectors are shifted
     toward YAC by delta x (YAC-OAC axis vector) and the full permutation test
     is re-run, showing what displacement this design can detect.
  F  Observed (0.124) vs bootstrap (0.412) TCOA median position: decompose the
     bootstrap by resampling only TCOA, only the reference groups, or all.
  G  Out-of-sample validity of the YAC-OAC axis: leave-one-out positions of
     reference participants, YAC-vs-OAC separability, split-half stability of
     the axis direction.

Usage (from this folder):
  python permutation_calibration_and_positive_controls.py                 # everything
  python permutation_calibration_and_positive_controls.py --parts A B     # quick parts only
  python permutation_calibration_and_positive_controls.py --quick         # small counts, smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import revision_common as rc

OUT = rc.out_dir("permutation_calibration")


# --------------------------------------------------------------------- workers
def _perm_chunk(args):
    young, pool, n_ref, n_perm, seed, keep_groups = args
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_perm)
    ref_med = np.empty(n_perm)
    test_med = np.empty(n_perm)
    for i in range(n_perm):
        order = rng.permutation(len(pool))
        old, test = pool[order[:n_ref]], pool[order[n_ref:]]
        t, o, _ = rc.score_groups(young, old, test)
        ref_med[i] = np.median(o["position"])
        test_med[i] = np.median(t["position"])
        diffs[i] = test_med[i] - ref_med[i]
    return (diffs, ref_med, test_med) if keep_groups else (diffs,)


def permutation_null(young, ref, test, n_perm, seed, workers, keep_groups=False):
    """Permute pooled ref+test labels (young fixed), full refit every time."""
    pool_arr = np.vstack([ref, test])
    n_ref = len(ref)
    seeds = np.random.SeedSequence(seed).spawn(workers)
    sizes = [n_perm // workers + (1 if k < n_perm % workers else 0) for k in range(workers)]
    jobs = [(young, pool_arr, n_ref, s, sd, keep_groups) for s, sd in zip(sizes, seeds)]
    if workers > 1:
        with Pool(workers) as p:
            parts = p.map(_perm_chunk, jobs)
    else:
        parts = [_perm_chunk(j) for j in jobs]
    return tuple(np.concatenate([part[k] for part in parts]) for k in range(len(parts[0])))


def _boot_chunk(args):
    young, old, tcoa, n, seed, mode = args
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        y, o, t = young, old, tcoa
        if mode in ("all", "reference_only"):
            y = young[rng.integers(0, len(young), len(young))]
            o = old[rng.integers(0, len(old), len(old))]
        if mode in ("all", "tcoa_only"):
            t = tcoa[rng.integers(0, len(tcoa), len(tcoa))]
        scores, _, _ = rc.score_groups(y, o, t)
        out[i] = np.median(scores["position"])
    return out


def bootstrap(young, old, tcoa, n, seed, workers, mode):
    seeds = np.random.SeedSequence(seed).spawn(workers)
    sizes = [n // workers + (1 if k < n % workers else 0) for k in range(workers)]
    jobs = [(young, old, tcoa, s, sd, mode) for s, sd in zip(sizes, seeds)]
    if workers > 1:
        with Pool(workers) as p:
            return np.concatenate(p.map(_boot_chunk, jobs))
    return np.concatenate([_boot_chunk(j) for j in jobs])


# ------------------------------------------------------------------------ parts
def part_a(arrays):
    published = pd.read_csv(rc.METHOD_DIR / "03_inference" / "primary_phenotype_inference.csv")
    rows = []
    for est in rc.ESTIMATORS:
        t, o, _ = rc.score_groups(arrays["YAC"], arrays["OAC"], arrays["TCOA"], est)
        for metric in ("position", "deviation", "relative_distance_youth", "rsa_fisher_difference"):
            reducer = np.median if metric in ("position", "deviation") else np.mean
            ours = float(reducer(t[metric]) - reducer(o[metric]))
            rows.append({"estimator": est, "metric": metric, "reproduced_tcoa_minus_oac": ours})
    rep = pd.DataFrame(rows)
    pub = published.rename(columns={"axis_estimator": "estimator"})[
        ["estimator", "metric", "observed_tcoa_minus_oac"]
    ]
    rep = rep.merge(pub, on=["estimator", "metric"], how="left")
    robust = pd.read_csv(rc.METHOD_DIR / "03_inference" / "robustness_inference.csv")
    robust = robust.rename(columns={"axis_estimator": "estimator"})
    if "observed_tcoa_minus_oac" in robust.columns:
        rep = rep.merge(
            robust[["estimator", "metric", "observed_tcoa_minus_oac"]].rename(
                columns={"observed_tcoa_minus_oac": "published_robustness"}),
            on=["estimator", "metric"], how="left")
        rep["observed_tcoa_minus_oac"] = rep["observed_tcoa_minus_oac"].fillna(rep["published_robustness"])
        rep = rep.drop(columns="published_robustness")
    rep = rep.rename(columns={"observed_tcoa_minus_oac": "published_tcoa_minus_oac"})
    rep["abs_difference"] = (rep["reproduced_tcoa_minus_oac"] - rep["published_tcoa_minus_oac"]).abs()
    rep.to_csv(OUT / "A_reproduction_check.csv", index=False)
    print(rep.round(4).to_string(index=False))
    return rep


def part_b(arrays, reproduction):
    import matplotlib.pyplot as plt

    null = np.load(rc.METHOD_DIR / "03_inference" / "permutation_distributions_by_estimator.npz")
    obs = reproduction.set_index(["estimator", "metric"])["reproduced_tcoa_minus_oac"]
    rows = []
    for est in rc.ESTIMATORS:
        for metric in ("position", "deviation", "relative_distance_youth", "rsa_fisher_difference"):
            d = null[f"{est}__{metric}"]
            o = obs[(est, metric)]
            two_sided = metric == "deviation"
            rows.append({
                "estimator": est, "metric": metric,
                "observed_tcoa_minus_oac": o,
                "null_mean": d.mean(), "null_median": np.median(d), "null_sd": d.std(ddof=1),
                "null_2_5": np.percentile(d, 2.5), "null_97_5": np.percentile(d, 97.5),
                "observed_minus_null_mean": o - d.mean(),
                "observed_percentile_in_null": 100 * np.mean(d < o),
                "tail_rule": "(1 + #{|null| >= |obs|})/(N+1)" if two_sided else "(1 + #{null >= obs})/(N+1)",
                "p_published_rule": float((1 + np.sum(np.abs(d) >= abs(o))) / (len(d) + 1)) if two_sided
                                    else rc.upper_tail_p(d, o),
                "p_lower_tail_for_reference": float((1 + np.sum(d <= o)) / (len(d) + 1)),
                "n_permutations": len(d),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "B_null_distribution_summary.csv", index=False)
    print(summary[["estimator", "metric", "observed_tcoa_minus_oac", "null_mean", "null_median",
                   "observed_percentile_in_null", "p_published_rule"]].round(3).to_string(index=False))

    # Supplementary figure: primary estimator, three youth-resemblance metrics.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, metric in zip(axes, rc.YOUTH_METRICS):
        d = null[f"{rc.PRIMARY}__{metric}"]
        s = summary.query("estimator == @rc.PRIMARY and metric == @metric").iloc[0]
        ax.hist(d, bins=60, color=rc.NULL_FILL, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=rc.INK2, linewidth=1, linestyle=":")
        ax.axvline(s.null_mean, color=rc.ACCENT2, linewidth=2)
        ax.axvline(s.null_median, color=rc.ACCENT2, linewidth=1.2, linestyle="--")
        ax.axvline(s.observed_tcoa_minus_oac, color=rc.ACCENT, linewidth=2.4)
        ax.set_title(rc.METRIC_LABELS[metric], fontsize=11, color=rc.INK, loc="left")
        ax.set_xlabel("TCOA − OAC under label permutation", fontsize=9, color=rc.INK2)
        ymax = ax.get_ylim()[1]
        ax.text(s.observed_tcoa_minus_oac, ymax * 0.97,
                f" observed {s.observed_tcoa_minus_oac:+.3f}\n one-sided p = {s.p_published_rule:.3f}",
                color=rc.ACCENT, fontsize=8.5, va="top")
        ax.text(0.98, 0.70, f"null mean {s.null_mean:+.3f}\nnull median {s.null_median:+.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=rc.ACCENT2)
        rc.style_axes(ax)
    axes[0].set_ylabel("Permutations (of 10,000)", fontsize=9, color=rc.INK2)
    fig.suptitle("Permutation null distributions, primary Euclidean geometric-median analysis  "
                 "(blue = observed, orange solid/dashed = null mean/median, dotted = 0)",
                 fontsize=9.5, color=rc.INK2, x=0.01, ha="left")
    fig.savefig(OUT / "FigS_permutation_nulls_primary.png", dpi=300)
    plt.close(fig)

    # Same, all estimators (supplement / response letter).
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.4), constrained_layout=True)
    for r, est in enumerate(rc.ESTIMATORS):
        for c, metric in enumerate(rc.YOUTH_METRICS):
            ax = axes[r, c]
            d = null[f"{est}__{metric}"]
            s = summary.query("estimator == @est and metric == @metric").iloc[0]
            ax.hist(d, bins=60, color=rc.NULL_FILL, edgecolor="white", linewidth=0.4)
            ax.axvline(0, color=rc.INK2, linewidth=1, linestyle=":")
            ax.axvline(s.null_mean, color=rc.ACCENT2, linewidth=2)
            ax.axvline(s.observed_tcoa_minus_oac, color=rc.ACCENT, linewidth=2.4)
            ax.set_title(f"{rc.ESTIMATOR_LABELS[est]} · {rc.METRIC_LABELS[metric]}", fontsize=9, loc="left")
            ax.text(0.98, 0.95, f"obs {s.observed_tcoa_minus_oac:+.3f}\nnull mean {s.null_mean:+.3f}\np = {s.p_published_rule:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, color=rc.INK2)
            rc.style_axes(ax)
    fig.savefig(OUT / "FigS_permutation_nulls_all_estimators.png", dpi=200)
    plt.close(fig)
    return summary


def part_c(arrays, n_perm, seed, workers):
    diffs, ref_med, test_med = permutation_null(
        arrays["YAC"], arrays["OAC"], arrays["TCOA"], n_perm, seed, workers, keep_groups=True)
    t, o, _ = rc.score_groups(arrays["YAC"], arrays["OAC"], arrays["TCOA"])
    res = {
        "n_permutations": int(n_perm),
        "permuted_OAC_median_position_mean": float(ref_med.mean()),
        "permuted_TCOA_median_position_mean": float(test_med.mean()),
        "null_difference_mean": float(diffs.mean()),
        "observed_OAC_median_position": float(np.median(o["position"])),
        "observed_TCOA_median_position": float(np.median(t["position"])),
        "p_upper_tail_this_run": rc.upper_tail_p(diffs, float(np.median(t["position"]) - np.median(o["position"]))),
    }
    (OUT / "C_null_mechanism.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame({"perm_oac_median": ref_med, "perm_tcoa_median": test_med, "difference": diffs}).to_csv(
        OUT / "C_null_mechanism_replicates.csv.gz", index=False)
    print(json.dumps(res, indent=2))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    bins = np.linspace(min(ref_med.min(), test_med.min()), max(ref_med.max(), test_med.max()), 70)
    ax.hist(ref_med, bins=bins, color=rc.GROUP_COLORS["OAC"], alpha=0.75, label="Permuted 'OAC' (builds the axis origin)")
    ax.hist(test_med, bins=bins, color=rc.GROUP_COLORS["TCOA"], alpha=0.75, label="Permuted 'TCOA' (scored out-of-sample)")
    ax.axvline(res["observed_OAC_median_position"], color=rc.GROUP_COLORS["OAC"], linewidth=2.2, linestyle="--")
    ax.axvline(res["observed_TCOA_median_position"], color=rc.GROUP_COLORS["TCOA"], linewidth=2.2, linestyle="--")
    ax.set_xlabel("Group median phenotype position (OAC = 0, YAC = 1)", fontsize=9, color=rc.INK2)
    ax.set_ylabel("Permutations", fontsize=9, color=rc.INK2)
    ax.set_title("Why the position null is shifted positive", fontsize=11, loc="left", color=rc.INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.text(0.01, 0.97, "dashed = observed group medians", transform=ax.transAxes, fontsize=8, color=rc.INK2, va="top")
    rc.style_axes(ax)
    fig.savefig(OUT / "FigR_null_mechanism.png", dpi=250)
    plt.close(fig)
    return res


def part_d(arrays, n_splits, n_perm, seed, workers):
    rng = np.random.default_rng(seed)
    yac, oac = arrays["YAC"], arrays["OAC"]
    rows = []
    for s in range(n_splits):
        idx = rng.permutation(len(yac))
        axis_young, held_out = yac[idx[:8]], yac[idx[8:]]
        t, o, _ = rc.score_groups(axis_young, oac, held_out)
        observed = float(np.median(t["position"]) - np.median(o["position"]))
        (null,) = permutation_null(axis_young, oac, held_out, n_perm, seed + 1000 + s, workers)
        rows.append({"split": s, "heldout_yac_median_position": float(np.median(t["position"])),
                     "oac_median_position": float(np.median(o["position"])),
                     "observed_difference": observed, "null_mean": float(null.mean()),
                     "p_upper_tail": rc.upper_tail_p(null, observed), "n_permutations": n_perm})
        print(f"  split {s + 1}/{n_splits}: diff={observed:+.3f}  null mean={null.mean():+.3f}  p={rows[-1]['p_upper_tail']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "D_positive_control_YAC_vs_OAC.csv", index=False)
    summary = {
        "n_splits": n_splits, "n_permutations_per_split": n_perm,
        "median_p": float(df.p_upper_tail.median()),
        "proportion_p_below_.05": float((df.p_upper_tail < 0.05).mean()),
        "median_heldout_yac_position": float(df.heldout_yac_median_position.median()),
        "median_null_mean": float(df.null_mean.median()),
    }
    (OUT / "D_positive_control_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return df


def part_e(arrays, deltas, n_perm, seed, workers):
    yac, oac, tcoa = arrays["YAC"], arrays["OAC"], arrays["TCOA"]
    _, _, model = rc.score_groups(yac, oac, tcoa)
    raw_axis_vector = model.direction * model.scale          # YAC-OAC vector in raw feature units
    rows = []
    for k, delta in enumerate(deltas):
        shifted = tcoa + delta * raw_axis_vector
        t, o, _ = rc.score_groups(yac, oac, shifted)
        observed = float(np.median(t["position"]) - np.median(o["position"]))
        (null,) = permutation_null(yac, oac, shifted, n_perm, seed + 2000 + k, workers)
        rows.append({"injected_shift_delta": delta, "tcoa_median_position": float(np.median(t["position"])),
                     "observed_difference": observed, "null_mean": float(null.mean()),
                     "p_upper_tail": rc.upper_tail_p(null, observed), "n_permutations": n_perm})
        print(f"  delta={delta:.2f}: TCOA median={rows[-1]['tcoa_median_position']:+.3f}  diff={observed:+.3f}  "
              f"null mean={null.mean():+.3f}  p={rows[-1]['p_upper_tail']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E_positive_control_injected_shift.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    ax.plot(df.injected_shift_delta, df.p_upper_tail, color=rc.ACCENT, linewidth=2, marker="o", markersize=7)
    ax.axhline(0.05, color=rc.INK2, linewidth=1, linestyle="--")
    ax.text(df.injected_shift_delta.max(), 0.055, "p = .05", ha="right", va="bottom", fontsize=8, color=rc.INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Injected TCOA shift toward YAC (fraction of the OAC→YAC axis)", fontsize=9, color=rc.INK2)
    ax.set_ylabel("One-sided permutation p", fontsize=9, color=rc.INK2)
    ax.set_title("Positive control: can the position test detect a real shift?", fontsize=10.5, loc="left")
    rc.style_axes(ax)
    fig.savefig(OUT / "FigR_positive_control_injected_shift.png", dpi=250)
    plt.close(fig)
    return df


def part_f(arrays, n_boot, seed, workers):
    yac, oac, tcoa = arrays["YAC"], arrays["OAC"], arrays["TCOA"]
    t, _, _ = rc.score_groups(yac, oac, tcoa)
    positions = np.sort(t["position"])
    saved = pd.read_csv(rc.METHOD_DIR / "03_inference" / "bootstrap_replicates.csv.gz")
    saved = saved.loc[saved["axis_estimator"].eq(rc.PRIMARY), "median_tcoa_position"].to_numpy()
    results = {"observed_tcoa_median_position": float(np.median(positions)),
               "observed_tcoa_mean_position": float(np.mean(positions)),
               "tcoa_positions_sorted": [round(float(v), 4) for v in positions],
               "saved_pipeline_bootstrap_mean_of_medians": float(saved.mean()),
               "saved_pipeline_bootstrap_median_of_medians": float(np.median(saved))}
    draws = {}
    for k, mode in enumerate(("tcoa_only", "reference_only", "all")):
        draws[mode] = bootstrap(yac, oac, tcoa, n_boot, seed + 3000 + k, workers, mode)
        results[f"{mode}_bootstrap_mean_of_medians"] = float(draws[mode].mean())
        results[f"{mode}_bootstrap_median_of_medians"] = float(np.median(draws[mode]))
        results[f"{mode}_ci_2_5"] = float(np.percentile(draws[mode], 2.5))
        results[f"{mode}_ci_97_5"] = float(np.percentile(draws[mode], 97.5))
        print(f"  {mode}: mean of medians={draws[mode].mean():+.3f}", flush=True)
    (OUT / "F_bootstrap_decomposition.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(draws).to_csv(OUT / "F_bootstrap_decomposition_replicates.csv.gz", index=False)
    print(json.dumps(results, indent=2))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    ax.scatter(positions, np.zeros_like(positions), s=60, color=rc.GROUP_COLORS["TCOA"], edgecolor="white", linewidth=1.5, zorder=3)
    ax.axvline(np.median(positions), color=rc.INK, linewidth=1.6, label=f"median {np.median(positions):.3f}")
    ax.axvline(np.mean(positions), color=rc.INK2, linewidth=1.2, linestyle="--", label=f"mean {np.mean(positions):.3f}")
    ax.set_yticks([])
    ax.set_xlabel("TCOA phenotype position (OAC = 0, YAC = 1)", fontsize=9, color=rc.INK2)
    ax.set_title("The 15 observed TCOA positions", fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False)
    rc.style_axes(ax)
    ax = axes[1]
    bins = np.linspace(-0.6, 1.4, 70)
    for mode, color, label in (("tcoa_only", rc.GROUP_COLORS["TCOA"], "resample TCOA only (axis fixed)"),
                               ("reference_only", rc.GROUP_COLORS["OAC"], "resample YAC+OAC only (TCOA fixed)"),
                               ("all", rc.ACCENT, "resample all (as in manuscript)")):
        ax.hist(draws[mode], bins=bins, histtype="step", linewidth=2, color=color,
                label=f"{label}: mean {draws[mode].mean():.3f}")
    ax.axvline(np.median(positions), color=rc.INK, linewidth=1.6)
    ax.set_xlabel("Bootstrap TCOA median position", fontsize=9, color=rc.INK2)
    ax.set_title("Where the 0.124 → 0.412 gap comes from", fontsize=10.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    rc.style_axes(ax)
    fig.savefig(OUT / "FigR_bootstrap_vs_observed_median.png", dpi=250)
    plt.close(fig)
    return results


def part_g(arrays, seed):
    """Out-of-sample validity of the YAC-OAC axis itself.

    The positive control (D) failed, so this part asks why: are YAC and OAC
    separable on the 28 features at all, and where does each reference
    participant land when scored by an axis fitted without them (leave-one-out)?
    """
    from scipy.stats import mannwhitneyu
    import matplotlib.pyplot as plt

    _, names, _ = rc.load_features()
    yac, oac, tcoa = arrays["YAC"], arrays["OAC"], arrays["TCOA"]
    t_full, o_full, _ = rc.score_groups(yac, oac, tcoa)
    y_full, _, _ = rc.score_groups(yac, oac, yac)

    loo = {"YAC": {"position": [], "deviation": []}, "OAC": {"position": [], "deviation": []}}
    for g in ("YAC", "OAC"):
        for i in range(15):
            Y = np.delete(yac, i, 0) if g == "YAC" else yac
            O = np.delete(oac, i, 0) if g == "OAC" else oac
            s, _, _ = rc.score_groups(Y, O, arrays[g][i:i + 1])
            loo[g]["position"].append(float(s["position"][0]))
            loo[g]["deviation"].append(float(s["deviation"][0]))

    X = np.vstack([yac, oac]); lab = np.r_[np.zeros(15), np.ones(15)]
    correct = 0
    for i in range(30):
        keep = np.arange(30) != i
        s, _, _ = rc.score_groups(X[keep & (lab == 0)], X[keep & (lab == 1)], X[i:i + 1])
        correct += int((1 if s["distance_old"][0] < s["distance_young"][0] else 0) == lab[i])

    pvals = np.array([mannwhitneyu(yac[:, j], oac[:, j]).pvalue for j in range(yac.shape[1])])

    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(500):
        iy, io = rng.permutation(15), rng.permutation(15)
        _, _, m1 = rc.score_groups(yac[iy[:7]], oac[io[:7]], tcoa)
        _, _, m2 = rc.score_groups(yac[iy[7:]], oac[io[7:]], tcoa)
        a1, a2 = m1.direction * m1.scale, m2.direction * m2.scale
        cosines.append(float(a1 @ a2 / np.linalg.norm(a1) / np.linalg.norm(a2)))

    res = {
        "in_sample_median_position": {"YAC": float(np.median(y_full["position"])),
                                      "OAC": float(np.median(o_full["position"])),
                                      "TCOA (always out-of-sample)": float(np.median(t_full["position"]))},
        "leave_one_out_median_position": {g: float(np.median(loo[g]["position"])) for g in loo},
        "deviation_median": {"OAC in-sample": float(np.median(o_full["deviation"])),
                             "OAC leave-one-out": float(np.median(loo["OAC"]["deviation"])),
                             "YAC in-sample": float(np.median(y_full["deviation"])),
                             "YAC leave-one-out": float(np.median(loo["YAC"]["deviation"])),
                             "TCOA": float(np.median(t_full["deviation"]))},
        "loo_nearest_endpoint_classification_YAC_vs_OAC": f"{correct}/30",
        "n_features_YAC_vs_OAC_mannwhitney_p_lt_.05": int((pvals < 0.05).sum()),
        "smallest_feature_p": float(pvals.min()),
        "split_half_axis_direction_cosine_median": float(np.median(cosines)),
        "split_half_axis_direction_cosine_IQR": [float(np.percentile(cosines, 25)), float(np.percentile(cosines, 75))],
    }
    (OUT / "G_axis_out_of_sample_validity.json").write_text(json.dumps(res, indent=2))
    rows = [{"group": g, "participant_index": i, "loo_position": loo[g]["position"][i],
             "loo_deviation": loo[g]["deviation"][i]} for g in loo for i in range(15)]
    pd.DataFrame(rows).to_csv(OUT / "G_leave_one_out_reference_scores.csv", index=False)
    pd.DataFrame({"feature": names, "mannwhitney_p_YAC_vs_OAC": pvals}).sort_values(
        "mannwhitney_p_YAC_vs_OAC").to_csv(OUT / "G_univariate_YAC_vs_OAC.csv", index=False)
    print(json.dumps(res, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True, sharey=True)
    jitter = np.random.default_rng(0)
    panels = ((axes[0], "In-sample (as in manuscript)",
               {"YAC": y_full["position"], "OAC": o_full["position"], "TCOA": t_full["position"]}),
              (axes[1], "Leave-one-out (each person scored by an axis built without them)",
               {"YAC": np.array(loo["YAC"]["position"]), "OAC": np.array(loo["OAC"]["position"]),
                "TCOA": t_full["position"]}))
    for ax, title, data in panels:
        for k, g in enumerate(("YAC", "OAC", "TCOA")):
            v = data[g]
            ax.scatter(k + jitter.uniform(-0.12, 0.12, len(v)), v, s=46, color=rc.GROUP_COLORS[g],
                       edgecolor="white", linewidth=1.2, zorder=3)
            ax.hlines(np.median(v), k - 0.28, k + 0.28, color=rc.INK, linewidth=2, zorder=4)
            ax.text(k + 0.3, np.median(v), f"{np.median(v):.2f}", fontsize=8.5, va="center", color=rc.INK)
        for yv in (0, 0.5, 1):
            ax.axhline(yv, color=rc.INK2, linewidth=0.8, linestyle=":")
        ax.set_xticks([0, 1, 2], ["YAC", "OAC", "TCOA"])
        ax.set_title(title, fontsize=9.5, loc="left")
        rc.style_axes(ax)
    axes[0].set_ylabel("Phenotype position (OAC = 0, YAC = 1)", fontsize=9, color=rc.INK2)
    fig.savefig(OUT / "FigR_axis_in_sample_vs_leave_one_out.png", dpi=250)
    plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=list("ABCDEFG"), choices=list("ABCDEFG"))
    ap.add_argument("--n-perm", type=int, default=2000, help="permutations for parts C and E")
    ap.add_argument("--n-splits", type=int, default=20, help="random YAC splits for part D")
    ap.add_argument("--n-perm-split", type=int, default=1000, help="permutations per split, part D")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap reps per mode, part F")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_perm, a.n_splits, a.n_perm_split, a.n_boot = 100, 2, 100, 100
    _, _, arrays = rc.load_features()
    t0 = time.time()
    rep = part_a(arrays)                      # always needed for B
    if "B" in a.parts:
        print("\n== B: saved null distributions"); part_b(arrays, rep)
    if "C" in a.parts:
        print("\n== C: null mechanism"); part_c(arrays, a.n_perm, a.seed, a.workers)
    if "D" in a.parts:
        print("\n== D: positive control, YAC vs OAC"); part_d(arrays, a.n_splits, a.n_perm_split, a.seed, a.workers)
    if "E" in a.parts:
        print("\n== E: positive control, injected shift"); part_e(arrays, a.deltas, a.n_perm, a.seed, a.workers)
    if "F" in a.parts:
        print("\n== F: bootstrap decomposition"); part_f(arrays, a.n_boot, a.seed, a.workers)
    if "G" in a.parts:
        print("\n== G: out-of-sample validity of the axis"); part_g(arrays, a.seed)
    print(f"\nDone in {time.time() - t0:.0f} s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
