"""Reviewer 1, comment 2.2 - is the TCOA off-axis deviation directionally
coherent, or just greater scatter?

For every participant the off-axis residual vector is
    r_i = z_i - q_i,   q_i = g_O + s_i * a
(standardized 28-feature vector minus its projection onto the YAC-OAC axis),
exactly as in the manuscript's deviation metric; deviation_i = RMS(r_i).

Coherence of a group = mean pairwise cosine similarity of its members'
residual vectors (equivalently, the length of the group's mean unit residual:
mean pairwise cos = (n * |mean unit vector|^2 - 1) / (n - 1)).

Test statistic: coherence(TCOA) - coherence(OAC).
Inference: 10,000 OAC/TCOA label permutations with full refitting
(winsorization, standardization, endpoints, axis), identical to the
manuscript's deviation test, so the in-sample (OAC) vs out-of-sample (TCOA)
asymmetry is present in the null as well. Reported one-sided (the reviewer's
prediction: TCOA > OAC) and two-sided.

Benchmarks: YAC coherence (in-sample, as an axis-defining group) and
leave-one-out OAC coherence (each OAC scored by an axis fitted without them).

Also saved: the signed mean residual per feature per group, i.e. the shared
direction, if any (feeds the minor-point request for signed feature-level
differences).

Usage:  python deviation_directional_coherence.py [--n-perm 10000] [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

import revision_common as rc

OUT = rc.out_dir("deviation_coherence")


def residuals(scores, model):
    proj = model.old_endpoint + scores["position"][:, None] * model.direction
    return scores["standardized"] - proj


def coherence(R):
    U = R / np.linalg.norm(R, axis=1, keepdims=True)
    n = len(U)
    s = U.sum(axis=0)
    return float((s @ s - n) / (n * (n - 1)))


def group_stats(young, old, test, estimator=rc.PRIMARY):
    t, o, m = rc.score_groups(young, old, test, estimator)
    return residuals(t, m), residuals(o, m), m


def _perm_chunk(args):
    young, pool, n_old, n, seed, estimator = args
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        idx = rng.permutation(len(pool))
        rt, ro, _ = group_stats(young, pool[idx[:n_old]], pool[idx[n_old:]], estimator)
        out[k] = coherence(rt) - coherence(ro)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_perm = 200
    t0 = time.time()
    _, names, A = rc.load_features()
    yac, oac, tcoa = A["YAC"], A["OAC"], A["TCOA"]

    summary_rows, nulls = [], {}
    for est in rc.ESTIMATORS:
        rt, ro, model = group_stats(yac, oac, tcoa, est)
        ry_scores, _, _ = rc.score_groups(yac, oac, yac, est)
        ry = residuals(ry_scores, model)
        observed = coherence(rt) - coherence(ro)
        pool = np.vstack([oac, tcoa])
        seeds = np.random.SeedSequence(a.seed).spawn(a.workers)
        sizes = [a.n_perm // a.workers + (1 if k < a.n_perm % a.workers else 0) for k in range(a.workers)]
        jobs = [(yac, pool, len(oac), s, sd, est) for s, sd in zip(sizes, seeds)]
        if a.workers > 1:
            with Pool(a.workers) as p:
                null = np.concatenate(p.map(_perm_chunk, jobs))
        else:
            null = _perm_chunk(jobs[0])
        nulls[est] = null

        # leave-one-out OAC benchmark (primary only is enough, cheap anyway)
        loo = []
        for i in range(len(oac)):
            s, _, m = rc.score_groups(yac, np.delete(oac, i, 0), oac[i:i + 1], est)
            loo.append(residuals(s, m)[0])
        loo = np.array(loo)

        row = {
            "estimator": est,
            "coherence_TCOA": coherence(rt), "coherence_OAC_in_sample": coherence(ro),
            "coherence_YAC_in_sample": coherence(ry), "coherence_OAC_leave_one_out": coherence(loo),
            "observed_TCOA_minus_OAC": observed,
            "null_mean": float(null.mean()), "null_median": float(np.median(null)),
            "null_2_5": float(np.percentile(null, 2.5)), "null_97_5": float(np.percentile(null, 97.5)),
            "p_one_sided_TCOA_gt_OAC": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
            "p_two_sided": float((1 + np.sum(np.abs(null - null.mean()) >= abs(observed - null.mean()))) / (len(null) + 1)),
            "n_permutations": len(null),
        }
        summary_rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

        if est == rc.PRIMARY:
            signed = pd.DataFrame({
                "feature": names,
                "mean_residual_TCOA": rt.mean(0), "mean_residual_OAC": ro.mean(0), "mean_residual_YAC": ry.mean(0),
                "median_residual_TCOA": np.median(rt, 0), "median_residual_OAC": np.median(ro, 0),
                "sign_agreement_TCOA": (np.sign(rt) == np.sign(rt.mean(0))).mean(0),
            })
            signed["TCOA_minus_OAC_mean_residual"] = signed.mean_residual_TCOA - signed.mean_residual_OAC
            signed.sort_values("TCOA_minus_OAC_mean_residual", key=np.abs, ascending=False).to_csv(
                OUT / "signed_mean_residual_by_feature.csv", index=False)
            pair = lambda R: (lambda U: U @ U.T)(R / np.linalg.norm(R, axis=1, keepdims=True))
            pd.DataFrame(pair(rt)).to_csv(OUT / "pairwise_cosine_TCOA.csv", index=False)
            pd.DataFrame(pair(ro)).to_csv(OUT / "pairwise_cosine_OAC.csv", index=False)
            primary = (row, null, rt, ro, ry, loo)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "coherence_permutation_results.csv", index=False)
    np.savez_compressed(OUT / "coherence_permutation_nulls.npz", **nulls)

    import matplotlib.pyplot as plt
    row, null, rt, ro, ry, loo = primary
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]

    def offdiag(R):
        U = R / np.linalg.norm(R, axis=1, keepdims=True)
        C = U @ U.T
        return C[np.triu_indices(len(C), 1)]

    data = [("YAC\n(in-sample)", offdiag(ry), rc.GROUP_COLORS["YAC"]),
            ("OAC\n(in-sample)", offdiag(ro), rc.GROUP_COLORS["OAC"]),
            ("OAC\n(leave-one-out)", offdiag(loo), "#c9b59a"),
            ("TCOA", offdiag(rt), rc.GROUP_COLORS["TCOA"])]
    jit = np.random.default_rng(0)
    for k, (lab, v, col) in enumerate(data):
        ax.scatter(k + jit.uniform(-0.18, 0.18, len(v)), v, s=9, color=col, alpha=0.55, linewidth=0)
        ax.hlines(v.mean(), k - 0.3, k + 0.3, color=rc.INK, linewidth=2.2)
        ax.text(k + 0.32, v.mean(), f"{v.mean():+.3f}", fontsize=8.5, va="center")
    ax.axhline(0, color=rc.INK2, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(data)), [d[0] for d in data], fontsize=8.5)
    ax.set_ylabel("Pairwise cosine of off-axis residuals", fontsize=9, color=rc.INK2)
    ax.set_title("Do group members deviate in a shared direction?", fontsize=10.5, loc="left")
    rc.style_axes(ax)
    ax = axes[1]
    ax.hist(null, bins=60, color=rc.NULL_FILL, edgecolor="white", linewidth=0.4)
    ax.axvline(null.mean(), color=rc.ACCENT2, linewidth=2)
    ax.axvline(row["observed_TCOA_minus_OAC"], color=rc.ACCENT, linewidth=2.4)
    ax.text(0.02, 0.95, f"observed {row['observed_TCOA_minus_OAC']:+.3f}\nnull mean {null.mean():+.3f}\n"
                        f"one-sided p = {row['p_one_sided_TCOA_gt_OAC']:.4f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=rc.INK)
    ax.set_xlabel("Coherence, TCOA − OAC, under label permutation", fontsize=9, color=rc.INK2)
    ax.set_title("Permutation test (10,000, full refit)", fontsize=10.5, loc="left")
    rc.style_axes(ax)
    fig.savefig(OUT / "FigR_deviation_coherence.png", dpi=250)
    plt.close(fig)
    print(f"\nDone in {time.time() - t0:.0f} s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
