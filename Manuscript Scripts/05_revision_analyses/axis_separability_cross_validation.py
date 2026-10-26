"""Can ANY version of the 28-feature network representation separate YAC from
OAC out of sample?  Decides whether a "sturdier" age axis is even possible.

Motivation: the 2.1 positive control showed the manuscript's YAC-OAC axis
does not generalise (leave-one-out YAC ~0.06, OAC ~0.65). Before trying to
rebuild the axis, test - with proper cross-validation - whether the age signal
exists in the available features at all.

Feature sets (all from results/robust_median/01_dataset, no new preprocessing):
  pooled_28        manuscript features (all matrices, EC and EO weighted equally)
  EC_28 / EO_28    eyes-closed / eyes-open only
  EC_EO_56         EC and EO side by side (lets the classifier use condition-specific effects)
  state_equal_28   each participant's HMM states weighted equally (mean of state means)
                   rather than by number of visits; EC/EO averaged
  variability_28   SD of each feature across the participant's matrices (dynamic FC
                   variability), EC/EO averaged
  pooled_plus_variability_56

HMMs were fitted per participant (5-7 states) and state labels are not
matched across people, so state-specific features cannot be compared and are
not used.

Classifiers, all evaluated leave-one-participant-out (scaling fitted inside the fold):
  axis     the manuscript's own pipeline (radial winsorization + geometric-median
           endpoints); score = leave-one-out phenotype position
  lda      Ledoit-Wolf shrinkage LDA
  logreg   L2 logistic regression (C = 1)
Metric: LOO AUC; p from 1,000 label permutations (whole LOO repeated per permutation).
Secondary contrast: TCOA vs OAC with the same machinery, for context.

Usage:  python axis_separability_cross_validation.py [--n-perm 1000] [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import revision_common as rc

OUT = rc.out_dir("axis_separability")


def build_feature_sets() -> tuple[dict[str, pd.DataFrame], list[str]]:
    _, names, _ = rc.load_features()
    base = rc.METHOD_DIR / "01_dataset"
    pooled = pd.read_csv(base / "participant_features.csv", dtype={"participant": str}).set_index("participant")
    mode = pd.read_csv(base / "participant_mode_features.csv", dtype={"participant": str})
    dyn = pd.read_csv(base / "dynamic_network_features.csv.gz", dtype={"participant": str})
    groups = pooled["group"]

    ec = mode[mode["mode"].eq("EC")].set_index("participant")[names]
    eo = mode[mode["mode"].eq("EO")].set_index("participant")[names]
    state_means = dyn.groupby(["participant", "mode", "state"])[names].mean()
    state_equal = state_means.groupby(["participant", "mode"]).mean().groupby("participant").mean()
    variability = dyn.groupby(["participant", "mode"])[names].std(ddof=1).groupby("participant").mean()

    sets = {
        "pooled_28": pooled[names],
        "EC_28": ec,
        "EO_28": eo,
        "EC_EO_56": pd.concat([ec.add_prefix("EC__"), eo.add_prefix("EO__")], axis=1),
        "state_equal_28": state_equal,
        "variability_28": variability,
        "pooled_plus_variability_56": pd.concat([pooled[names], variability.add_prefix("SD__")], axis=1),
    }
    sets = {k: v.loc[groups.index].assign(group=groups) for k, v in sets.items()}
    state_equal.to_csv(OUT / "state_equal_weighted_features.csv")   # reused for R1 2.6
    variability.to_csv(OUT / "dynamic_variability_features.csv")
    return sets, names


# ---------------------------------------------------------------- LOO scoring
def loo_scores(X: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    """Out-of-sample score per participant; higher = more like class 1 (YAC)."""
    s = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        if method == "axis":
            young, old = X[tr & (y == 1)], X[tr & (y == 0)]
            t, _, _ = rc.score_groups(young, old, X[i:i + 1])
            s[i] = t["position"][0]
        else:
            clf = (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto") if method == "lda"
                   else LogisticRegression(C=1.0, max_iter=5000))
            model = make_pipeline(StandardScaler(), clf).fit(X[tr], y[tr])
            s[i] = model.decision_function(X[i:i + 1])[0]
    return s


def _perm_job(args):
    X, y, method, seeds = args
    out = []
    for sd in seeds:
        yp = np.random.default_rng(sd).permutation(y)
        out.append(roc_auc_score(yp, loo_scores(X, yp, method)))
    return out


def loo_auc_with_permutation(X, y, method, n_perm, seed, workers):
    observed = roc_auc_score(y, loo_scores(X, y, method))
    seeds = np.random.SeedSequence(seed).generate_state(n_perm)
    chunks = [(X, y, method, seeds[k::workers]) for k in range(workers)]
    if workers > 1:
        with Pool(workers) as p:
            null = np.concatenate(p.map(_perm_job, chunks))
    else:
        null = np.array(_perm_job(chunks[0]))
    p = (1 + np.count_nonzero(null >= observed)) / (n_perm + 1)
    return observed, float(p), null


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()))
    ap.add_argument("--methods", nargs="+", default=["axis", "lda", "logreg"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_perm = 20
    t0 = time.time()
    sets, _ = build_feature_sets()

    rows, nulls = [], {}
    contrasts = [("YAC_vs_OAC", "YAC", "OAC")] + [("TCOA_vs_OAC", "TCOA", "OAC")]
    for contrast, pos_group, neg_group in contrasts:
        set_names = list(sets) if contrast == "YAC_vs_OAC" else ["pooled_28", "EC_EO_56", "variability_28"]
        for set_name in set_names:
            df = sets[set_name]
            sub = df[df["group"].isin([pos_group, neg_group])]
            X = sub.drop(columns="group").to_numpy(float)
            y = (sub["group"] == pos_group).to_numpy(int)
            pu = np.array([mannwhitneyu(X[y == 1, j], X[y == 0, j]).pvalue for j in range(X.shape[1])])
            for method in a.methods:
                if contrast == "TCOA_vs_OAC" and method == "axis":
                    continue                       # the axis is defined by YAC/OAC only
                auc, p, null = loo_auc_with_permutation(X, y, method, a.n_perm, a.seed, a.workers)
                nulls[f"{contrast}__{set_name}__{method}"] = null
                rows.append({"contrast": contrast, "feature_set": set_name, "n_features": X.shape[1],
                             "method": method, "loo_auc": auc, "perm_p": p,
                             "null_auc_95th": float(np.percentile(null, 95)),
                             "n_univariate_p_lt_.05": int((pu < 0.05).sum()),
                             "expected_by_chance": round(0.05 * X.shape[1], 1),
                             "n_permutations": a.n_perm})
                print(f"{contrast:12s} {set_name:28s} {method:7s} AUC={auc:.3f}  p={p:.3f}  "
                      f"(univariate p<.05: {(pu < 0.05).sum()}/{X.shape[1]})  [{time.time() - t0:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "separability_results.csv", index=False)
    np.savez_compressed(OUT / "separability_permutation_nulls.npz", **nulls)

    import matplotlib.pyplot as plt
    yo = res[res.contrast.eq("YAC_vs_OAC")]
    order = list(dict.fromkeys(yo.feature_set))
    colors = {"axis": rc.ACCENT, "lda": rc.ACCENT2, "logreg": "#1baf7a"}
    labels = {"axis": "Manuscript axis (LOO position)", "lda": "Shrinkage LDA", "logreg": "L2 logistic"}
    fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
    width = 0.26
    for k, m in enumerate(a.methods):
        d = yo[yo.method.eq(m)].set_index("feature_set").reindex(order)
        x = np.arange(len(order)) + (k - 1) * width
        ax.bar(x, d.loo_auc, width=width - 0.03, color=colors[m], label=labels[m])
        for xi, (auc, p) in zip(x, zip(d.loo_auc, d.perm_p)):
            ax.text(xi, auc + 0.015, f"{auc:.2f}" + ("*" if p < 0.05 else ""), ha="center", fontsize=7.5, color=rc.INK)
    ax.axhline(0.5, color=rc.INK2, linewidth=1, linestyle="--")
    ax.text(len(order) - 0.5, 0.505, "chance", ha="right", va="bottom", fontsize=8, color=rc.INK2)
    ax.set_xticks(np.arange(len(order)), [o.replace("_", " ") for o in order], fontsize=8, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Leave-one-out AUC, YAC vs OAC", fontsize=9, color=rc.INK2)
    ax.set_title("Is there an out-of-sample age signal in the network-level FC features?  (* permutation p < .05)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    rc.style_axes(ax)
    fig.savefig(OUT / "FigR_YAC_vs_OAC_separability.png", dpi=250)
    plt.close(fig)
    print(f"\nDone in {time.time() - t0:.0f} s. Outputs in {OUT}")


if __name__ == "__main__":
    main()
