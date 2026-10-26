"""Does the number of FC matrices per participant confound the group results?

Background: the separability check found that dynamic-FC variability (SD of each
feature across a participant's matrices) separates the groups almost
perfectly. That variability is almost entirely determined by how many
matrices (HMM state visits) a participant contributed, and the count differs
sharply by group. This script documents the count by group, its link to
variability and to off-axis deviation, and whether the TCOA-OAC deviation
difference survives adjustment. Relevant to R1 2.2 (estimation-noise
alternative), R1 2.6 (HMM/averaging), and the recommended points on retained
data and ICC per group.

Usage:  python matrix_count_confound.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import revision_common as rc

OUT = rc.out_dir("matrix_count")


def ols(y, X, names):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (resid @ resid) / df)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return {n: {"b": float(b), "se": float(s), "p": float(pp)} for n, b, s, pp in
            zip(["intercept"] + names, beta, se, p)} | {"r2": float(r2), "df": int(df)}


def main():
    base = rc.METHOD_DIR / "01_dataset"
    names = pd.read_csv(base / "feature_metadata.csv")["feature_name"].tolist()
    dyn = pd.read_csv(base / "dynamic_network_features.csv.gz", dtype={"participant": str})
    counts = pd.read_csv(base / "matrix_counts.csv", dtype={"participant": str})
    scores = pd.read_csv(rc.METHOD_DIR / "02_phenotype" / "participant_phenotype_scores.csv", dtype={"participant": str})
    scores = scores[scores["axis_estimator"].eq(rc.PRIMARY)].set_index("participant")

    t = counts.pivot(index="participant", columns="mode", values="n_matrices").rename(
        columns={"EC": "n_matrices_EC", "EO": "n_matrices_EO"})
    t["n_matrices"] = t.sum(axis=1)
    t["group"] = dyn.groupby("participant")["group"].first()
    t["n_states_EC"] = dyn[dyn["mode"].eq("EC")].groupby("participant")["state"].nunique()
    t["n_states_EO"] = dyn[dyn["mode"].eq("EO")].groupby("participant")["state"].nunique()
    t["mean_feature_SD"] = dyn.groupby(["participant", "mode"])[names].std(ddof=1).groupby("participant").mean().mean(axis=1)
    t["deviation"] = scores["analysis_deviation"]
    t["position"] = scores["analysis_position"]
    t.to_csv(OUT / "participant_matrix_counts_and_scores.csv")

    by_group = t.groupby("group")[["n_matrices_EC", "n_matrices_EO", "n_matrices", "n_states_EC", "n_states_EO",
                                   "mean_feature_SD", "deviation"]].agg(["median", "min", "max"])
    by_group.to_csv(OUT / "by_group_summary.csv")
    res = {
        "kruskal_n_matrices_p": float(stats.kruskal(*[t.n_matrices[t.group == g] for g in ("YAC", "OAC", "TCOA")]).pvalue),
        "mannwhitney_n_matrices_TCOA_vs_OAC_p": float(stats.mannwhitneyu(t.n_matrices[t.group == "TCOA"], t.n_matrices[t.group == "OAC"]).pvalue),
        "spearman_variability_vs_n_matrices_all": float(stats.spearmanr(t.mean_feature_SD, t.n_matrices)[0]),
        "spearman_deviation_vs_n_matrices": {g: [float(v) for v in stats.spearmanr(t.deviation[t.group == g], t.n_matrices[t.group == g])]
                                             for g in ("YAC", "OAC", "TCOA")},
    }
    older = t[t.group.isin(["OAC", "TCOA"])].copy()
    tc = (older.group == "TCOA").to_numpy(float)
    res["deviation_on_group"] = ols(older.deviation.to_numpy(), tc[:, None], ["TCOA"])
    res["deviation_on_group_plus_n_matrices"] = ols(older.deviation.to_numpy(), np.column_stack([tc, older.n_matrices]), ["TCOA", "n_matrices"])
    res["log_deviation_on_group_plus_log_n_matrices"] = ols(np.log(older.deviation.to_numpy()), np.column_stack([tc, np.log(older.n_matrices)]), ["TCOA", "log_n_matrices"])
    lo, hi = older.n_matrices[tc == 1].min(), older.n_matrices[tc == 1].max()
    res["n_OAC_within_TCOA_count_range"] = int(((tc == 0) & older.n_matrices.between(lo, hi)).sum())
    (OUT / "matrix_count_confound.json").write_text(json.dumps(res, indent=2))
    print(by_group.round(4).T.to_string())
    print(json.dumps(res, indent=2))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for g in ("YAC", "OAC", "TCOA"):
        s = t[t.group == g]
        axes[0].scatter(s.n_matrices, s.mean_feature_SD, s=48, color=rc.GROUP_COLORS[g], edgecolor="white", linewidth=1.2, label=g)
        axes[1].scatter(s.n_matrices, s.deviation, s=48, color=rc.GROUP_COLORS[g], edgecolor="white", linewidth=1.2, label=g)
    axes[0].set_ylabel("Dynamic FC variability (mean SD)", fontsize=9, color=rc.INK2)
    axes[1].set_ylabel("Off-axis deviation (manuscript)", fontsize=9, color=rc.INK2)
    axes[0].set_title(f"Variability tracks matrix count (ρ = {res['spearman_variability_vs_n_matrices_all']:.2f})", fontsize=10, loc="left")
    axes[1].set_title("Deviation vs matrix count", fontsize=10, loc="left")
    for ax in axes:
        ax.set_xlabel("FC matrices per participant (EC + EO)", fontsize=9, color=rc.INK2)
        ax.legend(fontsize=8, frameon=False)
        rc.style_axes(ax)
    fig.savefig(OUT / "FigR_matrix_count_confound.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
