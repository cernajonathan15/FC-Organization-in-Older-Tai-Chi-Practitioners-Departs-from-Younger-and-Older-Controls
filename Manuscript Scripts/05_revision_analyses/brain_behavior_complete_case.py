"""Reviewer 1, comments 2.4 and 2.5 (and Reviewer 2, exploratory correlations,
practice hours) - brain-behavior correlations, group behavior, dose-response.

2.4
  A  Re-report the manuscript's TCOA correlations (off-axis deviation vs RBANS,
     Mini-BESTest, reverse-scored GDS; Euclidean and Mahalanobis) with
     two-sided p, 95% CI (Fisher z), Spearman rho, and Holm / Benjamini-Hochberg
     adjustment across the 3 primary (Euclidean) tests.
  B  Complete-case sensitivity: Mini-BESTest without the imputed value (413),
     RBANS without values that cannot be traced to the source sheet.
  C  Specificity check: the same correlations within OAC.
  D  Leave-one-out (jackknife) range of r within TCOA (influence at n = 15).
  E  GDS distribution per group (floor effects).
  F  Scatter plots with least-squares lines (requested by Reviewer 2 too).
2.5
  G  Demographics-and-behavior comparison of YAC, OAC, TCOA (all available and
     complete-case), with TCOA-vs-OAC tests unadjusted and age+sex-adjusted.
  H  Cumulative practice hours vs position and deviation (TCOA, n = 15).

Inputs: data/participant_demographics_behavior.csv (flags imputed
Mini-BESTest values and RBANS values without a source record). Brain scores:
primary Euclidean geometric-median scores recomputed with revision_common, checked
against the pipeline's behavior_analysis_data.csv; Mahalanobis deviations from
the Mahalanobis pipeline's saved scores.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import revision_common as rc

OUT = rc.out_dir("brain_behavior")
OUTCOMES = [("final_score_RBANS", "RBANS", 1), ("final_score_miniBEST", "Mini-BESTest", 1),
            ("final_score_GDS", "GDS (reverse-scored)", -1)]


def scores():
    F, _, A = rc.load_features()
    t, o, _ = rc.score_groups(A["YAC"], A["OAC"], A["TCOA"])
    y, _, _ = rc.score_groups(A["YAC"], A["OAC"], A["YAC"])
    rows = []
    for g, s in (("YAC", y), ("OAC", o), ("TCOA", t)):
        ids = F.loc[F.group.eq(g), "participant"].astype(int).to_numpy()
        rows.append(pd.DataFrame({"participant": ids, "position": s["position"], "deviation": s["deviation"]}))
    sc = pd.concat(rows)
    maha = pd.read_csv(rc.MAHALANOBIS_DIR / "02_phenotype" / "participant_phenotype_scores.csv")
    maha = maha[maha.axis_estimator.eq("geometric_median")][["participant", "analysis_deviation"]]
    sc = sc.merge(maha.rename(columns={"analysis_deviation": "deviation_mahalanobis"}), on="participant", how="left")
    demo = pd.read_csv(rc.DATA_DIR / "participant_demographics_behavior.csv")
    d = demo.merge(sc, on="participant")
    pipe = pd.read_csv(rc.METHOD_DIR / "03_inference" / "behavior_analysis_data.csv")
    chk = d.merge(pipe[["participant", "analysis_deviation"]], on="participant")
    assert np.allclose(chk.deviation, chk.analysis_deviation), "deviation does not reproduce the pipeline"
    return d


def corr_row(x, y, label, sample, predictor):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    r, p2 = stats.pearsonr(x, y)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    p1 = stats.t.sf(t, n - 2)
    rho, prho = stats.spearmanr(x, y)
    return {"sample": sample, "predictor": predictor, "outcome": label, "n": n, "pearson_r": r,
            "ci95_low": lo, "ci95_high": hi, "t": t, "df": n - 2, "p_one_sided_positive": p1,
            "p_two_sided": p2, "spearman_rho": rho, "spearman_p_two_sided": prho}


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj


def bh(p):
    p = np.asarray(p); o = np.argsort(p)[::-1]; m = len(p); adj = np.empty(m); run = 1
    for k, i in enumerate(o):
        run = min(run, p[i] * m / (m - k)); adj[i] = run
    return adj


def main():
    d = scores()
    d.to_csv(OUT / "analysis_table.csv", index=False)
    tc, oc = d[d.group.eq("TCOA")], d[d.group.eq("OAC")]
    rows = []

    # A. primary + Mahalanobis
    for col, lab, sign in OUTCOMES:
        rows.append(corr_row(tc.deviation, sign * tc[col], lab, "TCOA (as published)", "deviation (Euclidean)"))
    for col, lab, sign in OUTCOMES:
        rows.append(corr_row(tc.deviation_mahalanobis, sign * tc[col], lab, "TCOA (as published)", "deviation (Mahalanobis)"))
    prim = [i for i, r in enumerate(rows) if r["predictor"] == "deviation (Euclidean)"]
    ph = holm([rows[i]["p_two_sided"] for i in prim]); pb = bh([rows[i]["p_two_sided"] for i in prim])
    for k, i in enumerate(prim):
        rows[i]["p_two_sided_holm_3tests"] = ph[k]; rows[i]["p_two_sided_BH_3tests"] = pb[k]

    # B. complete case
    cc_mb = tc[~tc.miniBEST_imputed]
    rows.append(corr_row(cc_mb.deviation, cc_mb.final_score_miniBEST, "Mini-BESTest", "TCOA complete-case (413 excluded)", "deviation (Euclidean)"))
    cc_rb = tc[tc.RBANS_source_check.eq("matches_source")]
    rows.append(corr_row(cc_rb.deviation, cc_rb.final_score_RBANS, "RBANS", "TCOA verified RBANS only", "deviation (Euclidean)"))

    # C. OAC specificity (all available and complete-case)
    for col, lab, sign in OUTCOMES:
        rows.append(corr_row(oc.deviation, sign * oc[col], lab, "OAC (all available)", "deviation (Euclidean)"))
    o_mb = oc[~oc.miniBEST_imputed]
    rows.append(corr_row(o_mb.deviation, o_mb.final_score_miniBEST, "Mini-BESTest", "OAC complete-case", "deviation (Euclidean)"))
    o_rb = oc[oc.RBANS_source_check.eq("matches_source")]
    rows.append(corr_row(o_rb.deviation, o_rb.final_score_RBANS, "RBANS", "OAC verified RBANS only", "deviation (Euclidean)"))

    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "A_C_correlations.csv", index=False)

    # D. jackknife range
    jk = []
    for col, lab, sign in OUTCOMES:
        rs = [stats.pearsonr(np.delete(tc.deviation.to_numpy(), i), np.delete(sign * tc[col].to_numpy(), i))[0] for i in range(len(tc))]
        k = int(np.argmin(rs))
        jk.append({"outcome": lab, "r_full": stats.pearsonr(tc.deviation, sign * tc[col])[0], "r_loo_min": min(rs),
                   "r_loo_max": max(rs), "most_influential_participant": int(tc.participant.iloc[k])})
    jk = pd.DataFrame(jk); jk.to_csv(OUT / "D_jackknife.csv", index=False)

    # E. GDS distribution
    gds = d.groupby("group").final_score_GDS.agg(n="size", mean="mean", sd="std", median="median", min="min", max="max",
                                                 n_zero=lambda s: int((s == 0).sum()), n_le1=lambda s: int((s <= 1).sum()),
                                                 skew=lambda s: float(stats.skew(s)))
    gds_counts = pd.crosstab(d.group, d.final_score_GDS)
    gds.to_csv(OUT / "E_GDS_distribution.csv"); gds_counts.to_csv(OUT / "E_GDS_counts.csv")

    # G. group comparisons
    gc = []
    def describe(s):
        return f"{s.mean():.1f} ± {s.std(ddof=1):.1f}; {s.median():.1f} [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}] (n={len(s)})"
    variants = [("final_score_RBANS", "RBANS (all values)", lambda f: f),
                ("final_score_RBANS", "RBANS (source-verified only)", lambda f: f[f.RBANS_source_check.eq("matches_source")]),
                ("final_score_miniBEST", "Mini-BESTest (with imputed)", lambda f: f),
                ("final_score_miniBEST", "Mini-BESTest (complete-case)", lambda f: f[~f.miniBEST_imputed]),
                ("final_score_GDS", "GDS (raw; lower = better)", lambda f: f),
                ("age", "Age (years)", lambda f: f)]
    for col, lab, filt in variants:
        f = filt(d)
        g = {k: f.loc[f.group.eq(k), col].astype(float) for k in ("YAC", "OAC", "TCOA")}
        kw = stats.kruskal(*g.values())
        row = {"measure": lab, **{f"{k}": describe(v) for k, v in g.items()}, "kruskal_p_3groups": kw.pvalue}
        for a_, b_ in (("TCOA", "OAC"), ("TCOA", "YAC"), ("OAC", "YAC")):
            mw = stats.mannwhitneyu(g[a_], g[b_], alternative="two-sided")
            wt = stats.ttest_ind(g[a_], g[b_], equal_var=False)
            diff = g[a_].values[:, None] - g[b_].values[None, :]
            row[f"{a_}_vs_{b_}_mean_diff"] = g[a_].mean() - g[b_].mean()
            row[f"{a_}_vs_{b_}_welch_p"] = wt.pvalue
            row[f"{a_}_vs_{b_}_mannwhitney_p"] = mw.pvalue
            row[f"{a_}_vs_{b_}_cliffs_delta"] = float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)
        if col != "age":
            older = f[f.group.isin(["OAC", "TCOA"])]
            X = np.column_stack([np.ones(len(older)), older.group.eq("TCOA"), older.age - older.age.mean(), older.sex.eq("F")]).astype(float)
            yv = older[col].to_numpy(float)
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ beta
            XtXi = np.linalg.inv(X.T @ X)
            h = np.einsum("ij,jk,ik->i", X, XtXi, X)
            meat = (X * (resid / (1 - h))[:, None] ** 2).T @ X
            se = np.sqrt(np.diag(XtXi @ meat @ XtXi))       # HC3
            tval = beta[1] / se[1]
            row["TCOA_vs_OAC_age_sex_adjusted_diff"] = beta[1]
            row["TCOA_vs_OAC_age_sex_adjusted_p_HC3"] = float(2 * stats.t.sf(abs(tval), len(yv) - X.shape[1]))
        gc.append(row)
    gc = pd.DataFrame(gc); gc.to_csv(OUT / "G_group_behavior_comparison.csv", index=False)
    sexc = pd.crosstab(d.group, d.sex)

    # H. practice hours
    ph_rows = []
    hrs = tc.tc_practice_hours.astype(float)
    for m in ("position", "deviation"):
        r, p = stats.pearsonr(hrs, tc[m]); rl, pl = stats.pearsonr(np.log(hrs), tc[m]); rho, pr = stats.spearmanr(hrs, tc[m])
        ph_rows.append({"metric": m, "n": len(tc), "spearman_rho": rho, "spearman_p": pr, "pearson_r": r, "pearson_p": p,
                        "pearson_r_log_hours": rl, "pearson_p_log_hours": pl})
    ph_df = pd.DataFrame(ph_rows); ph_df.to_csv(OUT / "H_practice_hours.csv", index=False)
    hours_desc = {"mean": hrs.mean(), "sd": hrs.std(ddof=1), "median": hrs.median(), "min": hrs.min(), "max": hrs.max(),
                  "cv": hrs.std(ddof=1) / hrs.mean()}

    # F. scatter plots
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
    for r_, (grp, frame) in enumerate((("TCOA", tc), ("OAC", oc))):
        for c_, (col, lab, sign) in enumerate(OUTCOMES):
            ax = axes[r_, c_]
            yv = frame[col].to_numpy(float); xv = frame.deviation.to_numpy(float)
            flag = frame.miniBEST_imputed.to_numpy() if col == "final_score_miniBEST" else (
                ~frame.RBANS_source_check.eq("matches_source").to_numpy() if col == "final_score_RBANS" else np.zeros(len(frame), bool))
            ax.scatter(xv[~flag], yv[~flag], s=46, color=rc.GROUP_COLORS[grp], edgecolor="white", linewidth=1.2, zorder=3)
            if flag.any():
                ax.scatter(xv[flag], yv[flag], s=46, facecolor="white", edgecolor=rc.GROUP_COLORS[grp], linewidth=1.6, zorder=3,
                           label="imputed / unverified")
                ax.legend(fontsize=7.5, frameon=False, loc="best")
            b1, b0 = np.polyfit(xv, yv, 1); xs = np.linspace(xv.min(), xv.max(), 50)
            ax.plot(xs, b0 + b1 * xs, color=rc.INK, linewidth=1.6)
            rr, pp = stats.pearsonr(xv, sign * yv)
            ax.set_title(f"{grp} · {lab.replace(' (reverse-scored)', '')}", fontsize=9.5, loc="left")
            ax.text(0.02, 0.04 if sign > 0 else 0.90, f"r = {rr:+.2f} ({'reverse-scored' if sign < 0 else 'raw'})\ntwo-sided p = {pp:.3f}",
                    transform=ax.transAxes, fontsize=8, color=rc.INK2)
            if col == "final_score_GDS":
                ax.set_ylabel("GDS (raw; higher = more symptoms)", fontsize=8.5, color=rc.INK2)
            if r_ == 1:
                ax.set_xlabel("Off-axis deviation (RMS standardized units)", fontsize=8.5, color=rc.INK2)
            rc.style_axes(ax)
    fig.savefig(OUT / "FigS_brain_behavior_scatter.png", dpi=250)
    plt.close(fig)

    summary = {"n_primary_correlation_tests": 3, "n_reported_in_table2": 6, "practice_hours": hours_desc,
               "sex_counts": sexc.to_dict("index")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(corr[["sample", "predictor", "outcome", "n", "pearson_r", "ci95_low", "ci95_high", "p_one_sided_positive",
                "p_two_sided", "spearman_rho", "spearman_p_two_sided"]].round(3).to_string(index=False))
    print(corr.loc[prim, ["outcome", "p_two_sided", "p_two_sided_holm_3tests", "p_two_sided_BH_3tests"]].round(3).to_string(index=False))
    print(jk.round(3).to_string(index=False)); print(gds.round(2).to_string()); print(gds_counts.to_string())
    print(gc.T.to_string()); print(ph_df.round(3).to_string(index=False)); print(json.dumps(hours_desc, default=float))


if __name__ == "__main__":
    main()
