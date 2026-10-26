"""Supplementary Figure S6 (brain-behavior scatter plots) and Supplementary
Table S2 (group behavioral comparisons) for comments R1 2.4/2.5 and R2 Major 5/7.

Run after brain_behavior_complete_case.py and brain_behavior_multiple_imputation.py.
Plots show observed (corrected raw) scores only; imputed values are not drawn.
r and p in each panel title are the multiple-imputation estimates reported in
the manuscript.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import revision_common as rc

OUT = rc.out_dir("brain_behavior")


def main():
    import matplotlib.pyplot as plt

    d = pd.read_csv(OUT / "analysis_table.csv")
    demo = pd.read_csv(rc.DATA_DIR / "participant_demographics_behavior.csv")
    d = d.drop(columns=[c for c in ("RBANS_raw_corrected", "miniBEST_observed") if c in d]).merge(
        demo[["participant", "RBANS_raw_corrected", "miniBEST_observed"]], on="participant")
    mi = pd.read_csv(OUT / "MI_correlations.csv")
    mi = mi[mi.method.str.startswith("multiple") & mi.predictor.eq("deviation")]
    panels = [("RBANS_raw_corrected", "RBANS", "RBANS total (sum of index scores)"),
              ("miniBEST_observed", "Mini-BESTest", "Mini-BESTest total"),
              ("final_score_GDS", "GDS (reverse-scored)", "GDS (raw; higher = more symptoms)")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), constrained_layout=True)
    for r_, grp in enumerate(("TCOA", "OAC")):
        f = d[d.group.eq(grp)]
        for c_, (col, lab, ylab) in enumerate(panels):
            ax = axes[r_, c_]
            g = f.dropna(subset=[col])
            x, y = g.deviation.to_numpy(float), g[col].to_numpy(float)
            ax.scatter(x, y, s=48, color=rc.GROUP_COLORS[grp], edgecolor="white", linewidth=1.2, zorder=3)
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b0 + b1 * xs, color=rc.INK, linewidth=1.6)
            row = mi[(mi["sample"] == grp) & (mi.outcome == lab)].iloc[0]
            nobs = len(g)
            short = lab.replace(" (reverse-scored)", "")
            sign_note = " (reverse-scored)" if "GDS" in lab else ""
            ax.set_title(f"{grp} · {short}\nr{sign_note} = {row.estimate:.2f}, p = {row.p_two_sided:.3f}"
                         f"{'' if nobs == len(f) else f'  (n observed = {nobs}/{len(f)})'}",
                         fontsize=9, loc="left", color=rc.INK)
            ax.set_ylabel(ylab, fontsize=8.5, color=rc.INK2)
            if r_ == 1:
                ax.set_xlabel("Off-axis deviation (RMS standardized units)", fontsize=8.5, color=rc.INK2)
            rc.style_axes(ax)
    fig.savefig(OUT / "FigS6_brain_behavior_scatter.png", dpi=300)
    plt.close(fig)

    # Table S2
    gc = pd.read_csv(OUT / "MI_group_comparisons.csv")
    gc = gc[gc.method.str.startswith("multiple")]
    obs_cols = {"RBANS": "RBANS_raw_corrected", "Mini-BESTest": "miniBEST_observed",
                "GDS (raw; lower = better)": "final_score_GDS"}
    rows = []
    for meas, col in obs_cols.items():
        row = {"Measure": meas}
        for g in ("YAC", "OAC", "TCOA"):
            s = d.loc[d.group.eq(g), col].dropna()
            row[f"{g} mean ± SD (n observed)"] = f"{s.mean():.1f} ± {s.std(ddof=1):.1f} ({len(s)})"
        for contrast in ("TCOA - OAC", "TCOA - YAC", "OAC - YAC", "TCOA - OAC, adjusted for age + sex"):
            q = gc[(gc.outcome == meas) & (gc.contrast == contrast)].iloc[0]
            row[f"{contrast}: difference [95% CI]"] = f"{q.estimate:.1f} [{q.ci95_low:.1f}, {q.ci95_high:.1f}]"
            row[f"{contrast}: p"] = f"{q.p_two_sided:.3f}"
        rows.append(row)
    for col_, lab in (("age", "Age (years)"),):
        row = {"Measure": lab}
        for g in ("YAC", "OAC", "TCOA"):
            s = d.loc[d.group.eq(g), col_]
            row[f"{g} mean ± SD (n observed)"] = f"{s.mean():.1f} ± {s.std(ddof=1):.1f} ({len(s)})"
        rows.append(row)
    sex = d.groupby("group").sex.apply(lambda s: f"{(s == 'F').sum()}/{(s == 'M').sum()}")
    rows.append({"Measure": "Sex (F/M)", **{f"{g} mean ± SD (n observed)": sex[g] for g in ("YAC", "OAC", "TCOA")}})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "TableS2_behavior_group_comparisons.csv", index=False)
    print(tab.T.to_string())


if __name__ == "__main__":
    main()
