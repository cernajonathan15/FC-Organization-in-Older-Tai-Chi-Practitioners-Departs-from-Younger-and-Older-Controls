"""Figure 1, panel B: participant off-axis deviation by group.

Reads the primary-estimator phenotype scores written by the main age-axis
pipeline (robust_median branch) and draws one box per group with every
participant overlaid; radially winsorized participants are circled and
labelled. Panel A is the phenotype-position figure produced by
`03_age_axis_analysis` (stage 4).

Output: <results>/revision_analyses/figure1/Figure1_panelB_deviation.png

Usage:  python figure1_panel_b_deviation.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import revision_common as rc  # noqa: E402

OUT = rc.out_dir("figure1")
COLORS = {"YAC": "#3b82f6", "OAC": "#f97316", "TCOA": "#10b981"}


def main() -> None:
    scores = pd.read_csv(
        rc.METHOD_DIR / "02_phenotype" / "participant_phenotype_scores.csv",
        dtype={"participant": str},
    )
    scores = scores[scores.is_primary_estimator]

    fig, ax = plt.subplots(figsize=(1697 / 200, 1247 / 200), dpi=200)
    rng = np.random.default_rng(7)
    for k, group in enumerate(("YAC", "OAC", "TCOA")):
        d = scores[scores.group.eq(group)]
        values = d.analysis_deviation.to_numpy()
        bp = ax.boxplot(
            values, positions=[k], widths=0.52, patch_artist=True, showfliers=False,
            medianprops=dict(color="#f97316", linewidth=1.2),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            boxprops=dict(edgecolor="#8c8c8c", linewidth=1.2),
        )
        bp["boxes"][0].set_facecolor(COLORS[group])
        bp["boxes"][0].set_alpha(0.32)
        x = k + rng.uniform(-0.09, 0.09, len(values))
        ax.scatter(x, values, s=42, color=COLORS[group], edgecolor="white",
                   linewidth=0.6, zorder=3)
        for xi, vi, pid, winsorized in zip(x, values, d.participant, d.is_winsorized):
            if winsorized:
                ax.scatter(xi, vi, s=110, facecolor="none", edgecolor="black",
                           linewidth=1.4, zorder=4)
                ax.annotate(pid, (xi, vi), xytext=(5, 5),
                            textcoords="offset points", fontsize=8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["YAC", "OAC", "TCOA"], fontsize=12)
    ax.set_xlim(-0.5, 2.5)
    ax.tick_params(labelsize=12)
    ax.set_ylabel("Off-axis deviation (RMS standardized units)", fontsize=12)
    ax.set_title("Participant off-axis deviation with robust median inference",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "Figure1_panelB_deviation.png", dpi=200)
    plt.close(fig)
    print(scores.groupby("group").analysis_deviation.median())


if __name__ == "__main__":
    main()
