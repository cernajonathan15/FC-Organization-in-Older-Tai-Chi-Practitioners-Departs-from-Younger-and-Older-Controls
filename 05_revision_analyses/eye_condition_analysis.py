"""Reviewer 1, 3a, and Reviewer 2, Major 4 - do results differ between
eyes-closed (EC) and eyes-open (EO)?

Runs the primary analysis (Euclidean geometric median, radial winsorization,
full refit in every permutation) separately on the EC-only and EO-only
28-feature vectors saved by the pipeline (participant_mode_features.csv),
and on the pooled vectors used in the manuscript.

For each version, it reports the TCOA - OAC median difference in position
(one-sided upper-tail p) and in off-axis deviation (two-sided p), using the
manuscript's tail rules. It also reports:
  - agreement between the EC and EO deviation scores across participants
    (Spearman);
  - agreement of the feature-level TCOA - OAC differences between EC and EO
    (Pearson across the 28 features).

Usage:  python eye_condition_analysis.py [--n-perm 5000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import revision_common as rc

OUT = rc.out_dir("eye_condition")


def arrays_from(frame, names):
    return {g: frame.loc[frame.group.eq(g), names].to_numpy(float) for g in ("YAC", "OAC", "TCOA")}


def stats_for(A):
    t, o, _ = rc.score_groups(A["YAC"], A["OAC"], A["TCOA"])
    return (float(np.median(t["position"]) - np.median(o["position"])),
            float(np.median(t["deviation"]) - np.median(o["deviation"])), t, o)


def perm(A, n, rng):
    pool = np.vstack([A["OAC"], A["TCOA"]]); nO = len(A["OAC"])
    pos, dev = np.empty(n), np.empty(n)
    for k in range(n):
        idx = rng.permutation(len(pool))
        B = {"YAC": A["YAC"], "OAC": pool[idx[:nO]], "TCOA": pool[idx[nO:]]}
        pos[k], dev[k], _, _ = stats_for(B)
    return pos, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    F, names, pooled = rc.load_features()
    modes = pd.read_csv(rc.METHOD_DIR / "01_dataset" / "participant_mode_features.csv", dtype={"participant": str})
    versions = {"pooled (manuscript)": pooled,
                "EC only": arrays_from(modes[modes["mode"].eq("EC")], names),
                "EO only": arrays_from(modes[modes["mode"].eq("EO")], names)}
    rows, dev_scores, feat_diff = [], {}, {}
    for lab, A in versions.items():
        p_obs, d_obs, t, o = stats_for(A)
        pn, dn = perm(A, a.n_perm, rng)
        rows.append({"version": lab,
                     "OAC_median_position": float(np.median(o["position"])), "TCOA_median_position": float(np.median(t["position"])),
                     "position_TCOA_minus_OAC": p_obs, "position_null_mean": float(pn.mean()),
                     "position_p_one_sided": float((1 + np.sum(pn >= p_obs)) / (a.n_perm + 1)),
                     "OAC_median_deviation": float(np.median(o["deviation"])), "TCOA_median_deviation": float(np.median(t["deviation"])),
                     "deviation_TCOA_minus_OAC": d_obs, "deviation_null_mean": float(dn.mean()),
                     "deviation_p_two_sided": float((1 + np.sum(np.abs(dn) >= abs(d_obs))) / (a.n_perm + 1)),
                     "n_permutations": a.n_perm})
        dev_scores[lab] = np.r_[o["deviation"], t["deviation"]]
        feat_diff[lab] = np.median(t["standardized"], 0) - np.median(o["standardized"], 0)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
    res = pd.DataFrame(rows); res.to_csv(OUT / "eye_condition_results.csv", index=False)
    agree = {
        "spearman_EC_vs_EO_deviation_older_adults": float(stats.spearmanr(dev_scores["EC only"], dev_scores["EO only"]).statistic),
        "pearson_EC_vs_EO_feature_level_TCOA_minus_OAC": float(stats.pearsonr(feat_diff["EC only"], feat_diff["EO only"])[0]),
    }
    (OUT / "eye_condition_agreement.json").write_text(json.dumps(agree, indent=2))
    print(json.dumps(agree, indent=2))


if __name__ == "__main__":
    main()
