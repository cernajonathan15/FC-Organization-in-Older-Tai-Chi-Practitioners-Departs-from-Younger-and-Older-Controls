"""Reviewer 1, comment 2.3 - age (and sex) of the two older groups.

  A  Formal TCOA-vs-OAC age comparison (Welch t, Mann-Whitney, permutation)
     and sex distribution per group (Fisher exact, chi-square across 3 groups).
  B  Within-older-adult correlations of age with phenotype position and with
     off-axis deviation (pooled n = 30 and within each group; Pearson + Spearman).
  C  Age-adjusted sensitivity analysis for position and deviation:
     OLS  score ~ group + age  with a Freedman-Lane permutation p for group
     (10,000 permutations), plus the same with age and sex.
     NOTE: these OLS/Freedman-Lane tests treat the scores as fixed, so they do
     NOT account for OACs being in-sample (axis-defining) while TCOAs are
     out-of-sample. For position this bias is large (see 2.1), so C is
     reported for transparency only.
  D  Calibrated age-adjusted test (the one to report): the age-adjusted group
     coefficient (score ~ group + age [+ sex]) is the statistic, and OAC/TCOA
     labels are permuted together with each participant's age and sex, with
     the full pipeline (winsorization, axis, scoring) refitted in every
     permutation, exactly as for the manuscript's deviation test.

Scores are the manuscript's primary Euclidean geometric-median scores,
recomputed with the pipeline's own functions (revision_common).
Demographics come from data/participant_demographics_behavior.csv, a copy
of Sheet3 of PCD_TCOA_subj_info.xlsx (the analysed cohort) and the TCOA
practice-hours file.

Usage:  python age_and_sex_adjustment.py [--n-perm 10000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import revision_common as rc

OUT = rc.out_dir("age_sex")


def scores_table():
    F, _, A = rc.load_features()
    t, o, _ = rc.score_groups(A["YAC"], A["OAC"], A["TCOA"])
    y, _, _ = rc.score_groups(A["YAC"], A["OAC"], A["YAC"])
    pos = np.r_[y["position"], o["position"], t["position"]]
    dev = np.r_[y["deviation"], o["deviation"], t["deviation"]]
    order = np.r_[F.index[F.group.eq("YAC")], F.index[F.group.eq("OAC")], F.index[F.group.eq("TCOA")]]
    s = pd.DataFrame({"participant": F.loc[order, "participant"].astype(int).values,
                      "group": F.loc[order, "group"].values, "position": pos, "deviation": dev})
    demo = pd.read_csv(rc.DATA_DIR / "participant_demographics_behavior.csv")
    return s.merge(demo.drop(columns="group"), on="participant", how="left")


def perm_diff(a, b, n, rng, reducer=np.mean):
    obs = reducer(a) - reducer(b)
    pool = np.r_[a, b]
    null = np.empty(n)
    for k in range(n):
        idx = rng.permutation(len(pool))
        null[k] = reducer(pool[idx[:len(a)]]) - reducer(pool[idx[len(a):]])
    return obs, float((1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1))


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    return beta, se, resid


def freedman_lane(y, group, covars, n, rng):
    """Permutation p for the group coefficient, adjusting for covariates."""
    Xr = np.column_stack([np.ones_like(y)] + covars)
    X = np.column_stack([Xr, group])
    beta, se, _ = ols(y, X)
    t_obs = beta[-1] / se[-1]
    br, _, resid_r = ols(y, Xr)
    fitted_r = y - resid_r
    null = np.empty(n)
    for k in range(n):
        yp = fitted_r + rng.permutation(resid_r)
        b, s, _ = ols(yp, X)
        null[k] = b[-1] / s[-1]
    p = float((1 + np.sum(np.abs(null) >= abs(t_obs))) / (n + 1))
    return {"b_group_TCOA_minus_OAC": float(beta[-1]), "se": float(se[-1]), "t": float(t_obs),
            "p_parametric": float(2 * stats.t.sf(abs(t_obs), len(y) - X.shape[1])),
            "p_freedman_lane": p, "b_age_per_year": float(beta[1]) if covars else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d = scores_table()
    d.to_csv(OUT / "participant_scores_with_demographics.csv", index=False)
    older = d[d.group.isin(["OAC", "TCOA"])].copy()
    tc, oc = older[older.group.eq("TCOA")], older[older.group.eq("OAC")]
    res = {}

    # A. age and sex
    w = stats.ttest_ind(tc.age, oc.age, equal_var=False)
    mw = stats.mannwhitneyu(tc.age, oc.age, alternative="two-sided")
    obs, pp = perm_diff(tc.age.to_numpy(), oc.age.to_numpy(), a.n_perm, rng)
    sd_pooled = np.sqrt((tc.age.var(ddof=1) + oc.age.var(ddof=1)) / 2)
    res["age"] = {g: {"mean": float(x.age.mean()), "sd": float(x.age.std(ddof=1)),
                      "median": float(x.age.median()), "min": float(x.age.min()), "max": float(x.age.max())}
                  for g, x in d.groupby("group")}
    res["age_TCOA_vs_OAC"] = {"mean_difference": float(obs), "welch_t": float(w.statistic),
                              "welch_df": float(w.df), "welch_p": float(w.pvalue),
                              "mannwhitney_U": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
                              "permutation_p_two_sided": pp, "cohens_d": float(obs / sd_pooled)}
    ct = pd.crosstab(d.group, d.sex)
    res["sex_counts"] = ct.to_dict("index")
    res["sex_TCOA_vs_OAC_fisher_p"] = float(stats.fisher_exact(ct.loc[["TCOA", "OAC"]].to_numpy())[1])
    res["sex_three_groups_chi2_p"] = float(stats.chi2_contingency(ct.to_numpy())[1])

    # B. age correlations
    corr = []
    for subset, frame in (("older pooled (n=30)", older), ("OAC", oc), ("TCOA", tc)):
        for metric in ("position", "deviation"):
            r, p = stats.pearsonr(frame.age, frame[metric])
            rho, ps = stats.spearmanr(frame.age, frame[metric])
            corr.append({"subset": subset, "metric": metric, "n": len(frame),
                         "pearson_r": r, "pearson_p_two_sided": p, "spearman_rho": rho, "spearman_p_two_sided": ps})
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / "age_correlations.csv", index=False)

    # C. adjusted group effects
    adj = []
    g = older.group.eq("TCOA").astype(float).to_numpy()
    age_c = (older.age - older.age.mean()).to_numpy()
    female = older.sex.eq("F").astype(float).to_numpy()
    for metric in ("position", "deviation"):
        y = older[metric].to_numpy()
        for label, cov in (("unadjusted", []), ("age", [age_c]), ("age + sex", [age_c, female])):
            row = freedman_lane(y, g, cov, a.n_perm, rng)
            row.update({"metric": metric, "covariates": label})
            adj.append(row)
        # age-residualized median difference (pooled older adults)
        resid = y - np.polyval(np.polyfit(age_c, y, 1), age_c)
        o2, p2 = perm_diff(resid[g == 1], resid[g == 0], a.n_perm, rng, reducer=np.median)
        adj.append({"metric": metric, "covariates": "age-residualized median difference",
                    "b_group_TCOA_minus_OAC": float(o2), "p_freedman_lane": p2})
    adj = pd.DataFrame(adj)
    adj.to_csv(OUT / "age_adjusted_group_effects.csv", index=False)

    # D. calibrated: refit axis in every permutation, covariates travel with participants
    F, _, A = rc.load_features()
    feat = {"OAC": A["OAC"], "TCOA": A["TCOA"]}
    pid = {g: F.loc[F.group.eq(g), "participant"].astype(int).to_numpy() for g in ("OAC", "TCOA")}
    demo = d.set_index("participant")
    pool_X = np.vstack([feat["OAC"], feat["TCOA"]])
    pool_age = demo.loc[np.r_[pid["OAC"], pid["TCOA"]], "age"].to_numpy(float)
    pool_fem = demo.loc[np.r_[pid["OAC"], pid["TCOA"]], "sex"].eq("F").to_numpy(float)

    def coef(order, covset):
        o_idx, t_idx = order[:15], order[15:]
        ts, os_, _ = rc.score_groups(A["YAC"], pool_X[o_idx], pool_X[t_idx])
        out = {}
        grp = np.r_[np.zeros(15), np.ones(15)]
        idx = np.r_[o_idx, t_idx]
        agec = pool_age[idx] - pool_age.mean()
        cov = [agec] if covset == "age" else [agec, pool_fem[idx]]
        X = np.column_stack([np.ones(30)] + cov + [grp])
        for metric in ("position", "deviation"):
            y = np.r_[os_[metric], ts[metric]]
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            out[metric] = b[-1]
        return out

    calib = []
    ident = np.arange(30)
    for covset in ("age", "age + sex"):
        obs_c = coef(ident, covset)
        null = {m: np.empty(a.n_perm) for m in obs_c}
        for k in range(a.n_perm):
            c = coef(rng.permutation(30), covset)
            for m in c:
                null[m][k] = c[m]
        for m in obs_c:
            calib.append({"metric": m, "covariates": covset, "adjusted_b_TCOA_minus_OAC": float(obs_c[m]),
                          "null_mean": float(null[m].mean()),
                          "p_two_sided_manuscript_rule": float((1 + np.sum(np.abs(null[m]) >= abs(obs_c[m]))) / (a.n_perm + 1)),
                          "p_one_sided_upper": float((1 + np.sum(null[m] >= obs_c[m])) / (a.n_perm + 1)),
                          "n_permutations": a.n_perm})
    calib = pd.DataFrame(calib)
    calib.to_csv(OUT / "age_adjusted_calibrated_permutation.csv", index=False)
    print(calib.round(4).to_string(index=False))

    (OUT / "age_sex_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(corr.round(3).to_string(index=False))
    print(adj.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
