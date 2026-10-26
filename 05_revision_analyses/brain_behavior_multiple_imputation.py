"""Reviewer 1, comments 2.4 / 2.5 - redo the behavioral analyses with corrected
raw scores and proper multiple imputation (as the Methods state), and compare
with the published group-mean-imputed values and with complete-case data.

Data corrections (see data/participant_demographics_behavior.csv):
  RBANS_raw_corrected  : from the RBANS grading workbook. Values that were
                         filled in (e.g. seven OAC at 520) are replaced by the
                         graders' scores where they exist, and set to missing
                         where no raw score exists (YAC 103, 111, 112, 116;
                         OAC 202, 206, 207, 214).
  miniBEST_observed    : group-mean-imputed values set back to missing
                         (111, 202, 221, 413).

Multiple imputation: chained equations (scikit-learn IterativeImputer,
BayesianRidge, posterior sampling), m = 100 imputations, all 45 participants.
The imputation model includes RBANS, Mini-BESTest, GDS, age, sex, group (YAC/
OAC/TCOA dummies), and the analysis variables (Euclidean and Mahalanobis
off-axis deviation), so that the imputations respect the associations being
tested. Mini-BESTest imputations are bounded to 0-28.

Pooled with Rubin's rules: r and its CI on the Fisher-z scale; the p value
from the pooled regression slope (Barnard-Rubin df), which equals the Pearson
t-test when no values are missing; group differences (Welch) and age+sex-adjusted differences (OLS, HC3) on
the raw scale.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

import revision_common as rc

warnings.filterwarnings("ignore")
OUT = rc.out_dir("brain_behavior")
M = 100
SEED = 20261021


def load():
    base = pd.read_csv(OUT / "analysis_table.csv")          # written by brain_behavior_complete_case.py
    demo = pd.read_csv(rc.DATA_DIR / "participant_demographics_behavior.csv")
    d = base[["participant", "group", "age", "sex", "deviation", "deviation_mahalanobis", "position",
              "final_score_RBANS", "final_score_miniBEST", "final_score_GDS"]].merge(
        demo[["participant", "RBANS_raw_corrected", "miniBEST_observed"]], on="participant")
    return d


def impute(d, k):
    X = pd.DataFrame({
        "rbans": d.RBANS_raw_corrected, "minibest": d.miniBEST_observed, "gds": d.final_score_GDS,
        "age": d.age, "female": d.sex.eq("F").astype(float),
        "oac": d.group.eq("OAC").astype(float), "tcoa": d.group.eq("TCOA").astype(float),
        "dev": d.deviation, "dev_m": d.deviation_mahalanobis,
    })
    imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=30,
                           random_state=SEED + k, min_value=[0, 0] + [-np.inf] * 7, max_value=[np.inf, 28] + [np.inf] * 7)
    Z = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    out = d.copy()
    out["rbans_i"], out["minibest_i"] = Z.rbans.values, Z.minibest.values
    return out


def fisher_pool(rs, n):
    z = np.arctanh(np.asarray(rs)); m = len(z)
    qbar, U, B = z.mean(), 1 / (n - 3), z.var(ddof=1)
    T = U + (1 + 1 / m) * B
    return pool_scalar(qbar, T, U, B, m, 1e6, back=np.tanh)   # Fisher z is normal-theory


def pool_scalar(qbar, T, U, B, m, dfcom, back=lambda v: v):
    lam = (1 + 1 / m) * B / T if T > 0 else 0.0
    if lam <= 1e-12:
        df = dfcom
    else:
        df_old = (m - 1) / lam ** 2
        df_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lam)
        df = 1 / (1 / df_old + 1 / df_obs)
    t = qbar / np.sqrt(T)
    half = stats.t.ppf(0.975, df) * np.sqrt(T)
    return {"estimate": float(back(qbar)), "ci95_low": float(back(qbar - half)), "ci95_high": float(back(qbar + half)),
            "p_two_sided": float(2 * stats.t.sf(abs(t), df)), "p_one_sided_positive": float(stats.t.sf(t, df)),
            "df": float(df), "fraction_missing_info": float(lam)}


def pool_list(est, var, dfcom):
    est, var = np.asarray(est), np.asarray(var); m = len(est)
    qbar, U, B = est.mean(), var.mean(), est.var(ddof=1)
    return pool_scalar(qbar, U + (1 + 1 / m) * B, U, B, m, dfcom)


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return a.mean() - b.mean(), va + vb


def adjusted(f, col):
    older = f[f.group.isin(["OAC", "TCOA"])]
    X = np.column_stack([np.ones(len(older)), older.group.eq("TCOA"), older.age - older.age.mean(),
                         older.sex.eq("F")]).astype(float)
    y = older[col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X); h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    V = XtXi @ ((X * (e / (1 - h))[:, None] ** 2).T @ X) @ XtXi
    return beta[1], V[1, 1], len(y) - X.shape[1]


def main():
    d = load()
    outcomes = [("rbans", "RBANS", 1, "final_score_RBANS", "RBANS_raw_corrected"),
                ("minibest", "Mini-BESTest", 1, "final_score_miniBEST", "miniBEST_observed"),
                ("gds", "GDS (reverse-scored)", -1, "final_score_GDS", "final_score_GDS")]
    imps = [impute(d, k) for k in range(M)]
    for f in imps:
        f["gds_i"] = f.final_score_GDS

    corr_rows, grp_rows = [], []
    for sample in ("TCOA", "OAC"):
        for pred in ("deviation", "deviation_mahalanobis"):
            if sample == "OAC" and pred == "deviation_mahalanobis":
                continue
            for key, lab, sign, pub_col, obs_col in outcomes:
                base = {"sample": sample, "predictor": pred, "outcome": lab}
                s = d[d.group.eq(sample)]
                r, p = stats.pearsonr(s[pred], sign * s[pub_col])
                corr_rows.append({**base, "method": "published (group-mean imputed)", "n": len(s), "estimate": r,
                                  "p_two_sided": p, "p_one_sided_positive": stats.t.sf(r * np.sqrt((len(s) - 2) / (1 - r * r)), len(s) - 2)})
                cc = s.dropna(subset=[obs_col])
                r, p = stats.pearsonr(cc[pred], sign * cc[obs_col])
                lo, hi = np.tanh(np.arctanh(r) + np.array([-1.96, 1.96]) / np.sqrt(len(cc) - 3))
                corr_rows.append({**base, "method": "complete-case (corrected raw)", "n": len(cc), "estimate": r,
                                  "ci95_low": lo, "ci95_high": hi, "p_two_sided": p,
                                  "p_one_sided_positive": stats.t.sf(r * np.sqrt((len(cc) - 2) / (1 - r * r)), len(cc) - 2)})
                rs, bs, vs = [], [], []
                for f in imps:
                    xx = f.loc[f.group.eq(sample), pred].to_numpy(float); yy = sign * f.loc[f.group.eq(sample), key + "_i"].to_numpy(float)
                    rs.append(stats.pearsonr(xx, yy)[0])
                    lr = stats.linregress(xx, yy); bs.append(lr.slope); vs.append(lr.stderr ** 2)
                fz = fisher_pool(rs, len(s))                     # pooled r and CI (Fisher z)
                sl = pool_list(bs, vs, len(s) - 2)               # pooled slope test = Pearson t-test when nothing is missing
                corr_rows.append({**base, "method": f"multiple imputation (m={M})", "n": len(s), "estimate": fz["estimate"],
                                  "ci95_low": fz["ci95_low"], "ci95_high": fz["ci95_high"], "p_two_sided": sl["p_two_sided"],
                                  "p_one_sided_positive": sl["p_one_sided_positive"], "df": sl["df"],
                                  "fraction_missing_info": sl["fraction_missing_info"]})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "MI_correlations.csv", index=False)

    for key, lab, sign, pub_col, obs_col in outcomes:
        for a_, b_ in (("TCOA", "OAC"), ("TCOA", "YAC"), ("OAC", "YAC")):
            base = {"outcome": lab.replace(" (reverse-scored)", " (raw; lower = better)"), "contrast": f"{a_} - {b_}"}
            A, B = d[d.group.eq(a_)], d[d.group.eq(b_)]
            for meth, col in (("published (group-mean imputed)", pub_col), ("complete-case (corrected raw)", obs_col)):
                aa, bb = A[col].dropna(), B[col].dropna()
                w = stats.ttest_ind(aa, bb, equal_var=False)
                grp_rows.append({**base, "method": meth, "n_a": len(aa), "n_b": len(bb), "estimate": aa.mean() - bb.mean(),
                                 "p_two_sided": w.pvalue, "mean_a": aa.mean(), "mean_b": bb.mean()})
            est, var = zip(*[welch(f.loc[f.group.eq(a_), key + "_i"], f.loc[f.group.eq(b_), key + "_i"]) for f in imps])
            grp_rows.append({**base, "method": f"multiple imputation (m={M})", "n_a": len(A), "n_b": len(B),
                             "mean_a": float(np.mean([f.loc[f.group.eq(a_), key + "_i"].mean() for f in imps])),
                             "mean_b": float(np.mean([f.loc[f.group.eq(b_), key + "_i"].mean() for f in imps])),
                             **pool_list(est, var, len(A) + len(B) - 2)})
        # TCOA - OAC adjusted for age + sex
        base = {"outcome": lab.replace(" (reverse-scored)", " (raw; lower = better)"), "contrast": "TCOA - OAC, adjusted for age + sex"}
        for meth, col in (("published (group-mean imputed)", pub_col), ("complete-case (corrected raw)", obs_col)):
            f = d.dropna(subset=[col]); b, v, dfr = adjusted(f, col)
            grp_rows.append({**base, "method": meth, "estimate": b, "p_two_sided": float(2 * stats.t.sf(abs(b / np.sqrt(v)), dfr))})
        res = [adjusted(f, key + "_i") for f in imps]
        grp_rows.append({**base, "method": f"multiple imputation (m={M})", **pool_list([r[0] for r in res], [r[1] for r in res], res[0][2])})
    grp = pd.DataFrame(grp_rows)
    grp.to_csv(OUT / "MI_group_comparisons.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["sample", "predictor", "outcome", "method", "n", "estimate", "ci95_low", "ci95_high", "p_one_sided_positive", "p_two_sided"]
    print(corr[cols].round(3).to_string(index=False))
    print(grp[["outcome", "contrast", "method", "mean_a", "mean_b", "estimate", "ci95_low", "ci95_high", "p_two_sided"]].round(3).to_string(index=False))
    imputed_vals = pd.DataFrame({"participant": d.participant, "group": d.group,
                                 "RBANS_missing": d.RBANS_raw_corrected.isna(), "miniBEST_missing": d.miniBEST_observed.isna(),
                                 "RBANS_MI_mean": np.mean([f.rbans_i for f in imps], 0), "RBANS_MI_sd": np.std([f.rbans_i for f in imps], 0),
                                 "miniBEST_MI_mean": np.mean([f.minibest_i for f in imps], 0), "miniBEST_MI_sd": np.std([f.minibest_i for f in imps], 0)})
    imputed_vals = imputed_vals[imputed_vals.RBANS_missing | imputed_vals.miniBEST_missing]
    imputed_vals.to_csv(OUT / "MI_imputed_values_summary.csv", index=False)
    print(imputed_vals.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
