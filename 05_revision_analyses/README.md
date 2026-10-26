# 05 · Revision analyses

Analyses added in response to peer review. Every script imports the scoring
functions of `03_age_axis_analysis` **unchanged** (via `revision_common.py`)
and reads that pipeline's saved outputs, so the main results are never
recomputed differently. Run both pipelines (`03`, `04`) first.

Outputs go to `results/revision_analyses/<analysis>/`. Private inputs
(demographics, corrected behavioral scores) are read from `data/`.

| Script | Question | Output folder | Manuscript |
|---|---|---|---|
| `permutation_calibration_and_positive_controls.py` | Is the permutation null correctly calibrated, and can the design detect a real shift? Includes the tail rule, null centring, YAC-vs-OAC and injected-shift positive controls, bootstrap decomposition, leave-one-out axis scores | `permutation_calibration/` | Suppl. §4, Figs S1–S4, Table S4 |
| `deviation_directional_coherence.py` | Do TCOA off-axis departures share a direction or only scatter more? | `deviation_coherence/` | Suppl. §5, Fig S5 |
| `age_and_sex_adjustment.py` | Do the TCOA–OAC age gap or sex explain the results? (Welch/Mann–Whitney, Fisher; OLS with Freedman–Lane permutation) | `age_sex/` | Methods 2.1, Results |
| `eye_condition_analysis.py` | Do results hold separately for eyes closed and eyes open? | `eye_condition/` | Suppl. §1, Table S2 |
| `winsorization_sensitivity_and_dispersion.py` | Ledoit–Wolf details; who was winsorized; results without winsorization; per-group dispersion; feature medians/MADs | `winsorization_dispersion/` | Suppl. §2, §3, §5, §7 (Tables S3, S5, S7) |
| `brain_behavior_complete_case.py` | Two-sided brain–behavior statistics, CIs, multiplicity, jackknife, group behavior, practice hours | `brain_behavior/` | Suppl. §6 |
| `brain_behavior_multiple_imputation.py` | Same correlations with multiple imputation (MICE, m = 100, Rubin's rules) | `brain_behavior/` | Table 2 |
| `brain_behavior_supplement_outputs.py` | Scatter plots and group-behavior table | `brain_behavior/` | Fig S6, Table S6 |
| `figure1_panel_b_deviation.py` | Off-axis deviation by group | `figure1/` | Figure 1B |
| `axis_separability_cross_validation.py` | Can any version of the features separate YAC from OAC out of sample? | `axis_separability/` | supporting check |
| `matrix_count_confound.py` | Does the number of FC matrices per participant confound the deviation result? | `matrix_count/` | supporting check |

## Run order

```bash
python permutation_calibration_and_positive_controls.py
python deviation_directional_coherence.py
python age_and_sex_adjustment.py
python eye_condition_analysis.py
python winsorization_sensitivity_and_dispersion.py
python brain_behavior_complete_case.py          # writes analysis_table.csv used next
python brain_behavior_multiple_imputation.py
python brain_behavior_supplement_outputs.py
python figure1_panel_b_deviation.py
python axis_separability_cross_validation.py
python matrix_count_confound.py
```

Most scripts accept `--help`; the permutation-heavy ones take `--n-perm`.
