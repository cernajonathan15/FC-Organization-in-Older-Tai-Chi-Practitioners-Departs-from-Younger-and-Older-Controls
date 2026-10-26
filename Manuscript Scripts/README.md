# Functional Connectivity Organization in Older Tai Chi Practitioners Diverges from Younger and Older Control Phenotypes

Analysis code for:

> Cerna, J., Gupta, P., He, M., Ziegelman, L., Hu, Y., & Hernandez, M. E.
> *Functional Connectivity Organization in Older Tai Chi Practitioners Diverges
> from Younger and Older Control Phenotypes.* Frontiers in Aging Neuroscience
> (Brief Research Report, under revision).

The study asks whether resting-state, source-localized EEG functional
connectivity (FC) in older Tai Chi practitioners (TCOA) looks "younger"
(closer to younger adults, YAC) or instead departs from the age continuum
defined by younger and older controls (OAC). Each participant is described by
28 within- and between-network FC features (7 Schaefer networks). A reference
axis is fitted between the YAC and OAC geometric medians, and every participant
is scored on two quantities:

- **position** along the older-to-younger axis, and
- **off-axis deviation**, the distance perpendicular to that axis.

Group differences are tested with permutation tests (full refitting in every
permutation) and bootstrap confidence intervals, with sensitivity analyses for
the estimator, distance metric, outlier handling, eye condition, and age/sex.

---

## Repository structure

```text
.
├── 01_preprocessing/                 EEG → source space → HMM states → FC matrices
│   ├── step1_sensor_preprocessing.ipynb
│   ├── step2_source_reconstruction.ipynb
│   ├── step3_source_to_parcel.ipynb
│   ├── step4_orthogonalization_and_state_number.ipynb
│   ├── step5_hmm_fitting_and_fc_matrices.ipynb
│   └── names.txt                     participant IDs processed by the notebooks
│
├── 02_data_quality/                  recording length, epoch retention, ICC
│   └── recording_length_and_icc.ipynb
│
├── 03_age_axis_analysis/             MAIN ANALYSIS (position and off-axis deviation)
│   ├── run_pipeline.py               runs stages 1–4
│   ├── analysis_config.py            all settings and paths
│   ├── analysis_utils.py             axis fitting, scoring, winsorization
│   ├── 01_build_participant_features.py
│   ├── 02_calculate_phenotype_scores.py
│   ├── 03_run_inference_and_behavior.py
│   ├── 04_create_results_workbook.py
│   ├── 05_miscellaneous_analysis_and_figures.py
│   └── tests/                        unit tests on synthetic data
│
├── 04_mahalanobis_sensitivity/       same pipeline, Ledoit–Wolf Mahalanobis geometry
│
├── 05_revision_analyses/             additional analyses added during peer review
│   ├── revision_common.py            shared loaders; imports 03_age_axis_analysis unchanged
│   └── *.py                          one script per analysis (see table below)
│
├── data/                             NOT distributed — see data/README.md
├── results/                          created by the scripts (git-ignored)
├── requirements.txt                  analysis environment (folders 02–05)
├── requirements-preprocessing.txt    preprocessing environment (folder 01)
├── CITATION.cff
└── LICENSE
```

Each folder has its own `README.md` with details.

---

## Workflow

```text
raw EEG ─► 01_preprocessing ─► per-participant FC matrices (data/out/subjects)
                                        │
                ┌───────────────────────┼─────────────────────────┐
                ▼                       ▼                         ▼
        02_data_quality       03_age_axis_analysis     04_mahalanobis_sensitivity
                                        │                         │
                                        └──────────┬──────────────┘
                                                   ▼
                                         05_revision_analyses
```

1. **Preprocessing** (`01_preprocessing/`). The pipeline of Cerna et al. (2024),
   run once per participant and eye condition: filtering, ICA (ICLabel), epoching,
   signed eLORETA source reconstruction on *fsaverage*, Schaefer-100 parcellation,
   leakage correction (orthogonalization of the signed source signals, then
   amplitude envelopes), HMM state estimation, and state-wise FC matrices. Output: `data/out/subjects/<id>/<EC|EO>/<id>_thresholded_matrices_raw.npz`.
2. **Data quality** (`02_data_quality/`). Usable recording length per participant
   and ICC(2,1) stability of the FC matrices.
3. **Main analysis** (`03_age_axis_analysis/`). Builds the 28-feature vectors, fits
   the YAC–OAC axis, scores position and deviation, runs permutation/bootstrap
   inference and brain–behavior correlations, and writes the results workbook
   and figures. The manuscript reports the **`robust-median`** branch.
4. **Distance-metric sensitivity** (`04_mahalanobis_sensitivity/`).
5. **Revision analyses** (`05_revision_analyses/`). These read the outputs of steps 3–4.

### Quick start

```bash
pip install -r requirements.txt

# main analysis, branch reported in the manuscript
cd 03_age_axis_analysis
python run_pipeline.py --method robust-median            # production (10,000 perm / 5,000 boot)
python run_pipeline.py --method robust-median --quick    # fast check (100 / 100)
python -m unittest discover -s tests                     # unit tests, no data needed

# Mahalanobis sensitivity
cd ../04_mahalanobis_sensitivity && python run_pipeline.py

# revision analyses (run after the two pipelines)
cd ../05_revision_analyses
python permutation_calibration_and_positive_controls.py
python brain_behavior_complete_case.py && python brain_behavior_multiple_imputation.py
# ... see 05_revision_analyses/README.md for the full order
```

All paths default to folders inside this repository and can be changed with
environment variables (`FAPA_INPUT_ROOT`, `FAPA_OUTPUT_ROOT`,
`FAPA_BEHAVIOR_FILE`, `TCFC_DATA_DIR`). Random seed 42 is used throughout.

---

## Where each result comes from

| Manuscript item | Script | Output (under `results/`) |
|---|---|---|
| Table 1 – position and deviation inference, robustness | `03_age_axis_analysis` stage 3 | `robust_median/03_inference/primary_phenotype_inference.csv`, `robustness_inference.csv` |
| Figure 1A – participant phenotype position | `03_age_axis_analysis` stage 4 | `robust_median/04_final_results/` |
| Figure 1B – off-axis deviation by group | `05_revision_analyses/figure1_panel_b_deviation.py` | `revision_analyses/figure1/` |
| Figure 2 – feature-level contributions to off-axis deviation | `03_age_axis_analysis/05_miscellaneous_analysis_and_figures.py` | `robust_median/04_final_results/` |
| Table 2 – brain–behavior correlations (multiple imputation) | `05_revision_analyses/brain_behavior_multiple_imputation.py` | `revision_analyses/brain_behavior/MI_correlations.csv` |
| Age/sex-adjusted results (Results text) | `05_revision_analyses/age_and_sex_adjustment.py` | `revision_analyses/age_sex/` |
| Suppl. Section 1 – data quality, eye condition (Tables S1–S2) | `02_data_quality/recording_length_and_icc.ipynb`, `05_revision_analyses/eye_condition_analysis.py` | `revision_analyses/eye_condition/` |
| Suppl. Section 2 – winsorization sensitivity (Table S3) | `05_revision_analyses/winsorization_sensitivity_and_dispersion.py` | `revision_analyses/winsorization_dispersion/` |
| Suppl. Section 3 – Mahalanobis / Ledoit–Wolf | `04_mahalanobis_sensitivity`, `winsorization_sensitivity_and_dispersion.py` (part A) | `robust_median/mahalanobis_ledoit_wolf/` |
| Suppl. Section 4 – permutation calibration, positive controls (Figs S1–S4, Table S4) | `05_revision_analyses/permutation_calibration_and_positive_controls.py` | `revision_analyses/permutation_calibration/` |
| Suppl. Section 5 – directional coherence, dispersion (Fig S5, Table S5) | `deviation_directional_coherence.py`, `winsorization_sensitivity_and_dispersion.py` | `revision_analyses/deviation_coherence/`, `.../winsorization_dispersion/` |
| Suppl. Section 6 – group behavior, scatter plots (Table S6, Fig S6) | `brain_behavior_supplement_outputs.py` | `revision_analyses/brain_behavior/` |
| Suppl. Section 7 – feature medians and MADs (Table S7) | `winsorization_sensitivity_and_dispersion.py` (part D) | `revision_analyses/winsorization_dispersion/` |

---

## Data availability

Participant EEG recordings, connectivity matrices, demographics, and behavioral
scores are **not** included, in line with the study's IRB approval (University
of Illinois Urbana-Champaign). As stated in the manuscript, the data supporting
the conclusions will be made available by the authors upon reasonable request.
`data/README.md` describes the file layout the code expects.

## Related work

Preprocessing, source localization and HMM/FC estimation follow:

> Cerna, J., et al. (2024). *Brain Sciences*, 14(9), 901.
> https://pmc.ncbi.nlm.nih.gov/articles/PMC11430092/
> Code: https://github.com/cernajonathan15/Tai-Chi-Practice-Buffers-Aging-Effects-in-Functional-Brain-Connectivity-

The code from the initial submission of this manuscript (network-based
statistic notebooks, `Manuscript Scripts/`) is preserved in this repository's
git history.

## Citation

See `CITATION.cff`. Please cite the article above if you use this code.

## License

MIT. See `LICENSE`.

## Contact

Jonathan Cerna, University of Illinois Urbana-Champaign: cerna3@illinois.edu
