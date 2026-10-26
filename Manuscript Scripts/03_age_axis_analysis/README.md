# 03 · Main analysis: age-phenotype axis

Fits a reference axis from the younger-adult (YAC) and older-adult (OAC)
control groups in a 28-feature network FC space (7 within-network and 21
between-network features, Schaefer-100, 7 networks) and scores every participant
on

- **position** along the OAC → YAC axis (0 = OAC endpoint, 1 = YAC endpoint), and
- **off-axis deviation**: root-mean-square distance perpendicular to the axis.

TCOAs never contribute to fitting the axis. Inference uses OAC/TCOA label
permutations with the whole procedure refitted in every permutation, plus
stratified bootstrap confidence intervals.

## Stages

| Stage | Script | Output folder |
|---|---|---|
| 1 | `01_build_participant_features.py`: validate matrices, average to 28 features per participant (EC and EO weighted equally) | `results/<branch>/01_dataset/` |
| 2 | `02_calculate_phenotype_scores.py`: robust standardization, outlier handling, axis fit (geometric median primary; coordinate median and mean as sensitivity), scores | `results/<branch>/02_phenotype/` |
| 3 | `03_run_inference_and_behavior.py`: permutation tests, bootstrap CIs, estimator robustness, reference jackknife, behavior correlations | `results/<branch>/03_inference/` |
| 4 | `04_create_results_workbook.py`: results workbook and figures | `results/<branch>/04_final_results/` |
| 5 | `05_miscellaneous_analysis_and_figures.py`: descriptive feature-level decomposition of the TCOA–OAC deviation difference (no hypothesis tests) | `results/<branch>/04_final_results/` |

## Branches

- **`robust-median` (reported in the manuscript).** Pooled YAC+OAC multivariate
  radial winsorization (3.5 normal-consistent MADs); TCOA–OAC median
  differences; bootstrapped TCOA medians; mean-based results as sensitivity.
- `position-winsor`. Alternative outlier handling: no radial winsorization;
  phenotype position capped at label-blind limits pooled over the 30 older adults.

## Running

```bash
python run_pipeline.py --method robust-median          # all stages, production settings
python run_pipeline.py --method robust-median --quick  # 100 permutations / 100 bootstraps
python run_pipeline.py --from-stage 2 --to-stage 4     # stage range, both branches
python run_pipeline.py --list-stages
python 05_miscellaneous_analysis_and_figures.py        # after stage 4 (robust-median)
python -m unittest discover -s tests                   # synthetic-data unit tests
```

Production settings: 10,000 permutations, 5,000 bootstraps, seed 42.

## Paths and settings

All settings live in `analysis_config.py`. Paths default to this repository
and can be overridden:

| Variable | Default |
|---|---|
| `FAPA_INPUT_ROOT` | `data/out/subjects` (FC matrices from `01_preprocessing`) |
| `FAPA_OUTPUT_ROOT` | `results` |
| `FAPA_BEHAVIOR_FILE` | `data/RBANS_miniBEST_GDS.xlsx` (sheet `Sheet1`) |
| `FAPA_N_PERMUTATIONS`, `FAPA_N_BOOTSTRAPS`, `FAPA_RANDOM_SEED` | 10000, 5000, 42 |
