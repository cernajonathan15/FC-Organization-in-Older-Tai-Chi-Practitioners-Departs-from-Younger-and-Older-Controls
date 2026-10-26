# 04 · Distance-metric sensitivity: Ledoit–Wolf Mahalanobis

Repeats the `robust-median` analysis of `03_age_axis_analysis` with a
Mahalanobis geometry instead of Euclidean distance. Stages 1 and 3 are
identical to the main pipeline; `analysis_config.py`, `analysis_utils.py`,
stages 2 and 4, and `run_pipeline.py` differ. The folder is self-contained so
that the sensitivity run is reproducible exactly as executed.

After robust standardization, a Ledoit–Wolf shrinkage covariance is estimated
from the 30 YAC+OAC reference participants only. Endpoints, axis projection and
off-axis deviation are then computed in the covariance-whitened space, which is
equivalent to Mahalanobis distance in the standardized feature space. TCOAs
never contribute to the covariance or endpoints. The covariance, whitening and
scoring are refitted in every permutation and bootstrap replicate.

```bash
python run_pipeline.py            # production: 10,000 permutations, 5,000 bootstraps, seed 42
python run_pipeline.py --quick    # 100 / 100
python -m unittest discover -s tests
```

Outputs: `results/robust_median/mahalanobis_ledoit_wolf/`. Reported in
Supplementary Section 3 and as a robustness check in Table 1.
