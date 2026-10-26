# 01 · Preprocessing (EEG → FC matrices)

Notebooks from the pipeline described in Cerna et al. (2024, *Brain Sciences*
14(9):901). They are run in order, once per participant and eye condition
(EC = eyes closed, EO = eyes open). Paths are relative to this folder:
`../data/in/` for inputs, `../data/out/subjects/` for outputs.

| Step | Notebook | What it does | Main output per participant/condition |
|---|---|---|---|
| 1 | `step1_sensor_preprocessing.ipynb` | Channel mapping, filtering, bad-channel handling, ICA (Infomax) with ICLabel component rejection, interpolation, average reference, epoching | `<id>_epoched.fif` |
| 2 | `step2_source_reconstruction.ipynb` | Forward model on *fsaverage*, noise covariance, eLORETA inverse solution with normal-oriented (signed) sources | source estimates (`stc/`) |
| 3 | `step3_source_to_parcel.ipynb` | Schaefer-2018 100-parcel / 7-network signed time series | parcel time series |
| 4 | `step4_orthogonalization_and_state_number.ipynb` | Symmetric orthogonalization of the signed parcel signals (leakage correction), then Hilbert amplitude envelopes; HMM fits over 3–16 states with AIC/BIC | `orth.npy`, `aic_bic_<id>_<mode>.txt` |
| 5 | `step5_hmm_fitting_and_fc_matrices.ipynb` | HMM at the selected state number, state-wise FC matrices, thresholding, within/between-network connectivity | `<id>_thresholded_matrices_raw.npz` (input to `03_age_axis_analysis`) |

`names.txt` lists the 45 participant IDs processed (15 YAC: 1xx, 15 OAC: 2xx,
15 TCOA: 4xx).

Inputs not distributed here: raw recordings (`../data/in/subjects/<id>/<mode>/`),
the electrode montage file (`../data/in/MFPRL_UPDATED_V2.sfp`). *fsaverage* is
downloaded by MNE on first use.

Environment: `pip install -r ../requirements-preprocessing.txt`. Steps 3–4 use
CuPy for GPU acceleration; an NVIDIA GPU with CUDA is expected.

Note: step 4 asks for a starting participant/condition/state at runtime
(`input()`), so that long HMM model-order searches can be resumed.
