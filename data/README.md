# data/ (not distributed)

Participant data are not shared publicly (IRB). They are available from the
authors upon reasonable request. The code expects this layout:

```text
data/
├── in/
│   ├── subjects/<id>/<EC|EO>/            raw EEG recordings            (01_preprocessing)
│   └── MFPRL_UPDATED_V2.sfp              electrode montage             (01_preprocessing)
├── out/
│   └── subjects/<id>/<EC|EO>/
│       ├── <id>_epoched.fif              cleaned epochs                (02_data_quality)
│       ├── <id>_correlation_matrices.npy.npz  windowed FC matrices     (02_data_quality, ICC)
│       └── <id>_thresholded_matrices_raw.npz  state-wise FC matrices   (03, 04)
├── RBANS_miniBEST_GDS.xlsx               behavioral scores, sheet "Sheet1"   (03, 04)
└── participant_demographics_behavior.csv age, sex, corrected behavioral scores (05)
```

Participant IDs: YAC 101–120, OAC 202–221, TCOA 401–416 (45 analysed; see
`01_preprocessing/names.txt`). A different location can be used by setting
`FAPA_INPUT_ROOT`, `FAPA_BEHAVIOR_FILE` and `TCFC_DATA_DIR`.
