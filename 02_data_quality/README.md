# 02 · Data quality

`recording_length_and_icc.ipynb`

1. **Recording length.** Reads every `<id>_epoched.fif` and reports the number
   of retained epochs and usable recording time per participant, condition and
   group.
2. **ICC(2,1).** Two-way random-effects, single-measure ICC of the edge-wise FC
   pattern across the windows assigned to each HMM state, per participant and
   condition, summarised by group. Writes `icc_per_participant.csv` and
   `icc_summary_by_group.csv`.

Reads from `../data/out/subjects/`. Reported in Supplementary Section 1.
