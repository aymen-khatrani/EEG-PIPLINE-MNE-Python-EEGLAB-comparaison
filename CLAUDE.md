# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Research internship project (MIMLAB, Boğaziçi University, supervisor Prof. Pınar S. Özbay) to reproduce
in Python (MNE) an EEG-fMRI preprocessing pipeline originally built in EEGLAB (MATLAB, FMRIB/FASTR
plugin), add a machine-learning component for ICA artifact-component classification (benchmarked
against ICLabel), and quantitatively compare EEGLAB vs. Python outputs. Full rationale and research
questions are in `EEG_fMRI_Project_Proposal.pdf`.

The EEGLAB pipeline being reproduced (in order): import BrainVision data → remove gradient artifacts
(FMRIB FASTR) → detect QRS on ECG and remove BCG/pulse artifacts (Optimal Basis Set, 4 components) →
resample to 250 Hz → re-reference to average → band-pass 1–40 Hz + 50 Hz notch → ICA (extended
Infomax) with ICLabel-based component rejection.

## Commands

There is no build system, test suite, or linter in this repo — it is a research notebook project.

Set up the environment (pinned versions in `requirements.txt`):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebook headlessly to validate changes end-to-end:
```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace Pipeline_EEG_fMRI_Python.ipynb
```

`mne-icalabel` pulls in `torch`; if it's unavailable, the notebook degrades gracefully (`HAS_ICLABEL`
flag) and falls back to a kurtosis/topography heuristic for component labelling.

## Data layout

Real lab data lives under `sub-38/raw/`, organized BIDS-like (not full BIDS — no `.json` sidecars):
- `eeg/EEG-fMRI_Assignment 1/sub-38_task-rest_run-0{1,2}.{vhdr,vmrk,dat}` — BrainVision EEG, 32 channels
  (31 EEG in 10-20 layout + 1 `ECG`), 5000 Hz sampling, recorded inside the scanner.
- `func/sub-38_task-rest_run-0{1,2}_bold.nii` — fMRI BOLD runs.
- `anat/` — T1w anatomical.
- `physio/*.puls` / `*.resp` — cardiac/respiration traces from the scanner's physio recording.
- `preprocessed_3mm/` — AFNI-preprocessed fMRI (`errts...+tlrc.BRIK/HEAD`).
- `EEG-fMRI_Assignment 1/Data Note.docx` — lab's own marker/trigger documentation. Trigger semantics in
  this codebase are intentionally re-derived from the `.vmrk` annotations themselves (periodicity
  analysis) rather than read from this file, so the detection logic generalizes to other subjects/runs.

Marker convention found in the `.vmrk` files (confirmed by direct inspection, not the Data Note): `Sync
On` fires every ~2 s regardless of acquisition (Brain Products SyncBox clock-sync pulse, keeps the EEG
amplifier's 5000 Hz clock locked to the scanner clock — required for gradient-artifact template
alignment); `Response,R128` fires once per fMRI volume (TR-locked, ~3 s here) and is the actual scan
trigger used to anchor gradient-artifact correction (AAS/FASTR) — equivalent to the marker referenced in
Niazy et al. 2005 (FASTR/OBS) and Allen et al. 2000 (AAS).

## Code architecture

Two parallel implementations of the same pipeline exist:
- `Pipeline_EEG_fMRI_Python.ipynb` — the primary, narrative artifact. Each preprocessing step is its own
  markdown+code cell pair, annotated with its EEGLAB GUI equivalent, and intermediate `Raw` states are
  kept in a `pipeline_states` dict (mirrors EEGLAB's habit of saving a `.set` after each stage). Beyond
  preprocessing it also contains: Phase 2 (EEGLAB↔Python quantitative comparison: PSD, Morlet TFR, ICA
  topography matching, non-parametric stats), Phase 3 (ICA-component feature extraction + supervised
  classifier trained/benchmarked against ICLabel), Phase 4 (band-power time-frequency analysis on the
  cleaned signal).
- `eeg_fmri_pipeline.py` — an earlier, simpler linear script version of just the preprocessing steps.
  Treat the notebook as the source of truth; the script is not kept in sync with it.

All pipeline parameters (paths, filter cutoffs, ICA method, sampling rates) are centralized in one
config cell near the top of the notebook — change values there rather than hardcoding inline further
down. The gradient-artifact correction (`aas_gradient_correction`) and the QRS-based BCG/OBS correction
are hand-implemented in the notebook (not single MNE calls) because MNE has no built-in FASTR/OBS
equivalent; this is the "from-scratch" implementation the project proposal (O1, O3) explicitly asks for,
so prefer extending these functions over replacing them with a black-box library call.

Outputs are written to `results/figures/`, `results/tables/`, and cleaned `.fif` files directly under
`results/` — name these consistently with `{subject}_{task}_{run}_preprocessed_raw.fif`.