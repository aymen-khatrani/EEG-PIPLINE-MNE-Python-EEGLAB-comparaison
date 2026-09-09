# EEG-fMRI Preprocessing Pipeline

Research internship project (MIMLAB, Boğaziçi University, supervisor Prof. Pınar S. Özbay) reproducing
in Python (MNE) an EEG-fMRI preprocessing pipeline originally built in EEGLAB (MATLAB, FMRIB/FASTR
plugin), with a machine-learning component for ICA artifact-component classification benchmarked
against ICLabel.

## Contents

- `Pipeline_EEG_fMRI_English.ipynb` — main notebook: preprocessing pipeline, EEGLAB↔Python comparison,
  ICA-component classifier, band-power analysis.
- `PIPELINE_Protocoles_EN.txt` — written protocol description.
- `Presentation_EEG_fMRI_EN.pdf` — project presentation slides.
- `requirements.txt` — pinned Python dependencies.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebook end-to-end:

```bash
jupyter nbconvert --to notebook --execute --inplace Pipeline_EEG_fMRI_English.ipynb
```
