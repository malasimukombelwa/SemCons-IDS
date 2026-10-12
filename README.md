# SemCons-IDS — research code archive (audit required)

Research notebooks for the CICIDS2017 numeric–semantic intrusion detection study and its later CSE-CIC-IDS2018 reanalyses.

## Status

- `notebooks/core/`: comparative v8, resumable v10, post-hoc explainability v12.
- `notebooks/archive/`: historical development notebooks; **not a single run-ready pipeline**.
- `src/semcons_corrected_pipeline_utils.py`: saved-model evaluation and post-hoc adaptation helpers.
- `reports/`: provenance/integrity summaries, not performance verification.
- All notebook outputs and execution counts were stripped for safe sharing; code cells are preserved.
- No raw dataset or trained-model checkpoints are included.

## Reproduction status

These notebooks expect Google Colab, Drive paths, the original datasets, and saved checkpoints/adapters. They have **not** been run end-to-end on Narval or validated against published tables in this package. The original source artifacts and dependency versions should be documented before declaring reproducibility.

## Security

Review code, outputs, notebook links, Git history, and access rights before making this repository public. `.gitignore` does not protect tracked files. Only publish datasets/models under appropriate licenses.
