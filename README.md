# Age Classification with NiLearn on the NeuroDev Dataset

A secondary analysis of the [NiLearn development fMRI dataset](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_development_fmri.html) (a public OpenNeuro dataset). The aim is to classify participants into two groups — **children** (ages 3–13) and **adults** (ages 18–39) — based on resting-state fMRI functional connectivity features extracted with NiLearn.

The pipeline uses a BASC 64-ROI brain parcellation to compute per-subject correlation matrices, then trains a linear Support Vector Classifier with cross-validation and evaluates it on a held-out test set.

---

## Quick Start

```bash
uv sync
uv run invoke fetch
uv run invoke run
```

---

## Setup

```bash
uv sync
```

Creates a `.venv` and installs all dependencies from `pyproject.toml`.

---

## Task Overview

| Task | Description |
|---|---|
| `fetch` | Download development fMRI dataset and BASC atlas via NiLearn |
| `run-connectivity` | Extract per-subject functional connectivity matrices (chunk: subject index) |
| `run-classify` | Train and evaluate linear SVC; save metrics and predictions |
| `run-notebooks` | Execute visualization notebook; save figures to `output_data/` |
| `run` | Full pipeline in order |
| `run-smoke` | Fast end-to-end check (2 subjects, skips full classification) |
| `clean` | Remove all computed outputs |
| `clean-connectivity` | Remove per-subject `.npy` files and `connectomes.npz` |
| `clean-classify` | Remove `classification_results.json` and predictions |
| `clean-source` | Remove downloaded NiLearn data from `source_data/` |

Use `invoke --list` or `invoke --help <task>` for full descriptions.

---

## Folder Structure

| Folder / File | Description |
|---|---|
| `analysis/connectivity.py` | Per-subject connectivity extraction (NiftiLabelsMasker + ConnectivityMeasure) |
| `analysis/classification.py` | SVC training, CV, permutation test, and prediction saving |
| `notebooks/` | Visualization notebook (`figure_classification.ipynb`) |
| `source_data/` | Downloaded data — see [`source_data/CONTENT.md`](source_data/CONTENT.md) |
| `output_data/` | Generated results and figures — see [`output_data/CONTENT.md`](output_data/CONTENT.md) |
| `tasks.py` | Invoke task definitions |
| `invoke.yaml` | Config: paths |

---

## Design Principles

- **Analysis in code, visualization in notebooks.** Connectivity extraction and classification live in `analysis/`; the notebook only reads results and produces figures.
- **Idempotent steps.** `run-connectivity` skips subjects whose `.npy` file already exists. `run-classify` skips if `classification_results.json` exists.
- **Smoke test.** `invoke run-smoke` processes 2 subjects end-to-end to verify the pipeline is wired correctly.

---

## Philosophy

Inspired by Uncle Iroh from *Avatar: The Last Airbender* — simple, reproducible, and warm.
