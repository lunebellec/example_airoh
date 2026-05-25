# Source Data

Downloaded automatically by `invoke fetch` via NiLearn's built-in fetchers into `source_data/`:

- **`development_fmri/`** — NiLearn development fMRI dataset (155 subjects: children ages 3–13 and adults ages 18–39). We use a balanced subset of 66 subjects (32 adults + 34 children).
- **`basc_multiscale_2015/`** — BASC multiscale brain parcellation atlas, resolution=64 (64 ROIs).

To remove downloaded data: `invoke clean-source`

📝 Note: NiLearn data files are **ignored by Git** (see `.gitignore`).
