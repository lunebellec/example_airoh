from pathlib import Path
from invoke import task


@task
def fetch(c):
    """Download the development fMRI dataset and BASC atlas via NiLearn."""
    from nilearn import datasets
    source_dir = c.config.get("source_data_dir")
    print("Fetching development fMRI dataset...")
    datasets.fetch_development_fmri(n_subjects=66, data_dir=source_dir, reduce_confounds=False)
    print("Fetching BASC multiscale atlas (resolution=64)...")
    datasets.fetch_atlas_basc_multiscale_2015(data_dir=source_dir, resolution=64)
    print("All data ready.")


@task
def run_connectivity(c, subjects=None, smoke=False):
    """Extract per-subject functional connectivity matrices (chunk: subject index)."""
    from analysis.connectivity import list_subjects, extract_subject_connectome, aggregate_connectomes
    from airoh.utils import ensure_dir_exist

    source_dir = c.config.get("source_data_dir")
    output_dir = Path(c.config.get("output_data_dir"))
    ensure_dir_exist(c, "output_data_dir")

    all_subjects = list_subjects(source_dir)
    if smoke:
        all_subjects = all_subjects[:2]
    if subjects:
        all_subjects = subjects.split(",")

    for subj in all_subjects:
        out = output_dir / f"connectome_sub{subj}.npy"
        if out.exists():
            print(f"Skipping subject {subj} (output exists)")
            continue
        extract_subject_connectome(subj, source_dir, output_dir)

    aggregate_connectomes(source_dir, output_dir)


@task(pre=[run_connectivity])
def run_classify(c, smoke=False):
    """Train and evaluate linear SVC classifier on connectivity features."""
    from analysis.classification import run_classification

    output_dir = Path(c.config.get("output_data_dir"))
    results_file = output_dir / "classification_results.json"

    if smoke:
        print("Smoke: skipping full SVC classification (insufficient subjects)")
        return
    if results_file.exists():
        print("Skipping classification (results already exist)")
        return

    run_classification(output_dir)


@task
def run_notebooks(c):
    """Generate figures using notebooks."""
    from airoh.utils import run_notebooks as airoh_run_notebooks, ensure_dir_exist

    notebooks_dir = Path(c.config.get("notebooks_dir"))
    output_dir = Path(c.config.get("output_data_dir")).resolve()
    ensure_dir_exist(c, "output_data_dir")
    airoh_run_notebooks(c, notebooks_dir, output_dir, keys=["source_data_dir", "output_data_dir"])


@task(pre=[fetch, run_connectivity, run_classify, run_notebooks])
def run(c):
    """Full pipeline."""
    print("Pipeline complete.")


@task
def run_smoke(c):
    """Smoke test: fetch + 2 subjects connectivity + classification check (no notebooks)."""
    fetch(c)
    run_connectivity(c, smoke=True)
    run_classify(c, smoke=True)


@task
def clean_connectivity(c):
    """Remove per-subject connectome files and the combined connectomes.npz."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "connectome_sub*.npy")
    clean_folder(c, "output_data_dir", "connectomes.npz")


@task
def clean_classify(c):
    """Remove classification results and predictions."""
    from airoh.utils import clean_folder
    clean_folder(c, "output_data_dir", "classification_results.json")
    clean_folder(c, "output_data_dir", "classification_predictions.npz")


@task(pre=[clean_connectivity, clean_classify])
def clean(c):
    """Remove all computed outputs."""
    pass


@task
def clean_source(c):
    """Remove all downloaded source data (NiLearn cache)."""
    from airoh.utils import clean_folder
    clean_folder(c, "source_data_dir", "nilearn_data")
