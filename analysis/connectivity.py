from pathlib import Path
import numpy as np


def list_subjects(source_dir):
    """Return list of subject index strings for the balanced 66-subject subset."""
    return [f"{i:03d}" for i in range(66)]


def extract_subject_connectome(subject_idx, source_dir, output_dir):
    """Extract vectorized correlation matrix for one subject and save as .npy."""
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.connectome import ConnectivityMeasure

    source_dir, output_dir = Path(source_dir), Path(output_dir)
    idx = int(subject_idx)

    dataset = datasets.fetch_development_fmri(n_subjects=66, data_dir=str(source_dir), reduce_confounds=False)
    basc = datasets.fetch_atlas_basc_multiscale_2015(data_dir=str(source_dir), resolution=64)

    masker = NiftiLabelsMasker(
        labels_img=basc.maps, standardize="zscore_sample",
        memory="nilearn_cache", resampling_target="data",
        detrend=True, verbose=0,
    )
    corr = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)

    ts = masker.fit_transform(dataset.func[idx], confounds=dataset.confounds[idx])
    connectome = corr.fit_transform([ts])[0]
    np.save(output_dir / f"connectome_sub{subject_idx}.npy", connectome)
    print(f"Subject {subject_idx}: done")


def aggregate_connectomes(source_dir, output_dir):
    """Collect all per-subject .npy files into connectomes.npz with labels."""
    import pandas as pd
    from nilearn import datasets

    source_dir, output_dir = Path(source_dir), Path(output_dir)

    npy_files = sorted(output_dir.glob("connectome_sub*.npy"))
    if not npy_files:
        raise FileNotFoundError("No per-subject connectome files found in output_data/")

    indices = [int(f.stem.replace("connectome_sub", "")) for f in npy_files]
    all_features = np.array([np.load(f) for f in npy_files])

    dataset = datasets.fetch_development_fmri(n_subjects=66, data_dir=str(source_dir), reduce_confounds=False)
    pheno = pd.DataFrame(dataset.phenotypic).head(66)
    y = pheno.iloc[indices]["Child_Adult"].values

    np.savez_compressed(output_dir / "connectomes.npz", X=all_features, y=y)
    print(f"Saved connectomes.npz: {len(all_features)} subjects, {all_features.shape[1]} features")
