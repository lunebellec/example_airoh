from pathlib import Path
import numpy as np
import json


def run_classification(output_dir):
    """Train linear SVC on connectivity features; save metrics and predictions."""
    from sklearn.model_selection import (
        train_test_split, cross_val_predict, cross_val_score, permutation_test_score
    )
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    output_dir = Path(output_dir)
    data = np.load(output_dir / "connectomes.npz", allow_pickle=True)
    X, y = data["X"], data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=123
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scl = scaler.transform(X_train)
    X_test_scl = scaler.transform(X_test)

    svc = SVC(kernel="linear", class_weight="balanced")

    y_pred_cv = cross_val_predict(svc, X_train_scl, y_train, groups=y_train, cv=3)
    cv_acc = cross_val_score(svc, X_train_scl, y_train, groups=y_train, cv=3)

    score, perm_scores, pvalue = permutation_test_score(
        svc, X_train_scl, y_train, cv=3, scoring="accuracy", n_jobs=2, n_permutations=100
    )

    svc.fit(X_train_scl, y_train)
    y_pred_test = svc.predict(X_test_scl)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Undo vectorization of SVC weights for brain visualization
    n_features = X.shape[1]
    n_roi = int(round((1 + (1 + 8 * n_features) ** 0.5) / 2))
    feat_matrix = np.zeros((n_roi, n_roi))
    triu_idx = np.triu_indices(n_roi, k=1)
    feat_matrix[triu_idx] = svc.coef_[0]
    feat_matrix = feat_matrix + feat_matrix.T

    results = {
        "cv_accuracy_per_fold": cv_acc.tolist(),
        "cv_accuracy_mean": float(cv_acc.mean()),
        "test_accuracy": float(test_acc),
        "permutation_score": float(score),
        "permutation_pvalue": float(pvalue),
        "n_subjects": int(len(X)),
        "n_features": int(n_features),
    }
    with open(output_dir / "classification_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez_compressed(
        output_dir / "classification_predictions.npz",
        y_train=y_train, y_test=y_test,
        y_pred_cv=y_pred_cv, y_pred_test=y_pred_test,
        feat_matrix=feat_matrix,
    )

    print(f"CV accuracy: {cv_acc.mean():.3f} | Test accuracy: {test_acc:.3f} | p = {pvalue:.4f}")
    print(classification_report(y_test, y_pred_test))
