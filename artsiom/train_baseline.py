"""
train_baseline.py
=================
Trains Random Forest and SVM baselines with Leave-One-Subject-Out (LOSO)
cross-validation and reports Accuracy, F1-score, and Confusion Matrix.

Leave-One-Subject-Out rationale
--------------------------------
With 60 participants the model must generalise across individuals.  LOSO
tests each subject as an unseen hold-out while training on all others –
this is the gold standard evaluation protocol for BCI/fNIRS studies.

Usage (CLI)
-----------
    python -m artsiom.train_baseline --data_dir path/to/snirf

Usage (API)
-----------
    from artsiom.train_baseline import build_dataset, run_loso

    X, y, groups = build_dataset("path/to/snirf")
    results = run_loso(X, y, groups)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from artsiom.data_loader import BHMentalDatabaseHelper
from artsiom.features import extract_features
from artsiom.preprocessing import run_preprocessing_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label constants  (mental=1, rest=0)
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, int] = {"mental": 1, "rest": 0}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _reduce_to_fixed_features(
    epochs,
    feature_funcs=None,
) -> np.ndarray:
    """Return a (n_epochs, 12) matrix: 6 stats × {HbO mean, HbR mean}.

    Each of the 6 statistics is averaged across all good channels for HbO
    and HbR independently, giving a consistent feature vector regardless of
    how many channels survive SCI rejection per subject.
    """
    from artsiom.features import VECTORISED_FEATURES
    if feature_funcs is None:
        feature_funcs = VECTORISED_FEATURES

    data_hbo = epochs.get_data(picks="hbo").astype(np.float64)  # (E, C, T)
    data_hbr = epochs.get_data(picks="hbr").astype(np.float64)

    feats = []
    names = []
    for fn, func in feature_funcs.items():
        hbo_ch = func(data_hbo)  # (E, C)
        hbr_ch = func(data_hbr)
        feats.append(hbo_ch.mean(axis=1, keepdims=True))  # (E, 1)
        feats.append(hbr_ch.mean(axis=1, keepdims=True))
        names += [f"hbo_{fn}_mean", f"hbr_{fn}_mean"]

    return np.concatenate(feats, axis=1).astype(np.float32), names  # (E, 12)


def build_dataset(
    data_dir: str | Path,
    condition: str | None = None,
    sci_threshold: float = 0.7,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all subjects, preprocess, and extract features.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing SNIRF files.
    condition : str | None
        ``"ST"`` or ``"DT"`` to restrict to one condition; ``None`` = all.
    sci_threshold : float
        SCI threshold for channel rejection (default 0.7).
    verbose : bool
        MNE verbosity.

    Returns
    -------
    X : np.ndarray, shape (n_total_epochs, n_features)
    y : np.ndarray, shape (n_total_epochs,)
        Integer labels: 1 = mental, 0 = rest.
    groups : np.ndarray, shape (n_total_epochs,)
        Subject index per epoch (used by LeaveOneGroupOut).
    """
    helper = BHMentalDatabaseHelper(data_dir, condition=condition)
    subjects = helper.get_subject_list()
    logger.info("Found %d subjects.", len(subjects))

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []

    for sid in subjects:
        logger.info("--- Processing subject %d / %d ---", sid + 1, len(subjects))
        try:
            raw = helper.load_subject(sid, verbose=verbose)
            epochs = run_preprocessing_pipeline(
                raw, sci_threshold=sci_threshold
            )

            if len(epochs) == 0:
                logger.warning("Subject %d: no valid epochs, skipping.", sid)
                continue

            X_sub, feat_names = _reduce_to_fixed_features(epochs)

            # Labels from epoch events
            # epochs.events[:, 2] holds the integer event code
            # EPOCH_EVENT_ID: mental=1, rest=2  -> remap to 1/0
            event_codes = epochs.events[:, 2]
            # Invert the event_id dict from the epochs object
            inv_event_id = {v: k for k, v in epochs.event_id.items()}
            y_sub = np.array(
                [LABEL_MAP.get(inv_event_id.get(c, ""), -1) for c in event_codes]
            )

            # Drop any epochs with unknown labels
            valid_mask = y_sub >= 0
            X_sub = X_sub[valid_mask]
            y_sub = y_sub[valid_mask]

            all_X.append(X_sub)
            all_y.append(y_sub)
            all_groups.append(np.full(len(y_sub), sid, dtype=int))

            logger.info(
                "Subject %d: %d epochs (mental=%d, rest=%d)",
                sid,
                len(y_sub),
                (y_sub == 1).sum(),
                (y_sub == 0).sum(),
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Subject %d failed: %s", sid, exc, exc_info=True)

    if not all_X:
        raise RuntimeError("No valid data found – check data_dir and SNIRF files.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    logger.info(
        "Dataset built: %d epochs, %d features. Classes: mental=%d, rest=%d",
        X.shape[0],
        X.shape[1],
        (y == 1).sum(),
        (y == 0).sum(),
    )
    return X, y, groups


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines (scaler + classifier)."""
    return {
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: dict[str, Pipeline] | None = None,
) -> dict[str, dict]:
    """Run Leave-One-Subject-Out cross-validation for each model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label vector (0 / 1).
    groups : np.ndarray
        Subject index per sample (used to define the leave-one-out splits).
    models : dict | None
        Mapping of model-name -> sklearn Pipeline.  Defaults to RF + SVM.

    Returns
    -------
    results : dict
        ``results[model_name]`` contains:
            - ``"per_fold"``    : list of per-subject dicts with acc / f1
            - ``"accuracy"``    : mean accuracy across subjects
            - ``"f1"``          : mean F1 (macro) across subjects
            - ``"confusion_matrix"`` : summed confusion matrix (int array)
            - ``"y_true"``      : concatenated ground-truth labels
            - ``"y_pred"``      : concatenated predictions
    """
    if models is None:
        models = get_models()

    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(X, y, groups)
    logger.info("LOSO: %d folds (subjects).", n_splits)

    results: dict[str, dict] = {}

    for model_name, pipeline in models.items():
        logger.info("=== Model: %s ===", model_name)
        t0 = time.time()

        per_fold: list[dict] = []
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            logo.split(X, y, groups)
        ):
            test_subject = groups[test_idx[0]]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Clone pipeline for each fold to avoid data leakage
            from sklearn.base import clone
            fold_pipeline = clone(pipeline)
            fold_pipeline.fit(X_train, y_train)
            y_pred = fold_pipeline.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            per_fold.append(
                {"subject": test_subject, "accuracy": acc, "f1_macro": f1}
            )
            y_true_all.append(y_test)
            y_pred_all.append(y_pred)

            logger.info(
                "  Fold %3d | Subject %3d | Acc=%.3f | F1=%.3f",
                fold_idx + 1,
                test_subject,
                acc,
                f1,
            )

        y_true_cat = np.concatenate(y_true_all)
        y_pred_cat = np.concatenate(y_pred_all)

        mean_acc = float(np.mean([f["accuracy"] for f in per_fold]))
        mean_f1 = float(np.mean([f["f1_macro"] for f in per_fold]))
        cm = confusion_matrix(y_true_cat, y_pred_cat, labels=[0, 1])

        elapsed = time.time() - t0
        logger.info(
            "%s -> Mean Acc=%.4f | Mean F1=%.4f | Time=%.1fs",
            model_name,
            mean_acc,
            mean_f1,
            elapsed,
        )

        results[model_name] = {
            "per_fold": per_fold,
            "accuracy": mean_acc,
            "f1": mean_f1,
            "confusion_matrix": cm,
            "y_true": y_true_cat,
            "y_pred": y_pred_cat,
        }

    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: dict[str, dict]) -> None:
    """Pretty-print a summary table of LOSO results to stdout."""
    print("\n" + "=" * 60)
    print("  LOSO Cross-Validation Results")
    print("=" * 60)

    for model_name, res in results.items():
        print(f"\n  {model_name}")
        print(f"    Mean Accuracy : {res['accuracy']:.4f}")
        print(f"    Mean F1 (macro): {res['f1']:.4f}")
        print("\n    Confusion Matrix (rows=true, cols=pred):")
        print(f"               Pred: Rest  Mental")
        cm = res["confusion_matrix"]
        print(f"    True: Rest      {cm[0, 0]:5d}   {cm[0, 1]:5d}")
        print(f"    True: Mental    {cm[1, 0]:5d}   {cm[1, 1]:5d}")

        # Per-subject table
        df = pd.DataFrame(res["per_fold"])
        df = df.set_index("subject").sort_index()
        print(f"\n    Per-Subject Summary:")
        print(df.to_string())

    print("\n" + "=" * 60)


def save_results(results: dict[str, dict], out_dir: str | Path) -> None:
    """Save per-fold CSV and confusion matrix for each model.

    Parameters
    ----------
    results : dict
        Output of :func:`run_loso`.
    out_dir : str | Path
        Directory where files will be written.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, res in results.items():
        # Per-fold CSV
        df = pd.DataFrame(res["per_fold"])
        csv_path = out_dir / f"{model_name}_loso_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved: %s", csv_path)

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res["confusion_matrix"],
            display_labels=["Rest", "Mental"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{model_name} – LOSO Confusion Matrix")
        fig.tight_layout()
        fig_path = out_dir / f"{model_name}_confusion_matrix.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", fig_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF/SVM baselines on fNIRS data with LOSO CV."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing participant SNIRF files.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=["ST", "DT", "SITTING", "WALKING"],
        help="Filter by condition: SITTING/ST or WALKING/DT. Default: all.",
    )
    parser.add_argument(
        "--sci_threshold",
        type=float,
        default=0.7,
        help="SCI channel rejection threshold (default: 0.7).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory to save CSV and confusion matrix plots.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose MNE output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Building dataset from: %s", args.data_dir)
    X, y, groups = build_dataset(
        data_dir=args.data_dir,
        condition=args.condition,
        sci_threshold=args.sci_threshold,
        verbose=args.verbose,
    )

    logger.info("Running LOSO cross-validation …")
    results = run_loso(X, y, groups)

    print_report(results)
    save_results(results, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
