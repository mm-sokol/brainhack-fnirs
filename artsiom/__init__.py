"""
artsiom – fNIRS classification pipeline for BrainHack Warsaw 2026.

Modules
-------
data_loader     : SNIRF loading + event remapping via BHMentalDatabaseHelper
preprocessing   : SCI rejection, OD, MBLL, IIR filter, epoching
features        : Statistical feature extraction (mean/std/peak/slope/skew/kurt)
train_baseline  : Random Forest & SVM with LOSO cross-validation

Quick start
-----------
    python -m artsiom.train_baseline --data_dir /path/to/snirf --out_dir results/
"""

from artsiom.data_loader import BHMentalDatabaseHelper, EPOCH_EVENT_ID  # noqa: F401
from artsiom.train_baseline import LABEL_MAP  # noqa: F401
from artsiom.features import extract_features, features_to_dataframe  # noqa: F401
from artsiom.preprocessing import run_preprocessing_pipeline  # noqa: F401
from artsiom.train_baseline import build_dataset, run_loso, print_report  # noqa: F401
