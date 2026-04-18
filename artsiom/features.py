"""
features.py
===========
Statistical feature extraction from fNIRS epochs.

Features extracted **per channel** for both HbO and HbR:
    - Mean
    - Standard deviation
    - Peak (max absolute value, signed)
    - Slope (linear regression coefficient)
    - Skewness
    - Kurtosis

The result is a flat feature vector of shape
    (n_epochs, n_channels_hbo * 6 + n_channels_hbr * 6)

Performance
-----------
All six features are computed with **fully vectorised NumPy operations**
over the entire (n_epochs, n_channels, n_times) array at once.
A subject with 40 epochs × 280 channels × 162 timepoints now takes
< 1 second instead of > 60 seconds.

Usage
-----
    from artsiom.features import extract_features

    X, feature_names = extract_features(epochs)
    # X.shape -> (n_epochs, n_features)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vectorised feature extraction over (n_epochs, n_channels, n_times)
# ---------------------------------------------------------------------------

def _vec_mean(data: np.ndarray) -> np.ndarray:
    """(E, C, T) -> (E, C)"""
    return data.mean(axis=2)


def _vec_std(data: np.ndarray) -> np.ndarray:
    """(E, C, T) -> (E, C)  – sample std (ddof=1)"""
    return data.std(axis=2, ddof=1)


def _vec_peak(data: np.ndarray) -> np.ndarray:
    """(E, C, T) -> (E, C)  – signed value at max |amplitude|"""
    idx = np.argmax(np.abs(data), axis=2)          # (E, C)
    E, C, _ = data.shape
    e_idx = np.arange(E)[:, None] * np.ones((1, C), dtype=int)
    c_idx = np.arange(C)[None, :] * np.ones((E, 1), dtype=int)
    return data[e_idx, c_idx, idx]                 # (E, C)


def _vec_slope(data: np.ndarray) -> np.ndarray:
    """(E, C, T) -> (E, C)  – OLS slope via closed-form formula.

    slope = (T * sum(x*y) - sum(x)*sum(y)) / (T * sum(x^2) - sum(x)^2)
    where x = [0, 1, ..., T-1]  (same for every epoch/channel).
    """
    T = data.shape[2]
    x = np.arange(T, dtype=np.float64)
    sum_x  = x.sum()                    # scalar
    sum_x2 = (x ** 2).sum()            # scalar
    denom  = T * sum_x2 - sum_x ** 2   # scalar

    # (E, C) sums
    sum_y  = data.sum(axis=2)
    sum_xy = (data * x[None, None, :]).sum(axis=2)

    return (T * sum_xy - sum_x * sum_y) / denom   # (E, C)


def _vec_skewness(data: np.ndarray) -> np.ndarray:
    """(E, C, T) -> (E, C)  – Fisher skewness (bias=False)."""
    T = data.shape[2]
    mu  = data.mean(axis=2, keepdims=True)
    dev = data - mu
    std = dev.std(axis=2, ddof=1, keepdims=True).clip(min=1e-12)
    m3  = (dev ** 3).mean(axis=2)
    s3  = (std ** 3).squeeze(axis=2)
    # bias correction factor: sqrt(T*(T-1)) / (T-2)
    if T > 2:
        corr = np.sqrt(T * (T - 1)) / (T - 2)
    else:
        corr = 1.0
    return corr * m3 / s3


def _vec_kurtosis(data: np.ndarray) -> np.ndarray:
    """(E, C, T) -> (E, C)  – excess kurtosis (Fisher, bias=False)."""
    T = data.shape[2]
    mu  = data.mean(axis=2, keepdims=True)
    dev = data - mu
    std = dev.std(axis=2, ddof=1, keepdims=True).clip(min=1e-12)
    m4  = (dev ** 4).mean(axis=2)
    s4  = (std ** 4).squeeze(axis=2)
    kurt_biased = m4 / s4 - 3.0
    # bias correction (same as scipy's kurtosis(bias=False))
    if T > 3:
        corr = (T - 1) / ((T - 2) * (T - 3)) * ((T + 1) * kurt_biased + 6)
    else:
        corr = kurt_biased
    return corr


# Registry – order defines column order in the feature matrix
VECTORISED_FEATURES: dict[str, callable] = {
    "mean":     _vec_mean,
    "std":      _vec_std,
    "peak":     _vec_peak,
    "slope":    _vec_slope,
    "skewness": _vec_skewness,
    "kurtosis": _vec_kurtosis,
}

# Keep the old per-signal dict for backwards compatibility (explore.py)
FEATURE_FUNCS = VECTORISED_FEATURES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_from_block(
    data: np.ndarray,        # (n_epochs, n_channels, n_times)
    ch_names: list[str],
    feat_funcs: dict[str, callable] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract all features from one chromophore block.

    Returns
    -------
    X_block : np.ndarray, shape (n_epochs, n_channels * n_features)
    names   : list[str]
    """
    if feat_funcs is None:
        feat_funcs = VECTORISED_FEATURES

    E, C, _ = data.shape
    blocks: list[np.ndarray] = []
    names:  list[str]        = []

    for feat_name, func in feat_funcs.items():
        feat = func(data.astype(np.float64))  # (E, C)
        blocks.append(feat)
        names.extend(f"{ch}_{feat_name}" for ch in ch_names)

    # Stack: each func gives (E, C); interleave by channel so output is
    # [ch0_mean, ch0_std, ..., ch1_mean, ch1_std, ...]
    stacked = np.stack(blocks, axis=2)         # (E, C, n_feats)
    X_block = stacked.reshape(E, C * len(feat_funcs))
    # Rebuild names in same channel-major order
    names_ordered = [
        f"{ch}_{fn}" for ch in ch_names for fn in feat_funcs.keys()
    ]
    return X_block.astype(np.float32), names_ordered


def extract_features(
    epochs,
    picks_hbo: str = "hbo",
    picks_hbr: str = "hbr",
    feature_funcs: dict[str, callable] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract features from all epochs for both HbO and HbR.

    All six features are computed with vectorised NumPy — O(1) Python loops
    regardless of the number of channels or epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Preprocessed epochs.
    picks_hbo, picks_hbr : str
        Channel type strings for MNE's ``get_data()``.
    feature_funcs : dict | None
        Custom feature registry.  Defaults to the six standard features.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_features)
    feature_names : list[str]
    """
    if feature_funcs is None:
        feature_funcs = VECTORISED_FEATURES

    # ---- channel names (excluding bads) ----------------------------------
    bads    = set(epochs.info.get("bads", []))
    all_ch  = epochs.ch_names
    ch_hbo  = [c for c in all_ch if "hbo" in c.lower() and c not in bads]
    ch_hbr  = [c for c in all_ch if "hbr" in c.lower() and c not in bads]

    # ---- raw data --------------------------------------------------------
    data_hbo = epochs.get_data(picks=picks_hbo)  # (E, n_hbo, T)
    data_hbr = epochs.get_data(picks=picks_hbr)  # (E, n_hbr, T)

    assert data_hbo.shape[1] == len(ch_hbo), (
        f"HbO shape mismatch: {data_hbo.shape[1]} vs {len(ch_hbo)} names")
    assert data_hbr.shape[1] == len(ch_hbr), (
        f"HbR shape mismatch: {data_hbr.shape[1]} vs {len(ch_hbr)} names")

    # ---- vectorised feature extraction -----------------------------------
    X_hbo, names_hbo = _extract_from_block(data_hbo, ch_hbo, feature_funcs)
    X_hbr, names_hbr = _extract_from_block(data_hbr, ch_hbr, feature_funcs)

    X = np.concatenate([X_hbo, X_hbr], axis=1)
    feature_names = names_hbo + names_hbr

    logger.info(
        "Feature extraction complete: %d epochs x %d features.",
        X.shape[0], X.shape[1],
    )
    return X, feature_names


# Keep for backwards compatibility with older call sites
def extract_features_single_epoch(
    epoch_data: np.ndarray,
    ch_names: list[str],
    feature_funcs: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Thin wrapper: extract features from a single (n_ch, n_t) array."""
    data_3d = epoch_data[np.newaxis, ...]   # (1, C, T)
    X, names = _extract_from_block(data_3d, ch_names, feature_funcs)
    return X[0], names


def features_to_dataframe(
    X: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray | None = None,
    subject_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert feature matrix to a labelled DataFrame."""
    df = pd.DataFrame(X, columns=feature_names)
    if labels is not None:
        df.insert(0, "label", labels)
    if subject_ids is not None:
        df.insert(0, "subject_id", subject_ids)
    return df
