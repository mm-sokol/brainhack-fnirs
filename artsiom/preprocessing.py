"""
preprocessing.py
================
fNIRS preprocessing pipeline following Arek's recommendations:

    1. Channel rejection based on Scalp Coupling Index (SCI ≥ 0.7)
    2. Optical Density (OD) calculation
    3. Modified Beer-Lambert Law (MBLL) -> HbO / HbR
    4. IIR band-pass filter  0.01 – 0.3 Hz
    5. Epoching around task onsets

Each step is exposed as an independent function so that it can be unit-tested
or swapped out without touching the rest of the pipeline.

Usage
-----
    from artsiom.preprocessing import run_preprocessing_pipeline
    from artsiom.data_loader import BHMentalDatabaseHelper, EPOCH_EVENT_ID

    helper = BHMentalDatabaseHelper("data/")
    raw = helper.load_subject(0)
    epochs = run_preprocessing_pipeline(raw)
"""

from __future__ import annotations

import logging
from itertools import compress

import mne
import numpy as np

from artsiom.data_loader import EPOCH_EVENT_ID, EPOCH_TMAX, EPOCH_TMIN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters (override by passing kwargs to each function)
# ---------------------------------------------------------------------------
SCI_THRESHOLD: float = 0.7          # keep channels with SCI ≥ threshold
PPF: float = 0.1                    # partial path-length factor for MBLL
L_FREQ: float = 0.01                # IIR high-pass cut-off (Hz)
H_FREQ: float = 0.3                 # IIR low-pass cut-off  (Hz)
REJECT_CRITERIA: dict = dict(hbo=80e-6, hbr=80e-6)  # µM artifact threshold


# ---------------------------------------------------------------------------
# Step 1 – SCI-based channel rejection
# ---------------------------------------------------------------------------

def reject_channels_by_sci(
    raw_od: mne.io.BaseRaw,
    threshold: float = SCI_THRESHOLD,
) -> mne.io.BaseRaw:
    """Mark channels with SCI < *threshold* as bad.

    Parameters
    ----------
    raw_od : mne.io.BaseRaw
        Raw data already converted to optical density.
    threshold : float
        Channels whose SCI is **below** this value are marked bad.
        Arek's recommendation: 0.7.

    Returns
    -------
    mne.io.BaseRaw
        Same object (mutated in place) with ``raw_od.info["bads"]`` updated.
    """
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    bad_channels = list(compress(raw_od.ch_names, sci < threshold))

    n_bad = len(bad_channels)
    n_total = len(raw_od.ch_names)
    logger.info(
        "SCI rejection (threshold=%.2f): %d / %d channels marked bad: %s",
        threshold,
        n_bad,
        n_total,
        bad_channels,
    )

    raw_od.info["bads"] = bad_channels
    return raw_od


# ---------------------------------------------------------------------------
# Step 2 – Intensity -> Optical Density
# ---------------------------------------------------------------------------

def intensity_to_od(raw_intensity: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Convert raw intensity to optical density.

    Parameters
    ----------
    raw_intensity : mne.io.BaseRaw
        Raw data with channel type ``fnirs_cw_amplitude``.

    Returns
    -------
    mne.io.BaseRaw
        New raw object with channel type ``fnirs_od``.
    """
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    logger.info("Converted intensity to optical density.")
    return raw_od


# ---------------------------------------------------------------------------
# Step 3 – MBLL: OD -> haemoglobin concentrations
# ---------------------------------------------------------------------------

def od_to_haemo(raw_od: mne.io.BaseRaw, ppf: float = PPF) -> mne.io.BaseRaw:
    """Apply Modified Beer-Lambert Law to get HbO / HbR concentrations.

    Parameters
    ----------
    raw_od : mne.io.BaseRaw
        Optical density data (bad channels already set).
    ppf : float
        Partial path-length factor (default 0.1 as in MNE tutorials).

    Returns
    -------
    mne.io.BaseRaw
        Raw object with channel types ``hbo`` and ``hbr``.
    """
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=ppf)
    logger.info("Applied MBLL (ppf=%.2f). Channels: %s", ppf, raw_haemo.ch_names[:4])
    return raw_haemo


# ---------------------------------------------------------------------------
# Step 4 – IIR band-pass filter
# ---------------------------------------------------------------------------

def apply_bandpass_filter(
    raw_haemo: mne.io.BaseRaw,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
) -> mne.io.BaseRaw:
    """Apply an IIR (Butterworth) band-pass filter in place.

    Parameters
    ----------
    raw_haemo : mne.io.BaseRaw
        Haemoglobin concentration data.
    l_freq : float
        High-pass cut-off in Hz (default 0.01).
    h_freq : float
        Low-pass cut-off in Hz (default 0.3).

    Returns
    -------
    mne.io.BaseRaw
        Filtered data (mutated in place).
    """
    raw_haemo.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )
    logger.info("Applied IIR band-pass filter [%.3f – %.3f Hz].", l_freq, h_freq)
    return raw_haemo


# ---------------------------------------------------------------------------
# Step 5 – Epoching
# ---------------------------------------------------------------------------

def epoch_data(
    raw_haemo: mne.io.BaseRaw,
    event_id: dict[str, int] | None = None,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    reject: dict | None = None,
    baseline: tuple | None = (None, 0),
    reject_by_annotation: bool = False,
) -> mne.Epochs:
    """Segment the continuous signal into epochs.

    Parameters
    ----------
    raw_haemo : mne.io.BaseRaw
        Filtered haemoglobin concentration data with annotations.
    event_id : dict | None
        Mapping of label -> event code.  Defaults to ``EPOCH_EVENT_ID``.
    tmin, tmax : float
        Epoch window in seconds relative to stimulus onset.
        Defaults: -2 s to +15 s.
    reject : dict | None
        Amplitude-based rejection thresholds (µM).  Set to ``None`` to
        disable amplitude rejection.
    baseline : tuple | None
        Baseline correction interval.  Default ``(None, 0)`` means from
        the start of the epoch to the onset.

    Returns
    -------
    mne.Epochs
        Epoched data.  Bad epochs are dropped automatically.
    """
    if event_id is None:
        event_id = EPOCH_EVENT_ID
    # NOTE: reject=None means no amplitude rejection (deliberate, not a bug).
    # Pass REJECT_CRITERIA explicitly if you want amplitude-based dropping.

    events, found_event_id = mne.events_from_annotations(raw_haemo, verbose=False)

    # Keep only events whose labels appear in our event_id dict
    valid_labels = {v: k for k, v in found_event_id.items() if k in
                    set(event_id.keys())}
    if not valid_labels:
        logger.warning(
            "No matching event labels found. Expected %s, got %s.",
            set(event_id.keys()),
            set(found_event_id.keys()),
        )
        return mne.Epochs.__new__(mne.Epochs)

    # Remap found_event_id to match our integer codes
    remapped = {label: event_id[label] for label in found_event_id
                if label in event_id}

    epochs = mne.Epochs(
        raw_haemo,
        events,
        event_id=remapped,
        tmin=tmin,
        tmax=tmax,
        reject=reject,
        reject_by_annotation=reject_by_annotation,
        proj=False,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    n_dropped = len(epochs.drop_log) - len(epochs)
    logger.info(
        "Epoching complete: %d epochs kept, %d dropped. "
        "Event counts: %s",
        len(epochs),
        n_dropped,
        {k: (epochs.events[:, 2] == v).sum() for k, v in remapped.items()},
    )
    return epochs


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    raw_intensity: mne.io.BaseRaw,
    sci_threshold: float = SCI_THRESHOLD,
    ppf: float = PPF,
    l_freq: float = L_FREQ,
    h_freq: float = H_FREQ,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    reject: dict | None = None,
) -> mne.Epochs:
    """Run the complete preprocessing pipeline on one subject's raw data.

    Steps
    -----
    1. Convert intensity -> OD
    2. SCI-based channel rejection
    3. OD -> haemoglobin (MBLL)
    4. IIR band-pass filter
    5. Epoching

    Parameters
    ----------
    raw_intensity : mne.io.BaseRaw
        Raw CW-amplitude data as returned by :class:`BHMentalDatabaseHelper`.

    Returns
    -------
    mne.Epochs
        Ready-to-use epoch object.
    """
    logger.info("=== Preprocessing pipeline start ===")

    # Step 1 & 2: OD + SCI rejection
    raw_od = intensity_to_od(raw_intensity)
    raw_od = reject_channels_by_sci(raw_od, threshold=sci_threshold)

    # Step 3: MBLL
    raw_haemo = od_to_haemo(raw_od, ppf=ppf)

    # Step 4: Filter
    raw_haemo = apply_bandpass_filter(raw_haemo, l_freq=l_freq, h_freq=h_freq)

    # Step 5: Epoch
    epochs = epoch_data(raw_haemo, tmin=tmin, tmax=tmax, reject=reject)

    logger.info("=== Preprocessing pipeline complete ===")
    return epochs
