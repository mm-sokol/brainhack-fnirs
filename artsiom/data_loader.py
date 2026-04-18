"""
data_loader.py
==============
Loads fNIRS data from SNIRF files and maps event codes to meaningful labels.

Event mapping (as specified by team lead):
    1 -> "mental"  (mental calculation / cognitive load)
    2 -> "rest"    (rest / baseline)

Usage
-----
    from artsiom.data_loader import BHMentalDatabaseHelper

    helper = BHMentalDatabaseHelper(data_dir="path/to/snirf/files")
    subjects = helper.get_subject_list()
    raw = helper.load_subject(subject_id=0)
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event mapping
# ---------------------------------------------------------------------------
EVENT_ID_MAP: dict[str, str] = {
    "1": "mental",
    "2": "rest",
    "1.0": "mental",
    "2.0": "rest",
}

# Numeric event-id dict used by mne.Epochs
EPOCH_EVENT_ID: dict[str, int] = {
    "mental": 1,
    "rest": 2,
}

# Default epoch duration in seconds relative to onset
EPOCH_TMIN: float = -2.0
EPOCH_TMAX: float = 19.0

# Default stimulus duration (seconds) – used when annotations have no duration
# Paper: "Epoching (19 s)" — 20 MC + 20 REST × 19 s = 760 s ≈ 12.7 min per session
DEFAULT_STIM_DURATION: float = 19.0


# ---------------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------------
class BHMentalDatabaseHelper:
    """Utility class for reading SNIRF participant files and mapping events.

    Parameters
    ----------
    data_dir : str | Path
        Root directory that contains SNIRF files.  The helper scans
        recursively for ``*.snirf`` files and sorts them by name so that
        ``subject_id`` maps predictably to a file.
    condition : str | None
        Optional filter: ``"ST"`` (sitting) or ``"DT"`` (dual-task / walking).
        When set, only files whose path contains the condition string are
        returned.  ``None`` means all files.
    """

    def __init__(self, data_dir: str | Path, condition: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.condition = condition
        self._files: list[Path] = self._discover_files()

        if not self._files:
            logger.warning(
                "No SNIRF files found under '%s' (condition=%s)",
                self.data_dir,
                condition,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_subject_list(self) -> list[int]:
        """Return list of integer subject indices (0-based)."""
        return list(range(len(self._files)))

    def get_subject_path(self, subject_id: int) -> Path:
        """Return the file path for a given subject index."""
        return self._files[subject_id]

    def load_subject(
        self,
        subject_id: int,
        preload: bool = True,
        verbose: bool = False,
    ) -> mne.io.BaseRaw:
        """Load a single subject's raw data from SNIRF.

        Parameters
        ----------
        subject_id : int
            Zero-based index into the sorted list of SNIRF files.
        preload : bool
            Whether to preload data into memory (default ``True``).
        verbose : bool
            MNE verbosity.

        Returns
        -------
        mne.io.BaseRaw
            Raw fNIRS object with annotations renamed to ``"mental"`` /
            ``"rest"``.  Unknown annotation codes are dropped.
        """
        path = self._files[subject_id]
        logger.info("Loading subject %d from: %s", subject_id, path)

        raw = mne.io.read_raw_snirf(str(path), preload=preload, verbose=verbose)
        raw = self._remap_annotations(raw)
        return raw

    def load_all(
        self,
        verbose: bool = False,
    ) -> list[tuple[int, mne.io.BaseRaw]]:
        """Load all subjects.

        Returns
        -------
        list of (subject_id, raw) tuples.
        """
        result = []
        for sid in self.get_subject_list():
            try:
                raw = self.load_subject(sid, verbose=verbose)
                result.append((sid, raw))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load subject %d: %s", sid, exc)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_files(self) -> list[Path]:
        files = sorted(self.data_dir.rglob("*.snirf"))
        if self.condition:
            files = [f for f in files if self.condition in str(f)]
        return files

    @staticmethod
    def _remap_annotations(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Rename annotation descriptions and drop unknown codes.

        Also fills in ``DEFAULT_STIM_DURATION`` for any annotation whose
        duration is 0 (or very close to 0).
        """
        keep_indices: list[int] = []
        new_descriptions: list[str] = []

        for i, desc in enumerate(raw.annotations.description):
            mapped = EVENT_ID_MAP.get(str(desc).strip())
            if mapped is not None:
                keep_indices.append(i)
                new_descriptions.append(mapped)

        # Build new annotations from scratch (keeps only valid events)
        onsets = raw.annotations.onset[keep_indices]
        durations = raw.annotations.duration[keep_indices]
        # Fill zero durations with the default stimulus duration
        durations = durations.copy()
        durations[durations < 0.1] = DEFAULT_STIM_DURATION

        new_annots = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=new_descriptions,
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(new_annots)
        logger.debug(
            "Annotations remapped: %d events kept out of original set.",
            len(keep_indices),
        )
        return raw
