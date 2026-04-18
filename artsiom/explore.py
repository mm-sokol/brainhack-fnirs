"""
explore.py
==========
Generates a visual report of the fNIRS data at **every preprocessing stage**,
plus dataset-level statistics.

All figures are saved to  reports/figures/artsiom/

Usage
-----
    uv run python -m artsiom.explore \
        --sitting_dir  "data/Sitting-20260416T142558Z-3-001/Sitting" \
        --walking_dir  "data/Walking-20260416T142602Z-3-001/Walking" \
        --subject_idx  0        # which participant to use for single-subject plots
"""

from __future__ import annotations

import argparse
import logging
import warnings
from itertools import compress
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
import numpy as np

warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

logger = logging.getLogger(__name__)

OUT_DIR = Path("reports/figures/artsiom")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
COL_HBO   = "#E63946"   # red
COL_HBR   = "#1D3557"   # dark blue
COL_REST  = "#457B9D"
COL_TASK  = "#E9C46A"
BG        = "#F8F9FA"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] saved -> {path}")


def _shade_events(ax: plt.Axes, raw: mne.io.BaseRaw, tmax_s: float = 600) -> None:
    """Draw coloured spans for mental / rest blocks (first tmax_s seconds)."""
    for ann in raw.annotations:
        onset = ann["onset"]
        if onset > tmax_s:
            break
        dur   = ann["duration"]
        label = ann["description"]
        colour = COL_TASK if label == "mental" else COL_REST
        ax.axvspan(onset, onset + dur, alpha=0.15, color=colour, linewidth=0)


def _load_and_stage(snirf_path: Path):
    """Return dict of raw objects at each preprocessing stage."""
    from artsiom.data_loader import EVENT_ID_MAP, DEFAULT_STIM_DURATION
    import mne

    # --- Stage 0: raw intensity ---
    raw_int = mne.io.read_raw_snirf(str(snirf_path), preload=True, verbose=False)

    # remap annotations
    keep, new_desc = [], []
    for i, d in enumerate(raw_int.annotations.description):
        mapped = EVENT_ID_MAP.get(str(d).strip())
        if mapped:
            keep.append(i)
            new_desc.append(mapped)
    onsets    = raw_int.annotations.onset[keep]
    durations = raw_int.annotations.duration[keep].copy()
    durations[durations < 0.1] = DEFAULT_STIM_DURATION
    raw_int.set_annotations(mne.Annotations(onsets, durations, new_desc,
                                            orig_time=raw_int.annotations.orig_time))

    # --- Stage 1: optical density ---
    raw_od = mne.preprocessing.nirs.optical_density(raw_int.copy())

    # SCI
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    bad_chs = list(compress(raw_od.ch_names, sci < 0.7))
    raw_od.info["bads"] = bad_chs

    # --- Stage 2: haemoglobin ---
    raw_hb = mne.preprocessing.nirs.beer_lambert_law(raw_od.copy(), ppf=0.1)

    # --- Stage 3: filtered ---
    raw_filt = raw_hb.copy().filter(
        l_freq=0.01, h_freq=0.3, method="iir",
        iir_params=dict(order=4, ftype="butter"), verbose=False
    )

    return dict(raw_int=raw_int, raw_od=raw_od,
                sci=sci, bad_chs=bad_chs,
                raw_hb=raw_hb, raw_filt=raw_filt)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 – dataset overview (all subjects)
# ═════════════════════════════════════════════════════════════════════════════

def fig_dataset_overview(sitting_dir: Path, walking_dir: Path) -> None:
    print("\n[1/7] Dataset overview ...")
    # Collect all SNIRF files by scanning from data root
    data_root = sitting_dir.parent if sitting_dir.name in ("Sitting", "SITTING") else Path("data")
    all_snirfs = sorted(data_root.rglob("*.snirf"))
    sit_files = [f for f in all_snirfs if "SITTING" in f.name.upper()]
    wal_files = [f for f in all_snirfs if "WALKING" in f.name.upper()]

    import re
    ids_sit = set(m.group(1) for f in sit_files for m in [re.search(r'_(\d+)_', f.name)] if m)
    ids_wal = set(m.group(1) for f in wal_files for m in [re.search(r'_(\d+)_', f.name)] if m)

    # file sizes as proxy for recording length
    sit_sizes = [f.stat().st_size / 1e6 for f in sit_files]
    wal_sizes = [f.stat().st_size / 1e6 for f in wal_files]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Dataset Overview", fontsize=13, fontweight="bold", y=1.01)

    # bar: number of files
    ax = axes[0]
    ax.bar(["Sitting (ST)", "Walking (DT)"],
           [len(ids_sit), len(ids_wal)],
           color=[COL_REST, COL_TASK], width=0.5)
    ax.set_title("Unique participants per condition")
    ax.set_ylabel("Count")
    for bar, v in zip(ax.patches, [len(ids_sit), len(ids_wal)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha="center", fontweight="bold")

    ax = axes[1]
    sit_sizes = [f.stat().st_size / 1e6 for f in sit_files]
    wal_sizes = [f.stat().st_size / 1e6 for f in wal_files]
    ax.hist(sit_sizes, bins=15, alpha=0.7, color=COL_REST, label=f"Sitting (n={len(sit_files)})")
    ax.hist(wal_sizes, bins=15, alpha=0.7, color=COL_TASK, label=f"Walking (n={len(wal_files)})")
    ax.set_title("SNIRF file size distribution")
    ax.set_xlabel("File size (MB)")
    ax.set_ylabel("Count")
    ax.legend()

    # trial structure schematic
    ax = axes[2]
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title("Trial structure (1 participant, 40 blocks)")
    colours = [COL_TASK if i % 2 == 0 else COL_REST for i in range(40)]
    for i, c in enumerate(colours):
        ax.barh(0.5, 1, left=i, height=0.6, color=c, alpha=0.85)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL_TASK, label="Mental (19 s)"),
                        Patch(color=COL_REST, label="Rest (19 s)")],
              loc="upper right")
    ax.set_xlabel("Block index (0–39)")
    ax.set_yticks([])

    fig.tight_layout()
    _save(fig, "1_dataset_overview.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 – raw intensity signal
# ═════════════════════════════════════════════════════════════════════════════

def fig_raw_intensity(stages: dict, subject_name: str) -> None:
    print("[2/7] Raw intensity ...")
    raw = stages["raw_int"]
    data, times = raw.get_data(return_times=True)
    tmax = min(600, times[-1])
    mask = times <= tmax
    chs  = raw.ch_names[:6]          # first 6 channels

    fig, axes = plt.subplots(len(chs), 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Stage 1 – Raw CW Amplitude  |  {subject_name}", fontsize=12, fontweight="bold")

    for i, ch in enumerate(chs):
        idx = raw.ch_names.index(ch)
        ax  = axes[i]
        ax.plot(times[mask], data[idx, mask], lw=0.6, color="#555")
        _shade_events(ax, raw, tmax_s=tmax)
        ax.set_ylabel(ch, fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, "2_raw_intensity.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 – optical density + SCI
# ═════════════════════════════════════════════════════════════════════════════

def fig_optical_density_sci(stages: dict, subject_name: str) -> None:
    print("[3/7] Optical density + SCI ...")
    raw_od = stages["raw_od"]
    sci    = stages["sci"]
    bad    = stages["bad_chs"]

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[2, 1], figure=fig)
    fig.suptitle(f"Stage 2 – Optical Density & SCI  |  {subject_name}",
                 fontsize=12, fontweight="bold")

    # OD timeseries (top left)
    ax0 = fig.add_subplot(gs[0, 0])
    data, times = raw_od.get_data(return_times=True)
    tmax = min(600, times[-1])
    mask = times <= tmax
    for i in range(min(6, data.shape[0])):
        ax0.plot(times[mask], data[i, mask], lw=0.5, alpha=0.7)
    _shade_events(ax0, raw_od, tmax_s=tmax)
    ax0.set_title("OD – first 6 channels (first 10 min)")
    ax0.set_ylabel("OD (a.u.)")

    # OD mean per channel (bottom left)
    ax1 = fig.add_subplot(gs[1, 0])
    ch_means = data.mean(axis=1)
    colours = ["#E63946" if c in bad else "#2a9d8f" for c in raw_od.ch_names]
    ax1.bar(range(len(ch_means)), ch_means, color=colours, width=0.8)
    ax1.set_title("Mean OD per channel  (red = SCI-bad)")
    ax1.set_xlabel("Channel index")
    ax1.set_ylabel("Mean OD")

    # SCI histogram (right, spanning both rows)
    ax2 = fig.add_subplot(gs[:, 1])
    colours_sci = ["#E63946" if s < 0.7 else "#2a9d8f" for s in sci]
    ax2.barh(range(len(sci)), sci, color=colours_sci, height=0.8)
    ax2.axvline(0.7, color="k", ls="--", lw=1.5, label="threshold 0.7")
    ax2.set_title("Scalp Coupling Index\nper channel")
    ax2.set_xlabel("SCI")
    ax2.set_ylabel("Channel index")
    ax2.legend(fontsize=8)
    n_bad = sum(s < 0.7 for s in sci)
    ax2.text(0.72, len(sci)-1, f"{n_bad} bad channels",
             color="#E63946", fontsize=9, va="top")

    fig.tight_layout()
    _save(fig, "3_optical_density_sci.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 – HbO / HbR after MBLL
# ═════════════════════════════════════════════════════════════════════════════

def fig_haemoglobin(stages: dict, subject_name: str) -> None:
    print("[4/7] HbO / HbR (MBLL) ...")
    raw_hb = stages["raw_hb"]
    data, times = raw_hb.get_data(return_times=True)
    tmax = min(600, times[-1])
    mask = times <= tmax

    hbo_idx = [i for i, c in enumerate(raw_hb.ch_names) if "hbo" in c][:4]
    hbr_idx = [i for i, c in enumerate(raw_hb.ch_names) if "hbr" in c][:4]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"Stage 3 – Haemoglobin Concentration (MBLL)  |  {subject_name}",
                 fontsize=12, fontweight="bold")

    for i, idx in enumerate(hbo_idx):
        axes[0].plot(times[mask], data[idx, mask]*1e6, lw=0.7,
                     color=COL_HBO, alpha=0.6 + 0.1*i,
                     label=raw_hb.ch_names[idx] if i == 0 else "")
    _shade_events(axes[0], raw_hb, tmax_s=tmax)
    axes[0].set_title("HbO (first 4 channels)")
    axes[0].set_ylabel("Concentration (µM)")

    for i, idx in enumerate(hbr_idx):
        axes[1].plot(times[mask], data[idx, mask]*1e6, lw=0.7,
                     color=COL_HBR, alpha=0.6 + 0.1*i)
    _shade_events(axes[1], raw_hb, tmax_s=tmax)
    axes[1].set_title("HbR (first 4 channels)")
    axes[1].set_ylabel("Concentration (µM)")
    axes[1].set_xlabel("Time (s)")

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL_TASK, alpha=0.3, label="Mental"),
                         Patch(color=COL_REST, alpha=0.3, label="Rest")],
               loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, "4_haemoglobin_mbll.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 – before vs after IIR filter (PSD)
# ═════════════════════════════════════════════════════════════════════════════

def fig_filter_psd(stages: dict, subject_name: str) -> None:
    print("[5/7] Filter comparison (PSD) ...")
    raw_hb   = stages["raw_hb"]
    raw_filt = stages["raw_filt"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    fig.suptitle(f"Stage 4 – IIR Band-pass Filter Effect  |  {subject_name}",
                 fontsize=12, fontweight="bold")

    for ax, raw, label in zip(axes, [raw_hb, raw_filt],
                               ["Before filter", "After filter (0.01–0.3 Hz)"]):
        hbo_idx = mne.pick_types(raw.info, fnirs=True)
        psd = raw.compute_psd(method="welch", fmin=0.001, fmax=2.0,
                               picks=hbo_idx, verbose=False)
        freqs = psd.freqs
        power = 10 * np.log10(psd.get_data().mean(axis=0) + 1e-30)
        ax.plot(freqs, power, color=COL_HBO, lw=1.2, label="HbO")
        ax.axvspan(0.01, 0.3, alpha=0.12, color="green", label="Pass band")
        ax.axvline(0.01, color="green", ls="--", lw=1)
        ax.axvline(0.30, color="green", ls="--", lw=1)
        ax.set_title(label)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_xlim(0, 1.5)
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "5_filter_psd.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 6 – epoched ERPs (mental vs rest)
# ═════════════════════════════════════════════════════════════════════════════

def fig_epochs_erp(stages: dict, subject_name: str) -> None:
    print("[6/7] Epoched ERP (mental vs rest) ...")
    from artsiom.preprocessing import epoch_data
    from artsiom.data_loader import EPOCH_EVENT_ID

    raw_filt = stages["raw_filt"]
    try:
        epochs = epoch_data(raw_filt, tmin=-2, tmax=19, reject=None)
    except Exception as e:
        print(f"    [WARN]  epoching failed: {e}")
        return

    if len(epochs) == 0:
        print("    [WARN]  no valid epochs found")
        return

    hbo_chs = [c for c in epochs.ch_names if "hbo" in c]
    hbr_chs = [c for c in epochs.ch_names if "hbr" in c]
    times   = epochs.times

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle(f"Stage 5 – Epoched HRF  |  {subject_name}",
                 fontsize=12, fontweight="bold")

    for row, (chs, chtype, colour) in enumerate([
        (hbo_chs, "HbO", COL_HBO),
        (hbr_chs, "HbR", COL_HBR),
    ]):
        for col, cond in enumerate(["mental", "rest"]):
            ax = axes[row][col]
            if cond not in epochs.event_id:
                ax.set_visible(False)
                continue
            ep = epochs[cond].get_data(picks=chs)  # (n_ep, n_ch, n_t)
            mean_signal = ep.mean(axis=(0, 1)) * 1e6   # µM
            sem_signal  = ep.mean(axis=1).std(axis=0) * 1e6 / np.sqrt(len(ep))

            ax.plot(times, mean_signal, color=colour, lw=1.8,
                    label=f"mean ± SEM  (n={len(ep)})")
            ax.fill_between(times,
                            mean_signal - sem_signal,
                            mean_signal + sem_signal,
                            alpha=0.25, color=colour)
            ax.axvline(0, ls="--", color="k", lw=1)
            ax.axhline(0, ls=":", color="gray", lw=0.8)
            ax.set_title(f"{chtype}   |   {cond.capitalize()}")
            ax.set_xlabel("Time rel. onset (s)")
            ax.set_ylabel("Concentration (µM)")
            ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "6_epochs_erp.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 7 – feature distribution across all subjects (fast, 3 subjects)
# ═════════════════════════════════════════════════════════════════════════════

def fig_feature_distribution(sitting_dir: Path, n_subjects: int = 3) -> None:
    print(f"[7/7] Feature distributions ({n_subjects} subjects) ...")
    from artsiom.data_loader import BHMentalDatabaseHelper
    from artsiom.preprocessing import run_preprocessing_pipeline
    from artsiom.features import FEATURE_FUNCS
    from artsiom.train_baseline import LABEL_MAP

    helper = BHMentalDatabaseHelper(sitting_dir, condition="SITTING")

    # Compute channel-averaged stats per epoch so feature size is fixed
    # regardless of how many channels survive SCI rejection.
    feat_labels = list(FEATURE_FUNCS.keys())   # mean, std, peak, slope, skew, kurt
    from scipy import stats as _stats
    import numpy as _np

    def _epoch_summary(data_3d):
        """data_3d: (n_epochs, n_channels, n_times) -> (n_epochs, 6) averaged features"""
        n_ep, n_ch, n_t = data_3d.shape
        out = _np.zeros((n_ep, 6))
        for e in range(n_ep):
            d = data_3d[e]            # (n_ch, n_t)
            out[e, 0] = d.mean()
            out[e, 1] = d.std()
            out[e, 2] = _np.abs(d).max()
            x = _np.arange(n_t, dtype=float)
            slopes = [_np.polyfit(x, d[c], 1)[0] for c in range(n_ch)]
            out[e, 3] = _np.mean(slopes)
            out[e, 4] = _stats.skew(d.ravel())
            out[e, 5] = _stats.kurtosis(d.ravel())
        return out

    all_X_hbo, all_X_hbr, all_y = [], [], []

    for sid in range(min(n_subjects, len(helper.get_subject_list()))):
        try:
            raw    = helper.load_subject(sid)
            epochs = run_preprocessing_pipeline(raw)
            if len(epochs) == 0:
                continue
            data_hbo = epochs.get_data(picks="hbo")
            data_hbr = epochs.get_data(picks="hbr")
            inv  = {v: k for k, v in epochs.event_id.items()}
            y    = _np.array([LABEL_MAP.get(inv.get(c, ""), -1)
                              for c in epochs.events[:, 2]])
            mask = y >= 0
            all_X_hbo.append(_epoch_summary(data_hbo[mask]))
            all_X_hbr.append(_epoch_summary(data_hbr[mask]))
            all_y.append(y[mask])
        except Exception as e:
            print(f"    subject {sid} failed: {e}")

    if not all_X_hbo:
        print("    [WARN]  no data collected")
        return

    X_hbo = np.concatenate(all_X_hbo, axis=0)   # (total_epochs, 6)
    X_hbr = np.concatenate(all_X_hbr, axis=0)
    y     = np.concatenate(all_y, axis=0)

    # 6 subplots: 3 HbO features + 3 HbR features
    feat_sel = [(X_hbo, "HbO"), (X_hbo, "HbO"), (X_hbo, "HbO"),
                (X_hbr, "HbR"), (X_hbr, "HbR"), (X_hbr, "HbR")]
    feat_idx = [0, 1, 3, 0, 1, 3]   # mean, std, slope for each chrom
    feat_sfx = ["mean", "std", "slope", "mean", "std", "slope"]
    feature_sel = list(zip(feat_sel, feat_idx, feat_sfx))

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle(f"Feature Distributions – Mental vs Rest  ({n_subjects} subjects)",
                 fontsize=12, fontweight="bold")

    for ax, ((X_mat, chrom), fi, sfx) in zip(axes.flat, feature_sel):
        vals_m = X_mat[y == 1, fi]
        vals_r = X_mat[y == 0, fi]
        ax.hist(vals_r, bins=20, alpha=0.6, color=COL_REST,  label="Rest",   density=True)
        ax.hist(vals_m, bins=20, alpha=0.6, color=COL_TASK,  label="Mental", density=True)
        ax.set_title(f"{chrom} – {sfx} (channel avg)", fontsize=9)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "7_feature_distributions.png")


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="fNIRS data exploration report.")
    parser.add_argument("--sitting_dir", type=str,
        default="data",
        help="Root data dir or specific sitting dir (SNIRF files found recursively).")
    parser.add_argument("--walking_dir", type=str,
        default="data",
        help="Root data dir or specific walking dir (SNIRF files found recursively).")
    parser.add_argument("--subject_idx", type=int, default=0,
        help="Index (0-based) of the participant used for single-subject plots.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    sitting_dir = Path(args.sitting_dir)
    walking_dir = Path(args.walking_dir)

    # Find all sitting SNIRF files across the whole data/ tree
    snirf_files = sorted(
        f for f in sitting_dir.rglob("*.snirf") if "SITTING" in f.name.upper()
    )
    if not snirf_files:
        # fallback: any snirf in the tree
        snirf_files = sorted(sitting_dir.rglob("*.snirf"))
    if not snirf_files:
        print(f"ERROR: no SNIRF files found under {sitting_dir}")
        return

    chosen = snirf_files[args.subject_idx]
    subject_name = chosen.stem.split("_")[4] if len(chosen.stem.split("_")) > 4 else chosen.stem
    print(f"\n{'='*60}")
    print(f"  fNIRS Exploration Report")
    print(f"  Single-subject example: {chosen.name}")
    print(f"  Output directory:       {OUT_DIR.resolve()}")
    print(f"{'='*60}\n")

    # ── load all stages for one subject ────────────────────────────────────
    print("[0/7] Loading & staging subject ...")
    stages = _load_and_stage(chosen)
    print(f"       Bad channels (SCI<0.7): {stages['bad_chs'] or 'none'}")

    # ── figures ────────────────────────────────────────────────────────────
    fig_dataset_overview(sitting_dir, walking_dir)
    fig_raw_intensity(stages, subject_name)
    fig_optical_density_sci(stages, subject_name)
    fig_haemoglobin(stages, subject_name)
    fig_filter_psd(stages, subject_name)
    fig_epochs_erp(stages, subject_name)
    fig_feature_distribution(sitting_dir, n_subjects=5)

    print(f"\n[DONE]  Done!  All figures saved to  {OUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
