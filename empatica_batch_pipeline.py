#!/usr/bin/env python3
"""
Empatica (E4 / EmbracePlus-style) batch preprocessing + analysis aligned to a protocol timeline.

v12 (physiology-paper upgrades + plot correctness):
- Keeps legend placement consistent with v8 for ALL plots.
- Correlation heatmaps are fully standardized (title schema + figsize + typography).
- Front-matter and summary text is wrapped to never cross page borders.
- Block summary tables are scaled to stay within page bounds.
- Participant pages in the all-sessions report follow the requested title schema.

Core pipeline features:
- Timeline is REQUIRED: you must pass --timeline-csv (protocol timeline).
- Minute features expanded with robust statistics (mean/std/iqr/p95 where meaningful) and within-minute variability.
- QC becomes legible: minute-level QC flags are visualized as a heatmap (minutes × flags) + validity summary.
- Time-series plotting is QC-aware: one primary signal line (valid minutes) + grey context for all minutes; no confusing double lines.
- Phase-level interpretation pages: phase-wise distributions + baseline-normalized deltas (where baseline exists).
- Cross-modal relationships: Spearman correlation matrix on phase means + scatter examples.
- PDF remains publication-ready: one figure per page, consistent head/foot room, and session_id in every title.

Outputs per session:
- features_minute.csv
- features_phase.csv
- minute_qc_flags.csv
- signal_qc_summary.csv
- qc_summary.json
- output_schema.json (+ .py)
- report.pdf

Run example (single session):
  python empatica_batch_pipeline.py \
    --session-dir /path/to/P01_D01_DIM-MOR \
    --timeline-csv /path/to/timeline_by_minutes.csv \
    --outdir /path/to/results

Run example (batch):
  python empatica_batch_pipeline.py \
    --sessions-root ../datasets/transform/empatica_test \
    --timeline-csv ../datasets/transform/master_files/timeline_by_minutes.csv \
    --timeline-tz Europe/Berlin \
    --outdir ../results/empatica/



Notes:
- ENMO = Euclidean Norm Minus One (in g): max(0, sqrt(x^2+y^2+z^2) - 1).
- Plausibility ranges are for *minute-level features*, not raw samples; edit PLAUSIBILITY_RANGES for your cohort.

Author: ChatGPT (research pipeline refactor)
"""
from __future__ import annotations

import argparse
import json
import math
import datetime
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---- Plot styling defaults (publication) ----
PLOT_FONTS = {
    'TITLE': 13,
    'AXIS_LABEL': 11,
    'TICK': 10,
    'LEGEND': 9,
    'ANNOT': 9,
}

# Correlation heatmap style (keep consistent across all correlation pages)
CORR_FIGSIZE = (11, 8)
CORR_CELL_FONTSIZE = 8
# Matplotlib global defaults for paper-ready figures
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': PLOT_FONTS['TICK'],
    'axes.titlesize': PLOT_FONTS['TITLE'],
    'axes.labelsize': PLOT_FONTS['AXIS_LABEL'],
    'xtick.labelsize': PLOT_FONTS['TICK'],
    'ytick.labelsize': PLOT_FONTS['TICK'],
    'legend.fontsize': PLOT_FONTS['LEGEND'],
    'axes.titlepad': 4.0,
})

PRIMARY_BLUE = "#0072B2"  # Okabe-Ito (colorblind-safe)
SECONDARY_ORANGE = "#D55E00"
ACCENT_GREEN = "#009E73"
NEUTRAL_GRAY = "#4D4D4D"

from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch

# -------------------------------
# NeuroKit2 (required)
# -------------------------------
try:
    import neurokit2 as nk  # type: ignore
except Exception as e:
    raise SystemExit(
        "NeuroKit2 is required for this pipeline (SCR/EDA processing depends on it).\n"
        "Install it in your environment and retry:\n"
        "  pip install neurokit2\n"
        f"Original import error: {e}"
    )
HAVE_NK = True

def format_plot_title(scope: str, session_id: str, fig_title: str) -> str:
    """One-line publication title used across Matplotlib.
    scope: 'session' or 'all'

    NOTE: fig_title is expected to be the *figure descriptor* (e.g., 'Heart rate (bpm) — Mean'),
    not a pre-prefixed full title. If a caller accidentally passes a full title, we sanitize it.
    """
    # Defensive: avoid duplicated prefixes like 'Wearable Physiology Wearable Physiology ...'
    ft = str(fig_title).strip()
    for prefix in ("Wearable Physiology –", "Wearable Physiology -", "Wearable Physiology"):
        if ft.startswith(prefix):
            # Drop everything up to and including the first '|', if present
            if "|" in ft:
                ft = ft.split("|", 1)[1].strip()
            else:
                ft = ft[len(prefix):].strip(" -–|")
            break

    if scope.lower() in ("all", "all_sessions", "combined"):
        return f"Wearable Physiology – All sessions | {ft}"
    return f"Wearable Physiology – {session_id} | {ft}"





def _title_all_sessions(tail: str) -> str:
    """Standardized title schema for cohort figures."""
    tail = (tail or "").strip()
    for pref in [
        "Wearable Physiology – All Sessions |",
        "Wearable Physiology - All Sessions |",
        "Wearable Physiology – All sessions |",
        "Wearable Physiology - All sessions |",
        "Wearable Physiology All Sessions |",
        "Wearable Physiology All sessions |",
    ]:
        if tail.startswith(pref):
            tail = tail[len(pref):].strip()
    return f"Wearable Physiology All sessions | {tail}"

def _read_square_csv_matrix(path: Path):
    """Read a square matrix CSV with first column as index."""
    import pandas as pd
    df = pd.read_csv(path, index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _plot_placeholder_heatmap(pdf, title: str, message: str = "Insufficient data", figsize=(10.5, 7.5)):
    """Render a placeholder page so PDF matches HTML figure slots."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.02, 0.98, title, ha="left", va="top", fontsize=PLOT_FONTS.get("TITLE", 16), weight="bold")
    if message:
        ax.text(0.02, 0.90, message, ha="left", va="top", fontsize=PLOT_FONTS.get("SUBTITLE", 12))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

def _save_heatmap_or_placeholder(pdf, mat, title: str, subtitle: str = "", *, cmap: str = "coolwarm",
                                 vmin=None, vmax=None, annotate: bool = True, fmt: str = ".2f",
                                 figsize=(10.5, 7.5)):
    """Save a heatmap page; if mat is empty/None, save placeholder instead."""
    if mat is None:
        _plot_placeholder_heatmap(pdf, title, message=(subtitle or "Insufficient data"), figsize=figsize)
        return
    try:
        if hasattr(mat, "empty") and mat.empty:
            _plot_placeholder_heatmap(pdf, title, message=(subtitle or "Insufficient data"), figsize=figsize)
            return
    except Exception:
        pass
    plot_corr_heatmap(
        pdf,
        mat,
        title=title,
        subtitle=subtitle,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annotate=annotate,
        fmt=fmt,
        figsize=figsize,
    )


def is_all_sessions_id(session_id: str) -> bool:
    """Heuristic: treat IDs like 'all', 'all_sessions', 'combined', or already-prefixed 'Wearable Physiology – All sessions'
    as cohort scope.
    """
    s = str(session_id or "").strip().lower()
    if not s:
        return True
    # normalize dash variants
    s = s.replace("–", "-").replace("—", "-")
    return (
        s.startswith("all")
        or "all sessions" in s
        or "all_sessions" in s
        or "combined" in s
        or "cohort" in s
        or "wearable physiology - all sessions" in s
        or "wearable physiology - all session" in s
        or "wearable physiology - all" in s
    )



def format_plausibility_frontmatter_lines(
    *,
    keys: List[str] | None = None,
    indent: str = "  • ",
) -> List[str]:
    """Format plausibility ranges for PDF front matter (minute-level features).
    If keys is provided, only include those metrics (in that order when possible).
    """
    if not PLAUSIBILITY_RANGES:
        return []
    if keys:
        ordered = []
        for k in keys:
            if k in PLAUSIBILITY_RANGES and k not in ordered:
                ordered.append(k)
        # also include any remaining ranges not in keys
        for k in PLAUSIBILITY_RANGES.keys():
            if k not in ordered:
                ordered.append(k)
    else:
        ordered = list(PLAUSIBILITY_RANGES.keys())

    out: List[str] = []
    for k in ordered:
        meta = PLAUSIBILITY_RANGES.get(k)
        if not meta:
            continue
        lo = meta.get("min", None)
        hi = meta.get("max", None)
        unit = meta.get("unit", "")
        label = meta.get("label", k)
        if lo is None and hi is None:
            continue
        rng = f"[{lo}, {hi}]"
        if unit:
            rng += f" {unit}"
        out.append(f"{indent}{label}: {rng}")
    return out


def _wrap_block(lines: List[str], *, width: int = 110) -> str:
    """Wrap a list of lines for PDF front-matter pages so text never runs beyond page border.

    Keeps indentation (e.g., "  • ") while wrapping.
    """
    import textwrap

    out: List[str] = []
    for ln in lines:
        if not ln:
            out.append("")
            continue
        # Preserve leading whitespace/bullets
        m = re.match(r"^(\s*[•\-\*]?\s*)", ln)
        prefix = m.group(1) if m else ""
        body = ln[len(prefix):]
        wrapped = textwrap.fill(
            body,
            width=width,
            subsequent_indent=prefix,
            initial_indent=prefix,
            break_long_words=True,
            break_on_hyphens=True,
        )
        out.append(wrapped)
    return "\n".join(out)

def _safe_pdf_call(pdf: PdfPages, fn, *args, **kwargs) -> None:
    """Render a PDF page safely; on error, add a readable error page and continue."""
    try:
        fn(pdf, *args, **kwargs)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
        ax.axis("off")
        ax.text(
            0.02, 0.98,
            "A figure failed to render, but the report continues.\n\n"
            + f"Function: {getattr(fn, '__name__', str(fn))}\n"
            + f"Error: {type(e).__name__}: {e}",
            va="top", ha="left", fontsize=10, family="monospace"
        )
        pdf.savefig(fig)
        plt.close(fig)

# =========================================================
# Plausibility ranges (EDIT HERE)
# =========================================================
# Scientific sanity gates (minute-level). Values outside are flagged as out-of-bounds (oob),
# not automatically removed from the raw feature table.

# -------------------------------
# Tunable parameters (printed in PDF front-matter)
# -------------------------------
# Movement QC: minutes with ENMO p95 above this threshold are flagged as 'high_motion'.
HIGH_MOTION_ENMO_P95_THRESHOLD_G: float = 0.30

# Gap detection for raw sample streams (used in QC summaries).
GAP_DETECTION_THRESHOLD_SECONDS: float = 2.0

# Phase-level correlation metrics (used in reports and exported matrices).
PHASE_CORR_METRICS: List[str] = [
    "hr_bpm_mean",
    "eda_tonic_mean",
    "eda_phasic_p95",
    "temp_smooth_C_mean",
    "acc_enmo_g_p95",
    "steps_count",
    "scr_count",
]


# Minute-level correlation metrics. Defaults to the same set as phase-level correlation.
MINUTE_CORR_METRICS: List[str] = list(PHASE_CORR_METRICS)

# Correlation robustness settings (tunable; printed in report front-matter).
BOOTSTRAP_N: int = 300
BOOTSTRAP_BLOCK_MINUTES: int | str = "auto"
BOOTSTRAP_BLOCK_MINUTES_MIN: int = 5
BOOTSTRAP_BLOCK_MINUTES_MAX: int = 30
BOOTSTRAP_SEED: int = 42

# Partial correlation control variable (motion proxy). If missing, partial corr is skipped gracefully.
PARTIAL_CONTROL_VAR: str = "acc_enmo_g_p95"

# Do not draw plausibility min/max lines in plots (put ranges in front matter only)
SHOW_PLAUSIBILITY_IN_PLOTS: bool = False

PLAUSIBILITY_RANGES: Dict[str, Dict[str, Any]] = {
    # HR from PPG (bpm). Resting adults typically ~40–120. Exercise can exceed; tune per protocol.
    "hr_bpm_mean": {"min": 35.0, "max": 180.0, "units": "bpm", "desc": "Minute-mean heart rate from PPG."},
    "hr_bpm_std": {"min": 0.0, "max": 40.0, "units": "bpm", "desc": "Within-minute HR standard deviation."},

    # EDA tonic/phasic (µS). Typical tonic ~0–20 in many settings; artifacts can be huge.
    "eda_tonic_mean": {"min": 0.0, "max": 30.0, "units": "uS", "desc": "Minute-mean EDA tonic component."},
    "eda_phasic_mean": {"min": -5.0, "max": 15.0, "units": "uS", "desc": "Minute-mean EDA phasic component."},
    "eda_phasic_p95": {"min": 0.0, "max": 20.0, "units": "uS", "desc": "95th percentile of EDA phasic component within minute (amplitude proxy)."},
    "scr_count": {"min": 0.0, "max": 60.0, "units": "count/min", "desc": "SCR peak count per minute."},

    # Skin temperature (°C). Wrist temp in lab often ~20–40.
    "temp_smooth_C_mean": {"min": 15.0, "max": 42.0, "units": "C", "desc": "Minute-mean smoothed skin temperature."},
    "temp_slope_Cps_mean": {"min": -0.05, "max": 0.05, "units": "C/s", "desc": "Minute-mean slope of smoothed temperature."},

    # Motion: ENMO in g; activity is unitless proxy (RMS of accel magnitude highpass).
    "acc_enmo_g_mean": {"min": 0.0, "max": 3.0, "units": "g", "desc": "Minute-mean ENMO in g."},
    "acc_enmo_g_p95": {"min": 0.0, "max": 6.0, "units": "g", "desc": "95th percentile ENMO within minute."},
    "acc_activity_mean": {"min": 0.0, "max": 5.0, "units": "a.u.", "desc": "Minute-mean activity proxy."},

    # Steps per minute.
    "steps_count": {"min": 0.0, "max": 250.0, "units": "steps/min", "desc": "Steps per minute."},
}

# =========================================================
# Utilities
# =========================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read CSV robustly (handles weird encodings, stray columns)."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def ensure_datetime_utc(df: pd.DataFrame, ts_col_candidates: List[str]) -> pd.DataFrame:
    """
    Ensure a `datetime_utc` column exists.

    Accepted timestamp representations:
      - integer/float seconds since epoch
      - integer/float microseconds since epoch
      - numeric strings representing the above
      - ISO-like datetime strings

    The Empatica exports often store epoch timestamps as *numeric strings*; we detect that.
    """
    df = df.copy()
    ts_col = pick_first_existing(df, ts_col_candidates + ["unix_timestamp_us", "timestamp", "datetime", "date"])
    if ts_col is None:
        raise ValueError(
            f"Timestamp column not found. Tried: {ts_col_candidates + ['unix_timestamp_us','timestamp','datetime','date']}"
        )

    # Some exports contain repeated header rows mid-file (e.g., 'unix_timestamp_us' appears as a data value).
    # Drop those rows deterministically before timestamp parsing.
    df = df.loc[df[ts_col].astype(str) != str(ts_col)].copy()

    s = df[ts_col]

    # If object but looks numeric, coerce to numeric first.
    if not np.issubdtype(s.dtype, np.number):
        s_num = pd.to_numeric(s, errors="coerce")
        frac_numeric = float(s_num.notna().mean())
        if frac_numeric > 0.95:
            s = s_num

    if np.issubdtype(s.dtype, np.number):
        x = pd.to_numeric(s, errors="coerce")
        if x.dropna().empty:
            raise ValueError("All timestamps are NaN.")
        med = float(np.nanmedian(x))
        # Heuristic: microseconds vs seconds
        if med > 1e12:
            dt = pd.to_datetime(x, unit="us", utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(x, unit="s", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(s, errors="coerce", utc=True)

    if dt.isna().any():
        bad = df.loc[dt.isna(), ts_col].head(5).tolist()
        raise ValueError(f"Failed parsing timestamps in {ts_col}. Examples: {bad}")

    df["datetime_utc"] = dt
    return df

def minute_floor_utc(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, utc=True).dt.floor("min")

def empirical_sampling_rate(df: pd.DataFrame) -> float:
    """Estimate sampling rate from median delta time (Hz)."""
    if "datetime_utc" not in df.columns:
        return float("nan")
    t = df["datetime_utc"].astype("int64").to_numpy(dtype=np.int64) / 1e9
    if len(t) < 5:
        return float("nan")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) < 3:
        return float("nan")
    med = float(np.median(dt))
    return float(1.0 / med) if med > 0 else float("nan")

def detect_gaps(df: pd.DataFrame, fs_hz: float, *, gap_s: float = 2.0) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Detect time gaps larger than gap_s seconds (default 2s), returns list of (start,end,duration_s)."""
    if not np.isfinite(fs_hz) or fs_hz <= 0 or "datetime_utc" not in df.columns or len(df) < 3:
        return []
    t = df["datetime_utc"].sort_values().to_numpy()
    dt = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
    gaps = np.where(dt > gap_s)[0]
    out = []
    for i in gaps:
        out.append((pd.Timestamp(t[i]), pd.Timestamp(t[i+1]), float(dt[i])))
    return out

def robust_iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 - q1)

def summarize_signal(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(float)
    x_f = x[np.isfinite(x)]
    if x_f.size == 0:
        return {"n": float(x.size), "n_finite": 0.0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "p05": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": float(x.size),
        "n_finite": float(x_f.size),
        "mean": float(np.mean(x_f)),
        "std": float(np.std(x_f, ddof=1)) if x_f.size > 1 else 0.0,
        "min": float(np.min(x_f)),
        "p05": float(np.percentile(x_f, 5)),
        "median": float(np.median(x_f)),
        "p95": float(np.percentile(x_f, 95)),
        "max": float(np.max(x_f)),
    }

def mad_outlier_fraction(x: np.ndarray, *, k: float = 10.0) -> float:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return 0.0
    z = 0.6745 * (x - med) / mad
    return float(np.mean(np.abs(z) > k))

def flatline_fraction(x: np.ndarray, *, eps: float, min_run: int) -> float:
    """Fraction of samples in (near) flatline runs."""
    x = x.astype(float)
    finite = np.isfinite(x)
    x2 = np.where(finite, x, np.nan)
    if np.all(~finite):
        return 1.0
    d = np.abs(np.diff(x2))
    same = np.concatenate([[False], (d < eps)])
    # Count runs of True
    run = 0
    mask = np.zeros_like(same, dtype=bool)
    for i, v in enumerate(same):
        if v:
            run += 1
        else:
            if run >= min_run:
                mask[i-run:i] = True
            run = 0
    if run >= min_run:
        mask[len(same)-run:len(same)] = True
    return float(np.mean(mask))

# =========================================================
# Modality processing
# =========================================================

def eda_process(eda: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {"method": "neurokit2"}
    df = eda.copy()
    val_col = pick_first_existing(df, ["eda_uS", "eda", "EDA"])
    if val_col is None:
        raise ValueError("EDA value column not found (expected eda_uS / eda / EDA).")
    fs = empirical_sampling_rate(df)
    info["fs_hz_empirical"] = fs
    x = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=float)
    info["n_missing_raw"] = int(np.isnan(x).sum())

    if not (np.isfinite(fs) and fs > 0):
        raise ValueError("Cannot process EDA: invalid empirical sampling rate.")

    x_clean = nk.eda_clean(x, sampling_rate=fs)
    decomp = nk.eda_phasic(x_clean, sampling_rate=fs)
    peaks, _ = nk.eda_peaks(x_clean, sampling_rate=fs)
    df["eda_clean"] = x_clean
    df["eda_tonic"] = decomp["EDA_Tonic"].to_numpy(dtype=float)
    df["eda_phasic"] = decomp["EDA_Phasic"].to_numpy(dtype=float)
    df["scr_peaks"] = peaks["SCR_Peaks"].to_numpy(dtype=int)
    info["nk_ok"] = True
    return df, info

def bvp_process(bvp: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {"method": "neurokit2"}
    df = bvp.copy()
    val_col = pick_first_existing(df, ["bvp", "BVP", "bvp_nW"])
    if val_col is None:
        raise ValueError("BVP value column not found (expected bvp/BVP).")
    fs = empirical_sampling_rate(df)
    info["fs_hz_empirical"] = fs
    x = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=float)
    info["n_missing_raw"] = int(np.isnan(x).sum())

    if HAVE_NK and np.isfinite(fs):
        try:
            clean = nk.ppg_clean(x, sampling_rate=fs)
            signals, _ = nk.ppg_process(clean, sampling_rate=fs)
            df["bvp_clean"] = clean
            # Per-sample HR
            df["hr_bpm"] = signals["PPG_Rate"].to_numpy(dtype=float)
            # Use NeuroKit's signal quality if available (varies by version)
            if "PPG_Quality" in signals.columns:
                df["ppg_quality"] = signals["PPG_Quality"].to_numpy(dtype=float)
            info["nk_ok"] = True
            return df, info
        except Exception as e:
            info["nk_ok"] = False
            info["nk_error"] = str(e)

    raise ValueError(
        "NeuroKit2 PPG processing failed; this pipeline requires NeuroKit2 to work reliably. "
        "Check your NeuroKit2 version and input BVP sampling rate/format."
    )

def temp_process(temp: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    df = temp.copy()
    val_col = pick_first_existing(df, ["temperature_C", "temperature", "temp_C"])
    if val_col is None:
        raise ValueError("Temperature value column not found.")
    fs = empirical_sampling_rate(df)
    info["fs_hz_empirical"] = fs
    x = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=float)
    info["n_missing_raw"] = int(np.isnan(x).sum())

    # Savitzky-Golay smoothing (adaptive window)
    win = 11 if len(x) >= 11 else (len(x) // 2) * 2 + 1
    if win < 5:
        df["temp_smooth_C"] = x
        df["temp_slope_Cps"] = np.nan
        return df, info
    df["temp_smooth_C"] = signal.savgol_filter(np.nan_to_num(x), win, 2)
    dt = df["datetime_utc"].diff().dt.total_seconds().to_numpy(dtype=float)
    dt[0] = np.nan
    df["temp_slope_Cps"] = np.gradient(df["temp_smooth_C"].to_numpy(dtype=float)) / dt
    return df, info

def acc_process(acc: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    df = acc.copy()
    x_col = pick_first_existing(df, ["x_g", "x"])
    y_col = pick_first_existing(df, ["y_g", "y"])
    z_col = pick_first_existing(df, ["z_g", "z"])
    if not (x_col and y_col and z_col):
        raise ValueError("Accelerometer columns missing (expected x_g,y_g,z_g).")
    fs = empirical_sampling_rate(df)
    info["fs_hz_empirical"] = fs
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=float)
    mag = np.sqrt(x*x + y*y + z*z)
    df["acc_mag_g"] = mag
    # ENMO: remove gravity (1 g) and clip to 0
    df["acc_enmo_g"] = np.maximum(0.0, mag - 1.0)

    # Activity proxy: high-pass filter magnitude, then RMS (unitless a.u.)
    if np.isfinite(fs) and fs > 1:
        hp = 0.25  # Hz: removes slow drift/posture
        b, a = signal.butter(2, hp / (fs / 2), btype="highpass")
        mag_hp = signal.filtfilt(b, a, np.nan_to_num(mag))
        s2 = signal.savgol_filter(mag_hp**2, 11 if len(mag_hp) >= 11 else 5, 2)
        df["acc_activity"] = np.sqrt(np.maximum(0, s2))
    else:
        df["acc_activity"] = np.nan
    return df, info

def steps_process(steps: pd.DataFrame) -> pd.DataFrame:
    df = steps.copy()
    val_col = pick_first_existing(df, ["steps", "step_count"])
    if val_col is None:
        raise ValueError("Steps column missing.")
    df["steps"] = pd.to_numeric(df[val_col], errors="coerce")
    return df

# =========================================================
# Timeline alignment
# =========================================================

def load_timeline(timeline_csv: Path, *, timeline_tz: str) -> pd.DataFrame:
    tl = pd.read_csv(timeline_csv)
    colmap = {c.lower().strip(): c for c in tl.columns}

    def require(aliases: List[str]) -> str:
        for a in aliases:
            if a in colmap:
                return colmap[a]
        raise ValueError(f"timeline missing one of {aliases}. Found: {list(tl.columns)}")

    c_datetime = require(["datetime", "minute_utc", "datetime_utc"])
    c_session = require(["session id", "session_id"])
    c_part = require(["participant id", "participant_id", "participant"])
    c_minute = require(["minute index", "minute_index", "protocol_minute"])
    c_block = require(["protocol block", "protocol_block"])
    c_phase = require(["protocol phase", "protocol_phase"])
    # expected_fan_mode is optional (older timelines may not include it)
    c_fan = None
    for _a in ["expected fan mode", "expected_fan_mode"]:
        if _a in colmap:
            c_fan = colmap[_a]
            break

    out = pd.DataFrame({
        "datetime_local": tl[c_datetime],
        "session_id": tl[c_session].astype(str),
        "participant": tl[c_part].astype(str),
        "minute_index": tl[c_minute],
        "protocol_block": tl[c_block],
        "protocol_phase": tl[c_phase],
        "expected_fan_mode": (tl[c_fan] if c_fan is not None else "unknown"),
    })

    dt_raw = pd.to_datetime(out["datetime_local"], errors="coerce")
    if dt_raw.isna().any():
        bad = out.loc[dt_raw.isna(), "datetime_local"].head(5).tolist()
        raise ValueError(f"timeline datetime parsing failed. Examples: {bad}")

    # If tz-naive: interpret as local timeline_tz; if tz-aware: convert to UTC directly.
    if getattr(dt_raw.dt, "tz", None) is None:
        dt_utc = dt_raw.dt.tz_localize(timeline_tz).dt.tz_convert("UTC")
    else:
        dt_utc = dt_raw.dt.tz_convert("UTC")
    out["minute_utc"] = dt_utc.dt.floor("min")
    out = out.drop(columns=["datetime_local"])
    return out

def group_minute(df: pd.DataFrame) -> pd.core.groupby.generic.DataFrameGroupBy:
    df = df.copy()
    df["minute_utc"] = minute_floor_utc(df["datetime_utc"])
    return df.groupby("minute_utc")

def minute_agg_stats(g: pd.core.groupby.generic.DataFrameGroupBy, col: str) -> pd.DataFrame:
    """Return minute-level mean/std/p95/iqr for a sample-level column."""
    x = pd.to_numeric(g[col], errors="coerce")
    out = pd.DataFrame({
        f"{col}_mean": x.mean(),
        f"{col}_std": x.std(ddof=1),
        f"{col}_p95": x.quantile(0.95),
        f"{col}_iqr": x.apply(lambda s: np.subtract(*np.nanpercentile(s.to_numpy(dtype=float), [75, 25])) if s.notna().sum() >= 5 else np.nan),
    })
    return out

def build_minute_features(
    *,
    session_id: str,
    timeline_session: pd.DataFrame,
    eda_df: pd.DataFrame,
    bvp_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    steps_df: pd.DataFrame,
) -> pd.DataFrame:
    tl = timeline_session.copy().sort_values("minute_utc")
    tl["session_id"] = tl["session_id"].astype(str)
    tl["participant"] = tl["participant"].astype(str)

    # EDA
    eda_g = group_minute(eda_df)
    eda_tonic = minute_agg_stats(eda_g, "eda_tonic")
    eda_phasic = minute_agg_stats(eda_g, "eda_phasic")
    scr_count = eda_g["scr_peaks"].sum().rename("scr_count").to_frame()

    # HR (from BVP)
    bvp_g = group_minute(bvp_df)
    hr = minute_agg_stats(bvp_g, "hr_bpm")

    # Temp
    temp_g = group_minute(temp_df)
    temp_s = minute_agg_stats(temp_g, "temp_smooth_C")
    temp_sl = minute_agg_stats(temp_g, "temp_slope_Cps")

    # Acc
    acc_g = group_minute(acc_df)
    acc_act = minute_agg_stats(acc_g, "acc_activity")
    enmo = minute_agg_stats(acc_g, "acc_enmo_g")

    # Steps: already sparse; sum per minute
    steps_df = steps_df.copy()
    steps_df["minute_utc"] = minute_floor_utc(steps_df["datetime_utc"])
    steps_count = steps_df.groupby("minute_utc")["steps"].sum().rename("steps_count").to_frame()

    feat = (
        tl.merge(eda_tonic, on="minute_utc", how="left")
          .merge(eda_phasic, on="minute_utc", how="left")
          .merge(scr_count, on="minute_utc", how="left")
          .merge(hr, on="minute_utc", how="left")
          .merge(temp_s, on="minute_utc", how="left")
          .merge(temp_sl, on="minute_utc", how="left")
          .merge(acc_act, on="minute_utc", how="left")
          .merge(enmo, on="minute_utc", how="left")
          .merge(steps_count, on="minute_utc", how="left")
    )

    # Derive rates
    feat["scr_rate_per_min"] = feat["scr_count"]
    feat["steps_rate_per_min"] = feat["steps_count"]

    # Presence
    for base in ["eda_tonic_mean", "hr_bpm_mean", "temp_smooth_C_mean", "acc_activity_mean", "steps_count"]:
        feat[f"has_{base.split('_')[0] if base!='steps_count' else 'steps'}"] = feat[base].notna().astype(int)

    # High motion: based on ENMO p95 (more robust than mean)
    feat["high_motion"] = (feat["acc_enmo_g_p95"] > HIGH_MOTION_ENMO_P95_THRESHOLD_G).astype(int)  # ~vigorous movement threshold; tune if needed
    return feat

# =========================================================
# QC / validity
# =========================================================

def add_oob_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, meta in PLAUSIBILITY_RANGES.items():
        if col not in df.columns:
            continue
        lo, hi = meta["min"], meta["max"]
        df[f"{col}_oob"] = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
        df[f"{col}_oob"] = df[f"{col}_oob"].astype(int)
    return df

def add_validity_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = add_oob_flags(df)

    # "Valid" per signal: present and not oob for the primary minute-mean
    df["valid_hr_minute"] = (df["hr_bpm_mean"].notna() & (df.get("hr_bpm_mean_oob", 0) == 0)).astype(int)
    df["valid_eda_minute"] = (df["eda_tonic_mean"].notna() & (df.get("eda_tonic_mean_oob", 0) == 0)).astype(int)
    df["valid_temp_minute"] = (df["temp_smooth_C_mean"].notna() & (df.get("temp_smooth_C_mean_oob", 0) == 0)).astype(int)
    df["valid_acc_minute"] = (df["acc_enmo_g_mean"].notna() & (df.get("acc_enmo_g_mean_oob", 0) == 0)).astype(int)

    # Joint validity: all key signals present + in-range + not high motion
    df["valid_joint_minute"] = (
        (df["valid_hr_minute"] == 1) &
        (df["valid_eda_minute"] == 1) &
        (df["valid_temp_minute"] == 1) &
        (df["valid_acc_minute"] == 1) &
        (df["high_motion"] == 0)
    ).astype(int)
    return df

def phase_aggregate(df: pd.DataFrame, *, valid_only: bool) -> pd.DataFrame:
    d = df.copy()
    if valid_only:
        d = d.loc[d["valid_joint_minute"] == 1].copy()

    group_cols = ["session_id", "participant", "protocol_block", "protocol_phase"]
    if "expected_fan_mode" in d.columns:
        group_cols.append("expected_fan_mode")
    metric_cols = [c for c in d.columns if c.endswith("_mean") or c in ["scr_count", "scr_rate_per_min", "steps_count", "steps_rate_per_min"]]
    aggs = {c: ["mean", "std", "median"] for c in metric_cols if c in d.columns}

    out = d.groupby(group_cols).agg(aggs)
    out.columns = [f"{a}__{b}" for a, b in out.columns]
    out = out.reset_index()

    # coverage
    cov = d.groupby(group_cols)["minute_utc"].count().rename("n_minutes").reset_index()
    out = out.merge(cov, on=group_cols, how="left")
    out["valid_only"] = int(valid_only)
    return out

def aggregate_phase_qcaware(df: pd.DataFrame) -> pd.DataFrame:
    allp = phase_aggregate(df, valid_only=False)
    valp = phase_aggregate(df, valid_only=True)
    return pd.concat([allp, valp], ignore_index=True)

# =========================================================
# Output schema generation
# =========================================================

def schema_from_dataframe(file: str, file_description: str, df: pd.DataFrame, session_id: str) -> List[Dict[str, Any]]:
    out = []
    for col in df.columns:
        out.append({
            "file": file,
            "file_description": file_description,
            "column": col,
            "dtype": str(df[col].dtype),
            "description": "",
            "session_id": session_id,
        })
    return out

def write_output_schema(out_dir: Path, schema: List[Dict[str, Any]], *, prefix: str = "output_schema", write_py: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_py:
        (out_dir / f"{prefix}.py").write_text(
            "# Auto-generated by empatica_batch_pipeline_v3.py\n"
            f"{prefix.upper()} = " + json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )


# =========================================================
# Statistics helpers (matrices export)
# =========================================================

from scipy.stats import spearmanr

def spearman_corr_with_pvalues(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Spearman correlation matrix + p-values (pairwise, NaN-safe).

    Returns:
        corr_df: square DataFrame of rho
        pval_df: square DataFrame of p-values (two-sided)
    """
    cols = list(df.columns)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    pval = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for i, ci in enumerate(cols):
        xi = pd.to_numeric(df[ci], errors="coerce").to_numpy(dtype=float)
        for j, cj in enumerate(cols):
            if j < i:
                # symmetry
                corr.iat[i, j] = corr.iat[j, i]
                pval.iat[i, j] = pval.iat[j, i]
                continue
            xj = pd.to_numeric(df[cj], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 3:
                continue
            rho, pv = spearmanr(xi[mask], xj[mask])
            try:
                rho = float(rho)
                pv = float(pv)
            except Exception:
                rho, pv = float("nan"), float("nan")
            corr.iat[i, j] = rho
            pval.iat[i, j] = pv
            corr.iat[j, i] = rho
            pval.iat[j, i] = pv
    np.fill_diagonal(pval.values, 0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr, pval


def compute_phase_mean_matrices(
    df_minute: pd.DataFrame,
    metrics: List[str],
    *,
    valid_only: bool = True,
    clean_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (phase_means, spearman_corr, spearman_pvalues) on phase means."""
    d = df_minute.copy()
    if valid_only and ("valid_joint_minute" in d.columns):
        d = d.loc[d["valid_joint_minute"] == 1].copy()
    cols = [c for c in metrics if c in d.columns]
    if (not cols) or ("protocol_phase" not in d.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    g = d.groupby("protocol_phase")[cols].mean(numeric_only=True)

    if clean_labels:
        def clean(name: str) -> str:
            return name.replace("_mean", "").replace("__", "_").strip("_")
        g = g.rename(columns={c: clean(c) for c in g.columns})

    corr, pval = spearman_corr_with_pvalues(g)
    return g, corr, pval


# =========================================================
# Plotting
# =========================================================

@dataclass
class PlotStyle:
    dpi: int = 300
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 9

def _prep_ax(ax: plt.Axes, *, session_id: str, title: str, enable_grid: bool = True):
    """Standard axis styling for all non-heatmap plots."""
    # Grid: both horizontal and vertical, dashed, 40% opacity (publication-friendly)
    if enable_grid:
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.4)
    else:
        ax.grid(False)
    ax.tick_params(axis="both", labelsize=9)
    # Put the session id into the axis title (not figure suptitle) to avoid layout collisions.
    # Title schema:
    # - For combined/cohort pages we pass session_id as "Wearable Physiology – <Scope>" and expect " | " separator.
    # - For session pages we keep "<SESSION_ID> — <Title>".
    if str(session_id).startswith("Wearable Physiology –"):
        ax.set_title(f"{session_id} | {title}", fontsize=PLOT_FONTS.get("TITLE", 11), pad=8)
    else:
        ax.set_title(f"{session_id} — {title}", fontsize=PLOT_FONTS.get("TITLE", 11), pad=10)
def _shade_blocks_and_phases(ax: plt.Axes, df: pd.DataFrame):
    """
    Background protocol annotations.

    v6 policy:
    - NO phase labels.
    - NO phase shading.
    - Block-only alternating shading + block boundary lines.

    Required columns: minute_utc, protocol_block
    """
    if df.empty:
        return

    d = df.sort_values("minute_utc").copy()
    t = pd.to_datetime(d["minute_utc"], utc=True).dt.tz_convert(None)

    blocks = d["protocol_block"].astype(str).to_numpy()

    # run-length encode blocks
    segs = []
    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            segs.append((start, i))
            start = i

    # Alternate very light greys
    shades = ["0.96", "0.92"]
    for k, (s, e) in enumerate(segs):
        x0 = t.iloc[s]
        x1 = t.iloc[e - 1] + pd.Timedelta(minutes=1)
        ax.axvspan(x0, x1, color=shades[k % 2], alpha=0.55, zorder=0)

    # Block boundary lines
    for (s, e) in segs[1:]:
        x = t.iloc[s]
        ax.axvline(x, color="0.70", linewidth=0.8, alpha=0.9, zorder=1)



def _block_segments(df: pd.DataFrame):
    """Return block segments as list of (block_label, x0, x1) with tz-naive datetimes."""
    if df.empty:
        return []
    d = df.sort_values("minute_utc").copy()
    t = pd.to_datetime(d["minute_utc"], utc=True).dt.tz_convert(None)
    blocks = d["protocol_block"].astype(str).to_numpy()

    segs = []
    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            blk = blocks[start]
            x0 = t.iloc[start]
            x1 = t.iloc[i - 1] + pd.Timedelta(minutes=1)
            segs.append((blk, x0, x1))
            start = i
    return segs




def _phase_boundaries(df: pd.DataFrame) -> List[pd.Timestamp]:
    """Return tz-naive timestamps where protocol_phase changes (minute-level boundaries)."""
    if df.empty or "protocol_phase" not in df.columns:
        return []
    d = df.sort_values("minute_utc").copy()
    t = pd.to_datetime(d["minute_utc"], utc=True).dt.tz_convert(None)
    phases = d["protocol_phase"].astype(str).to_numpy()
    bounds: List[pd.Timestamp] = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            bounds.append(t.iloc[i])  # boundary at start of new phase
    return bounds


def _draw_protocol_ribbon(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    t: Union[pd.Series, pd.DatetimeIndex, np.ndarray, List[Any]],
    y0: float,
    y1: float,
    gap_minutes: float = 1.0,
):
    """
    Draw the protocol *ribbon* inside the reserved band [y0, y1] (DATA coordinates).

    Style target:
    - Blocks are colored rectangles.
    - Phase boundaries indicated by thin WHITE gaps inside the ribbon (no phase labels).
    - Block labels centered inside each block segment.

    This function is robust to x-axis being either:
    - datetime-like (Timestamp/DatetimeIndex/np.datetime64)
    - numeric (protocol minute index)
    """
    if df is None or df.empty:
        return

    # --- normalize x to 1D numpy array ---
    if isinstance(t, pd.Series):
        tt = t.reset_index(drop=True)
    else:
        tt = pd.Series(t).reset_index(drop=True)

    x = tt.to_numpy()

    # --- detect datetime-like vs numeric axis ---
    is_dt = (
        pd.api.types.is_datetime64_any_dtype(tt)
        or isinstance(tt.dtype, pd.DatetimeTZDtype)
    )
    if not is_dt and tt.dtype == object:
        # check first non-null element
        for v in x:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if isinstance(v, (pd.Timestamp, datetime.datetime, np.datetime64)):
                is_dt = True
            break

    # Guard: if x is numeric, do NOT treat as datetime
    if is_dt and pd.api.types.is_numeric_dtype(tt):
        is_dt = False
    one_step = pd.Timedelta(minutes=1) if is_dt else 1.0
    gap = pd.Timedelta(minutes=float(gap_minutes)) if is_dt else float(gap_minutes)

    d = df.sort_values("minute_utc").copy()
    blocks = d["protocol_block"].astype(str).to_numpy()

    # contiguous block segments (indices)
    segs: List[Tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            segs.append((blocks[start], start, i))
            start = i

    # Pastel block colors (consistent, manuscript-friendly)
    palette = [
        "#dbeafe",  # light blue
        "#fef3c7",  # light amber
        "#fde2e4",  # light pink
        "#dcfce7",  # light green
        "#ede9fe",  # light violet (fallback)
    ]

    def _add_rect(x0, x1, facecolor, zorder=0.8):
        if is_dt:
            x0n = mdates.date2num(pd.to_datetime(x0))
            x1n = mdates.date2num(pd.to_datetime(x1))
            rect = Rectangle(
                (x0n, y0),
                width=(x1n - x0n),
                height=(y1 - y0),
                transform=ax.transData,
                facecolor=facecolor,
                edgecolor="none",
                alpha=1.0,
                zorder=zorder,
            )
        else:
            rect = Rectangle(
                (float(x0), y0),
                width=float(x1) - float(x0),
                height=(y1 - y0),
                transform=ax.transData,
                facecolor=facecolor,
                edgecolor="none",
                alpha=1.0,
                zorder=zorder,
            )
        ax.add_patch(rect)

    for j, (blk, s, e) in enumerate(segs):
        x0 = x[s]
        x1 = x[e - 1] + one_step
        _add_rect(x0, x1, palette[j % len(palette)], zorder=0.8)

        # Block label
        try:
            xc = x0 + (x1 - x0) / 2
        except Exception:
            # numeric fallback
            xc = (float(x0) + float(x1)) / 2

        ax.text(
            xc,
            y0 + 0.55 * (y1 - y0),
            f"Block {blk}",
            ha="center",
            va="center",
            fontsize=9,
            color="0.20",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.6),
            zorder=5,
        )

    # Phase boundary gaps (white) inside ribbon
    bounds = _phase_boundaries(d)
    if bounds:
        for xb in bounds:
            x0 = xb - gap / 2
            x1 = xb + gap / 2
            _add_rect(x0, x1, "white", zorder=2.0)

    # Block boundary lines (thin) inside ribbon
    for (_, _, e) in segs[1:]:
        xb = x[e] if e < len(x) else (x[-1] + one_step)
        ax.plot([xb, xb], [y0, y1], color="0.35", linewidth=0.8, alpha=0.8, zorder=3)

def _add_block_brackets_data(
    ax: plt.Axes,
    x: pd.Series,
    block_series: pd.Series,
    y: float,
    *,
    fontsize: int = 9,
    text_offset_frac: float = 0.02,
) -> None:
    """Draw block brackets inside plot bounds (data coords).

    This intentionally avoids a solid top ribbon, so the burst stays visually rich.
    """
    blocks = block_series.astype(str).to_numpy()
    if len(blocks) == 0:
        return

    # indices where the block label changes (including ends)
    change = np.where(np.r_[True, blocks[1:] != blocks[:-1], True])[0]
    ymin, ymax = ax.get_ylim()
    text_offset = text_offset_frac * (ymax - ymin)

    for i in range(len(change) - 1):
        a = int(change[i])
        b = int(change[i + 1] - 1)
        if b <= a:
            continue
        x0 = x.iloc[a]
        x1 = x.iloc[b]
        if pd.isna(x0) or pd.isna(x1):
            continue

        bracket = FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=ax.transData,
            arrowstyle="<->",
            mutation_scale=11,
            linewidth=0.9,
            color="0.20",
            clip_on=True,
        )
        ax.add_patch(bracket)

        xm = x.iloc[(a + b) // 2]
        ax.text(
            xm,
            y + text_offset,
            f"Block {blocks[a]}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="0.20",
            clip_on=True,
        )

def _draw_block_header_band_in_axis(ax: plt.Axes, df: pd.DataFrame, y0: float = 0.92, y1: float = 1.0):
    """
    Draw a dedicated *internal* header band inside the plot axes (no extra stacked axes).

    The band is used exclusively for block annotations (Block 0/1/2/3) and block shading.
    We "mask" the data region in that band so the signal never competes with the labels.

    Parameters
    ----------
    ax : matplotlib Axes
    df : minute-level dataframe containing at least minute_utc and protocol_block
    y0, y1 : axes-fraction coordinates defining the vertical band [y0, y1]
    """
    segs = _block_segments(df)
    if not segs:
        return

    # Full white mask to reserve space (covers data lines behind it)
    x_min = segs[0][1]
    x_max = segs[-1][2]
    ax.axvspan(x_min, x_max, ymin=y0, ymax=y1, facecolor="white", edgecolor="none", alpha=1.0, zorder=3)

    # Block shading within the band
    shades = ["0.96", "0.92"]
    for k, (blk, x0, x1) in enumerate(segs):
        ax.axvspan(x0, x1, ymin=y0, ymax=y1, facecolor=shades[k % 2], edgecolor="none", alpha=1.0, zorder=4)

        xc = x0 + (x1 - x0) / 2
        ax.text(
            xc, (y0 + y1) / 2,
            f"Block {blk}",
            transform=ax.get_xaxis_transform(),  # x=data, y=axes fraction
            ha="center", va="center",
            fontsize=9, color="0.20",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8),
            zorder=0.8,
            clip_on=False,
        )

    # Block boundary lines (only inside the band)
    for (blk, x0, x1) in segs[1:]:
        ax.axvline(x0, ymin=y0, ymax=y1, color="0.65", linewidth=0.8, alpha=0.9, zorder=5)

def _draw_block_header_band_in_data(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    t: pd.Series,
    y0: float,
    y1: float,
):
    """
    Draw block shading + labels + boundary lines ONLY inside [y0, y1] in DATA coordinates.

    This is the robust way to create "headroom" without overlapping the signal:
    - we expand ylim to include [y0, y1]
    - then draw everything in data coords so neither the curve nor the legend gets covered.
    """
    if df.empty:
        return

    d = df.sort_values("minute_utc").copy()
    blocks = d["protocol_block"].astype(str).to_numpy()

    # Build contiguous block segments
    segs = []
    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            segs.append((blocks[start], start, i))
            start = i

    # Alternate subtle band shading per block (inside y0..y1)
    shades = ["0.96", "0.93"]
    for j, (blk, s, e) in enumerate(segs):
        x0 = t.iloc[s]
        x1 = t.iloc[e - 1] + pd.Timedelta(minutes=1)

        rect = Rectangle(
            (mdates.date2num(x0), y0),
            width=mdates.date2num(x1) - mdates.date2num(x0),
            height=(y1 - y0),
            transform=ax.transData,
            facecolor=shades[j % 2],
            edgecolor="none",
            alpha=1.0,
            zorder=0.8,
        )
        ax.add_patch(rect)

        # Block label centered in the band
        xc = x0 + (x1 - x0) / 2
        ax.text(
            xc,
            y0 + 0.55 * (y1 - y0),
            f"Block {blk}",
            ha="center",
            va="center",
            fontsize=9,
            color="0.25",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.6),
            zorder=5,
        )

    # Boundary lines only inside the band
    for (_, _, e) in segs[1:]:
        xb = t.iloc[e - 1] + pd.Timedelta(minutes=1)
        ax.plot([xb, xb], [y0, y1], color="0.65", linewidth=0.8, alpha=0.9, zorder=1)


def _add_block_labels_headroom(ax: plt.Axes, df: pd.DataFrame):
    """
    Add centered block labels in the headroom (no span arrows).

    Uses x in data coordinates and y in axes fraction to avoid transform bugs.
    """
    if df.empty:
        return

    d = df.sort_values("minute_utc").copy()
    t = pd.to_datetime(d["minute_utc"], utc=True).dt.tz_convert(None)
    blocks = d["protocol_block"].astype(str).to_numpy()

    segs = []
    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            segs.append((blocks[start], start, i))
            start = i

    y = 0.985  # inside axes top band (below title)
    for blk, s, e in segs:
        x0 = t.iloc[s]
        x1 = t.iloc[e - 1] + pd.Timedelta(minutes=1)


        # label centered
        xc = x0 + (x1 - x0) / 2
        label = f"Block {blk}"
        ax.text(xc, y, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color="0.20",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.6),
                clip_on=False, zorder=11)



def plot_timeseries_qcaware(
    pdf: PdfPages,
    df: pd.DataFrame,
    *,
    session_id: str,
    ycol: str,
    ylabel: str,
    valid_col: str,
    title: str,
    extra_hlines: Optional[List[Tuple[float, str]]] = None,
):
    """
    Plot minute-level time series with an *internal* headroom band inside the same axes.

    Requirements enforced (per feedback):
    - Headroom must be inside the plot boundary and must not overlap the signal or legend.
    - No stacked axes.
    - Block annotations only (no phase labels, no span arrows).
    - No vertical grid lines.
    - Thin curves.

    Implementation strategy:
    - Compute y-limits from valid data (robust percentiles).
    - Expand the top y-limit by a dedicated headroom height in *data coordinates*.
    - Draw block shading + block labels only inside that top headroom band (in data coords).
    - Place legend inside the headroom band (top-right) so it never overlaps the signal.
    - Do NOT allow extra horizontal reference lines (e.g., ENMO threshold) to expand y-limits;
      if a reference line is above the data-driven range, it is annotated in headroom instead.
    """
    if df.empty:
        return

    d = df.sort_values("minute_utc").copy()
    t = pd.to_datetime(d["minute_utc"], utc=True).dt.tz_convert(None)

    y_all = pd.to_numeric(d[ycol], errors="coerce")
    mask_valid = (d[valid_col] == 1) & y_all.notna()
    mask_invalid = (d[valid_col] != 1) & y_all.notna()

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.set_title(format_plot_title("session", session_id, title), fontsize=12, pad=4)

    # Background: keep plot area clean (no full-background shading)

    # --- Robust y-limits from valid data (compute BEFORE drawing headroom band and plotting lines) ---
    y_valid = y_all[mask_valid]
    if y_valid.notna().any():
        lo = float(np.nanpercentile(y_valid, 1))
        hi = float(np.nanpercentile(y_valid, 99))
    else:
        lo = float(np.nanpercentile(y_all, 1)) if y_all.notna().any() else 0.0
        hi = float(np.nanpercentile(y_all, 99)) if y_all.notna().any() else 1.0

    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        lo, hi = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            lo, hi = 0.0, 1.0

    yr = hi - lo
    pad = 0.10 * yr if yr > 0 else 1.0
    head = 0.18 * yr if yr > 0 else 0.2  # dedicated headroom height (data coords)

    y_data_top = hi + pad
    y_band0 = y_data_top
    y_band1 = y_data_top + head

    ax.set_ylim(lo - pad, y_band1)

    # --- Draw headroom band content in DATA coordinates (behind data/legend; never overlaps) ---
    # NOTE: zorder kept low so it cannot cover lines or legend.
    # Reserve headroom region in data coordinates and draw protocol ribbon (blocks + white phase gaps)
    ax.axhspan(y_band0, y_band1, color="white", alpha=1.0, zorder=0.6)
    _draw_protocol_ribbon(ax, d, t=t, y0=y_band0, y1=y_band1, gap_minutes=0.6)

    handles: List[Any] = []
    labels: List[str] = []

    # All minutes (faint)
    if y_all.notna().any():
        h_all, = ax.plot(t, y_all, linewidth=0.5, alpha=0.18, color="0.25", zorder=2)
        ax.scatter(t, y_all, s=5, alpha=0.12, color="0.25", zorder=2)
        handles.append(h_all)
        labels.append("all minutes")

    # Valid minutes (thin but prominent)
    if mask_valid.any():
        h_valid, = ax.plot(t[mask_valid], y_all[mask_valid], linewidth=0.9, alpha=0.95, color=PRIMARY_BLUE, zorder=3)
        ax.scatter(t[mask_valid], y_all[mask_valid], s=6, alpha=0.70, color=PRIMARY_BLUE, zorder=3)
        handles.append(h_valid)
        labels.append("valid minutes")

    # Invalid minutes (numeric but invalid)
    if mask_invalid.any():
        h_inv = ax.scatter(t[mask_invalid], y_all[mask_invalid], s=18, marker="x", alpha=0.65, color="0.25", zorder=4)
        handles.append(h_inv)
        labels.append("invalid")

    # Axes labels + time formatting
    ax.set_xlabel("time (UTC, minute)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Grid: both axes (requested)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.4)

# Extra reference lines:
    # - draw only if within the data region (<= y_data_top)
    # - otherwise annotate in headroom (no empty-space inflation)
    if extra_hlines:
        for yv, lab in extra_hlines:
            yv = float(yv)
            if yv <= y_data_top:
                ax.axhline(yv, linestyle="--", linewidth=0.8, alpha=0.55, zorder=1)
                ax.text(t.iloc[0], yv, f" {lab}", va="bottom", ha="left", fontsize=PLOT_FONTS['LEGEND'], alpha=0.7)
            else:
                ax.text(
                    t.iloc[0],
                    y_band0 + 0.08 * (y_band1 - y_band0),
                    f"{lab}: {yv:g} (above axis)",
                    fontsize=PLOT_FONTS['LEGEND'], alpha=0.8, ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4),
                    zorder=12,
                )

    # Legend: place inside the headroom band, top-right (never overlaps signal)
    # Legend placement: match reference style (top-right, just below the ribbon)
    ribbon_frac = head / ( (y_band1 - (lo - pad)) if (y_band1 - (lo - pad)) > 0 else 1.0 )
    y_anchor = max(0.0, 1.0 - min(0.22, 0.12 + ribbon_frac))  # heuristic
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.995, 0.90),  # top-right, just below ribbon

        frameon=True,
        fontsize=PLOT_FONTS['LEGEND'],
        borderpad=0.25,
        labelspacing=0.22,
        handlelength=1.4,
        handletextpad=0.45,
        framealpha=0.82,
    )
    ax.set_xlabel("time (UTC, minute)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(format_plot_title("session", session_id, title), fontsize=12, pad=4)

    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)




def plot_phase_distributions(pdf: PdfPages, df: pd.DataFrame, *, session_id: str, metric: str, ylabel: str):
    """
    Phase-wise distribution plot (valid minutes): violin + embedded box + jittered points + n + median label.

    Notes:
    - Uses valid_joint_minute only (joint QC), so phases compare like-for-like across modalities.
    """
    d = df.copy()
    d = d.loc[d["valid_joint_minute"] == 1].copy()
    d = d.sort_values("minute_utc")
    if d.empty or metric not in d.columns:
        return

    # Stable phase order: appearance order in the session timeline
    phases = d["protocol_phase"].astype(str).dropna()
    phases_order = list(dict.fromkeys(phases.tolist()))
    data: List[np.ndarray] = []
    ns: List[int] = []
    meds: List[float] = []

    for p in phases_order:
        arr = pd.to_numeric(d.loc[d["protocol_phase"].astype(str) == p, metric], errors="coerce").dropna().to_numpy(dtype=float)
        data.append(arr)
        ns.append(int(arr.size))
        meds.append(float(np.nanmedian(arr)) if arr.size else float("nan"))

    if sum(ns) < 5:
        return

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)

    xpos = np.arange(1, len(phases_order) + 1)

    parts = ax.violinplot(
        data,
        positions=xpos,
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in parts.get("bodies", []):
        pc.set_alpha(0.55)
        pc.set_edgecolor("0.35")
        pc.set_linewidth(0.6)

    ax.boxplot(
        data,
        positions=xpos,
        widths=0.20,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="0.25", linewidth=0.9, alpha=0.95),
        medianprops=dict(color="0.10", linewidth=1.3),
        whiskerprops=dict(color="0.25", linewidth=0.8),
        capprops=dict(color="0.25", linewidth=0.8),
    )

    # Jittered points (subsample)
    rng = np.random.default_rng(7)
    for i, arr in enumerate(data, start=1):
        if arr.size == 0:
            continue
        if arr.size > 250:
            arr = rng.choice(arr, size=250, replace=False)
        jitter = rng.normal(0, 0.06, size=arr.size)
        ax.scatter(i + jitter, arr, s=10, alpha=0.22, color="0.25", linewidths=0)

    # y-range for labels
    ymax = np.nanmax([np.nanmax(a) if a.size else np.nan for a in data])
    ymin = np.nanmin([np.nanmin(a) if a.size else np.nan for a in data])
    yr = float(ymax - ymin) if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin else 1.0
    # Reserve internal headroom so n/median labels stay inside plot boundary
    ax.set_ylim(ymin - 0.05 * yr, ymax + 0.22 * yr)

    for i, (n, med) in enumerate(zip(ns, meds), start=1):
        ax.text(i, ymax + 0.18 * yr, f"n={n}", ha="center", va="bottom", fontsize=PLOT_FONTS['LEGEND'], color="0.25")
        if math.isfinite(med):
            ax.text(i, ymax + 0.12 * yr, f"med={med:.2g}", ha="center", va="bottom", fontsize=PLOT_FONTS['LEGEND'], color="0.25")

    _prep_ax(ax, session_id=session_id, title=f"Phase-wise distribution (valid minutes): {metric}")
    ax.set_ylabel(ylabel, fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.set_xlabel("protocol_phase", fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.set_xticks(xpos)
    ax.set_xticklabels(phases_order, rotation=30, ha="right", fontsize=9)

    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)


def plot_phase_delta_from_baseline(pdf: PdfPages, df: pd.DataFrame, *, session_id: str, metric: str, ylabel: str, baseline_phase: str = "acclimation"):
    """
    Phase mean delta vs baseline with bootstrap 95% CI (valid minutes only).

    Output is defensible: raw minutes -> bootstrap distribution of mean difference.
    """
    d = df.copy()
    d = d.loc[d["valid_joint_minute"] == 1].copy()
    d = d.sort_values("minute_utc")
    if d.empty or metric not in d.columns:
        return

    phases = d["protocol_phase"].astype(str).dropna()
    phases_order = list(dict.fromkeys(phases.tolist()))
    if baseline_phase not in phases_order:
        baseline_phase = phases_order[0]

    base = pd.to_numeric(d.loc[d["protocol_phase"].astype(str) == baseline_phase, metric], errors="coerce").dropna().to_numpy(dtype=float)
    if base.size < 5:
        return

    rng = np.random.default_rng(17)

    rows = []
    for p in phases_order:
        arr = pd.to_numeric(d.loc[d["protocol_phase"].astype(str) == p, metric], errors="coerce").dropna().to_numpy(dtype=float)
        if arr.size < 3:
            rows.append((p, float("nan"), float("nan"), float("nan"), int(arr.size)))
            continue

        # bootstrap mean difference: mean(p) - mean(base)
        B = 1000
        bs = rng.choice(base, size=(B, base.size), replace=True).mean(axis=1)
        ps = rng.choice(arr, size=(B, arr.size), replace=True).mean(axis=1)
        diff = ps - bs
        est = float(np.mean(arr) - np.mean(base))
        lo, hi = np.percentile(diff, [2.5, 97.5]).tolist()
        rows.append((p, est, lo, hi, int(arr.size)))

    out = pd.DataFrame(rows, columns=["protocol_phase", "delta_mean", "ci_low", "ci_high", "n"])
    # Order as appearance
    out["protocol_phase"] = pd.Categorical(out["protocol_phase"], categories=phases_order, ordered=True)
    out = out.sort_values("protocol_phase")

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)

    x = np.arange(len(out)) + 1
    y = out["delta_mean"].to_numpy(dtype=float)
    yerr_low = y - out["ci_low"].to_numpy(dtype=float)
    yerr_high = out["ci_high"].to_numpy(dtype=float) - y
    yerr = np.vstack([yerr_low, yerr_high])

    ax.axhline(0.0, color="0.25", linewidth=1.0, alpha=0.7, zorder=1)

    ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, elinewidth=1.0, markersize=5, alpha=0.9, zorder=3)

    # n labels
    finite = np.isfinite(y)
    if finite.any():
        ymax = float(np.nanmax(np.abs(y[finite]))) or 1.0
    else:
        ymax = 1.0
    for xi, n in zip(x, out["n"].to_list()):
        ax.text(xi, (np.nanmax(y) if finite.any() else 0.0) + 0.08 * ymax, f"n={n}", ha="center", va="bottom", fontsize=PLOT_FONTS['LEGEND'], color="0.25")

    _prep_ax(ax, session_id=session_id, title=f"Phase mean delta vs baseline ({baseline_phase}): {metric}")
    ax.set_ylabel(ylabel, fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.set_xlabel("protocol_phase", fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.set_xticks(x)
    ax.set_xticklabels(out["protocol_phase"].astype(str).tolist(), rotation=30, ha="right", fontsize=9)

    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)



def plot_cohort_phase_distributions(
    pdf: PdfPages,
    df: pd.DataFrame,
    *,
    session_id: str,
    metric: str,
    ylabel: str,
    phase_order: Optional[List[str]] = None,
) -> None:
    """Cohort phase-wise distribution plot using ALL valid minutes across sessions."""
    d = df.copy()
    if "valid_joint_minute" in d.columns:
        d = d.loc[d["valid_joint_minute"] == 1].copy()
    if d.empty or metric not in d.columns or "protocol_phase" not in d.columns:
        return

    # Choose an order: provided order, else appearance order
    if phase_order is None:
        phases = d["protocol_phase"].astype(str).dropna()
        phase_order = list(dict.fromkeys(phases.tolist()))
    phase_order = [p for p in phase_order if (d["protocol_phase"].astype(str) == p).any()]
    if not phase_order:
        return

    data = []
    ns = []
    meds = []
    for p in phase_order:
        arr = pd.to_numeric(d.loc[d["protocol_phase"].astype(str) == p, metric], errors="coerce").dropna().to_numpy(dtype=float)
        data.append(arr)
        ns.append(int(arr.size))
        meds.append(float(np.nanmedian(arr)) if arr.size else float("nan"))
    if sum(ns) < 10:
        return

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    xpos = np.arange(1, len(phase_order) + 1)

    parts = ax.violinplot(data, positions=xpos, widths=0.85, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts.get("bodies", []):
        pc.set_alpha(0.55)
        pc.set_edgecolor("0.35")
        pc.set_linewidth(0.6)

    ax.boxplot(
        data,
        positions=xpos,
        widths=0.20,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="0.25", linewidth=0.9, alpha=0.95),
        medianprops=dict(color="0.10", linewidth=1.3),
        whiskerprops=dict(color="0.25", linewidth=0.8),
        capprops=dict(color="0.25", linewidth=0.8),
    )

    # Light jittered points (subsample)
    rng = np.random.default_rng(7)
    for i, arr in enumerate(data, start=1):
        if arr.size == 0:
            continue
        if arr.size > 300:
            arr = rng.choice(arr, size=300, replace=False)
        jitter = rng.normal(0, 0.06, size=arr.size)
        ax.scatter(i + jitter, arr, s=9, alpha=0.18, color="0.25", linewidths=0)

    ymax = np.nanmax([np.nanmax(a) if a.size else np.nan for a in data])
    ymin = np.nanmin([np.nanmin(a) if a.size else np.nan for a in data])
    yr = float(ymax - ymin) if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.05 * yr, ymax + 0.22 * yr)
    for i, (n, med) in enumerate(zip(ns, meds), start=1):
        ax.text(i, ymax + 0.18 * yr, f"n={n}", ha="center", va="bottom", fontsize=PLOT_FONTS['LEGEND'], color="0.25")
        if math.isfinite(med):
            ax.text(i, ymax + 0.12 * yr, f"med={med:.2g}", ha="center", va="bottom", fontsize=PLOT_FONTS['LEGEND'], color="0.25")

    _prep_ax(ax, session_id=session_id, title=f"{ylabel} — Phase distribution")
    ax.set_ylabel(ylabel, fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.set_xlabel("protocol_phase", fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.set_xticks(xpos)
    ax.set_xticklabels(phase_order, rotation=30, ha="right", fontsize=9)

    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)


def compute_cohort_phase_deltas_across_sessions(
    df: pd.DataFrame,
    metric: str,
    *,
    baseline_phase: str = "acclimation",
) -> pd.DataFrame:
    """Compute per-session phase mean deltas vs baseline.

    Returns a tidy table with columns:
    session_id, protocol_phase, baseline_mean, phase_mean, delta
    """
    d = df.copy()
    if "valid_joint_minute" in d.columns:
        d = d.loc[d["valid_joint_minute"] == 1].copy()
    if d.empty or metric not in d.columns or ("protocol_phase" not in d.columns) or ("session_id" not in d.columns):
        return pd.DataFrame()

    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])

    out_rows = []
    for sid, ds in d.groupby("session_id"):
        base = ds.loc[ds["protocol_phase"].astype(str) == str(baseline_phase), metric]
        if base.dropna().empty:
            continue
        base_mean = float(np.nanmean(base.to_numpy(dtype=float)))
        for ph, dph in ds.groupby(ds["protocol_phase"].astype(str)):
            ph_mean = float(np.nanmean(dph[metric].to_numpy(dtype=float)))
            out_rows.append(
                {
                    "session_id": str(sid),
                    "protocol_phase": str(ph),
                    "baseline_phase": str(baseline_phase),
                    "baseline_mean": base_mean,
                    "phase_mean": ph_mean,
                    "delta": ph_mean - base_mean,
                }
            )
    return pd.DataFrame(out_rows)


def plot_cohort_phase_delta_across_sessions(
    pdf: PdfPages,
    df: pd.DataFrame,
    *,
    session_id: str,
    metric: str,
    ylabel: str,
    baseline_phase: str = "acclimation",
    phase_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Plot distribution of per-session deltas vs baseline (phase means). Returns the tidy delta table."""
    deltas = compute_cohort_phase_deltas_across_sessions(df, metric, baseline_phase=baseline_phase)
    if deltas.empty:
        return deltas

    if phase_order is None:
        phase_order = list(dict.fromkeys(deltas["protocol_phase"].astype(str).tolist()))
    phase_order = [p for p in phase_order if (deltas["protocol_phase"].astype(str) == p).any()]
    if not phase_order:
        return deltas

    data = []
    ns = []
    meds = []
    for p in phase_order:
        arr = pd.to_numeric(deltas.loc[deltas["protocol_phase"].astype(str) == p, "delta"], errors="coerce").dropna().to_numpy(dtype=float)
        data.append(arr)
        ns.append(int(arr.size))
        meds.append(float(np.nanmedian(arr)) if arr.size else float("nan"))
    if sum(ns) < 5:
        return deltas

    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    xpos = np.arange(1, len(phase_order) + 1)

    parts = ax.violinplot(data, positions=xpos, widths=0.85, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts.get("bodies", []):
        pc.set_alpha(0.55)
        pc.set_edgecolor("0.35")
        pc.set_linewidth(0.6)

    ax.boxplot(
        data,
        positions=xpos,
        widths=0.20,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="0.25", linewidth=0.9, alpha=0.95),
        medianprops=dict(color="0.10", linewidth=1.3),
        whiskerprops=dict(color="0.25", linewidth=0.8),
        capprops=dict(color="0.25", linewidth=0.8),
    )

    # 0-line
    ax.axhline(0.0, linewidth=0.9, linestyle="--", color="0.25", alpha=0.7)

    ymax = np.nanmax([np.nanmax(a) if a.size else np.nan for a in data])
    ymin = np.nanmin([np.nanmin(a) if a.size else np.nan for a in data])
    yr = float(ymax - ymin) if np.isfinite(ymax) and np.isfinite(ymin) and ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.10 * yr, ymax + 0.22 * yr)
    for i, (n, med) in enumerate(zip(ns, meds), start=1):
        ax.text(i, ymax + 0.18 * yr, f"n_sessions={n}", ha="center", va="bottom", fontsize=PLOT_FONTS['LEGEND'], color="0.25")
        if math.isfinite(med):
            ax.text(i, ymax + 0.12 * yr, f"med={med:.2g}", ha="center", va="bottom", fontsize=PLOT_FONTS['LEGEND'], color="0.25")

    _prep_ax(ax, session_id=session_id, title=f"{ylabel} — Phase delta vs {baseline_phase}")
    ax.set_ylabel(f"Δ {ylabel}", fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.set_xlabel("protocol_phase", fontsize=PLOT_FONTS['AXIS_LABEL'])
    ax.set_xticks(xpos)
    ax.set_xticklabels(phase_order, rotation=30, ha="right", fontsize=9)

    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)

    return deltas


def plot_phase_correlation(pdf: PdfPages, df: pd.DataFrame, *, session_id: str, metrics: List[str]):
    """
    Phase-level Spearman correlation (valid minutes only), with cleaned labels and cell annotations.
    """
    d = df.copy()
    d = d.loc[d["valid_joint_minute"] == 1].copy()
    if d.empty:
        return

    cols = [c for c in metrics if c in d.columns]
    if len(cols) < 3:
        return

    # Phase-level mean matrix
    g = d.groupby("protocol_phase")[cols].mean(numeric_only=True)

    # Clean labels
    def clean(name: str) -> str:
        return name.replace("_mean", "").replace("__", "_").strip("_")

    g = g.rename(columns={c: clean(c) for c in g.columns})

    corr = g.corr(method="spearman")

    # Use the same heatmap renderer across *all* correlation pages (consistent figsize + typography)
    scope = "all" if is_all_sessions_id(session_id) else "session"
    title = format_plot_title(scope, session_id if scope == "session" else "", "Phase-level Spearman correlation (valid_only)")
    # symmetric range, center at 0, but keep readable
    vmax = float(np.nanmax(np.abs(corr.to_numpy())))
    vmax = max(0.3, min(1.0, vmax))
    plot_corr_heatmap(pdf, corr, title=title, vmin=-vmax, vmax=vmax, cmap="RdBu_r")



def plot_phase_correlation_pvalues(pdf: PdfPages, df: pd.DataFrame, *, session_id: str, metrics: List[str]):
    """
    Phase-level Spearman correlation p-values computed on phase-level means (valid minutes only).

    This is a *diagnostic* complement to the ρ matrix. It answers: 'how surprising is this ρ under H0?'
    (Still not a causal statement; interpret alongside protocol design + multiplicity.)
    """
    d = df.copy()
    d = d.loc[d["valid_joint_minute"] == 1].copy()
    if d.empty:
        return

    cols = [c for c in metrics if c in d.columns]
    if len(cols) < 3:
        return

    g = d.groupby("protocol_phase")[cols].mean(numeric_only=True)

    def clean(name: str) -> str:
        return name.replace("_mean", "").replace("__", "_").strip("_")

    g = g.rename(columns={c: clean(c) for c in g.columns})

    # Pairwise p-values
    names = list(g.columns)
    pmat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        xa = g[a].to_numpy(dtype=float)
        for j, b in enumerate(names):
            if j < i:
                pmat.iat[i, j] = pmat.iat[j, i]
                continue
            if i == j:
                pmat.iat[i, j] = 0.0
                continue
            xb = g[b].to_numpy(dtype=float)
            # Spearman on phase means; handle NaNs defensively
            mask = np.isfinite(xa) & np.isfinite(xb)
            if mask.sum() < 3:
                p = np.nan
            else:
                _, p = spearmanr(xa[mask], xb[mask])
            pmat.iat[i, j] = float(p) if p is not None else np.nan
            pmat.iat[j, i] = pmat.iat[i, j]

    scope = "all" if is_all_sessions_id(session_id) else "session"
    title = format_plot_title(scope, session_id if scope == "session" else "", "Phase-level correlation p-values of phase means")
    # cap at 0.10 for color readability; annotate actual numbers via cell text in plot_corr_heatmap
    plot_corr_heatmap(pdf, pmat, title=title, vmin=0.0, vmax=0.10, cmap="viridis_r", fmt="{:.3f}")



def _rank(x: np.ndarray) -> np.ndarray:
    """Rank-transform with average ranks; returns float array."""
    from scipy.stats import rankdata
    return rankdata(x, method="average").astype(float)


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Partial Spearman correlation between x and y controlling for z.

    Implementation:
      - rank-transform x,y,z
      - regress out z from x and y via least squares (1D)
      - Pearson correlation of residuals

    Returns np.nan if not enough samples or z has near-zero variance.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 5:
        return float("nan")

    xr = _rank(x[mask])
    yr = _rank(y[mask])
    zr = _rank(z[mask])

    zc = zr - zr.mean()
    vz = float(np.dot(zc, zc))
    if vz < 1e-12:
        return float("nan")

    bx = float(np.dot(xr - xr.mean(), zc) / vz)
    by = float(np.dot(yr - yr.mean(), zc) / vz)

    rx = (xr - xr.mean()) - bx * zc
    ry = (yr - yr.mean()) - by * zc

    denom = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Fast autocorrelation estimate up to max_lag (normalized), ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return np.full(max_lag + 1, np.nan, dtype=float)
    x = x - np.mean(x)
    var = float(np.dot(x, x))
    if var <= 1e-12:
        acf = np.zeros(max_lag + 1, dtype=float)
        acf[0] = 1.0
        return acf
    r = np.correlate(x, x, mode="full")[n - 1 : n + max_lag]
    return (r / var).astype(float)

def estimate_iat_minutes(x: np.ndarray, *, max_lag: int = 60) -> float:
    """Estimate integrated autocorrelation time (IAT) in minutes using initial positive sequence."""
    acf = _acf_1d(x, max_lag=max_lag)
    if not np.isfinite(acf[0]):
        return float("nan")
    s = 0.0
    for k in range(1, len(acf)):
        if not np.isfinite(acf[k]) or acf[k] <= 0:
            break
        s += float(acf[k])
    tau = 1.0 + 2.0 * s
    return float(max(1.0, tau))

def estimate_block_length_minutes(
    df_minute: pd.DataFrame,
    metrics: List[str],
    *,
    valid_col_candidates: Tuple[str, ...] = ("minute_valid_joint", "valid_joint", "minute_valid", "valid"),
    max_lag: int = 60,
    agg: str = "max",
) -> Tuple[int, Dict[str, Any]]:
    """Choose block length for block bootstrap based on autocorrelation in this dataset."""
    d = df_minute.copy()
    valid_col = next((c for c in valid_col_candidates if c in d.columns), None)
    if valid_col is not None:
        v = pd.to_numeric(d[valid_col], errors="coerce").fillna(0).astype(int)
        d = d[v == 1].copy()

    keep = [m for m in metrics if m in d.columns]
    diag: Dict[str, Any] = {"iat_minutes": {}, "method": "IAT(initial-positive-sequence)", "max_lag": int(max_lag), "agg": agg}
    if not keep or len(d) < 20:
        return int(BOOTSTRAP_BLOCK_MINUTES_MIN), diag

    for m in keep:
        x = pd.to_numeric(d[m], errors="coerce").to_numpy(dtype=float)
        diag["iat_minutes"][m] = estimate_iat_minutes(x, max_lag=max_lag)

    taus = [t for t in diag["iat_minutes"].values() if np.isfinite(t)]
    if not taus:
        return int(BOOTSTRAP_BLOCK_MINUTES_MIN), diag

    tau_ref = float(np.nanmax(taus) if agg == "max" else np.nanmedian(taus))
    block_len = int(np.clip(int(np.ceil(2.0 * tau_ref)), BOOTSTRAP_BLOCK_MINUTES_MIN, BOOTSTRAP_BLOCK_MINUTES_MAX))
    diag["tau_ref_minutes"] = tau_ref
    diag["block_len_minutes"] = block_len
    return block_len, diag

def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Sample indices 0..n-1 using contiguous blocks with replacement."""
    if n <= 0:
        return np.array([], dtype=int)
    block_len = max(1, int(block_len))
    starts = rng.integers(0, n, size=int(np.ceil(n / block_len)))
    idx: List[int] = []
    for s in starts:
        idx.extend([(int(s) + k) % n for k in range(block_len)])
        if len(idx) >= n:
            break
    return np.asarray(idx[:n], dtype=int)


def compute_minute_corr_bootstrap(
    df_minute: pd.DataFrame,
    metrics: List[str],
    *,
    valid_only: bool = True,
    n_boot: int = BOOTSTRAP_N,
    block_len_minutes: int = BOOTSTRAP_BLOCK_MINUTES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Minute-level Spearman correlation with block bootstrap to reduce autocorrelation bias.

    Returns (rho_mean, rho_lo, rho_hi) as DataFrames indexed/columned by metrics.
    """
    d = df_minute.copy()
    if valid_only and "minute_valid_joint" in d.columns:
        d = d[d["minute_valid_joint"] == 1].copy()

    keep = [c for c in metrics if c in d.columns and c != control_var]
    if len(keep) < 2:
        empty = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
        return empty, empty, empty

    d = d[keep].astype(float).dropna(axis=0, how="all")
    if len(d) < 8:
        empty = pd.DataFrame(index=keep, columns=keep, dtype=float)
        return empty, empty, empty

    X = d.to_numpy()
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    rhos = np.full((n_boot, len(keep), len(keep)), np.nan, dtype=float)
    for b in range(n_boot):
        idx = _block_bootstrap_indices(n, block_len_minutes, rng)
        xb = X[idx, :]
        db = pd.DataFrame(xb, columns=keep)
        cm = db.corr(method="spearman", min_periods=5).to_numpy()
        rhos[b, :, :] = cm

    rho_mean = np.nanmean(rhos, axis=0)
    rho_lo = np.nanpercentile(rhos, 2.5, axis=0)
    rho_hi = np.nanpercentile(rhos, 97.5, axis=0)

    return (
        pd.DataFrame(rho_mean, index=keep, columns=keep),
        pd.DataFrame(rho_lo, index=keep, columns=keep),
        pd.DataFrame(rho_hi, index=keep, columns=keep),
    )


def compute_minute_partial_corr_bootstrap(
    df_minute: pd.DataFrame,
    metrics: List[str],
    *,
    control_var: str = PARTIAL_CONTROL_VAR,
    valid_only: bool = True,
    n_boot: int = BOOTSTRAP_N,
    block_len_minutes: int = BOOTSTRAP_BLOCK_MINUTES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Minute-level partial Spearman correlation controlling for `control_var`, with block bootstrap.

    Returns (rho_mean, rho_lo, rho_hi). If control variable missing, returns empty frames.
    """
    if control_var not in df_minute.columns:
        empty = pd.DataFrame(dtype=float)
        return empty, empty, empty

    d = df_minute.copy()
    if valid_only and "minute_valid_joint" in d.columns:
        d = d[d["minute_valid_joint"] == 1].copy()

    keep = [c for c in metrics if (c in d.columns) and (c != control_var)]
    if len(keep) < 2:
        empty = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
        return empty, empty, empty

    cols = keep + [control_var]
    d = d[cols].astype(float).dropna(axis=0, how="all")
    if len(d) < 10:
        empty = pd.DataFrame(index=keep, columns=keep, dtype=float)
        return empty, empty, empty

    X = d.to_numpy()
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    rhos = np.full((n_boot, len(keep), len(keep)), np.nan, dtype=float)
    zcol = len(cols) - 1

    for b in range(n_boot):
        idx = _block_bootstrap_indices(n, block_len_minutes, rng)
        xb = X[idx, :]
        z = xb[:, zcol]
        for i in range(len(keep)):
            rhos[b, i, i] = 1.0
            for j in range(i + 1, len(keep)):
                r = _partial_spearman(xb[:, i], xb[:, j], z)
                rhos[b, i, j] = r
                rhos[b, j, i] = r

    rho_mean = np.nanmean(rhos, axis=0)
    rho_lo = np.nanpercentile(rhos, 2.5, axis=0)
    rho_hi = np.nanpercentile(rhos, 97.5, axis=0)

    return (
        pd.DataFrame(rho_mean, index=keep, columns=keep),
        pd.DataFrame(rho_lo, index=keep, columns=keep),
        pd.DataFrame(rho_hi, index=keep, columns=keep),
    )


def _fisher_z(r: float) -> float:
    r = float(r)
    if not np.isfinite(r):
        return float("nan")
    r = max(min(r, 0.999999), -0.999999)
    return float(np.arctanh(r))


def _inv_fisher_z(z: float) -> float:
    z = float(z)
    if not np.isfinite(z):
        return float("nan")
    return float(np.tanh(z))


def meta_analyze_corr_matrices(
    mats: List[pd.DataFrame],
    weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Meta-analyze correlation matrices using Fisher z transform.

    mats: list of same-shape DataFrames with same index/columns.
    weights: optional list; if None, equal weights.

    Returns DataFrame of pooled correlations.
    """
    if not mats:
        return pd.DataFrame(dtype=float)

    idx = mats[0].index
    cols = mats[0].columns
    if weights is None:
        weights = [1.0] * len(mats)

    Z = np.zeros((len(idx), len(cols)), dtype=float)
    W = np.zeros_like(Z)

    for mat, w in zip(mats, weights):
        a = mat.reindex(index=idx, columns=cols).to_numpy()
        z = np.vectorize(_fisher_z)(a)
        mask = np.isfinite(z)
        Z[mask] += w * z[mask]
        W[mask] += w

    with np.errstate(invalid="ignore", divide="ignore"):
        Zm = Z / W
    R = np.vectorize(_inv_fisher_z)(Zm)
    return pd.DataFrame(R, index=idx, columns=cols)


def plot_corr_heatmap(
    pdf: PdfPages,
    mat: pd.DataFrame,
    *,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
    fmt: str | None = None,
):
    """Generic correlation heatmap page."""
    if mat is None or mat.empty:
        fig, ax = plt.subplots(figsize=CORR_FIGSIZE, dpi=300)
        ax.axis("off")
        ax.text(0.02, 0.98, f"{title}\n\n(not enough data to compute)", va="top", ha="left", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=CORR_FIGSIZE, dpi=300)
    im = ax.imshow(mat.to_numpy(), vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")

    # contrast-aware annotation colors for readability (esp. p-value heatmaps)
    def _ann_color(val, cmap_, norm_):
        try:
            rgba = cmap_(norm_(val))
            # relative luminance
            r,g,b = rgba[0], rgba[1], rgba[2]
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            return 'black' if lum > 0.55 else 'white'
        except Exception:
            return 'black'

    ax.set_title(title, fontsize=PLOT_FONTS['TITLE'], pad=6)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in mat.columns], rotation=45, ha="right", fontsize=PLOT_FONTS['TICK'])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([c.replace("_", " ") for c in mat.index], fontsize=PLOT_FONTS['TICK'])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iat[i, j]
            if np.isfinite(val):
                s = (fmt.format(val) if (fmt and ('{' in fmt)) else (f"{val:.2f}" if fmt is None else fmt % val))
                ax.text(j, i, s, ha="center", va="center", fontsize=CORR_CELL_FONTSIZE)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)



def corr_matrix_diagnostics(mat: pd.DataFrame, *, name: str, n_obs: Optional[int] = None) -> List[str]:
    """Return human-readable diagnostics for a correlation matrix.

    This does NOT 'prove causality' — it checks numerical/scientific sanity:
    - symmetric
    - diagonal ~ 1
    - values within [-1, 1]
    - no fully-empty rows/cols
    """
    lines: List[str] = []
    if mat is None or mat.empty:
        return [f"{name}: matrix is empty (insufficient data / missing metrics)."]
    arr = mat.to_numpy(dtype=float)
    k = arr.shape[0]
    if arr.shape[0] != arr.shape[1]:
        lines.append(f"{name}: not square (shape={arr.shape}).")
    if n_obs is not None:
        lines.append(f"{name}: N={int(n_obs)} observations.")
    # symmetry
    try:
        sym_err = np.nanmax(np.abs(arr - arr.T))
        if np.isfinite(sym_err) and sym_err > 1e-6:
            lines.append(f"{name}: WARNING not symmetric (max |A-Aᵀ|={sym_err:.3g}).")
    except Exception:
        pass
    # diag
    try:
        diag = np.diag(arr)
        bad_diag = np.where(np.isfinite(diag) & (np.abs(diag - 1.0) > 0.05))[0]
        if bad_diag.size > 0:
            lines.append(f"{name}: WARNING diagonal deviates from 1 (indices={bad_diag.tolist()[:6]}...).")
    except Exception:
        pass
    # range
    try:
        mx = float(np.nanmax(arr))
        mn = float(np.nanmin(arr))
        if np.isfinite(mx) and mx > 1.0 + 1e-6:
            lines.append(f"{name}: WARNING max correlation > 1 (max={mx:.3f}).")
        if np.isfinite(mn) and mn < -1.0 - 1e-6:
            lines.append(f"{name}: WARNING min correlation < -1 (min={mn:.3f}).")
    except Exception:
        pass
    # empty rows/cols
    try:
        row_all_nan = np.where(np.all(~np.isfinite(arr), axis=1))[0]
        if row_all_nan.size > 0:
            bad = [str(mat.index[i]) for i in row_all_nan[:10]]
            lines.append(f"{name}: WARNING metrics with all-NaN rows/cols: {', '.join(bad)}.")
    except Exception:
        pass
    if not lines:
        lines = [f"{name}: OK (numerically sane)."]
    return lines


def top_correlations(mat: pd.DataFrame, *, top_k: int = 8, min_abs: float = 0.15) -> List[str]:
    """Extract top |rho| off-diagonal pairs."""
    if mat is None or mat.empty:
        return []
    arr = mat.to_numpy(dtype=float)
    idx = list(mat.index)
    pairs = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[1]):
            r = arr[i, j]
            if not np.isfinite(r):
                continue
            if abs(r) < min_abs:
                continue
            pairs.append((abs(r), r, idx[i], idx[j]))
    pairs.sort(reverse=True, key=lambda t: t[0])
    out = []
    for a, r, i, j in pairs[:top_k]:
        out.append(f"ρ={r:+.2f}  {i} ↔ {j}")
    return out


def write_text_page(pdf: PdfPages, *, title: str, lines: List[str], wrap_width: int = 112, fontsize: int = 11) -> None:
    """Write a wrapped text page (prevents crossing page borders)."""
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=4)
    ax.text(0.02, 0.95, _wrap_block(lines, width=wrap_width), va="top", ha="left", fontsize=fontsize)
    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)



def plot_block_summary_table(pdf: PdfPages, df: pd.DataFrame, *, session_id: str):
    """
    Block-level summary table (valid_joint minutes only).

    Columns:
    - n_valid_minutes
    - valid_pct_of_block
    - key metric means: HR, EDA tonic, Temp, ENMO, SCR, Steps
    """
    d = df.copy()
    if d.empty:
        return

    # block duration in minutes from timeline (all minutes)
    block_counts = d.groupby("protocol_block")["minute_utc"].size().rename("n_minutes_block").to_frame()

    v = d.loc[d["valid_joint_minute"] == 1].copy()
    if v.empty:
        return

    agg_cols = {
        "hr_bpm_mean": "HR (bpm)",
        "eda_tonic_mean": "EDA tonic (uS)",
        "temp_smooth_C_mean": "Temp (C)",
        "acc_enmo_g_p95": "ENMO p95 (g)",
        "scr_count": "SCR (count/min)",
        "steps_count": "Steps (count/min)",
    }
    present = {k: lab for k, lab in agg_cols.items() if k in v.columns}

    g = v.groupby("protocol_block")[list(present.keys())].mean(numeric_only=True)
    g = g.rename(columns=present)

    n_valid = v.groupby("protocol_block")["minute_utc"].size().rename("n_valid_minutes").to_frame()
    out = block_counts.join(n_valid, how="left").join(g, how="left")
    out["n_valid_minutes"] = out["n_valid_minutes"].fillna(0).astype(int)
    out["valid_pct_of_block"] = (100.0 * out["n_valid_minutes"] / out["n_minutes_block"]).round(1)

    out = out.reset_index().rename(columns={"protocol_block": "Block"})
    cols = ["Block", "n_minutes_block", "n_valid_minutes", "valid_pct_of_block"] + list(present.values())
    out = out[cols]

    # Render as a clean figure table
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.axis("off")

    _prep_ax(ax, session_id=session_id, title="Block-level summary (valid_joint minutes)")

    cell_text = []
    for _, row in out.iterrows():
        r = []
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                r.append(f"{val:.3g}" if c not in ["valid_pct_of_block"] else f"{val:.1f}")
            else:
                r.append(str(val))
        cell_text.append(r)

    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    # Make sure the table never crosses the page border (paper-safe)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.95, 1.25)
    try:
        table.auto_set_column_width(col=list(range(len(cols))))
    except Exception:
        pass

    # header styling
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("0.80")
        if r == 0:
            cell.set_facecolor("0.93")
            cell.set_text_props(weight="bold", color="0.15")

    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)

def plot_summary_page(pdf: PdfPages, qc: Dict[str, Any], *, session_id: str):
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.axis("off")
    lines = [
        f"Session ID: {qc.get('session_id','')}",
        f"Participant: {qc.get('participant','')}",
        f"Window (UTC, minute grid): {qc.get('start_minute_utc','')}  →  {qc.get('end_minute_utc','')}",
        f"Minutes: total={qc.get('minutes_total', 'NA')} | valid_joint={qc.get('minutes_valid_joint','NA')} ({qc.get('pct_valid_joint', float('nan')):.1f}%) | any-OOB={qc.get('pct_oob_any', float('nan')):.1f}%",
        "",
        f"Sampling rate estimate (Hz): ACC={qc.get('fs_acc', float('nan')):.3g} | EDA={qc.get('fs_eda', float('nan')):.3g} | BVP={qc.get('fs_bvp', float('nan')):.3g} | TEMP={qc.get('fs_temp', float('nan')):.3g}",
        f"Detected gaps >{GAP_DETECTION_THRESHOLD_SECONDS:g}s: ACC={qc.get('gaps_acc',0)} | EDA={qc.get('gaps_eda',0)} | BVP={qc.get('gaps_bvp',0)} | TEMP={qc.get('gaps_temp',0)}",
        "",
        f"Coverage (% minutes with data): EDA={qc.get('cov_eda',float('nan')):.1f} | HR={qc.get('cov_hr',float('nan')):.1f} | TEMP={qc.get('cov_temp',float('nan')):.1f} | ACC={qc.get('cov_acc',float('nan')):.1f} | STEPS={qc.get('cov_steps',float('nan')):.1f}",
        f"NeuroKit2 available: {qc.get('neurokit2_available', False)}",
        "",
        "Validity rule (minute-level):",
        "  valid_joint_minute = HR+EDA+TEMP+ACC present AND within plausibility ranges AND NOT high_motion.",
        "  high_motion uses ENMO p95 > {HIGH_MOTION_ENMO_P95_THRESHOLD_G:.2f} g (protocol-dependent; edit in code if needed).",
        "",
        "Tunable parameters (edit in script):",
        f"  • HIGH_MOTION_ENMO_P95_THRESHOLD_G = {HIGH_MOTION_ENMO_P95_THRESHOLD_G:.2f} g",
        f"  • GAP_DETECTION_THRESHOLD_SECONDS = {GAP_DETECTION_THRESHOLD_SECONDS:g} s",
        f"  • PHASE_CORR_METRICS = {', '.join(PHASE_CORR_METRICS)}",
        f"  • BOOTSTRAP_N = {BOOTSTRAP_N} | BOOTSTRAP_BLOCK_MINUTES = {BOOTSTRAP_BLOCK_MINUTES} (min={BOOTSTRAP_BLOCK_MINUTES_MIN}, max={BOOTSTRAP_BLOCK_MINUTES_MAX})",
        "",
        "Plausibility ranges (minute features):",
    ]
    # Append plausibility ranges compactly (stable ordering)
    try:
        pr = qc.get('plausibility_ranges', {}) or {}
        for k in sorted(pr.keys()):
            v = pr.get(k, {}) or {}
            lo = v.get('min', None)
            hi = v.get('max', None)
            if lo is None and hi is None:
                continue
            lines.append(f"  - {k}: [{'' if lo is None else lo}, {'' if hi is None else hi}]")
    except Exception:
        pass
    ax.text(0.02, 0.98, _wrap_block(lines, width=112), va="top", ha="left", fontsize=11)
    ax.set_title(format_plot_title("session", session_id, "Session QC summary"), fontsize=12, pad=4)
    # NOTE: keep Matplotlib defaults (no tight_layout) for consistent margins
    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)

# =========================================================
# Session processing
# =========================================================

def qc_eda(eda_df: pd.DataFrame, fs: float) -> Dict[str, Any]:
    x = eda_df["eda_clean"].to_numpy(dtype=float)
    base = summarize_signal(x)
    base["outlier_pct_mad10"] = 100.0 * mad_outlier_fraction(x, k=10.0)
    min_run = int(max(5, (fs if np.isfinite(fs) else 4.0) * 10))
    base["flatline_pct"] = 100.0 * flatline_fraction(x, eps=1e-4, min_run=min_run)
    base["negative_count"] = int(np.sum(np.isfinite(x) & (x < 0)))
    base["gt_20us_count"] = int(np.sum(np.isfinite(x) & (x > 20)))
    base["gt_50us_count"] = int(np.sum(np.isfinite(x) & (x > 50)))
    return base

def qc_modality_minute(minute_feat: pd.DataFrame, col: str) -> Dict[str, Any]:
    return summarize_signal(pd.to_numeric(minute_feat[col], errors="coerce").to_numpy(dtype=float))


def plot_minute_qc_flags_heatmap(pdf: PdfPages, minute_feat: pd.DataFrame, *, session_id: str) -> None:
    """Minute-level QC flags heatmap (minutes × flags).

    This is designed to be compact and legible:
    - Rows = protocol minutes
    - Columns = QC flags (valid_* , high_motion, *_oob)
    - Values = 0/1
    """
    if minute_feat is None or minute_feat.empty:
        return
    d = minute_feat.copy()

    # Choose columns in a sensible order
    preferred = [
        "high_motion",
        "valid_hr_minute",
        "valid_eda_minute",
        "valid_temp_minute",
        "valid_acc_minute",
        "valid_joint_minute",
    ]
    oob = [c for c in d.columns if c.endswith("_oob")]
    cols = [c for c in preferred if c in d.columns] + sorted(oob)

    if not cols:
        return

    # Build numeric matrix (minutes × flags)
    X = []
    minute_index = None
    if "protocol_minute" in d.columns:
        minute_index = pd.to_numeric(d["protocol_minute"], errors="coerce")
    elif "minute_index" in d.columns:
        minute_index = pd.to_numeric(d["minute_index"], errors="coerce")
    else:
        minute_index = pd.Series(np.arange(len(d)), index=d.index)

    order = np.argsort(np.asarray(minute_index.fillna(pd.Series(np.arange(len(d)))), dtype=float))
    d = d.iloc[order].reset_index(drop=True)

    mat = np.zeros((len(d), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        v = d[c]
        if v.dtype.kind in "biu":
            vv = v.to_numpy(dtype=float)
        else:
            vv = pd.to_numeric(v, errors="coerce").fillna(0).to_numpy(dtype=float)
        mat[:, j] = (vv > 0).astype(float)

    # Plot
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)

    ax.set_yticks([])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)

    # Add a light minute axis scale
    if len(d) <= 200:
        yticks = np.linspace(0, len(d) - 1, num=min(10, len(d)), dtype=int)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(i)) for i in yticks], fontsize=8)
        ax.set_ylabel("protocol_minute", fontsize=PLOT_FONTS['AXIS_LABEL'])

    _prep_ax(ax, session_id=session_id, title="QC flags heatmap (1 = flag true)", enable_grid=False)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout(pad=0.15)
    pdf.savefig(fig)
    plt.close(fig)


def process_one_session(
    *,
    session_id: str,
    session_dir: Path,
    timeline_session: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Analysis-only mode: allow running from precomputed
    # minute/phase feature CSVs when raw files are absent.
    # This preserves the original raw-signal pipeline while
    # making the script usable on archived outputs.
    # -------------------------------------------------
    raw_required = ["accelerometer.csv","eda.csv","bvp.csv","temperature.csv","steps.csv"]
    raw_present = all((session_dir / f).exists() for f in raw_required)
    precomp_min = session_dir / "features_minute.csv"
    if (not raw_present) and precomp_min.exists():
        minute_feat = pd.read_csv(precomp_min)
        if "minute_utc" in minute_feat.columns:
            minute_feat["minute_utc"] = pd.to_datetime(minute_feat["minute_utc"], utc=True, errors="coerce")
        if "session_id" not in minute_feat.columns:
            minute_feat["session_id"] = session_id
        # Ensure protocol_minute exists
        if "protocol_minute" not in minute_feat.columns or minute_feat["protocol_minute"].isna().all():
            if "minute_utc" in minute_feat.columns:
                minute_feat = minute_feat.sort_values("minute_utc")
            minute_feat["protocol_minute"] = np.arange(len(minute_feat), dtype=int)
        minute_feat = add_validity_flags(minute_feat)
        phase_feat = aggregate_phase_qcaware(minute_feat)

        # Try load qc_summary if present; otherwise create minimal QC
        qc_path = session_dir / "qc_summary.json"
        if qc_path.exists():
            try:
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
            except Exception:
                qc = {}
        else:
            qc = {}
        qc = {**qc, "session_id": session_id, "participant": str(timeline_session["participant"].iloc[0]) if len(timeline_session) else ""}

        # Write upgraded outputs
        minute_feat.to_csv(out_dir / "features_minute.csv", index=False)
        phase_feat.to_csv(out_dir / "features_phase.csv", index=False)

        # QC flags export (best-effort)
        qc_cols = ["minute_utc","session_id","participant","minute_index","protocol_block","protocol_phase","expected_fan_mode",
                   "high_motion","valid_hr_minute","valid_eda_minute","valid_temp_minute","valid_acc_minute","valid_joint_minute"]
        qc_cols += [c for c in minute_feat.columns if c.endswith("_oob")]
        minute_feat[[c for c in qc_cols if c in minute_feat.columns]].to_csv(out_dir / "minute_qc_flags.csv", index=False)

        # Matrices export
        corr_metrics = ["hr_bpm_mean","eda_tonic_mean","eda_phasic_p95","temp_smooth_C_mean","acc_enmo_g_p95","steps_count","scr_count"]
        phase_means_valid, phase_corr, phase_p = compute_phase_mean_matrices(minute_feat, corr_metrics, valid_only=True, clean_labels=True)
        if not phase_means_valid.empty:
            phase_means_valid.to_csv(out_dir / "phase_means_valid.csv", index=True)
        if not phase_corr.empty:
            phase_corr.to_csv(out_dir / "phase_corr_spearman.csv", index=True)
        if not phase_p.empty:
            phase_p.to_csv(out_dir / "phase_corr_spearman_pvalues.csv", index=True)

        # PDF report (regenerate with upgraded plots)
        with open(out_dir / f"{session_id}_report.pdf", 'wb') as _pdf_fh:
            with PdfPages(_pdf_fh) as pdf:
                _safe_pdf_call(pdf, plot_summary_page, qc, session_id=session_id)
                # Major timeseries (if present)
                for ycol, ylabel, vcol, title in [
                    ("eda_tonic_mean","EDA tonic (µS)","valid_eda_minute","EDA tonic (minute mean)"),
                    ("eda_phasic_p95","EDA phasic p95 (µS)","valid_eda_minute","EDA phasic (minute p95)"),
                    ("scr_count","SCR count (peaks/min)","valid_eda_minute","SCR count per minute"),
                    ("hr_bpm_mean","Heart rate (bpm)","valid_hr_minute","Heart rate (minute mean)"),
                    ("temp_smooth_C_mean","Skin temperature (°C)","valid_temp_minute","Skin temperature (minute mean)"),
                    ("acc_enmo_g_p95","ENMO p95 (g)","valid_acc_minute","Acceleration ENMO (minute p95)"),
                    ("steps_count","Steps (count/min)","has_steps","Steps per minute"),
                ]:
                    if ycol in minute_feat.columns:
                        _safe_pdf_call(
                            pdf,
                            plot_timeseries_qcaware,
                            minute_feat,
                            session_id=session_id,
                            ycol=ycol,
                            ylabel=ylabel,
                            valid_col=vcol if vcol in minute_feat.columns else None,
                            title=title,
                        )
                # QC flags heatmap removed (per feedback: not informative)
                for metric, ylabel in [
                    ("hr_bpm_mean", "Heart rate (bpm)"),
                    ("eda_tonic_mean", "EDA tonic (µS)"),
                    ("temp_smooth_C_mean", "Skin temperature (°C)"),
                    ("acc_enmo_g_p95", "ENMO p95 (g)"),
                    ("scr_count", "SCR count (peaks/min)"),
                    ("steps_count", "Steps (count/min)"),
                ]:
                    if metric in minute_feat.columns:
                        _safe_pdf_call(pdf, plot_phase_distributions, minute_feat, session_id=session_id, metric=metric, ylabel=ylabel)
                        _safe_pdf_call(pdf, plot_phase_delta_from_baseline, minute_feat, session_id=session_id, metric=metric, ylabel=ylabel)
                _safe_pdf_call(pdf, plot_block_summary_table, minute_feat, session_id=session_id)
                _safe_pdf_call(pdf, plot_phase_correlation, minute_feat, session_id=session_id, metrics=corr_metrics)

                # Determine bootstrap block length for this session (minutes). If BOOTSTRAP_BLOCK_MINUTES='auto',
                # use an autocorrelation-based recommendation; otherwise use the user-provided integer.
                try:
                    rec_block_len, rec_diag = estimate_block_length_minutes(minute_feat, MINUTE_CORR_METRICS)
                    bootstrap_block_len_used = rec_block_len if str(BOOTSTRAP_BLOCK_MINUTES).lower() == "auto" else int(BOOTSTRAP_BLOCK_MINUTES)
                    bootstrap_block_len_used = int(np.clip(bootstrap_block_len_used, BOOTSTRAP_BLOCK_MINUTES_MIN, BOOTSTRAP_BLOCK_MINUTES_MAX))
                    qc.setdefault("bootstrap", {})["block_len_minutes_used"] = bootstrap_block_len_used
                    qc.setdefault("bootstrap", {})["block_len_minutes_recommended"] = rec_block_len
                    qc.setdefault("bootstrap", {})["autocorr_diagnostics"] = rec_diag
                except Exception as e:
                    bootstrap_block_len_used = int(BOOTSTRAP_BLOCK_MINUTES_MIN)
                    qc.setdefault("warnings", []).append(f"Bootstrap block-length validation failed: {e!r}")
                # --- Deeper coupling analysis (minute-level, robust to autocorrelation) ---
                # Minute-level Spearman correlation with block bootstrap (valid minutes only).
                try:
                    rho_mean, rho_lo, rho_hi = compute_minute_corr_bootstrap(
                        minute_feat,
                        MINUTE_CORR_METRICS,
                        valid_only=True,
                        n_boot=BOOTSTRAP_N,
                        block_len_minutes=bootstrap_block_len_used,
                        seed=BOOTSTRAP_SEED,
                    )
                    if not rho_mean.empty:
                        rho_mean.to_csv(out_dir / "minute_corr_spearman_bootstrap_mean.csv")
                        rho_lo.to_csv(out_dir / "minute_corr_spearman_bootstrap_ci95_lo.csv")
                        rho_hi.to_csv(out_dir / "minute_corr_spearman_bootstrap_ci95_hi.csv")
                        _safe_pdf_call(
                            pdf,
                            plot_corr_heatmap,
                            rho_mean,
                            title=f"{format_plot_title('session', session_id, 'Minute-level Spearman correlation (block bootstrap, valid_only)')}",
                        )
                except Exception as e:
                    qc.setdefault("warnings", []).append(f"Minute-level bootstrap Spearman correlation failed: {e!r}")

                # Minute-level partial Spearman correlation controlling for motion proxy (ENMO).
                try:
                    prho_mean, prho_lo, prho_hi = compute_minute_partial_corr_bootstrap(
                        minute_feat,
                        MINUTE_CORR_METRICS,
                        control_var=PARTIAL_CONTROL_VAR,
                        valid_only=True,
                        n_boot=BOOTSTRAP_N,
                        block_len_minutes=bootstrap_block_len_used,
                        seed=BOOTSTRAP_SEED,
                    )
                    if prho_mean is not None and not prho_mean.empty:
                        prho_mean.to_csv(out_dir / "minute_partial_corr_spearman_bootstrap_mean.csv")
                        prho_lo.to_csv(out_dir / "minute_partial_corr_spearman_bootstrap_ci95_lo.csv")
                        prho_hi.to_csv(out_dir / "minute_partial_corr_spearman_bootstrap_ci95_hi.csv")
                        _safe_pdf_call(
                            pdf,
                            plot_corr_heatmap,
                            prho_mean,
                            title=f"{format_plot_title('session', session_id, f'Partial Spearman correlation (block bootstrap, control={PARTIAL_CONTROL_VAR})')}",
                        )
                except Exception as e:
                    qc.setdefault("warnings", []).append(f"Minute-level partial bootstrap correlation failed: {e!r}")



        return qc


    def load_modality(filename: str) -> pd.DataFrame:
        f = session_dir / filename
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")
        return read_csv_safely(f)

    acc = ensure_datetime_utc(load_modality("accelerometer.csv"), ts_col_candidates=["unix_timestamp_us"])
    eda = ensure_datetime_utc(load_modality("eda.csv"), ts_col_candidates=["unix_timestamp_us"])
    bvp = ensure_datetime_utc(load_modality("bvp.csv"), ts_col_candidates=["unix_timestamp_us"])
    temp = ensure_datetime_utc(load_modality("temperature.csv"), ts_col_candidates=["unix_timestamp_us"])
    steps = ensure_datetime_utc(load_modality("steps.csv"), ts_col_candidates=["unix_timestamp_us"])

    fs_acc = empirical_sampling_rate(acc)
    fs_eda = empirical_sampling_rate(eda)
    fs_bvp = empirical_sampling_rate(bvp)
    fs_temp = empirical_sampling_rate(temp)

    gaps_acc = detect_gaps(acc, fs_acc)
    gaps_eda = detect_gaps(eda, fs_eda)
    gaps_bvp = detect_gaps(bvp, fs_bvp)
    gaps_temp = detect_gaps(temp, fs_temp)

    acc_p, _ = acc_process(acc)
    eda_p, _ = eda_process(eda)
    bvp_p, _ = bvp_process(bvp)
    temp_p, _ = temp_process(temp)
    steps_p = steps_process(steps)

    minute_feat = build_minute_features(
        session_id=session_id,
        timeline_session=timeline_session,
        eda_df=eda_p,
        bvp_df=bvp_p,
        temp_df=temp_p,
        acc_df=acc_p,
        steps_df=steps_p,
    )
    minute_feat = add_validity_flags(minute_feat)
    phase_feat = aggregate_phase_qcaware(minute_feat)

    # -------------------------------
    # Matrices export (CSV)
    # -------------------------------
    corr_metrics = ["hr_bpm_mean","eda_tonic_mean","eda_phasic_p95","temp_smooth_C_mean","acc_enmo_g_p95","steps_count","scr_count"]
    phase_means_valid, phase_corr, phase_p = compute_phase_mean_matrices(minute_feat, corr_metrics, valid_only=True, clean_labels=True)
    if not phase_means_valid.empty:
        phase_means_valid.to_csv(out_dir / "phase_means_valid.csv", index=True)
    if not phase_corr.empty:
        phase_corr.to_csv(out_dir / "phase_corr_spearman.csv", index=True)
    if not phase_p.empty:
        phase_p.to_csv(out_dir / "phase_corr_spearman_pvalues.csv", index=True)



    # Write outputs
    minute_feat.to_csv(out_dir / "features_minute.csv", index=False)
    phase_feat.to_csv(out_dir / "features_phase.csv", index=False)

    # QC flags export
    qc_cols = ["minute_utc","session_id","participant","minute_index","protocol_block","protocol_phase","expected_fan_mode",
               "high_motion","valid_hr_minute","valid_eda_minute","valid_temp_minute","valid_acc_minute","valid_joint_minute"]
    qc_cols += [c for c in minute_feat.columns if c.endswith("_oob")]
    minute_feat[[c for c in qc_cols if c in minute_feat.columns]].to_csv(out_dir / "minute_qc_flags.csv", index=False)

    # Signal QC summary
    signal_qc_rows = [
        {"session_id": session_id, "participant": str(timeline_session["participant"].iloc[0]), "modality": "EDA_raw_clean", **qc_eda(eda_p, fs_eda)},
        {"session_id": session_id, "participant": str(timeline_session["participant"].iloc[0]), "modality": "HR_minute_mean", **qc_modality_minute(minute_feat, "hr_bpm_mean")},
        {"session_id": session_id, "participant": str(timeline_session["participant"].iloc[0]), "modality": "TEMP_minute_mean", **qc_modality_minute(minute_feat, "temp_smooth_C_mean")},
        {"session_id": session_id, "participant": str(timeline_session["participant"].iloc[0]), "modality": "ENMO_minute_mean", **qc_modality_minute(minute_feat, "acc_enmo_g_mean")},
    ]
    pd.DataFrame(signal_qc_rows).to_csv(out_dir / "signal_qc_summary.csv", index=False)

    cov = {
        "cov_eda": 100.0 * float(minute_feat["eda_tonic_mean"].notna().mean()),
        "cov_hr": 100.0 * float(minute_feat["hr_bpm_mean"].notna().mean()),
        "cov_temp": 100.0 * float(minute_feat["temp_smooth_C_mean"].notna().mean()),
        "cov_acc": 100.0 * float(minute_feat["acc_enmo_g_mean"].notna().mean()),
        "cov_steps": 100.0 * float(minute_feat["steps_count"].notna().mean()),
        "pct_valid_joint": 100.0 * float((minute_feat["valid_joint_minute"] == 1).mean()),
    }

    # Session-level coverage stats for report front-matter
    minutes_total = int(len(minute_feat))
    minutes_valid_joint = int((minute_feat['valid_joint_minute'] == 1).sum()) if 'valid_joint_minute' in minute_feat.columns else 0
    start_minute_utc = str(pd.to_datetime(minute_feat['minute_utc'].min(), utc=True)) if 'minute_utc' in minute_feat.columns and minutes_total else ''
    end_minute_utc = str(pd.to_datetime(minute_feat['minute_utc'].max(), utc=True)) if 'minute_utc' in minute_feat.columns and minutes_total else ''
    oob_cols = [c for c in minute_feat.columns if c.endswith('_oob')]
    pct_oob_any = 100.0 * float(minute_feat[oob_cols].any(axis=1).mean()) if oob_cols else 0.0
    qc = {
        "session_id": session_id,
        "participant": str(timeline_session["participant"].iloc[0]) if len(timeline_session) else "",
        "start_minute_utc": start_minute_utc,
        "end_minute_utc": end_minute_utc,
        "minutes_total": minutes_total,
        "minutes_valid_joint": minutes_valid_joint,
        "pct_oob_any": pct_oob_any,
        "neurokit2_available": HAVE_NK,
        "fs_acc": fs_acc,
        "fs_eda": fs_eda,
        "fs_bvp": fs_bvp,
        "fs_temp": fs_temp,
        "gaps_acc": int(len(gaps_acc)),
        "gaps_eda": int(len(gaps_eda)),
        "gaps_bvp": int(len(gaps_bvp)),
        "gaps_temp": int(len(gaps_temp)),
        **cov,
        "plausibility_ranges": PLAUSIBILITY_RANGES,
    }
    (out_dir / "qc_summary.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")

    # Output schema
    schema: List[Dict[str, Any]] = []
    schema += schema_from_dataframe("features_minute.csv", "Minute-aligned features (one row per protocol minute).", minute_feat, session_id)
    schema += schema_from_dataframe("features_phase.csv", "Phase aggregated features (all + valid_only).", phase_feat, session_id)
    # Matrices
    if (out_dir / "phase_means_valid.csv").exists():
        schema += schema_from_dataframe("phase_means_valid.csv", "Phase mean table (valid_joint minutes only).", pd.read_csv(out_dir / "phase_means_valid.csv"), session_id)
    if (out_dir / "phase_corr_spearman.csv").exists():
        schema += schema_from_dataframe("phase_corr_spearman.csv", "Spearman correlation matrix computed on phase means (valid_joint).", pd.read_csv(out_dir / "phase_corr_spearman.csv"), session_id)
    if (out_dir / "phase_corr_spearman_pvalues.csv").exists():
        schema += schema_from_dataframe("phase_corr_spearman_pvalues.csv", "P-values for Spearman correlation matrix on phase means (valid_joint).", pd.read_csv(out_dir / "phase_corr_spearman_pvalues.csv"), session_id)
    schema += schema_from_dataframe("minute_qc_flags.csv", "Minute-level QC flags (booleans).", pd.read_csv(out_dir / "minute_qc_flags.csv"), session_id)
    schema += schema_from_dataframe("signal_qc_summary.csv", "Signal-level QC summaries.", pd.read_csv(out_dir / "signal_qc_summary.csv"), session_id)
    # qc_summary keys
    for k, v in qc.items():
        if k == "plausibility_ranges":
            continue
        schema.append({"file":"qc_summary.json","file_description":"Session-level QC metadata (JSON).","column":k,"dtype":type(v).__name__,"description":"","session_id":session_id})
    qc['_output_schema'] = schema

    # PDF report
    with open(out_dir / f"{session_id}_report.pdf", 'wb') as _pdf_fh:
        with PdfPages(_pdf_fh) as pdf:
            plot_summary_page(pdf, qc, session_id=session_id)

            plot_timeseries_qcaware(pdf, minute_feat, session_id=session_id,
                                    ycol="eda_tonic_mean", ylabel="EDA tonic (µS)", valid_col="valid_eda_minute",
                                    title="EDA tonic (minute mean)")
            plot_timeseries_qcaware(pdf, minute_feat, session_id=session_id,
                                    ycol="eda_phasic_p95", ylabel="EDA phasic p95 (µS)", valid_col="valid_eda_minute",
                                    title="EDA phasic (minute p95)")
            plot_timeseries_qcaware(pdf, minute_feat, session_id=session_id,
                                    ycol="scr_count", ylabel="SCR count (peaks/min)", valid_col="valid_eda_minute",
                                    title="SCR count per minute")
            plot_timeseries_qcaware(pdf, minute_feat, session_id=session_id,
                                    ycol="hr_bpm_mean", ylabel="Heart rate (bpm)", valid_col="valid_hr_minute",
                                    title="Heart rate (minute mean)")
            plot_timeseries_qcaware(pdf, minute_feat, session_id=session_id,
                                    ycol="temp_smooth_C_mean", ylabel="Skin temperature (°C)", valid_col="valid_temp_minute",
                                    title="Skin temperature (minute mean)")
            plot_timeseries_qcaware(pdf, minute_feat, session_id=session_id,
                                    ycol="acc_enmo_g_p95", ylabel="ENMO p95 (g)", valid_col="valid_acc_minute",
                                    title="Acceleration ENMO (minute p95)",
                                    extra_hlines=[(HIGH_MOTION_ENMO_P95_THRESHOLD_G, "high_motion threshold")])
            plot_timeseries_qcaware(pdf, minute_feat, session_id=session_id,
                                    ycol="steps_count", ylabel="Steps (count/min)", valid_col="has_steps",
                                    title="Steps per minute")
            # QC flags heatmap removed (per feedback: not informative)

            # Phase interpretation pages (boxplots + baseline deltas)
            for metric, ylabel in [
                ("hr_bpm_mean", "Heart rate (bpm)"),
                ("eda_tonic_mean", "EDA tonic (µS)"),
                ("temp_smooth_C_mean", "Skin temperature (°C)"),
                ("acc_enmo_g_p95", "ENMO p95 (g)"),
                ("scr_count", "SCR count (peaks/min)"),
                ("steps_count", "Steps (count/min)"),
            ]:
                plot_phase_distributions(pdf, minute_feat, session_id=session_id, metric=metric, ylabel=ylabel)
                plot_phase_delta_from_baseline(pdf, minute_feat, session_id=session_id, metric=metric, ylabel=ylabel)

            plot_block_summary_table(pdf, minute_feat, session_id=session_id)

            plot_phase_correlation(pdf, minute_feat, session_id=session_id, metrics=["hr_bpm_mean","eda_tonic_mean","eda_phasic_p95","temp_smooth_C_mean","acc_enmo_g_p95","steps_count","scr_count"])

            # Complementary significance view
            plot_phase_correlation_pvalues(pdf, minute_feat, session_id=session_id, metrics=["hr_bpm_mean","eda_tonic_mean","eda_phasic_p95","temp_smooth_C_mean","acc_enmo_g_p95","steps_count","scr_count"])
    return qc

# =========================================================
# Batch driver
# =========================================================

def find_session_dirs(sessions_root: Path) -> List[Path]:
    out = []
    for p in sorted(sessions_root.iterdir()):
        if p.is_dir() and ((p / "eda.csv").exists() or (p / "features_minute.csv").exists()):
            out.append(p)
    return out


def _quantiles(x: np.ndarray, qs: List[float]) -> List[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [float("nan") for _ in qs]
    return [float(np.quantile(x, q)) for q in qs]


def generate_combined_results(
    *,
    out_root: Path,
    sessions: List[Tuple[str, Path]],
    timeline: pd.DataFrame,
    tz_name: str,
    combined_max_participants: int = 0,
):
    """
    Create a batch-level (multi-session) combined report + CSVs.

    Strategy:
    - Align sessions by protocol_minute (minute index within each session timeline).
    - For each metric, compute per-minute mean + [p10, p90] "color-burst envelope".
    - Plot one page per metric: envelope + mean, with block-only header band inside axis.
    - Export combined CSVs for downstream stats/manuscript tables.
    """
    if len(sessions) < 2:
        return

    combined_dir = out_root / "all_sessions"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # --- choose a template timeline (first session) for block layout ---
    sid0 = sessions[0][0]
    tl0 = timeline.loc[timeline["session_id"].astype(str) == str(sid0)].copy()
    if tl0.empty:
        return
    tl0 = tl0.sort_values("minute_utc")
    tl0["protocol_minute"] = np.arange(len(tl0), dtype=int)

    # Dataframe used only for shading + block labels (minute_utc must be tz-aware in source, we keep it)
    template_blocks = tl0[[c for c in ["minute_utc","protocol_block","protocol_phase","protocol_minute"] if c in tl0.columns]].copy()

    # --- load all sessions minute features ---
    dfs = []
    phase_dfs: List[pd.DataFrame] = []
    for sid, _ in sessions:
        f = out_root / sid / "features_minute.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if "session_id" not in d.columns:
            d["session_id"] = sid
        # Ensure minute_utc exists
        if "minute_utc" in d.columns:
            d["minute_utc"] = pd.to_datetime(d["minute_utc"], utc=True, errors="coerce")
        # Ensure protocol_minute exists
        if "protocol_minute" not in d.columns or d["protocol_minute"].isna().all():
            if "minute_utc" in d.columns:
                d = d.sort_values("minute_utc")
            d["protocol_minute"] = np.arange(len(d), dtype=int)
        d["protocol_minute"] = pd.to_numeric(d["protocol_minute"], errors="coerce").astype("Int64")
        dfs.append(d)
        # Also gather phase-level features if present (for combined CSV)
        try:
            fp = out_root / sid / "features_phase.csv"
            if fp.exists():
                dp = pd.read_csv(fp)
                if "session_id" not in dp.columns:
                    dp["session_id"] = sid
                phase_dfs.append(dp)
        except Exception:
            pass

    if not dfs:
        return
    all_min = pd.concat(dfs, ignore_index=True)

    # Write combined phase-level table (stacked across sessions) if available
    if phase_dfs:
        all_phase = pd.concat(phase_dfs, ignore_index=True)
        all_phase.to_csv(combined_dir / "all_sessions_features_phase.csv", index=False)



    # Prefer p95 ENMO if present
    metric_candidates = [
        ("hr_bpm_mean", "Heart rate (bpm)"),
        ("eda_tonic_mean", "EDA tonic (µS)"),
        ("eda_phasic_p95", "EDA phasic p95 (µS)"),
        ("scr_count", "SCR count (peaks/min)"),
        ("temp_smooth_C_mean", "Skin temperature (°C)"),
        ("acc_enmo_g_p95", "ENMO p95 (g)"),
        ("acc_enmo_g_mean", "ENMO mean (g)"),
        ("steps_count", "Steps (count/min)"),
    ]

    # Keep only existing metrics (dedupe ENMO choice)
    existing = []
    seen = set()
    for col, lab in metric_candidates:
        if col in all_min.columns and col not in seen:
            # If p95 exists, skip mean ENMO
            if col == "acc_enmo_g_mean" and "acc_enmo_g_p95" in all_min.columns:
                continue
            existing.append((col, lab))
            seen.add(col)

    combined_rows = []
    for col, lab in existing:
        g = all_min.groupby("protocol_minute")[col]
        rows = []
        for pm, series in g:
            q10, q50, q90 = _quantiles(series.to_numpy(), [0.10, 0.50, 0.90])
            mean = float(np.nanmean(pd.to_numeric(series, errors="coerce")))
            n = int(np.sum(pd.to_numeric(series, errors="coerce").notna()))
            rows.append({"protocol_minute": int(pm), f"{col}_mean": mean, f"{col}_p10": q10, f"{col}_p90": q90, f"{col}_n": n})
        if not rows:
            continue
        dfc = pd.DataFrame(rows).sort_values("protocol_minute")
        dfc.to_csv(combined_dir / f"all_sessions_minute_{col}.csv", index=False)
        combined_rows.append(dfc.set_index("protocol_minute"))

    # Write a wide combined CSV for convenience
    if combined_rows:
        wide = pd.concat(combined_rows, axis=1).reset_index()
        wide.to_csv(combined_dir / "all_sessions_features_minute.csv", index=False)

    # --- combined PDF ---
    pdf_path = combined_dir / "all_sessions_report.pdf"
    with open(pdf_path, 'wb') as _pdf_fh:
        with PdfPages(_pdf_fh) as pdf:

            def _extract_participant(session_id: str) -> str | None:
                m = re.match(r"(P\d{2})", str(session_id))
                return m.group(1) if m else None

            def _plot_combined_metric(
                *,
                title_line: str,
                df_all: pd.DataFrame,
                metric: str,
                ylabel: str,
                template_df: pd.DataFrame,
                pdf: PdfPages,
                max_traces: int = 140,
            ) -> None:
                """Cohort-style burst + envelope + mean (publication-oriented).

                Design goals:
                - Avoid the 'solid blob' look by using *many* thin, highly-transparent, colorful traces.
                - Keep block context without a heavy top ribbon: use in-axis brackets.
                - Keep the legend out of the data region.
                """
                if "protocol_minute" not in df_all.columns:
                    return

                dfm = df_all.copy()
                dfm["protocol_minute"] = pd.to_numeric(dfm["protocol_minute"], errors="coerce")
                dfm[metric] = pd.to_numeric(dfm[metric], errors="coerce")
                dfm = dfm.dropna(subset=["protocol_minute"]).sort_values("protocol_minute")

                # Choose which minutes count toward aggregation: prefer valid_joint if present
                valid_col = "valid_joint" if "valid_joint" in dfm.columns else None
                df_agg = dfm
                if valid_col is not None:
                    df_agg = dfm[dfm[valid_col].astype("Int64") == 1]

                # Aggregation: mean + p10/p50/p90 per minute
                g = df_agg.groupby("protocol_minute")[metric]
                rows = []
                for pm, series in g:
                    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                    q10, q50, q90 = _quantiles(arr, [0.10, 0.50, 0.90])
                    mean = float(np.nanmean(arr))
                    n = int(np.sum(np.isfinite(arr)))
                    rows.append({"protocol_minute": int(pm), "mean": mean, "p10": q10, "p50": q50, "p90": q90, "n": n})
                if not rows:
                    return
                stat = pd.DataFrame(rows).sort_values("protocol_minute")
                x = stat["protocol_minute"].to_numpy(dtype=float)

                # Robust y-lims from central mass
                y_data = stat[["mean", "p10", "p90"]].to_numpy(dtype=float).ravel()
                y_data = y_data[np.isfinite(y_data)]
                if y_data.size == 0:
                    return
                y_min = float(np.nanpercentile(y_data, 1))
                y_max = float(np.nanpercentile(y_data, 99))
                if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
                    y_min = float(np.nanmin(y_data))
                    y_max = float(np.nanmax(y_data))
                    if y_min == y_max:
                        y_min -= 1.0
                        y_max += 1.0
                pad = 0.10 * (y_max - y_min)
                y_min -= pad
                y_max += pad

                headroom = 0.14 * (y_max - y_min)

                fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
                ax.set_title(title_line, pad=2, fontsize=PLOT_FONTS['TITLE'])
                ax.set_ylim(y_min, y_max + headroom)
                fig.subplots_adjust(top=0.93, bottom=0.10, left=0.08, right=0.98)

                # Protocol ribbon (same style as individual reports)
                tmp = template_df.copy()
                tmp = tmp.sort_values("protocol_minute")
                t_series = tmp["protocol_minute"].astype(float).reset_index(drop=True)
                # Place ribbon in the reserved headroom band
                try:
                    y0_band = y_max + 0.20 * headroom
                    y1_band = y_max + 0.85 * headroom
                    if "protocol_block" in tmp.columns and "minute_utc" in tmp.columns:
                        _draw_protocol_ribbon(ax, tmp, t=t_series, y0=y0_band, y1=y1_band, gap_minutes=1.0)
                except Exception:
                    pass

                # Optional: a *very* light density layer (kept subtle to avoid a grey blob)
                try:
                    dens_src = df_agg[["protocol_minute", metric]].copy()
                    dens_src["protocol_minute"] = pd.to_numeric(dens_src["protocol_minute"], errors="coerce")
                    dens_src[metric] = pd.to_numeric(dens_src[metric], errors="coerce")
                    dens_src = dens_src.dropna()
                    if len(dens_src) > 0:
                        y_bins = 90
                        x_vals = dens_src["protocol_minute"].to_numpy(dtype=float)
                        y_vals = dens_src[metric].to_numpy(dtype=float)
                        y_vals = np.clip(y_vals, y_min, y_max)
                        x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
                        if np.isfinite(x_min) and np.isfinite(x_max) and x_min < x_max:
                            H, xedges, yedges = np.histogram2d(
                                x_vals,
                                y_vals,
                                bins=[int(max(60, min(320, x_max - x_min + 1))), y_bins],
                                range=[[x_min, x_max], [y_min, y_max]],
                            )
                            H = np.log1p(H)
                            ax.imshow(
                                H.T,
                                origin="lower",
                                aspect="auto",
                                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                cmap="Greys",
                                alpha=0.08,
                                zorder=1,
                                interpolation="bilinear",
                                rasterized=True,
                            )
                except Exception:
                    pass

                # Burst traces via LineCollection (fast, and avoids over-inking)
                try:
                    from matplotlib.collections import LineCollection
                    sess_ids = dfm["session_id"].dropna().unique().tolist() if "session_id" in dfm.columns else ["session"]
                    if len(sess_ids) > max_traces:
                        rng = np.random.default_rng(0)
                        sess_ids = rng.choice(sess_ids, size=max_traces, replace=False).tolist()

                    cmap = plt.get_cmap("turbo")
                    n_sess = max(len(sess_ids), 1)
                    segs = []
                    cols = []
                    for j, sid in enumerate(sess_ids):
                        dsi = dfm[dfm.get("session_id", sid) == sid][["protocol_minute", metric]].copy() if "session_id" in dfm.columns else dfm[["protocol_minute", metric]].copy()
                        dsi = dsi.dropna().sort_values("protocol_minute")
                        if len(dsi) < 2:
                            continue
                        xy = np.column_stack([dsi["protocol_minute"].to_numpy(dtype=float), dsi[metric].to_numpy(dtype=float)])
                        segs.append(xy)
                        cols.append(cmap(j / (n_sess - 1)) if n_sess > 1 else cmap(0.5))

                    if segs:
                        lc = LineCollection(
                            segs,
                            colors=cols,
                            linewidths=0.45,
                            alpha=0.08,
                            zorder=2,
                            rasterized=True,
                        )
                        ax.add_collection(lc)
                except Exception:
                    # Fallback: individual plot calls (slower but safe)
                    sess_ids = dfm["session_id"].dropna().unique().tolist() if "session_id" in dfm.columns else ["session"]
                    if len(sess_ids) > max_traces:
                        rng = np.random.default_rng(0)
                        sess_ids = rng.choice(sess_ids, size=max_traces, replace=False).tolist()
                    cmap = plt.get_cmap("turbo")
                    n_sess = max(len(sess_ids), 1)
                    for j, sid in enumerate(sess_ids):
                        dsi = dfm[dfm["session_id"] == sid][["protocol_minute", metric]].copy() if "session_id" in dfm.columns else dfm[["protocol_minute", metric]].copy()
                        dsi = dsi.dropna().sort_values("protocol_minute")
                        if dsi.empty:
                            continue
                        color = cmap(j / (n_sess - 1)) if n_sess > 1 else cmap(0.5)
                        ax.plot(dsi["protocol_minute"].to_numpy(), dsi[metric].to_numpy(), color=color, linewidth=0.55, alpha=0.14, rasterized=True, zorder=2)

                # Envelope + central tendency
                # Publication palette: mean (blue), median (orange), band (blue tint)
                ax.fill_between(
                    x,
                    stat["p10"].to_numpy(),
                    stat["p90"].to_numpy(),
                    color=PRIMARY_BLUE,
                    alpha=0.16,
                    zorder=3,
                    label="10–90th percentile band",
                )
                ax.plot(x, stat["p10"].to_numpy(), color=PRIMARY_BLUE, linewidth=0.7, alpha=0.35, zorder=3)
                ax.plot(x, stat["p90"].to_numpy(), color=PRIMARY_BLUE, linewidth=0.7, alpha=0.35, zorder=3)
                ax.plot(x, stat["p50"].to_numpy(), color=SECONDARY_ORANGE, linewidth=1.0, linestyle="--", zorder=4, label="Median (p50)")
                ax.plot(x, stat["mean"].to_numpy(), color=PRIMARY_BLUE, linewidth=1.4, zorder=5, label="Mean across sessions")

                # Plausibility min/max as dashed horizontal lines (if defined).
                # Make them visible by (a) drawing above the ribbon/percentile bands and (b) expanding y-limits if needed.
                if SHOW_PLAUSIBILITY_IN_PLOTS and metric in PLAUSIBILITY_RANGES:
                    lo = PLAUSIBILITY_RANGES[metric].get("min", None)
                    hi = PLAUSIBILITY_RANGES[metric].get("max", None)

                    # Draw lines (single legend label for the pair).
                    drew_label = False
                    if lo is not None and np.isfinite(lo):
                        ax.axhline(
                            float(lo),
                            linestyle="--",
                            linewidth=1.1,
                            alpha=0.85,
                            color="0.25",
                            zorder=6,
                            label=("Plausibility min/max" if not drew_label else None),
                        )
                        drew_label = True
                    if hi is not None and np.isfinite(hi):
                        ax.axhline(
                            float(hi),
                            linestyle="--",
                            linewidth=1.1,
                            alpha=0.85,
                            color="0.25",
                            zorder=6,
                            label=("Plausibility min/max" if not drew_label else None),
                        )

                    # Ensure plausibility lines are within view.
                    y0, y1 = ax.get_ylim()
                    vals = [v for v in [lo, hi] if (v is not None and np.isfinite(v))]
                    if vals:
                        ymin = min([y0] + [float(v) for v in vals])
                        ymax = max([y1] + [float(v) for v in vals])
                        span = max(ymax - ymin, 1e-6)
                        pad = 0.04 * span
                        ax.set_ylim(ymin - pad, ymax + pad)

                ax.set_xlabel("Protocol minute index", fontsize=PLOT_FONTS['AXIS_LABEL'])
                ax.set_ylabel(ylabel, fontsize=PLOT_FONTS['AXIS_LABEL'])
                ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
                ax.grid(False, axis="x")
                ax.margins(x=0.0)

                # Legend inside plot (compact) to avoid wasting right-side space
                handles, labels = ax.get_legend_handles_labels()
                # Place legend top-right but *below* the protocol ribbon band.
                try:
                    ymin_total, ymax_total = ax.get_ylim()
                    denom = max(ymax_total - ymin_total, 1e-9)
                    legend_y = (y0_band - 0.02 * headroom - ymin_total) / denom
                    legend_y = float(min(max(legend_y, 0.05), 0.98))
                except Exception:
                    legend_y = 0.88

                leg = ax.legend(
                    handles,
                    labels,
                    loc="upper right",
                    bbox_to_anchor=(0.995, legend_y),
                    frameon=True,
                    fontsize=PLOT_FONTS['LEGEND'],
                    borderpad=0.25,
                    labelspacing=0.22,
                    handlelength=1.4,
                    handletextpad=0.45,
                    framealpha=0.82,
                )
                if leg is not None:
                    leg.set_zorder(20)

                # Coverage note
                try:
                    n_sessions = int(dfm["session_id"].nunique()) if "session_id" in dfm.columns else 1
                    med_n = int(np.nanmedian(stat["n"].to_numpy()))
                    ax.text(
                        0.01,
                        0.02,
                        f"Sessions: {n_sessions} | Median n/min (valid): {med_n}",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=9,
                        color="0.25",
                    )
                except Exception:
                    pass

                # Keep default layout; do not reserve extra space for legends
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
                fig.tight_layout(pad=0.15)
                pdf.savefig(fig)
                plt.close(fig)
            # Cover / summary page (match inspiration style)
            fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
            ax.axis("off")
            # Front matter: cohort overview (publication-friendly)
            sessions_n = int(all_min['session_id'].nunique()) if 'session_id' in all_min.columns else len(dfs)
            participants_n = int(all_min['participant'].nunique()) if 'participant' in all_min.columns else int(np.nan)
            minutes_total = int(len(all_min))
            valid_col = 'valid_joint' if 'valid_joint' in all_min.columns else ('minute_valid_joint' if 'minute_valid_joint' in all_min.columns else None)
            valid_joint_pct = 100.0 * float(pd.to_numeric(all_min[valid_col], errors='coerce').fillna(0).astype(int).mean()) if valid_col else float('nan')
            pm_min = int(pd.to_numeric(all_min['protocol_minute'], errors='coerce').min()) if 'protocol_minute' in all_min.columns else 0
            pm_max = int(pd.to_numeric(all_min['protocol_minute'], errors='coerce').max()) if 'protocol_minute' in all_min.columns else 0
            oob_cols = [c for c in all_min.columns if str(c).endswith('__oob')]
            if oob_cols:
                oob_mat = all_min[oob_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
                any_oob_pct = 100.0 * float((np.nan_to_num(oob_mat) > 0).any(axis=1).mean())
            else:
                any_oob_pct = float('nan')
            blocks = sorted(set(all_min['protocol_block'].astype(str))) if 'protocol_block' in all_min.columns else []
            phases = sorted(set(all_min['protocol_phase'].astype(str))) if 'protocol_phase' in all_min.columns else []
            corr_metrics = [m for m in PHASE_CORR_METRICS if m in all_min.columns]
            bootstrap_block_len_rec, blk_diag = estimate_block_length_minutes(all_min, corr_metrics)
            bootstrap_block_len_used = bootstrap_block_len_rec if str(BOOTSTRAP_BLOCK_MINUTES).lower() == 'auto' else int(BOOTSTRAP_BLOCK_MINUTES)
            bootstrap_block_len_used = int(np.clip(bootstrap_block_len_used, BOOTSTRAP_BLOCK_MINUTES_MIN, BOOTSTRAP_BLOCK_MINUTES_MAX))
            ax.set_title(format_plot_title('all', '', 'All sessions overview'), fontsize=12, pad=4)
            info_lines = [
                f"Sessions: {sessions_n} | Participants: {participants_n if np.isfinite(participants_n) else 'NA'}",
                f"Protocol minute range: {pm_min}–{pm_max} | Rows (minute × session): {minutes_total:,}",
                f"Valid_joint minutes (overall): {valid_joint_pct:.1f}%",
                f"Any out-of-bounds (any_OOB): {any_oob_pct:.1f}%",
                "",
                "Coverage (minutes with non-missing feature):",
                f"  • HR: {100.0*float(pd.to_numeric(all_min.get('hr_bpm_mean'), errors='coerce').notna().mean()) if 'hr_bpm_mean' in all_min.columns else float('nan'):.1f}%",
                f"  • EDA tonic: {100.0*float(pd.to_numeric(all_min.get('eda_tonic_mean'), errors='coerce').notna().mean()) if 'eda_tonic_mean' in all_min.columns else float('nan'):.1f}%",
                f"  • Temp: {100.0*float(pd.to_numeric(all_min.get('temp_smooth_C_mean'), errors='coerce').notna().mean()) if 'temp_smooth_C_mean' in all_min.columns else float('nan'):.1f}%",
                f"  • ACC ENMO p95: {100.0*float(pd.to_numeric(all_min.get('acc_enmo_g_p95'), errors='coerce').notna().mean()) if 'acc_enmo_g_p95' in all_min.columns else float('nan'):.1f}%",
                f"  • Steps: {100.0*float(pd.to_numeric(all_min.get('steps_count'), errors='coerce').notna().mean()) if 'steps_count' in all_min.columns else float('nan'):.1f}%",
                "",
                "Valid_joint distribution across sessions:",
                (
                    f"  • Median (IQR): "
                    f"{np.nanmedian(all_min.groupby('session_id')[valid_col].apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).astype(int).mean()*100.0)) if valid_col and 'session_id' in all_min.columns else float('nan'):.1f}% "
                    f"({np.nanpercentile(all_min.groupby('session_id')[valid_col].apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).astype(int).mean()*100.0),25) if valid_col and 'session_id' in all_min.columns else float('nan'):.1f}–"
                    f"{np.nanpercentile(all_min.groupby('session_id')[valid_col].apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).astype(int).mean()*100.0),75) if valid_col and 'session_id' in all_min.columns else float('nan'):.1f}%)"
                ),
                "",
                "What you are seeing on the next pages:",
                "  • Thin session traces (burst) + 10–90% band + mean across sessions (valid_joint minutes only).",
                "  • Protocol context via the same block ribbon used in individual reports.",
                "  • Correlations computed on phase means (valid minutes only).",
                f"Bootstrap block length (minutes): {bootstrap_block_len_used} (auto-recommended: {bootstrap_block_len_rec})",
                "",
                f"Blocks: {', '.join(blocks) if blocks else 'NA'}",
                f"Phases: {', '.join(phases[:12]) + (' …' if len(phases) > 12 else '') if phases else 'NA'}",
            ]
            
            # Plausibility ranges are reported in front matter (not drawn on plots).
            _pl_keys = [
                "hr_bpm_mean",
                "eda_tonic_mean",
                "eda_phasic_p95",
                "temp_smooth_C_mean",
                "acc_enmo_g_p95",
                "steps_count",
                "scr_count",
            ]
            _pl_lines = format_plausibility_frontmatter_lines(keys=_pl_keys)
            if _pl_lines:
                info_lines += ["", "Plausibility ranges (minute features; used for OOB flags):"] + _pl_lines
            ax.text(0.02, 0.95, _wrap_block(info_lines, width=112), va="top", ha="left", fontsize=9)
            fig.tight_layout(pad=0.15)
            pdf.savefig(fig)
            plt.close(fig)

            # Cohort-wide pages: ALL sessions
            for col, lab in existing:
                if col not in all_min.columns:
                    continue
                _plot_combined_metric(
                    title_line=format_plot_title("all", "", f"{lab} (minute feature)"),
                    df_all=all_min,
                    metric=col,
                    ylabel=lab,
                    template_df=template_blocks.copy(),
                    pdf=pdf,
                )

            # Cohort-wide phase-level Spearman correlation (ALL sessions)
            try:
                plot_phase_correlation(
                    pdf,
                    all_min,
                    session_id="Wearable Physiology – All Sessions",
                    metrics=[
                        "hr_bpm_mean",
                        "eda_tonic_mean",
                        "eda_phasic_p95",
                        "temp_smooth_C_mean",
                        "acc_enmo_g_p95",
                        "steps_count",
                        "scr_count",
                    ],
                )

                plot_phase_correlation_pvalues(
                    pdf,
                    all_min,
                    session_id="Wearable Physiology – All Sessions",
                    metrics=[
                        "hr_bpm_mean",
                        "eda_tonic_mean",
                        "eda_phasic_p95",
                        "temp_smooth_C_mean",
                        "acc_enmo_g_p95",
                        "steps_count",
                        "scr_count",
                    ],
                )
            except Exception:
                # Never fail the whole combined report because of correlation plotting
                pass


            # --- Deeper coupling analysis (ALL sessions) ---
            # 1) Within-session minute-level bootstrap correlations + meta-analysis (Fisher-z pooling).
            try:
                mats = []
                weights = []
                for sid, g in all_min.groupby("session_id"):
                    rho_mean, _, _ = compute_minute_corr_bootstrap(
                        g,
                        MINUTE_CORR_METRICS,
                        valid_only=True,
                        n_boot=BOOTSTRAP_N,
                        block_len_minutes=bootstrap_block_len_used,
                        seed=BOOTSTRAP_SEED,
                    )
                    if rho_mean is not None and not rho_mean.empty:
                        mats.append(rho_mean)
                        # weight by number of valid minutes
                        w = float((g.get("minute_valid_joint", 1) == 1).sum()) if "minute_valid_joint" in g.columns else float(len(g))
                        weights.append(max(w, 1.0))
                meta_rho = meta_analyze_corr_matrices(mats, weights=weights) if mats else pd.DataFrame(dtype=float)
                if meta_rho is not None and not meta_rho.empty:
                    meta_rho.to_csv(combined_dir / "all_sessions_minute_corr_spearman_meta_mean.csv")
                    plot_corr_heatmap(
                        pdf,
                        meta_rho,
                        title=format_plot_title("all", "", "Minute-level Spearman correlation (within-session meta, block bootstrap)"),
                    )
            except Exception:
                pass

            # 2) Partial correlations controlling for motion proxy (ENMO), meta-analyzed across sessions.
            try:
                pmats = []
                pweights = []
                if PARTIAL_CONTROL_VAR in all_min.columns:
                    for sid, g in all_min.groupby("session_id"):
                        prho_mean, _, _ = compute_minute_partial_corr_bootstrap(
                            g,
                            MINUTE_CORR_METRICS,
                            control_var=PARTIAL_CONTROL_VAR,
                            valid_only=True,
                            n_boot=BOOTSTRAP_N,
                            block_len_minutes=bootstrap_block_len_used,
                            seed=BOOTSTRAP_SEED,
                        )
                        if prho_mean is not None and not prho_mean.empty:
                            pmats.append(prho_mean)
                            w = float((g.get("minute_valid_joint", 1) == 1).sum()) if "minute_valid_joint" in g.columns else float(len(g))
                            pweights.append(max(w, 1.0))
                    pmeta = meta_analyze_corr_matrices(pmats, weights=pweights) if pmats else pd.DataFrame(dtype=float)
                    if pmeta is not None and not pmeta.empty:
                        pmeta.to_csv(combined_dir / "all_sessions_minute_partial_corr_spearman_meta_mean.csv")
                        plot_corr_heatmap(
                            pdf,
                            pmeta,
                            title=format_plot_title("all", "", f"Partial Spearman correlation (within-session meta, control={PARTIAL_CONTROL_VAR})"),
                        )
            except Exception:
                pass

            # --- v17: guaranteed heatmap slots (PDF parity with HTML; render placeholders if empty) ---
            try:
                _phase_corr = pd.DataFrame(dtype=float)
                _phase_p = pd.DataFrame(dtype=float)
                p = combined_dir / "all_sessions_phase_corr_spearman.csv"
                if p.exists():
                    _phase_corr = _read_square_csv_matrix(p)
                p = combined_dir / "all_sessions_phase_corr_spearman_pvalues.csv"
                if p.exists():
                    _phase_p = _read_square_csv_matrix(p)

                _minute_meta = pd.DataFrame(dtype=float)
                p = combined_dir / "all_sessions_minute_corr_spearman_meta_mean.csv"
                if p.exists():
                    _minute_meta = _read_square_csv_matrix(p)

                _partial_meta = pd.DataFrame(dtype=float)
                p = combined_dir / "all_sessions_minute_partial_corr_spearman_meta_mean.csv"
                if p.exists():
                    _partial_meta = _read_square_csv_matrix(p)

                _save_heatmap_or_placeholder(
                    pdf, _minute_meta,
                    title=_title_all_sessions("Minute-level Spearman meta correlation (ρ) across sessions"),
                    subtitle="Within-session Spearman (minute features), meta-analysed across sessions (valid minutes only).",
                    vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f"
                )
                _save_heatmap_or_placeholder(
                    pdf, _partial_meta,
                    title=_title_all_sessions(f"Minute-level partial meta correlation (ρ, control {PARTIAL_CONTROL_VAR})"),
                    subtitle="Partial Spearman (minute features) controlling ENMO, meta-analysed across sessions.",
                    vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f"
                )
                _save_heatmap_or_placeholder(
                    pdf, _phase_corr,
                    title=_title_all_sessions("Pooled phase-level Spearman correlation (ρ) across sessions"),
                    subtitle="Spearman correlation on phase-level means pooled across all sessions (valid minutes only).",
                    vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f"
                )
            except Exception:
                pass

            # --- Correlation validation + key patterns (ALL sessions) ---
            try:
                # Recompute phase-level correlation matrix for diagnostics (same logic as plot_phase_correlation)
                phase_cols = [c for c in [
                    "hr_bpm_mean","eda_tonic_mean","eda_phasic_p95","temp_smooth_C_mean","acc_enmo_g_p95","steps_count","scr_count"
                ] if c in all_min.columns]
                dvalid = all_min.copy()
                if "valid_joint_minute" in dvalid.columns:
                    dvalid = dvalid[dvalid["valid_joint_minute"] == 1].copy()
                if phase_cols and "protocol_phase" in dvalid.columns and len(dvalid) > 0:
                    gph = dvalid.groupby("protocol_phase")[phase_cols].mean(numeric_only=True)
                    gph = gph.rename(columns={c: c.replace("_mean","").strip("_") for c in gph.columns})
                    phase_corr = gph.corr(method="spearman")
                    n_phase_obs = int(gph.shape[0])
                else:
                    phase_corr = pd.DataFrame(dtype=float)
                    n_phase_obs = None

                # meta_rho and pmeta may or may not exist depending on data; normalize
                _meta = meta_rho if "meta_rho" in locals() else pd.DataFrame(dtype=float)
                _pmeta = pmeta if "pmeta" in locals() else pd.DataFrame(dtype=float)

                diag_lines = []
                diag_lines += corr_matrix_diagnostics(phase_corr, name="Phase-level Spearman (phase means, valid_only)", n_obs=n_phase_obs)
                diag_lines += [""] + corr_matrix_diagnostics(_meta, name=f"Minute-level Spearman (within-session meta; block bootstrap={bootstrap_block_len_used}min)")
                diag_lines += [""] + corr_matrix_diagnostics(_pmeta, name=f"Partial Spearman (within-session meta; control={PARTIAL_CONTROL_VAR}; block bootstrap={bootstrap_block_len_used}min)")

                # Explain why partial matrix excludes the control variable
                diag_lines += [
                    "",
                    "Notes:",
                    f"  • Partial correlation matrix excludes the control variable ({PARTIAL_CONTROL_VAR}) by design;",
                    "    correlations involving the control variable are not estimated (ill-posed when control=variable).",
                    "  • These are association summaries; interpret alongside protocol phases and motion (ENMO).",
                ]

                write_text_page(
                    pdf,
                    title=format_plot_title("all", "", "Correlation validation (sanity checks)"),
                    lines=diag_lines,
                    wrap_width=112,
                    fontsize=10,
                )

                # Key patterns summary: strongest correlations
                top_lines = [
                    "Top associations (|ρ|) extracted from the correlation matrices (off-diagonal):",
                    "",
                    "Phase-level (phase means):",
                ]
                tops = top_correlations(phase_corr, top_k=8, min_abs=0.20)
                top_lines += [f"  • {t}" for t in (tops or ["(none above threshold)"])]
                top_lines += ["", "Minute-level (meta; block bootstrap mean):"]
                tops2 = top_correlations(_meta, top_k=8, min_abs=0.20)
                top_lines += [f"  • {t}" for t in (tops2 or ["(none above threshold)"])]
                top_lines += ["", f"Partial (meta; control={PARTIAL_CONTROL_VAR}):"]
                tops3 = top_correlations(_pmeta, top_k=8, min_abs=0.20)
                top_lines += [f"  • {t}" for t in (tops3 or ["(none above threshold)"])]

                write_text_page(
                    pdf,
                    title=format_plot_title("all", "", "Key coupling patterns (summary)"),
                    lines=top_lines,
                    wrap_width=112,
                    fontsize=10,
                )
            except Exception:
                pass


            # 3) Condition-stratified phase-level correlation (using expected_fan_mode if available).
            try:
                if "expected_fan_mode" in all_min.columns:
                    for fan_mode, g in all_min.groupby("expected_fan_mode"):
                        # Aggregate to phase within each session to avoid ecological bias (session_id x phase points)
                        cols = [c for c in PHASE_CORR_METRICS if c in g.columns]
                        if len(cols) < 2:
                            continue
                        agg = (
                            g[g.get("minute_valid_joint", 1) == 1]
                            .groupby(["session_id", "protocol_phase"], as_index=False)[cols]
                            .median(numeric_only=True)
                        )
                        if len(agg) < 6:
                            continue
                        mat = agg[cols].corr(method="spearman", min_periods=5)
                        plot_corr_heatmap(
                            pdf,
                            mat,
                            title=format_plot_title("all", "", f"Phase-level Spearman correlation (fan_mode={fan_mode})"),
                        )
                        mat.to_csv(combined_dir / f"all_sessions_phase_corr_spearman_fan_mode_{fan_mode}.csv")
            except Exception:
                pass



            # Export cohort-level phase matrices (CSV)
            try:
                corr_metrics = ["hr_bpm_mean","eda_tonic_mean","eda_phasic_p95","temp_smooth_C_mean","acc_enmo_g_p95","steps_count","scr_count"]
                phase_means_valid, phase_corr, phase_p = compute_phase_mean_matrices(all_min, corr_metrics, valid_only=True, clean_labels=True)
                if not phase_means_valid.empty:
                    phase_means_valid.to_csv(combined_dir / "all_sessions_phase_means_valid.csv", index=True)
                if not phase_corr.empty:
                    phase_corr.to_csv(combined_dir / "all_sessions_phase_corr_spearman.csv", index=True)
                if not phase_p.empty:
                    phase_p.to_csv(combined_dir / "all_sessions_phase_corr_spearman_pvalues.csv", index=True)
            except Exception:
                pass


# --- v16: Ensure PDF includes the full heatmap set present in the interactive report (even if empty) ---
            try:
                # Phase-level Spearman rho/p heatmaps (use freshly computed if available)
                _phase_corr = phase_corr if 'phase_corr' in locals() else pd.DataFrame(dtype=float)
                _phase_p = phase_p if 'phase_p' in locals() else pd.DataFrame(dtype=float)

                # Fallback to CSVs if variables are empty
                if (_phase_corr is None) or getattr(_phase_corr, 'empty', True):
                    p = combined_dir / "all_sessions_phase_corr_spearman.csv"
                    if p.exists():
                        _phase_corr = _read_square_csv_matrix(p)
                if (_phase_p is None) or getattr(_phase_p, 'empty', True):
                    p = combined_dir / "all_sessions_phase_corr_spearman_pvalues.csv"
                    if p.exists():
                        _phase_p = _read_square_csv_matrix(p)

                # Minute-level meta matrices (may have been saved earlier)
                _minute_meta = pd.DataFrame(dtype=float)
                p = combined_dir / "all_sessions_minute_corr_spearman_meta_mean.csv"
                if p.exists():
                    _minute_meta = _read_square_csv_matrix(p)

                _partial_meta = pd.DataFrame(dtype=float)
                p = combined_dir / "all_sessions_minute_partial_corr_spearman_meta_mean.csv"
                if p.exists():
                    _partial_meta = _read_square_csv_matrix(p)

                # Plot the heatmap slots in a stable order (parity with HTML report)
                _save_heatmap_or_placeholder(
                    pdf, _phase_corr,
                    title=_title_all_sessions("Phase-level Spearman correlation (ρ) of phase means"),
                    subtitle="Spearman correlation on phase-level means (valid minutes only).",
                    vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f"
                )
                _save_heatmap_or_placeholder(
                    pdf, _phase_p,
                    title=_title_all_sessions("Phase-level Spearman correlation p-values of phase means"),
                    subtitle="Two-sided p-values for Spearman ρ (phase-level means; valid minutes only).",
                    vmin=0, vmax=1, cmap="viridis_r", fmt=".3f", annotate=True
                )
                _save_heatmap_or_placeholder(
                    pdf, _minute_meta,
                    title=_title_all_sessions("Minute-level Spearman meta correlation (ρ) across sessions"),
                    subtitle="Within-session Spearman (minute features), meta-analysed across sessions (valid minutes only).",
                    vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f"
                )
                _save_heatmap_or_placeholder(
                    pdf, _partial_meta,
                    title=_title_all_sessions(f"Minute-level partial meta correlation (ρ, control {PARTIAL_CONTROL_VAR})"),
                    subtitle="Partial Spearman (minute features) controlling ENMO, meta-analysed across sessions.",
                    vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f"
                )
                _save_heatmap_or_placeholder(
                    pdf, _phase_corr,
                    title=_title_all_sessions("Pooled phase-level Spearman correlation (ρ) across sessions"),
                    subtitle="Spearman correlation on phase-level means pooled across all sessions (valid minutes only).",
                    vmin=-1, vmax=1, cmap="coolwarm", fmt=".2f"
                )
            except Exception:
                pass

            # Cohort-level phase boxplots (ALL valid minutes) + per-session delta distributions
            phase_order = None
            try:
                if "protocol_phase" in template_blocks.columns:
                    phase_order = list(dict.fromkeys(template_blocks["protocol_phase"].astype(str).dropna().tolist()))
            except Exception:
                phase_order = None

            for metric, ylabel in [
                ("hr_bpm_mean", "Heart rate (bpm)"),
                ("eda_tonic_mean", "EDA tonic (µS)"),
                ("temp_smooth_C_mean", "Skin temperature (°C)"),
                ("acc_enmo_g_p95", "ENMO p95 (g)"),
                ("scr_count", "SCR count (peaks/min)"),
                ("steps_count", "Steps (count/min)"),
            ]:
                if metric not in all_min.columns:
                    continue
                plot_cohort_phase_distributions(pdf, all_min, session_id="Wearable Physiology – All Sessions", metric=metric, ylabel=ylabel, phase_order=phase_order)
                try:
                    deltas = plot_cohort_phase_delta_across_sessions(
                        pdf,
                        all_min,
                        session_id="Wearable Physiology – All Sessions",
                        metric=metric,
                        ylabel=ylabel,
                        baseline_phase="acclimation",
                        phase_order=phase_order,
                    )
                    if deltas is not None and (not deltas.empty):
                        deltas.to_csv(combined_dir / f"all_sessions_phase_deltas_{metric}.csv", index=False)
                except Exception:
                    pass

            # Participant-level pages (optional; can be long but useful)
            all_min["_participant"] = all_min["session_id"].map(_extract_participant)
            parts = [p for p in all_min["_participant"].dropna().unique().tolist()]
            parts = sorted(parts)
            max_parts = int(combined_max_participants or 0)
            if max_parts > 0:
                parts = parts[:max_parts]
        

            # --- Phase-pattern overview: z-scored phase means across sessions (paper-friendly summary) ---
            try:
                # Compute session×phase means on valid_joint minutes
                if "minute_valid_joint" in all_min.columns:
                    base = all_min[all_min["minute_valid_joint"] == 1].copy()
                else:
                    base = all_min.copy()

                metrics = [m for m in PHASE_CORR_METRICS if m in base.columns]
                if ("session_id" in base.columns) and ("protocol_phase" in base.columns) and metrics:
                    g = base.groupby(["session_id", "protocol_phase"], dropna=False)[metrics].mean(numeric_only=True).reset_index()

                    # Z-score within session across phases (per metric)
                    def _zscore_within_session(df: pd.DataFrame) -> pd.DataFrame:
                        out = df.copy()
                        for m in metrics:
                            v = pd.to_numeric(out[m], errors="coerce")
                            mu = float(np.nanmean(v))
                            sd = float(np.nanstd(v, ddof=0))
                            if np.isfinite(sd) and sd > 1e-9:
                                out[m] = (v - mu) / sd
                            else:
                                out[m] = np.nan
                        return out

                    try:
                        gz = g.groupby("session_id", group_keys=False).apply(_zscore_within_session, include_groups=False)
                    except TypeError:
                        gz = g.groupby("session_id", group_keys=False).apply(_zscore_within_session)

                    # Average z-scores across sessions for each phase
                    phase_order = list(pd.unique(base["protocol_phase"].dropna()))
                    # Keep a stable order if protocol_phase looks like integers/strings
                    try:
                        phase_order = sorted(phase_order, key=lambda x: (str(x)))
                    except Exception:
                        pass

                    H = gz.groupby("protocol_phase", dropna=False)[metrics].mean(numeric_only=True).reindex(phase_order)

                    # Save for manuscript tables
                    zcsv = combined_dir / "all_sessions_phase_means_zscore.csv"
                    H.reset_index().to_csv(zcsv, index=False)

                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=CORR_FIGSIZE, dpi=300)
                    arr = H.to_numpy(dtype=float)
                    # Match correlation heatmap UI: diverging cmap, symmetric limits
                    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(np.nanmax(np.abs(arr))) else 1.0
                    vmax = max(0.8, min(2.5, vmax))
                    im = ax.imshow(arr, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto", interpolation="nearest")

                    ax.set_yticks(np.arange(len(H.index)))
                    ax.set_yticklabels([str(p) for p in H.index], fontsize=PLOT_FONTS["TICK"])
                    ax.set_xticks(np.arange(len(metrics)))
                    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=45, ha="right", fontsize=PLOT_FONTS["TICK"])

                    title = format_plot_title("all", "", "Phase pattern heatmap (z-scored within session, mean across sessions)")
                    ax.set_title(title, fontsize=PLOT_FONTS["TITLE"], pad=6)
                    ax.set_xlabel("Metric (z-score)", fontsize=PLOT_FONTS["AXIS_LABEL"])
                    ax.set_ylabel("Protocol phase", fontsize=PLOT_FONTS["AXIS_LABEL"])

                    # Annotate values (paper-friendly)
                    for i in range(arr.shape[0]):
                        for j in range(arr.shape[1]):
                            v = arr[i, j]
                            if not np.isfinite(v):
                                continue
                            txt_color = "white" if abs(v) > (0.55 * vmax) else "black"
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=txt_color)

                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cb.ax.tick_params(labelsize=9)
                    fig.tight_layout(pad=0.2)
                    pdf.savefig(fig)
                    plt.close(fig)
            except Exception:
                pass

            # Participant-level pages (optional; can be long but useful)
            for p in parts:
                dfp = all_min[all_min["_participant"] == p].copy()
                if dfp.empty:
                    continue
                for col, lab in existing:
                    if col not in dfp.columns:
                        continue
                    _plot_combined_metric(
                        # Participant-specific schema: "Wearable Physiology – P01 sessions | Heart rate (bpm)"
                        title_line=f"Wearable Physiology – {p} sessions | {lab}",
                        df_all=dfp,
                        metric=col,
                        ylabel=lab,
                        template_df=template_blocks.copy(),
                        pdf=pdf,
                    )

def _build_timeline_from_segments_used(session_id: str, session_dir: Path, *, timeline_tz: str) -> pd.DataFrame:
    """Fallback timeline builder when a protocol timeline CSV is not provided.

    This is *only* used when the user omits --timeline/--timeline-csv.
    We derive a continuous per-minute window from segments_used.csv.
    """
    seg_path = session_dir / "segments_used.csv"
    if not seg_path.exists():
        raise FileNotFoundError("No timeline provided and segments_used.csv not found to build a fallback timeline.")
    seg = pd.read_csv(seg_path)
    if "segment_start_utc" not in seg.columns or "segment_end_utc" not in seg.columns:
        raise ValueError("segments_used.csv missing required columns segment_start_utc/segment_end_utc.")
    starts = pd.to_datetime(seg["segment_start_utc"], utc=True, errors="coerce")
    ends = pd.to_datetime(seg["segment_end_utc"], utc=True, errors="coerce")
    t0 = starts.min()
    t1 = ends.max()
    if pd.isna(t0) or pd.isna(t1):
        raise ValueError("segments_used.csv has invalid datetimes; cannot build fallback timeline.")
    # minute grid (inclusive start, exclusive end)
    minutes = pd.date_range(t0.floor("min"), t1.ceil("min"), freq="1min", tz="UTC", inclusive="left")
    df = pd.DataFrame({"minute_utc": minutes})
    df["session_id"] = str(session_id)
    m = re.match(r"(P\d{2})", str(session_id))
    df["participant"] = m.group(1) if m else str(session_id)
    df["minute_local"] = df["minute_utc"].dt.tz_convert(timeline_tz)
    df["protocol_minute"] = np.arange(len(df), dtype=int)
    # Minimal protocol labels (unknown)
    df["phase"] = "unknown"
    df["block"] = "unknown"
    df["protocol_phase"] = "unknown"
    df["protocol_block"] = "unknown"
    return df


def main():
    ap = argparse.ArgumentParser(description="Empatica batch pipeline (v5, non-interactive, publication outputs).")

    # --- modern (v8+) style ---
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--session-dir", type=Path, help="Process a single session folder containing eda.csv, bvp.csv, accelerometer.csv, temperature.csv, steps.csv.")
    src.add_argument("--sessions-root", type=Path, help="Process a batch: a folder containing per-session subfolders.")

    ap.add_argument("--session-id", type=str, default=None, help="Override session_id (single-session mode only). Default: folder name.")
    ap.add_argument("--session-ids", type=str, nargs="*", default=None, help="Subset of session IDs to process (batch mode only). Default: all.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output root directory (modern CLI).")
    ap.add_argument("--timeline-csv", type=Path, default=None, help="Protocol minute timeline CSV (recommended).")
    ap.add_argument("--timeline-tz", type=str, default="Europe/Paris", help="Timezone for timeline datetime localization.")
    ap.add_argument("--combined-max-participants", type=int, default=0, help="Limit participant-wise pages in combined report (0 = no limit).")

    # --- legacy (v2) style aliases (kept to preserve older runs/scripts) ---
    ap.add_argument("--timeline", dest="legacy_timeline", type=str, default=None, help="[legacy] Path to timeline_by_minutes.csv (same as --timeline-csv).")
    ap.add_argument("--empatica_root", dest="legacy_empatica_root", type=str, default=None, help="[legacy] Root folder containing per-session folders.")
    ap.add_argument("--out_root", dest="legacy_out_root", type=str, default=None, help="[legacy] Output directory (same as --outdir).")
    ap.add_argument("--sessions_from_timeline", action="store_true", help="[legacy] Batch: process all session_ids listed in timeline.")
    ap.add_argument("--session_id", dest="legacy_session_id", default=None, type=str, help="[legacy] Process only this session_id.")
    ap.add_argument("--session_dir", dest="legacy_session_dir", default=None, type=str, help="[legacy] Directory for single-session mode.")

    args = ap.parse_args()

    # Helpful runtime banner to avoid running the wrong file version
    print(f"[empatica_pipeline] running file={__file__} | version=v5")

    # Decide mode
    legacy_mode = any([
        args.legacy_timeline is not None,
        args.legacy_empatica_root is not None,
        args.legacy_out_root is not None,
        args.legacy_session_id is not None,
        args.sessions_from_timeline,
        args.legacy_session_dir is not None,
    ])

    # Resolve timeline path
    timeline_path = args.timeline_csv
    if args.legacy_timeline is not None:
        timeline_path = Path(args.legacy_timeline)

    timeline: Optional[pd.DataFrame] = None
    if timeline_path is not None:
        timeline = load_timeline(Path(timeline_path), timeline_tz=args.timeline_tz)

    # Resolve output root
    out_root = args.outdir
    if args.legacy_out_root is not None:
        out_root = Path(args.legacy_out_root)
    if out_root is None:
        raise SystemExit("Missing output directory: provide --outdir (modern) or --out_root (legacy).")
    out_root.mkdir(parents=True, exist_ok=True)

    # Determine sessions to run
    sessions: List[Tuple[str, Path]] = []
    if legacy_mode:
        if args.legacy_empatica_root is None and args.legacy_session_dir is None:
            raise SystemExit("Legacy mode requires --empatica_root (or --session_dir for single-session).")
        emp_root = Path(args.legacy_empatica_root) if args.legacy_empatica_root is not None else None

        if args.legacy_session_id is not None:
            sid = str(args.legacy_session_id)
            sdir = Path(args.legacy_session_dir) if args.legacy_session_dir is not None else (emp_root if emp_root is not None else Path("."))
            if sdir.is_file():
                raise SystemExit(f"--session_dir points to a file, expected folder: {sdir}")
            sessions = [(sid, sdir)]
        else:
            if not args.sessions_from_timeline:
                raise SystemExit("Legacy batch mode: use --sessions_from_timeline, or provide --session_id for single-session.")
            if timeline is None:
                raise SystemExit("Legacy batch mode requires a timeline: provide --timeline/--timeline-csv.")
            if emp_root is None:
                raise SystemExit("Legacy batch mode requires --empatica_root.")
            found = find_session_dirs(emp_root)
            if not found:
                raise SystemExit(f"No session folders found under {emp_root} (expected subfolders containing eda.csv).")
            # Keep only those present in timeline
            tl_ids = set(timeline["session_id"].astype(str).unique().tolist())
            for sdir in found:
                sid = str(sdir.name)
                if sid in tl_ids:
                    sessions.append((sid, sdir))
            if not sessions:
                raise SystemExit("No sessions selected from timeline. Check empatica_root structure and timeline session_id values.")
    else:
        if args.session_dir is not None:
            sdir = args.session_dir
            if not sdir.exists():
                raise SystemExit(f"--session-dir does not exist: {sdir}")
            sid = args.session_id or sdir.name
            sessions = [(str(sid), sdir)]
            if args.session_ids is not None and len(args.session_ids) > 0:
                raise SystemExit("--session-ids is only valid with --sessions-root.")
        elif args.sessions_root is not None:
            root = args.sessions_root
            if not root.exists():
                raise SystemExit(f"--sessions-root does not exist: {root}")
            found = find_session_dirs(root)
            if not found:
                raise SystemExit(f"No session folders found under {root} (expected subfolders containing eda.csv).")
            ids = set(map(str, args.session_ids)) if args.session_ids else None
            for sdir in found:
                sid = str(sdir.name)
                if (ids is None) or (sid in ids):
                    sessions.append((sid, sdir))
            if not sessions:
                raise SystemExit("No sessions selected (check --session-ids).")
        else:
            raise SystemExit("Provide either modern inputs (--session-dir/--sessions-root) or legacy inputs (--empatica_root...).")

    # Process
    qc_all = []
    schema_entries: List[Dict[str, Any]] = []

    for sid, sdir in sessions:
        # session-specific timeline slice (or fallback)
        if timeline is not None:
            tl_s = timeline.loc[timeline["session_id"].astype(str) == str(sid)].copy()
            if tl_s.empty:
                raise SystemExit(f"Timeline has no rows for session_id={sid}.")
        else:
            tl_s = _build_timeline_from_segments_used(str(sid), sdir, timeline_tz=args.timeline_tz)

        out_dir = out_root / str(sid)
        qc = process_one_session(session_id=str(sid), session_dir=sdir, timeline_session=tl_s, out_dir=out_dir)

        sch = qc.get("_output_schema", [])
        if isinstance(sch, list) and sch:
            schema_entries.extend(sch)
        qc_all.append({k: v for k, v in qc.items() if k != "_output_schema"})

    # ---- Combined feature files (ALL sessions) ----
    # v7 policy: write *one* canonical long-form combined file per granularity
    # (minute + phase), using the all_sessions naming schema.
    #
    # We intentionally do NOT emit legacy duplicate filenames like
    # 'features_minute_ALL.csv.csv' to avoid confusion and downstream ambiguity.
    minute_all_frames: List[pd.DataFrame] = []
    phase_all_frames: List[pd.DataFrame] = []
    for sid, _sdir in sessions:
        s_out = out_root / str(sid)
        f_min = s_out / "features_minute.csv"
        f_ph = s_out / "features_phase.csv"
        if f_min.exists():
            try:
                dfm = pd.read_csv(f_min)
                if "session_id" not in dfm.columns:
                    dfm.insert(0, "session_id", str(sid))
                minute_all_frames.append(dfm)
            except Exception:
                pass
        if f_ph.exists():
            try:
                dfp = pd.read_csv(f_ph)
                if "session_id" not in dfp.columns:
                    dfp.insert(0, "session_id", str(sid))
                phase_all_frames.append(dfp)
            except Exception:
                pass

    all_dir = out_root / "all_sessions"
    all_dir.mkdir(parents=True, exist_ok=True)

    if minute_all_frames:
        minute_all = pd.concat(minute_all_frames, ignore_index=True)
        minute_all.to_csv(all_dir / "all_sessions_features_minute_long.csv", index=False)

    if phase_all_frames:
        phase_all = pd.concat(phase_all_frames, ignore_index=True)
        phase_all.to_csv(all_dir / "all_sessions_features_phase_long.csv", index=False)

    # Unified output schema for all sessions (deduplicate by file/column/dtype)
    dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in schema_entries:
        key = (str(row.get("file")), str(row.get("column")), str(row.get("dtype")))
        if key not in dedup:
            canon = dict(row)
            canon.pop("session_id", None)
            canon["sessions_present"] = []
            dedup[key] = canon
        sid = row.get("session_id")
        if sid is not None and sid not in dedup[key]["sessions_present"]:
            dedup[key]["sessions_present"].append(sid)

    merged_schema = list(dedup.values())
    merged_schema.sort(key=lambda r: (str(r.get("file")), str(r.get("column"))))
    write_output_schema(all_dir, merged_schema, prefix="output_schema_all_sessions", write_py=False)

    # Batch QC summary
    if len(qc_all) >= 1:
        pd.DataFrame(qc_all).to_csv(all_dir / "batch_qc_summary.csv", index=False)

    # Batch combined results (only if multiple sessions)
    if len(sessions) > 1 and timeline is not None:
        generate_combined_results(out_root=out_root, sessions=sessions, timeline=timeline, tz_name=args.timeline_tz, combined_max_participants=getattr(args, "combined_max_participants", 0))


if __name__ == "__main__":
    main()
