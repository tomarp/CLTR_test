#!/usr/bin/env python3
"""
Empatica interactive report builder (Plotly) for outputs produced by empatica_batch_pipeline_v14.py.

Goals (v3):
- Robust to Plotly version differences (avoid deprecated properties like titlefont / titleside).
- High-quality, publication-ready interactive figures (larger fonts, richer hover, consistent labels).
- Clean protocol annotation: a dedicated "protocol ribbon" subplot (row 1) so phase markers + legend never fight.
- Build BOTH per-session interactive reports and a COMPLETE combined (cohort) interactive report:
    * envelope plots (median + 10–90 band) by protocol minute across sessions
    * optional individual session traces hidden by default
    * combined phase-level Spearman correlation (pooled across sessions)

Inputs:
- --batch-out: directory produced by empatica_batch_pipeline_v14.py (contains session subfolders).
No other files are required. If you *do* want to override protocol annotation, you can pass --timeline-csv.

Outputs:
- <session_dir>/interactive_report.html
- <batch-out>/all_sessions/interactive_report.html   (when --build-combined)

Tested on Python 3.11+ and Plotly 5+ (works with older Plotly as we avoid deprecated keys).

USAGE
----
python empatica_interactive_report.py \
  --batch-out ../results/empatica/ \
  --build-combined \                                           
  --overwrite 

python empatica_interactive_report.py \
  --batch-out ../results/empatica/ \
  --build-combined \
  --overwrite

"""

from __future__ import annotations



def _dedupe_square_matrix(mat: pd.DataFrame, label_fn):
    """
    Deduplicate square matrices by *display-label identity*.

    Some upstream CSVs may include redundant metrics (e.g., temp_mean and temp_smooth_mean),
    which can collapse to the same publication label. For publication-grade heatmaps, we
    drop duplicates deterministically (keep the first occurrence) and keep the matrix square.

    Parameters
    ----------
    mat : DataFrame (square)
    label_fn : callable mapping original label -> display label (string)

    Returns
    -------
    (mat2, display_labels)
      - mat2: deduplicated square DataFrame
      - display_labels: list of display labels aligned with mat2 rows/cols
    """
    if mat is None or mat.empty:
        return mat, []

    # Ensure labels are strings
    cols = [str(c) for c in mat.columns]
    idx = [str(i) for i in mat.index]

    # If not square or index/cols mismatch, coerce to intersection order
    common = [c for c in cols if c in idx]
    if len(common) >= 2 and (len(common) != len(cols) or len(common) != len(idx)):
        mat = mat.loc[common, common]
        cols = common
        idx = common

    # Compute display labels from original
    disp = [label_fn(c) for c in cols]

    # Keep first occurrence for each display label
    keep_pos = []
    seen = set()
    for j, d in enumerate(disp):
        if d not in seen:
            keep_pos.append(j)
            seen.add(d)

    # If nothing to dedupe
    if len(keep_pos) == len(cols):
        return mat, disp

    kept_cols = [cols[j] for j in keep_pos]
    mat2 = mat.loc[kept_cols, kept_cols]
    disp2 = [disp[j] for j in keep_pos]
    return mat2, disp2




def _apply_heatmap_hover_style(fig):
    """
    Enforce a consistent crosshair / spike-line hover behavior for all heatmaps,
    matching the reference behavior used in Figure 15.
    """
    # Hover behavior
    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(namelength=-1),
    )
    # Crosshair-like spike lines (x and y)
    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikecolor="rgba(0,0,0,0.35)",
        spikethickness=1,
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="solid",
        spikecolor="rgba(0,0,0,0.35)",
        spikethickness=1,
    )
    return fig


import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Optional static export via Kaleido
try:
    import kaleido  # noqa: F401
    HAVE_KALEIDO = True
except Exception:
    HAVE_KALEIDO = False


import re

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.3.0.min.js"  # works broadly; pinned for reproducibility

# Okabe-Ito colorblind-safe palette
C_PRIMARY = "#0072B2"   # blue
C_ORANGE  = "#D55E00"
C_GREEN   = "#009E73"
C_PURPLE  = "#CC79A7"
C_SKY     = "#56B4E9"
C_YELLOW  = "#F0E442"
C_GRAY    = "#4D4D4D"
C_LIGHT   = "rgba(0,0,0,0.18)"

BLOCK_COLORS = ["rgba(0,0,0,0.06)", "rgba(0,0,0,0.10)"]  # very subtle
PHASE_GAP_COLOR = "rgba(255,255,255,1.0)"

DEFAULT_METRICS: List[Tuple[str, str, str]] = [
    ("hr_bpm_mean", "Heart rate (bpm)", "bpm"),
    ("eda_tonic_mean", "EDA tonic (µS)", "µS"),
    ("eda_phasic_p95", "EDA phasic p95 (µS)", "µS"),
    ("temp_smooth_C_mean", "Skin temperature (°C)", "°C"),
    ("acc_enmo_g_p95", "ENMO p95 (g)", "g"),
    ("steps_count", "Steps (count/min)", "count/min"),
    ("scr_count", "SCR count (peaks/min)", "count/min"),
]
# Map raw metric column -> publication-grade short label (used in correlation matrices etc.)
_METRIC_LABEL_MAP: Dict[str, str] = {m[0]: m[1] for m in DEFAULT_METRICS}

def _pub_label(name: str) -> str:
    """Publication-grade label for a metric/column name.

    Note: In this pipeline, phase-level tables often contain metric columns like `hr__mean`.
    We treat `__mean` as an internal aggregation suffix and do NOT surface "Mean" in axis labels,
    because the figure title already communicates the aggregation level.
    """
    if name in _METRIC_LABEL_MAP:
        return _METRIC_LABEL_MAP[name]

    if isinstance(name, str):
        # strip common aggregation suffixes for display
        base = name
        for suf in ["__mean", "__median", "__sd", "__std", "__iqr"]:
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        if base in _METRIC_LABEL_MAP:
            return _METRIC_LABEL_MAP[base]

        # also handle already-prettified names containing "Mean"
        base2 = base.replace(" Mean", "").replace(" mean", "")
        if base2 in _METRIC_LABEL_MAP:
            return _METRIC_LABEL_MAP[base2]

    return _pretty_feature_title(str(name)).replace(" Mean", "")

# ------------------------------------------------------------
# Heatmap label normalization (publication-grade)
# ------------------------------------------------------------

_DASH_CHARS = ["—", "–", "−"]  # em dash, en dash, minus

def _strip_dash_suffix(s: str) -> str:
    s = str(s).strip()
    # Normalize dash variants to em dash
    for d in _DASH_CHARS[1:]:
        s = s.replace(d, "—")
    # Remove trailing em-dash fragments and extra whitespace
    s = re.sub(r"\s*—\s*$", "", s).strip()
    return s

def _remove_stat_suffix(s: str) -> str:
    """Remove aggregation/stat suffixes from labels that leak into matrices."""
    s = _strip_dash_suffix(s)
    # Remove common summary suffixes like '— Minimum', '— Mean', '(mean)', etc.
    # Keep units if present e.g. '(°C)'
    stat_words = [
        "minimum","maximum","min","max","mean","median","std","sd","stdev","standard deviation",
        "iqr","p95","p90","p10","p05","p5","percentile","q95","q05",
        "rate per min","per min"
    ]
    # Remove em-dash suffix patterns
    m = re.match(r"^(.*?)(\s*—\s*)(.+)$", s)
    if m:
        left, _, right = m.group(1).strip(), m.group(2), m.group(3).strip()
        rlow = right.lower()
        if any(w in rlow for w in stat_words):
            s = left
        else:
            # if it's not a stat, keep full string
            s = f"{left} — {right}"
    # Remove parenthetical stat markers at the end
    s = re.sub(r"\s*\((mean|median|min|max|std|sd|iqr|p95|p90|p10|p05|p5)\)\s*$", "", s, flags=re.I).strip()
    return s

def _canonical_metric_key(s: str) -> Optional[str]:
    """Map a messy label to one of the canonical pipeline metric keys, if possible."""
    raw = _remove_stat_suffix(s)
    low = raw.lower()

    # Quick exits for known canonical keys
    if raw in _METRIC_LABEL_MAP:
        return raw

    # Normalize separators
    low = re.sub(r"[^a-z0-9]+", " ", low).strip()

    has = lambda w: (w in low)

    # EDA tonic/phasic
    if has("eda") or has("electrodermal") or has("gsr"):
        if has("tonic") or has("scl"):
            return "eda_tonic_mean"
        if has("phasic") or has("scr") or has("peak"):
            # if explicitly SCR count/rate, map there
            if has("count") or has("rate") or has("peaks"):
                return "scr_count"
            return "eda_phasic_p95"
        # generic EDA -> tonic as primary for label purposes
        return "eda_tonic_mean"

    # HR
    if has("heart") and has("rate") or has("hr") or has("bpm"):
        return "hr_bpm_mean"

    # Temperature
    if has("temperature") or has("temp") or has("skin temp") or has("skintemp"):
        return "temp_smooth_C_mean"

    # ENMO / acceleration
    if has("enmo") or has("acceleration") or has("acc") or has("motion"):
        return "acc_enmo_g_p95"

    # Steps
    if has("steps") or has("step"):
        return "steps_count"

    # SCR
    if has("scr") or (has("skin") and has("conductance") and has("response")) or has("peaks"):
        return "scr_count"

    return None

def _wrap_tick_label(s: str, max_chars: int = 18) -> str:
    """Insert <br> into long labels for readable heatmap ticks."""
    s = str(s)
    if len(s) <= max_chars:
        return s
    # Split on spaces and rebuild lines
    words = s.split()
    lines = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "<br>".join(lines)

def normalize_heatmap_ticktexts(raw_labels: List[str]) -> List[str]:
    """Normalize heatmap tick labels to a publication-grade, consistent scheme."""
    cleaned: List[str] = []
    for lab in raw_labels:
        s = _remove_stat_suffix(str(lab))
        # If looks like a raw column name, map via pub label (which uses DEFAULT_METRICS)
        if ("_" in s) or (s.lower() == s) or (s in _METRIC_LABEL_MAP):
            key = _canonical_metric_key(s) or s
            s2 = _pub_label(key)
        else:
            # human-ish label: try canonical mapping, otherwise keep cleaned string
            key = _canonical_metric_key(s)
            s2 = _pub_label(key) if key else s

        s2 = " ".join(str(s2).split())
        cleaned.append(s2)

    # Deduplicate while keeping order (append suffix if needed)
    counts: Dict[str, int] = {}
    out: List[str] = []
    for s in cleaned:
        k = s
        counts[k] = counts.get(k, 0) + 1
        if counts[k] == 1:
            out.append(_wrap_tick_label(s))
        else:
            out.append(_wrap_tick_label(f"{s} ({counts[k]})"))
    return out




# Figure typography
FONT_FAMILY = "Arial"
TITLE_SIZE = 24
SUBTITLE_SIZE = 16
AXIS_TITLE_SIZE = 18
HEATMAP_TICK_SIZE = 12
HEATMAP_TICK_ANGLE = -30

TICK_SIZE = 16
LEGEND_SIZE = 14
HOVER_LABEL_SIZE = 14

# Heatmap sizing + hover UI (keep all heatmaps visually consistent)
HEATMAP_WIDTH = 980
HEATMAP_HEIGHT = 760

def _apply_heatmap_hover_spikes(fig: "go.Figure") -> None:
    """Enable crosshair-style hover guide lines for heatmaps (matches the 'on-hover lines' UI)."""
    try:
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    except Exception:
        pass
# Plotly colorscales (picked for manuscript-friendly readability)
CORR_COLORSCALE = "RdBu"          # diverging, centered at 0
CORR_REVERSE = True              # keep negative=blue, positive=red (RdBu reversed)
POOLED_CORR_COLORSCALE = "BrBG"  # diverging alternative for pooled correlation
POOLED_CORR_REVERSE = False
PVALUE_COLORSCALE = "Cividis"    # perceptually-uniform for -log10(p)

# Static export (SVG/PDF) is handled server-side via Kaleido.
# If Kaleido is missing, reports still build, but the PDF/SVG download links won't appear.
STATIC_EXPORT_DIRNAME = "static_exports"
STATIC_EXPORT_SCALE = 2
EXPORT_STATIC_DEFAULT = False



# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _pretty_feature_title(col: str) -> str:
    """Turn pipeline-ish column names into publication-grade figure titles.

    This is intentionally conservative: it only prettifies obvious tokens and
    falls back to a cleaned title-cased string for unknown patterns.
    """
    if not col:
        return "Physiological Feature"

    raw = str(col).strip()
    toks = [t for t in re.split(r"[_\-\s]+", raw) if t]

    # Canonical modality/feature names
    MOD = {
        "hr": "Heart Rate",
        "bpm": "Heart Rate",
        "eda": "Electrodermal Activity (EDA)",
        "gsr": "Electrodermal Activity (EDA)",
        "temp": "Skin Temperature",
        "skintemp": "Skin Temperature",
        "temperature": "Skin Temperature",
        "bvp": "Blood Volume Pulse (BVP)",
        "ppg": "Blood Volume Pulse (BVP)",
        "acc": "Acceleration",
        "accel": "Acceleration",
        "ibi": "Inter‑beat Interval (IBI)",
        "rr": "Inter‑beat Interval (IBI)",
    }

    STAT = {
        "mean": "Mean",
        "avg": "Mean",
        "median": "Median",
        "std": "Standard deviation",
        "sd": "Standard deviation",
        "var": "Variance",
        "min": "Minimum",
        "max": "Maximum",
        "iqr": "Interquartile range",
        "mad": "Median absolute deviation",
    }

    UNIT = {
        "us": "µS",
        "µs": "µS",
        "degc": "°C",
        "c": "°C",
        "bpm": "bpm",
        "ms": "ms",
    }

    # Percentiles like p10, p90, perc10, pct90
    stat = None
    unit = None
    feature = None

    for t in toks:
        tl = t.lower()
        if feature is None and tl in MOD:
            feature = MOD[tl]
        if unit is None and tl in UNIT:
            unit = UNIT[tl]
        if stat is None and tl in STAT:
            stat = STAT[tl]
        if stat is None:
            m = re.fullmatch(r"p(\d{1,2})", tl)
            if m:
                stat = f"{int(m.group(1))}th percentile"
            m = re.fullmatch(r"(?:perc|pct)(\d{1,2})", tl)
            if stat is None and m:
                stat = f"{int(m.group(1))}th percentile"

    # If nothing matched, just humanize the raw column name.
    if feature is None:
        feature = raw.replace("_", " ").replace("-", " ").strip().title()

    # Compose
    if stat and unit:
        return f"{feature} — {stat} ({unit})"
    if stat:
        return f"{feature} — {stat}"
    if unit:
        return f"{feature} ({unit})"
    return feature


def _pretty_figure_title(fig_num: int, col_or_name: str, kind: str = "feature") -> str:
    """Uniform figure title template (keeps numbering stable, improves distinguishability)."""
    key = str(col_or_name).lower()

    if kind == "corr":
        return f"Figure {fig_num}: Phase‑level Spearman correlation (ρ) of phase means"
    if kind == "corr_p":
        return f"Figure {fig_num}: Phase‑level Spearman correlation p-values of phase means"
    if kind == "combined_corr":
        return f"Figure {fig_num}: Pooled phase‑level Spearman correlation (ρ) across sessions"
    if kind == "combined_p":
        return f"Figure {fig_num}: Pooled phase‑level Spearman correlation p-values across sessions"
    if kind == "delta_summary":
        return f"Figure {fig_num}: Combined phase delta summary (baseline-referenced phase means)"
    if kind == "dist":
        # distribution figures (violin/box) – keep generic
        return f"Figure {fig_num}: {_pretty_feature_title(col_or_name)}"

    return f"Figure {fig_num}: {_pretty_feature_title(col_or_name)}"


def _strip_report_prefix(title: str) -> str:
    """Remove redundant report prefixes from figure titles (defensive cleanup)."""
    if not isinstance(title, str):
        return str(title)
    t = title.strip()
    for pref in [
        "Wearable Physiology – All Sessions |",
        "Wearable Physiology - All Sessions |",
        "Wearable Physiology – All sessions |",
        "Wearable Physiology All sessions |",
        "Wearable Physiology –",
    ]:
        if t.startswith(pref):
            t = t[len(pref):].strip()
    return t


def _read_csv_safely(p: Path, **read_csv_kwargs) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_csv(p, **read_csv_kwargs)


def _collect_pipeline_hyperparameters(all_sessions_dir: Path) -> Dict[str, str]:
    """Best-effort extraction of pipeline hyperparameters from JSON artifacts in all_sessions.

    We prefer `qc_summary.json` (if present) because it reflects the values actually used
    by the batch pipeline run. If unavailable, we fall back to a small curated list of
    common tunables (so the report is never empty).
    """
    candidates = [
        all_sessions_dir / "qc_summary.json",
        all_sessions_dir / "qc_summary_all_sessions.json",
        all_sessions_dir / "report_meta.json",
        all_sessions_dir / "run_metadata.json",
        all_sessions_dir / "output_schema_all_sessions.json",
        all_sessions_dir / "output_schema.json",
    ]
    meta = None
    for p in candidates:
        if p.exists():
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                meta = None

    hp: Dict[str, str] = {}

    def _put(k: str, v) -> None:
        if v is None:
            return
        if isinstance(v, (dict, list)):
            try:
                hp[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                hp[k] = str(v)
        else:
            hp[k] = str(v)

    if isinstance(meta, dict):
        # common locations
        for key in ["tunable_parameters", "hyperparameters", "params", "config"]:
            if key in meta and isinstance(meta[key], dict):
                for kk, vv in meta[key].items():
                    _put(kk, vv)
        # sometimes stored flat
        for kk in [
            "BOOTSTRAP_N", "BOOTSTRAP_BLOCK_MINUTES", "PARTIAL_CONTROL_VAR",
            "GAP_DETECTION_THRESHOLD_SECONDS", "HIGH_MOTION_ENMO_P95_THRESHOLD_G",
        ]:
            if kk in meta:
                _put(kk, meta[kk])

    # Fallback: keep report informative even if JSON is missing.
    if not hp:
        _put("BOOTSTRAP_N", 300)
        _put("BOOTSTRAP_BLOCK_MINUTES", "auto or 10")
        _put("PARTIAL_CONTROL_VAR", "acc_enmo_g_p95")
        _put("GAP_DETECTION_THRESHOLD_SECONDS", 2.0)
        _put("HIGH_MOTION_ENMO_P95_THRESHOLD_G", 0.3)

    return hp

def _find_session_dirs(batch_out: Path) -> List[Path]:
    # session folder contains features_minute.csv
    out = []
    for d in sorted(batch_out.iterdir()):
        if not d.is_dir():
            continue
        if (d / "features_minute.csv").exists():
            out.append(d)
    return out

def _safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)

def _pct(numer: float, denom: float) -> float:
    try:
        return (100.0 * float(numer) / float(denom)) if float(denom) else 0.0
    except Exception:
        return 0.0


def _series_stats(y: pd.Series) -> Dict[str, float]:
    y = pd.to_numeric(y, errors="coerce")
    y = y[np.isfinite(y.to_numpy())]
    if len(y) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "p10": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "n": int(len(y)),
        "mean": float(np.nanmean(y)),
        "std": float(np.nanstd(y)),
        "min": float(np.nanmin(y)),
        "p10": float(np.nanpercentile(y, 10)),
        "p90": float(np.nanpercentile(y, 90)),
        "max": float(np.nanmax(y)),
    }


def _subtitle_timeseries(df: pd.DataFrame, *, ycol: str, ylabel: str, valid_col: str, session_id: str) -> str:
    n = len(df)
    if n == 0 or ycol not in df.columns:
        return f"{ylabel}: missing/empty."
    valid = pd.to_numeric(df.get(valid_col, 1), errors="coerce").fillna(0).astype(int)
    n_valid = int(valid.sum())
    y_valid = df.loc[valid == 1, ycol]
    st = _series_stats(y_valid)
    return (
        f"{ylabel}. Minute-aligned feature <b>{_safe_text(ycol)}</b>. QC-pass (<b>{_safe_text(valid_col)}</b>) = "
        f"<b>{n_valid}</b>/<b>{n}</b> minutes ({_pct(n_valid, n):.1f}%). "
        f"Valid-minute summary: mean={st['mean']:.3g}, sd={st['std']:.3g}, p10={st['p10']:.3g}, p90={st['p90']:.3g}. "
        f"Hover shows protocol block/phase and QC flag."
    )


def _minute_axis(df: pd.DataFrame) -> pd.Series:
    # prefer minute_index (protocol minute). fallback to 0..n-1
    for c in ["minute_index", "protocol_minute", "minute"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.arange(len(df)), name="minute_index")

def _infer_protocol_segments(df: pd.DataFrame) -> Tuple[List[Tuple[str, float, float]], List[float]]:
    """
    Return (block_segments, phase_boundaries) in x-axis units (protocol minutes).
    - block_segments: list of (block_label, x0, x1) with x1 exclusive-ish.
    - phase_boundaries: list of x positions where phase changes (start of new phase).
    """
    if df.empty:
        return [], []
    x = _minute_axis(df).reset_index(drop=True)
    d = df.copy()
    d["_x"] = x
    d = d.sort_values("_x").reset_index(drop=True)

    blocks = d["protocol_block"].astype(str).to_numpy() if "protocol_block" in d.columns else np.array(["0"] * len(d))
    phases = d["protocol_phase"].astype(str).to_numpy() if "protocol_phase" in d.columns else np.array(["0"] * len(d))

    # segments by block
    segs: List[Tuple[str, float, float]] = []
    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            blk = blocks[start]
            x0 = float(d["_x"].iloc[start])
            x1 = float(d["_x"].iloc[i - 1]) + 1.0
            segs.append((blk, x0, x1))
            start = i

    # boundaries by phase
    bounds: List[float] = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            bounds.append(float(d["_x"].iloc[i]))
    return segs, bounds

def _plotly_base_layout(title: str, subtitle: Optional[str] = None) -> Dict:
    """Base Plotly layout.

    v7.4 UI strategy:
    - Title + subtitle are rendered in the HTML card (not inside Plotly) so they never collide with hover/legend.
    - Plotly legend is disabled; we render a button-based external legend under each figure.
    """
    _ = title
    _ = subtitle

    return dict(
        title=dict(text=""),
        font=dict(family=FONT_FAMILY, size=TICK_SIZE, color="black"),
        # Compact margins: legend lives outside the Plotly canvas.
        margin=dict(l=55, r=30, t=28, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        showlegend=False,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.85)",  # transparent enough to see the curve underneath
            bordercolor="rgba(0,0,0,0.22)",
            font=dict(size=HOVER_LABEL_SIZE, color="black"),
            align="left",
            namelength=-1,
        ),
    )


def _legend_items_from_traces(fig: go.Figure) -> List[Dict[str, str]]:
    """Derive a concise external legend definition from trace metadata."""
    items: List[Dict[str, str]] = []
    seen: set[str] = set()
    for tr in fig.data:
        lg = getattr(tr, "legendgroup", None) or ""
        nm = getattr(tr, "name", None) or ""
        if not lg and not nm:
            continue
        key = lg or nm
        if key in seen:
            continue
        # Skip helper / invisible scaffolding traces
        if nm.lower().startswith("_helper"):
            continue

        col = None
        if hasattr(tr, "line") and tr.line and getattr(tr.line, "color", None):
            col = tr.line.color
        if col is None and hasattr(tr, "marker") and tr.marker and getattr(tr.marker, "color", None):
            col = tr.marker.color

        items.append({
            "label": nm or lg,
            "group": lg or nm,
            "color": col or "rgba(0,0,0,0.35)",
        })
        seen.add(key)
    return items

def _apply_axis_style(fig: go.Figure, *, x_title: str, y_title: str, row: int, col: int, showgrid: bool = True):
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        showgrid=showgrid,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        showgrid=showgrid,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
        row=row,
        col=col,
    )

def _add_protocol_ribbon(fig: go.Figure, *, block_segments: List[Tuple[str, float, float]], phase_bounds: List[float], row: int, col: int):
    """
    Adds a protocol ribbon in the specified subplot cell using shapes + annotations.
    Ribbon uses y in [0,1] and hides ticks.
    """
    # Block rectangles
    for i, (blk, x0, x1) in enumerate(block_segments):
        fig.add_shape(
            type="rect",
            xref=f"x{'' if (row,col)==(1,1) else fig._get_subplot_ref(row,col)[0][1:]}",
            yref=f"y{'' if (row,col)==(1,1) else fig._get_subplot_ref(row,col)[1][1:]}",
            x0=x0, x1=x1,
            y0=0.0, y1=1.0,
            line=dict(width=0),
            fillcolor=BLOCK_COLORS[i % len(BLOCK_COLORS)],
            layer="below",
            row=row, col=col
        )
        # block label
        xm = (x0 + x1) / 2.0
        fig.add_annotation(
            x=xm, y=0.5,
            xref=f"x{'' if (row,col)==(1,1) else fig._get_subplot_ref(row,col)[0][1:]}",
            yref=f"y{'' if (row,col)==(1,1) else fig._get_subplot_ref(row,col)[1][1:]}",
            text=f"Block {blk}",
            showarrow=False,
            font=dict(size=14, color="rgba(0,0,0,0.75)"),
            xanchor="center",
            yanchor="middle",
            row=row, col=col
        )

    # Phase boundaries as thin white gaps
    for xb in phase_bounds:
        fig.add_shape(
            type="line",
            x0=xb, x1=xb,
            y0=0.0, y1=1.0,
            xref=f"x{'' if (row,col)==(1,1) else fig._get_subplot_ref(row,col)[0][1:]}",
            yref=f"y{'' if (row,col)==(1,1) else fig._get_subplot_ref(row,col)[1][1:]}",
            line=dict(color=PHASE_GAP_COLOR, width=3),
            layer="above",
            row=row, col=col
        )

    # Ribbon axes style: no ticks
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=row, col=col, range=[0,1])
    fig.update_xaxes(showgrid=False, zeroline=False, row=row, col=col)

def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _spearman_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman correlation with pairwise complete observations.
    Uses pandas' rank-based corr for robustness.
    """
    if df.shape[0] < 3:
        return pd.DataFrame()
    return df.corr(method="spearman", numeric_only=True)



def build_corr_heatmap(title: str, corr: pd.DataFrame, *, colorscale: str = CORR_COLORSCALE, reversescale: bool = CORR_REVERSE) -> go.Figure:
    """
    Manuscript-friendly correlation UI (Fig 10 style):
    - square cells (scale-anchored axes)
    - diverging colors centered at 0 (zmid=0)
    - numeric labels rendered via a separate text scatter (robust across Plotly.js versions)
    - hover uses explicit x=..., y=... (no "y vs x")
    """
    if corr is None or corr.empty:
        fig = go.Figure()
        fig.update_layout(_plotly_base_layout(title, "No data (insufficient rows to compute correlation)."))
        fig.update_layout(height=820)
        return fig

    corr = corr.copy()

    # Ensure we are square + ordered consistently
    common = [c for c in corr.columns if c in corr.index]
    corr = corr.loc[common, common]

    raw_labels = corr.columns.tolist()

    # Deduplicate *by display label* (prevents '(2)' label suffixes in publication heatmaps).
    # Keep first occurrence deterministically.
    def _lbl(x):
        x = str(x).replace("__mean", "")
        key = _canonical_metric_key(x) or x
        return " ".join(str(_pub_label(key)).split())

    corr, disp = _dedupe_square_matrix(corr, _lbl)
    labels = [_wrap_tick_label(d) for d in disp]

    z = corr.to_numpy(dtype=float)

    hm = go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        zmin=-1,
        zmax=1,
        zmid=0,
        colorscale=colorscale,
        reversescale=reversescale,
        xgap=1,
        ygap=1,
        colorbar=dict(
            title=dict(text="Spearman ρ", font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE),
            len=0.85,
        ),
        hovertemplate="y=%{y}<br>x=%{x}<br>ρ=%{z:.3f}<extra></extra>",
    )

    fig = go.Figure(data=[hm])


    fig.update_layout(_plotly_base_layout(title, "Phase-level feature coupling (hover for details)"))

    fig.update_xaxes(
        tickangle=HEATMAP_TICK_ANGLE,
        automargin=True,
        ticks='outside',
        ticklen=4,

        tickfont=dict(size=HEATMAP_TICK_SIZE),
        showgrid=False,
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        spikecolor="rgba(0,0,0,0.35)",
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=HEATMAP_TICK_SIZE),
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        spikecolor="rgba(0,0,0,0.35)",
    )

    n = len(labels)
    base = 980
    px_per_var = 85
    size = int(max(base, min(1600, 220 + n * px_per_var)))
    fig.update_layout(width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)

    meta = fig.layout.meta if hasattr(fig.layout, "meta") else None
    meta = meta if isinstance(meta, dict) else {}
    meta["is_corr"] = True
    fig.update_layout(meta=meta)
    return fig




def build_pvalue_heatmap(title: str, pvals: pd.DataFrame, *, colorscale: str = PVALUE_COLORSCALE) -> go.Figure:
    """
    Heatmap of p-values (Spearman), matched to the correlation matrix UI (Fig 11 style).
    Improvements vs v2_3:
    - publication-grade axis labels via _pub_label
    - explicit hover: y=..., x=... (no "y vs x")
    - cell annotations (scientific notation where needed)
    - color uses -log10(p) for contrast, while hover/text shows the original p
    """
    if pvals is None or pvals.empty:
        fig = go.Figure()
        fig.update_layout(_plotly_base_layout(title, "No data."))
        fig.update_layout(height=820)
        return fig

    pv = pvals.copy().apply(pd.to_numeric, errors="coerce").clip(lower=0, upper=1)

    raw_x = pv.columns.tolist()
    raw_y = pv.index.tolist()

    # Keep matrix square + dedupe by display label (avoid '(2)' suffixes).
    def _lbl(x):
        x = str(x).replace("__mean", "")
        key = _canonical_metric_key(x) or x
        return " ".join(str(_pub_label(key)).split())

    pv, disp = _dedupe_square_matrix(pv, _lbl)
    xlab = [_wrap_tick_label(d) for d in disp]
    ylab = xlab

    z = pv.to_numpy(dtype=float)

    # Use -log10(p) for color while keeping p for display.
    zc = -np.log10(np.clip(z, 1e-12, 1.0))
    zc[~np.isfinite(zc)] = np.nan

    hm = go.Heatmap(
        z=zc,
        x=xlab,
        y=ylab,
        zmin=0,
        zmax=float(np.nanmax(zc)) if np.isfinite(np.nanmax(zc)) else 1,
        colorscale=colorscale,
        xgap=1,
        ygap=1,
        colorbar=dict(
            title=dict(text="-log10(p)", font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=TICK_SIZE),
            len=0.85,
        ),
        customdata=z,
        hovertemplate="y=%{y}<br>x=%{x}<br>p=%{customdata:.3g}<br>-log10(p)=%{z:.2f}<extra></extra>",
    )

    fig = go.Figure(data=[hm])


    fig.update_layout(_plotly_base_layout(title, "Spearman correlation p-values (phase-level means)"))
    fig.update_layout(width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)
    _apply_heatmap_hover_spikes(fig)

    fig.update_xaxes(
        tickangle=HEATMAP_TICK_ANGLE,
        automargin=True,
        ticks='outside',
        ticklen=4,

        tickfont=dict(size=HEATMAP_TICK_SIZE),
        showgrid=False,
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        spikecolor="rgba(0,0,0,0.35)",
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=HEATMAP_TICK_SIZE),
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikedash="dot",
        spikecolor="rgba(0,0,0,0.35)",
    )

    n = len(xlab)
    base = 980
    px_per_var = 85
    size = int(max(base, min(1600, 220 + n * px_per_var)))
    fig.update_layout(height=size, width=size)

    meta = fig.layout.meta if hasattr(fig.layout, "meta") else None
    meta = meta if isinstance(meta, dict) else {}
    meta["is_corr"] = True
    fig.update_layout(meta=meta)
    return fig



def build_qc_flags_heatmap(flags: pd.DataFrame, session_id: str) -> go.Figure:
    """Minute-level QC flags heatmap. Rows are flags, columns are protocol minutes."""
    if len(flags) == 0:
        return go.Figure()

    # choose binary-ish columns (exclude identifiers)
    exclude = {
        "minute_utc", "session_id", "participant", "minute_index",
        "protocol_block", "protocol_phase", "expected_fan_mode",
    }
    cand = [c for c in flags.columns if c not in exclude]
    # Keep oob / validity / motion columns preferentially
    ordered = []
    for key in ["valid_joint_minute", "valid_hr_minute", "valid_eda_minute", "valid_temp_minute", "valid_acc_minute", "high_motion"]:
        if key in cand:
            ordered.append(key)
    ordered += [c for c in cand if c.endswith("_oob") and c not in ordered]
    ordered += [c for c in cand if c not in ordered]

    mat = flags[ordered].apply(pd.to_numeric, errors="coerce").fillna(0)
    # convert to 0/1 for display
    mat = (mat > 0).astype(int)

    x = flags["minute_index"] if "minute_index" in flags.columns else list(range(len(flags)))
    y = [c.replace("_", " ") for c in ordered]

    fig = go.Figure(
        data=go.Heatmap(
            z=mat.T.values,
            x=x,
            y=y,
            zmin=0,
            zmax=1,
            colorbar=dict(title="flag"),
            hovertemplate="minute=%{x}<br>%{y}=%{z}<extra></extra>",
        )
    )
    fig.update_layout(_plotly_base_layout(f"{session_id} — QC flags by minute"))
    fig.update_layout(width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)
    _apply_heatmap_hover_spikes(fig)
    fig.update_xaxes(title="protocol minute")
    fig.update_yaxes(title="", automargin=True)
    return fig




def build_phase_distributions_boxgrid(
    minute: pd.DataFrame,
    *,
    session_id: str,
    metrics: List[Tuple[str, str, str]],
    valid_col: str = "valid_joint_minute",
) -> go.Figure:
    """
    Phase distributions (professional UI):
    - single plot with a dropdown to select metric (keeps the UI compact)
    - x-axis = protocol phase (categorical)
    - violin + embedded box + mean line (cleaner than raw box + jittered points)
    - optionally show only suspected outliers to avoid "confetti" plots
    """
    if len(minute) == 0 or "protocol_phase" not in minute.columns:
        return go.Figure()

    mdf = minute.copy()
    if valid_col in mdf.columns:
        mdf = mdf.loc[pd.to_numeric(mdf[valid_col], errors="coerce").fillna(0).astype(int) == 1].copy()
    if len(mdf) == 0:
        return go.Figure()

    phases = list(dict.fromkeys(mdf["protocol_phase"].astype(str).tolist()))  # preserve appearance order

    metric_items: List[Tuple[str, str]] = [(c, lab) for (c, lab, _u) in metrics if c in mdf.columns]
    if not metric_items:
        return go.Figure()

    fig = go.Figure()

    for i, (col, lab) in enumerate(metric_items):
        tmp = mdf[["protocol_phase", col]].copy()
        tmp["protocol_phase"] = tmp["protocol_phase"].astype(str)
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=[col])
        if len(tmp) == 0:
            # keep dropdown indexing stable (add an empty trace)
            fig.add_trace(go.Violin(x=[], y=[], name=lab, visible=(i == 0), showlegend=False))
            continue

        fig.add_trace(
            go.Violin(
                x=tmp["protocol_phase"],
                y=tmp[col],
                name=lab,
                visible=(i == 0),
                showlegend=False,
                points="suspectedoutliers",
                jitter=0.18,
                scalemode="width",
                spanmode="hard",
                box=dict(visible=True, width=0.25),
                meanline=dict(visible=True),
                line=dict(width=1.2),
                marker=dict(size=4, opacity=0.55),
                hovertemplate="phase=%{x}<br>" + f"{lab}=%{{y:.3g}}<extra></extra>",
            )
        )

    buttons = []
    for i, (_col, lab) in enumerate(metric_items):
        vis = [False] * len(metric_items)
        vis[i] = True
        buttons.append(
            dict(
                label=lab,
                method="update",
                args=[
                    {"visible": vis},
                    {"yaxis": {"title": {"text": lab}}},
                ],
            )
        )

    fig.update_xaxes(
        title="protocol phase",
        type="category",
        categoryorder="array",
        categoryarray=phases,
        tickangle=-25,
        tickfont=dict(size=HEATMAP_TICK_SIZE),
    )
    fig.update_yaxes(
        title=dict(text=metric_items[0][1], font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        automargin=True,
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )

    fig.update_layout(
        _plotly_base_layout(f"{session_id} — Phase distributions"),
        updatemenus=[
            dict(
                type="dropdown",
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                direction="down",
                showactive=True,
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.18)",
                borderwidth=1,
                font=dict(size=14),
            )
        ],
        margin=dict(l=65, r=30, t=70, b=60),
        height=720,
    )

    return fig




def build_phase_delta_boxgrid(phase_means: pd.DataFrame, *, session_id: str, metrics_mean_cols: List[str]) -> go.Figure:
    """
    Within-session delta vs baseline phase for phase-mean metrics.

    v2_4 improvement:
    - The previous subplot-per-metric layout produced a lot of tiny panels.
    - Here we keep a single panel + dropdown per metric (same UI style as phase distributions).
    - Each metric is shown as a baseline-centered bar chart across phases (Δ vs baseline).
    """
    if len(phase_means) == 0 or "protocol_phase" not in phase_means.columns:
        return go.Figure()

    p = phase_means.copy()

    # stable phase order: block then appearance
    if "protocol_block" in p.columns:
        p = p.sort_values(["protocol_block", "protocol_phase"], kind="stable")

    phases = list(dict.fromkeys(p["protocol_phase"].astype(str).tolist()))
    if not phases:
        return go.Figure()
    baseline = phases[0]

    # Ensure baseline exists
    base_row = p.loc[p["protocol_phase"].astype(str) == baseline, metrics_mean_cols].apply(pd.to_numeric, errors="coerce")
    if len(base_row) == 0:
        return go.Figure()
    base = base_row.iloc[0]

    metric_items: List[Tuple[str, str]] = []
    for c in metrics_mean_cols:
        base_name = c.replace("__mean", "")
        metric_items.append((c, _pub_label(base_name) + " — Mean"))

    fig = go.Figure()
    for i, (c, lab) in enumerate(metric_items):
        y = []
        for ph in phases:
            v = pd.to_numeric(p.loc[p["protocol_phase"].astype(str) == ph, c], errors="coerce")
            v = v.dropna()
            if len(v) == 0 or pd.isna(base.get(c)):
                y.append(np.nan)
            else:
                y.append(float(v.iloc[0] - base[c]))

        fig.add_trace(
            go.Bar(
                x=phases,
                y=y,
                name=lab,
                visible=(i == 0),
                showlegend=False,
                hovertemplate="phase=%{x}<br>" + f"Δ {lab}=%{{y:.3g}}<extra></extra>",
            )
        )

    buttons = []
    for i, (_c, lab) in enumerate(metric_items):
        vis = [False] * len(metric_items)
        vis[i] = True
        buttons.append(
            dict(
                label=lab,
                method="update",
                args=[
                    {"visible": vis},
                    {"yaxis": {"title": {"text": f'Δ {lab} vs {baseline}'}}},
                ],
            )
        )

    fig.update_layout(
        _plotly_base_layout(f"{session_id} — Phase mean deltas vs baseline"),
        updatemenus=[
            dict(
                type="dropdown",
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                direction="down",
                showactive=True,
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.18)",
                borderwidth=1,
                font=dict(size=14),
            )
        ],
        margin=dict(l=65, r=30, t=70, b=60),
        height=640,
        shapes=[
            dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0,
                 line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))
        ],
    )

    fig.update_xaxes(
        title="protocol phase",
        type="category",
        categoryorder="array",
        categoryarray=phases,
        tickangle=-25,
        tickfont=dict(size=HEATMAP_TICK_SIZE),
    )
    fig.update_yaxes(
        title=dict(text=f"Δ {metric_items[0][1]} vs {baseline}", font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        automargin=True,
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return fig



def build_timeseries_figure(
    df: pd.DataFrame,
    *,
    session_id: str,
    ycol: str,
    ylabel: str,
    valid_col: str = "valid_joint_minute",
    description: str = "",
    show_individual: bool = True,
) -> go.Figure:
    """
    Clean interactive time-series figure with a dedicated protocol ribbon (row 1),
    keeping the legend OUT of the ribbon (in the top headroom).
    Hover box is semi-transparent and compact (no long free-text inside hover).
    """
    d = df.copy()
    if d.empty or ycol not in d.columns:
        fig = go.Figure()
        fig.update_layout(_plotly_base_layout(f"{session_id} — {ycol}", "Missing column / empty data"))
        return fig

    x = _minute_axis(d)
    y = _coerce_numeric(d[ycol])

    if valid_col in d.columns:
        valid = pd.to_numeric(d[valid_col], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        valid = np.ones(len(d), dtype=int)

    # Protocol segments
    block_segs, phase_bounds = _infer_protocol_segments(d)

    # Customdata for hover: block, phase, QC flag
    custom_cols = []
    custom_names = []
    if "protocol_block" in d.columns:
        custom_cols.append(d["protocol_block"].astype(str).to_numpy())
        custom_names.append("block")
    if "protocol_phase" in d.columns:
        custom_cols.append(d["protocol_phase"].astype(str).to_numpy())
        custom_names.append("phase")
    if valid_col in d.columns:
        custom_cols.append(pd.to_numeric(d[valid_col], errors="coerce").fillna(0).astype(int).to_numpy())
        custom_names.append("qc")

    customdata = np.stack(custom_cols, axis=1) if custom_cols else None

    hovertemplate = "minute=%{x}<br>" + f"{ycol}=%{{y:.3g}}"
    if custom_names:
        for i, nm in enumerate(custom_names):
            hovertemplate += f"<br>{nm}=%{{customdata[{i}]}}"
    hovertemplate += "<extra></extra>"

    # Subplots: row 1 ribbon, row 2 data
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.14, 0.86],
    )

    # Ribbon dummy trace to ensure axes exist
    fig.add_trace(
        go.Scatter(
            x=[float(np.nanmin(x)), float(np.nanmax(x))],
            y=[0.5, 0.5],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1
    )
    _add_protocol_ribbon(fig, block_segments=block_segs, phase_bounds=phase_bounds, row=1, col=1)

    # All minutes (faint) for context
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            name="all minutes",
            line=dict(color=C_LIGHT, width=1),
            hovertemplate=hovertemplate,
            customdata=customdata,
            showlegend=show_individual,
            legendgroup="context",
        ),
        row=2, col=1
    )

    # Valid minutes (primary)
    y_valid = y.copy()
    y_valid[valid != 1] = np.nan
    fig.add_trace(
        go.Scatter(
            x=x, y=y_valid,
            mode="lines",
            name="QC-passed",
            line=dict(color=C_PRIMARY, width=2.6),
            hovertemplate=hovertemplate,
            customdata=customdata,
            legendgroup="primary",
        ),
        row=2, col=1
    )

    # Layout + axis style
    # Description is rendered in the HTML card; avoid duplicating it inside Plotly.
    fig.update_layout(_plotly_base_layout(f"{session_id} — {ylabel}", None))
    _apply_axis_style(fig, x_title="Protocol minute", y_title=ylabel, row=2, col=1, showgrid=True)

    # Ribbon axis titles off
    fig.update_xaxes(title=None, tickfont=dict(size=TICK_SIZE), row=1, col=1)
    fig.update_yaxes(title=None, row=1, col=1)

    # External legend spec (HTML)
    fig.update_layout(meta=dict(
        legend_items=_legend_items_from_traces(fig),
        legend_note="Single-session plots: toggle the faint context trace or the QC-passed primary trace.",
    ))

    return fig

def build_envelope_figure(
    df_all: pd.DataFrame,
    *,
    ycol: str,
    ylabel: str,
    description: str = "",
    valid_col: str = "valid_joint_minute",
    show_individual_hidden: bool = True,
) -> go.Figure:
    """
    Cohort-level envelope plot (protocol-minute aligned):
      - 10–90% band and median across sessions (QC-passed minutes)
      - scalable legend strategy for 80+ sessions:
            * legend items are PARTICIPANTS (P01..P20), not sessions
            * clicking Pxx toggles all traces for that participant (legend group)
      - hover includes minute + value + inferred block/phase at that minute
    """
    if df_all.empty or ycol not in df_all.columns:
        fig = go.Figure()
        fig.update_layout(_plotly_base_layout(f"Combined — {ycol}", "Missing column / empty data"))
        return fig

    d = df_all.copy()
    xcol = "minute_index"
    if xcol not in d.columns:
        d[xcol] = _minute_axis(d)

    # Validity filter for envelope (QC-passed)
    if valid_col in d.columns:
        d_valid = d.loc[pd.to_numeric(d[valid_col], errors="coerce").fillna(0).astype(int) == 1].copy()
    else:
        d_valid = d.copy()

    d_valid[ycol] = pd.to_numeric(d_valid[ycol], errors="coerce")

    if "session_id" not in d_valid.columns:
        d_valid["session_id"] = "S?"

    # Participant id (P01) inferred from session_id prefix
    d_valid["participant_id"] = d_valid["session_id"].astype(str).str.split("_").str[0].fillna("P?")

    # Protocol lookup at each minute: most common phase/block (assumes alignment across sessions)
    proto_cols = []
    if "protocol_block" in d_valid.columns:
        proto_cols.append("protocol_block")
    if "protocol_phase" in d_valid.columns:
        proto_cols.append("protocol_phase")

    proto_map = None
    if proto_cols:
        def _mode1(s: pd.Series):
            m = s.mode(dropna=True)
            if len(m) > 0:
                return str(m.iloc[0])
            return str(s.dropna().iloc[0]) if s.dropna().shape[0] else ""
        proto_map = (
            d_valid.groupby(xcol, as_index=True)[proto_cols]
            .agg(_mode1)
            .reset_index()
            .sort_values(xcol)
        )

    # Pivot session x minute (mean within minute)
    piv = d_valid.pivot_table(index=xcol, columns="session_id", values=ycol, aggfunc="mean")
    x = piv.index.to_numpy()

    # Envelope over sessions at each minute
    q10 = piv.quantile(0.10, axis=1, numeric_only=True)
    q90 = piv.quantile(0.90, axis=1, numeric_only=True)
    med = piv.median(axis=1, numeric_only=True)

    # Customdata for envelope hover (block, phase)
    if proto_map is not None and not proto_map.empty:
        pm = proto_map.set_index(xcol).reindex(piv.index)
        cd_cols = []
        cd_names = []
        if "protocol_block" in pm.columns:
            cd_cols.append(pm["protocol_block"].astype(str).to_numpy())
            cd_names.append("block")
        if "protocol_phase" in pm.columns:
            cd_cols.append(pm["protocol_phase"].astype(str).to_numpy())
            cd_names.append("phase")
        customdata_env = np.stack(cd_cols, axis=1) if cd_cols else None
        env_names = cd_names
    else:
        customdata_env = None
        env_names = []

    def _hover_env(prefix: str):
        ht = "minute=%{x}<br>" + prefix + "=%{y:.3g}"
        for i, nm in enumerate(env_names):
            ht += f"<br>{nm}=%{{customdata[{i}]}}"
        ht += "<extra></extra>"
        return ht

    fig = go.Figure()

    # Band (draw p90 then fill down to p10)
    fig.add_trace(go.Scatter(
        x=x, y=q90.to_numpy(),
        mode="lines",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
        name="p90",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=q10.to_numpy(),
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(0,114,178,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10–90% band",
        hovertemplate=_hover_env("p10"),
        customdata=customdata_env,
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=x, y=med.to_numpy(),
        mode="lines",
        line=dict(color=C_PRIMARY, width=3),
        name="median (QC-passed)",
        hovertemplate=_hover_env("median"),
        customdata=customdata_env,
    ))

    # Individual session traces (hidden by default) with participant-level legend control
    if show_individual_hidden:
        # create one dummy legend item per participant
        participants = sorted(d_valid["participant_id"].unique().tolist())
        for pid in participants:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(width=2, color="rgba(0,0,0,0.35)"),
                name=pid,
                legendgroup=pid,
                showlegend=True,
                visible="legendonly",
                hoverinfo="skip",
            ))

        # add all session traces, grouped by participant, but hidden and not cluttering legend
        session_to_pid = (
            d_valid[["session_id","participant_id"]]
            .drop_duplicates()
            .set_index("session_id")["participant_id"]
            .to_dict()
        )

        for sid in piv.columns.tolist():
            pid = session_to_pid.get(sid, "P?")
            ys = piv[sid]
            fig.add_trace(go.Scatter(
                x=x,
                y=ys.to_numpy(),
                mode="lines",
                line=dict(width=1.2, color="rgba(0,0,0,0.22)"),
                name=f"{sid}",
                legendgroup=pid,
                showlegend=False,
                visible="legendonly",
                hovertemplate=_hover_env(sid),
                customdata=customdata_env,
            ))

        # Grouping is preserved via legendgroup; external legend will toggle groups.
    # Description is rendered in the HTML card; avoid duplicating it inside Plotly.
    fig.update_layout(_plotly_base_layout(f"Combined — {ylabel}", None))

    fig.update_xaxes(
        title=dict(text="Protocol minute", font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False,
    )
    fig.update_yaxes(
        title=dict(text=ylabel, font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False,
    )

    fig.update_layout(meta=dict(
        legend_items=_legend_items_from_traces(fig),
        legend_note="Combined plots: click Pxx to show/hide all sessions for that participant.",
    ))

    return fig


# ------------------------------------------------------------
# Additional plots for v14 (correlations + phase pattern heatmap)
# ------------------------------------------------------------


def _heatmap_is_pvalues(title: str) -> bool:
    t = (title or "").lower()
    return ("p-value" in t) or ("pvalue" in t) or ("p values" in t)

def _heatmap_colorscale(title: str) -> str:
    # Diverging for correlations, sequential for p-values
    return "Viridis" if _heatmap_is_pvalues(title) else "RdBu"

def _heatmap_reversescale(title: str) -> bool:
    # For p-values, low p should look "strong" -> reverse Viridis so small values are darker
    return True if _heatmap_is_pvalues(title) else True

def _heatmap_hovertemplate(title: str) -> str:
    if _heatmap_is_pvalues(title):
        return "%{y} × %{x}<br>p=%{z:.4g}<extra></extra>"
    return "%{y} × %{x}<br>ρ=%{z:.3f}<extra></extra>"


def build_matrix_heatmap(df_mat: pd.DataFrame, *, title: str, description: str, zmin: Optional[float]=None, zmax: Optional[float]=None, fmt: str=".2f") -> go.Figure:
    """Generic annotated heatmap for square matrices."""
    mat = df_mat.copy()
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)

    # ensure square ordering
    cols = [c for c in mat.columns if c in mat.index]
    mat = mat.loc[cols, cols]

    x_vals = mat.columns.tolist()
    y_vals = mat.index.tolist()

    def _lbl(x):
        key = _canonical_metric_key(str(x)) or str(x)
        return " ".join(str(_pub_label(key)).split())

    mat, disp = _dedupe_square_matrix(mat, _lbl)
    z = mat.to_numpy(dtype=float)
    x_text = [_wrap_tick_label(d) for d in disp]
    y_text = x_text

    if zmin is None:
        zmin = float(np.nanmin(z)) if np.isfinite(z).any() else -1.0
    if zmax is None:
        zmax = float(np.nanmax(z)) if np.isfinite(z).any() else  1.0

    # Heatmap trace (NO cell annotations; hover shows values)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_vals,
        y=y_vals,
        zmin=zmin,
        zmax=zmax,
        colorscale=_heatmap_colorscale(title),
        reversescale=_heatmap_reversescale(title),
        colorbar=dict(title="", tickfont=dict(size=TICK_SIZE)),
        hovertemplate=_heatmap_hovertemplate(title),
    ))

    # Base UI layout (title/subtitle rendered in HTML card, not inside Plotly)
    fig.update_layout(**_plotly_base_layout(title, description))
    fig.update_layout(width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)
    _apply_heatmap_hover_spikes(fig)
    # Axis styling for single-panel figures (no subplot row/col)
    fig.update_xaxes(
        title=dict(text="", font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        tickangle=HEATMAP_TICK_ANGLE,
        tickmode="array",
        tickvals=x_vals,
        ticktext=x_text,
        automargin=True,
        ticks='outside',
        ticklen=4,

        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
    )
    fig.update_yaxes(
        title=dict(text="", font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        tickmode="array",
        tickvals=y_vals,
        ticktext=y_text,
        automargin=True,
        ticks='outside',
        ticklen=4,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
        autorange="reversed",  # matrix convention: first row at top
    )
    return fig


def build_phase_pattern_heatmap(zdf: pd.DataFrame, *, title: str, description: str) -> go.Figure:
    """Phase (rows) × metric (cols) z-scored heatmap, annotated."""
    # Expect columns: protocol_phase, metric1, metric2, ...
    df = zdf.copy()
    if "protocol_phase" in df.columns:
        df = df.set_index("protocol_phase")
    df.index = df.index.astype(str)

    # Choose numeric cols
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[num_cols]
    z = df.to_numpy(dtype=float)
    x_vals = df.columns.tolist()
    y_vals = df.index.tolist()

    x_text = normalize_heatmap_ticktexts(x_vals)

    # symmetric bounds for z-scores
    vmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 2.0
    vmax = max(1.0, min(4.0, vmax))
    vmin = -vmax

    # Heatmap trace (NO cell annotations; hover shows values)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_vals,
        y=y_vals,
        zmin=vmin,
        zmax=vmax,
        colorscale="RdBu",
        reversescale=False,
        colorbar=dict(title="z", tickfont=dict(size=TICK_SIZE)),
        hovertemplate="%{y} × %{x}<br>z=%{z:.2f}<extra></extra>",
    ))

    fig.update_layout(**_plotly_base_layout(title, description))
    fig.update_layout(width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT)
    _apply_heatmap_hover_spikes(fig)
    fig.update_xaxes(
        title=dict(text="", font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        tickangle=HEATMAP_TICK_ANGLE,
        tickmode="array",
        tickvals=x_vals,
        ticktext=x_text,
        automargin=True,
        ticks='outside',
        ticklen=4,

        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
    )
    fig.update_yaxes(
        title=dict(text="", font=dict(size=AXIS_TITLE_SIZE)),
        tickfont=dict(size=TICK_SIZE),
        automargin=True,
        ticks='outside',
        ticklen=4,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
        autorange="reversed",
    )
    return fig

def _html_header(title: str) -> str:
    return f"""<!doctype html>
<html><head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>{title}</title>
<script charset='utf-8' src='{PLOTLY_CDN}'></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 28px; background: #fff; }}
  h1 {{ font-size: 42px; margin: 0 0 6px 0; letter-spacing: -0.3px; }}
  .sub {{ font-size: 19px; color: rgba(0,0,0,0.72); margin: 0 0 14px 0; }}
  .meta {{ font-size: 16px; color: rgba(0,0,0,0.70); margin: 0 0 18px 0; line-height: 1.45; }}
  .toc a {{ text-decoration: none; }}
  .toc li {{ margin: 6px 0; font-size: 16px; }}
  .card {{ border: 1px solid rgba(0,0,0,0.10); border-radius: 16px; padding: 16px 16px 12px 16px; margin: 18px 0 24px 0; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }}
  .summary {{ border-left: 6px solid rgba(0,114,178,0.55); }}
  .kpi {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; }}
  .kpi .box {{ border:1px solid rgba(0,0,0,0.10); border-radius:14px; padding:10px 12px; min-width: 220px; }}
  .kpi .big {{ font-size: 22px; font-weight: 700; }}
  .kpi .lab {{ font-size: 14px; color: rgba(0,0,0,0.65); margin-top: 2px; }}
  hr {{ border: none; border-top: 1px solid rgba(0,0,0,0.12); margin: 18px 0; }}
  .figtitle {{ font-size:22px; margin:0 0 6px 0; line-height:1.25; word-break:break-word; }}
  .figsub {{ font-size:15px; margin: 0 0 12px 0; color: rgba(0,0,0,0.70); line-height:1.3; }}
  .dlbar {{ display:flex; align-items:center; gap:10px; margin: 0 0 10px 0; }}
  .dllab {{ font-size: 14px; color: rgba(0,0,0,0.70); }}
  .dlbtn {{ display:inline-block; padding:6px 10px; border-radius:999px; border:1px solid rgba(0,0,0,0.16); text-decoration:none; color: rgba(0,0,0,0.85); font-size: 14px; }}
  .dlbtn:hover {{ background: rgba(0,0,0,0.03); }}
  .legendbox {{ margin-top: 10px; padding-top: 12px; border-top: 1px dashed rgba(0,0,0,0.18); }}
  .legendhdr {{ font-size: 20px; margin: 0 0 10px 0; color: rgba(0,0,0,0.80); }}
  .legendnote {{ margin-top: 6px; font-size: 14px; color: rgba(0,0,0,0.65); }}
  .legendgrid {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
  .lgbtn {{ display:flex; align-items:center; gap:10px; padding:10px 14px; border-radius:999px; border:1px solid rgba(0,0,0,0.15); background: rgba(255,255,255,0.9); cursor:pointer; user-select:none; }}
  .lgbtn:hover {{ background: rgba(0,0,0,0.03); }}
  .lgbtn.off {{ opacity:0.38; }}
  .swatch {{ width:18px; height:18px; border-radius:5px; border:1px solid rgba(0,0,0,0.18); }}
</style>
</head><body>
<script>
function _toggleLegendGroup(plotDiv, group, btn) {{
  const gd = plotDiv;
  const data = gd.data || [];
  let indices = [];
  for (let i=0; i<data.length; i++) {{
    const lg = (data[i].legendgroup || data[i].name || '');
    if (lg === group) indices.push(i);
  }}
  if (indices.length === 0) return;
  let anyVisible = false;
  for (const i of indices) {{
    const v = data[i].visible;
    if (v === undefined || v === true) {{ anyVisible = true; break; }}
  }}
  const newVis = anyVisible ? 'legendonly' : true;
  Plotly.restyle(gd, {{visible: newVis}}, indices);
  if (newVis === true) btn.classList.remove('off');
  else btn.classList.add('off');
}}

function buildLegend(plotDivId, items, note) {{
  const plotDiv = document.getElementById(plotDivId);
  const host = document.getElementById(plotDivId + '_legend');
  if (!plotDiv || !host || !items || items.length === 0) return;
  const grid = host.querySelector('.legendgrid');
  grid.innerHTML = '';
  for (const it of items) {{
    const btn = document.createElement('div');
    btn.className = 'lgbtn';
    const sw = document.createElement('div');
    sw.className = 'swatch';
    sw.style.background = it.color || 'rgba(0,0,0,0.35)';
    const tx = document.createElement('div');
    tx.textContent = it.label;
    btn.appendChild(sw);
    btn.appendChild(tx);
    btn.addEventListener('click', () => _toggleLegendGroup(plotDiv, it.group, btn));
    grid.appendChild(btn);
  }}
  const nn = host.querySelector('.legendnote');
  nn.textContent = note || '';
}}
</script>
"""

def _plot_div(fig: go.Figure, div_id: str) -> str:
    """Render Plotly figure into a stable div id + an external legend container.

    UI strategy:
    - Plotly legend is always disabled (we render a pill-based legend below the plot canvas).
    - The HTML title/subtitle live outside the Plotly canvas, so exported SVG/PNG contains ONLY the plot.
    - For correlation heatmaps we add a "crosshair" hover guide (dashed lines) via a tiny JS handler.
    """
    # Hard-disable Plotly legend, even if a trace was created with showlegend=True.
    fig.update_layout(showlegend=False)
    for tr in fig.data:
        try:
            tr.showlegend = False
        except Exception:
            pass

    try:
        html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            default_height="650px",
            default_width="100%",
            div_id=div_id,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )
    except TypeError:
        html = fig.to_html(full_html=False, include_plotlyjs=False, default_height="650px", default_width="100%")
        m = re.search(r"<div id=\"([^\"]+)\" class=\"plotly-graph-div\"", html)
        if m:
            old = m.group(1)
            html = html.replace(f'id="{old}"', f'id="{div_id}"', 1)
            html = html.replace(f"Plotly.newPlot('{old}'", f"Plotly.newPlot('{div_id}'")
            html = html.replace(f"document.getElementById('{old}')", f"document.getElementById('{div_id}')")

    meta = fig.layout.meta if hasattr(fig.layout, "meta") else None
    meta = meta if isinstance(meta, dict) else {}
    legend_items = meta.get("legend_items", [])
    note = meta.get("legend_note", "")

    out = html

    # Optional: dashed crosshair for correlation heatmaps (gives the UI you showed in the screenshot).
    if meta.get("is_corr", False):
        out += f"""
<script>
(function() {{
  const gd = document.getElementById('{div_id}');
  if (!gd) return;

  const baseShapes = (gd.layout && gd.layout.shapes) ? gd.layout.shapes.slice() : [];

  function _setCrosshair(pt) {{
    // pt.x / pt.y are category labels for our heatmap; we use paper refs to span full panel.
    const x = pt.x;
    const y = pt.y;

    const cross = [
      {{
        type: 'line',
        xref: 'x', yref: 'paper',
        x0: x, x1: x, y0: 0, y1: 1,
        line: {{color: 'rgba(0,0,0,0.45)', width: 1, dash: 'dot'}},
        layer: 'above'
      }},
      {{
        type: 'line',
        xref: 'paper', yref: 'y',
        x0: 0, x1: 1, y0: y, y1: y,
        line: {{color: 'rgba(0,0,0,0.45)', width: 1, dash: 'dot'}},
        layer: 'above'
      }}
    ];

    Plotly.relayout(gd, {{shapes: baseShapes.concat(cross)}});
  }}

  gd.on('plotly_hover', function(ev) {{
    if (!ev || !ev.points || ev.points.length === 0) return;
    const p = ev.points[0];
    if (!p || p.x === undefined || p.y === undefined) return;
    _setCrosshair(p);
  }});

  gd.on('plotly_unhover', function() {{
    Plotly.relayout(gd, {{shapes: baseShapes}});
  }});
}})();
</script>
"""

    if not legend_items:
        return out

    legend_html = (
        f"<div class='legendbox' id='{div_id}_legend'>"
        f"<div class='legendhdr'>Legend (click to toggle)</div>"
        f"<div class='legendgrid'></div>"
        f"<div class='legendnote'></div>"
        f"</div>"
        f"<script>buildLegend('{div_id}', {json.dumps(legend_items)}, {json.dumps(note)});</script>"
    )
    return out + legend_html


def _figure_card_html(anchor: str, title: str, desc: str, fig: go.Figure) -> str:
    """HTML for one figure card.

    IMPORTANT (v8.3 UI fix):
    - The card wrapper keeps id=anchor for TOC navigation.
    - The Plotly div id MUST be unique and MUST NOT equal the card id, otherwise Plotly.newPlot()
      may target the wrapper div and overwrite the HTML title/subtitle.
    """
    # Ensure exported images (camera icon) contain only the plot, not the surrounding HTML header/subtitle.
    try:
        fig.update_layout(title=None)
    except Exception:
        pass

    plot_id = f"{anchor}__plot"

    # Optional static exports (SVG/PDF) are stored in fig.layout.meta by _export_static_assets().
    meta = fig.layout.meta if hasattr(fig.layout, "meta") else None
    meta = meta if isinstance(meta, dict) else {}
    svg_href = meta.get("export_svg")
    pdf_href = meta.get("export_pdf")
    dl = ""
    if svg_href or pdf_href:
        bits = []
        if svg_href:
            bits.append(f"<a class='dlbtn' href='{svg_href}' download>SVG</a>")
        if pdf_href:
            bits.append(f"<a class='dlbtn' href='{pdf_href}' download>PDF</a>")
        dl = (
            "<div class='dlbar'>"
            "<span class='dllab'>Download vector:</span> "
            + " ".join(bits)
            + "</div>"
        )

    return (
        f"<div class='card' id='{anchor}'>"
        f"<div class='figtitle'>{title}</div>"
        f"<div class='figsub'>{desc}</div>"
        f"{dl}"
        f"{_plot_div(fig, plot_id)}"
        f"</div>"
    )


def _export_static_assets(fig: go.Figure, out_dir: Path, stem: str, *, enabled: bool) -> None:
    """Export a figure to SVG + PDF next to the HTML (server-side via Kaleido).

    We keep the interactive HTML as the primary artifact. Static exports are a convenience for
    publication workflows and reliable PDF export (which plotly.js in-browser does not provide).
    
    On success, stores relative paths in fig.layout.meta: export_svg, export_pdf.
    """
    if not enabled or not HAVE_KALEIDO:
        return
    export_dir = out_dir / STATIC_EXPORT_DIRNAME
    export_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the exported vector contains only the plot.
    try:
        fig2 = fig.full_copy()
    except Exception:
        fig2 = fig
    try:
        fig2.update_layout(title=None)
    except Exception:
        pass

    svg_path = export_dir / f"{stem}.svg"
    pdf_path = export_dir / f"{stem}.pdf"

    try:
        pio.write_image(fig2, svg_path, format="svg", scale=STATIC_EXPORT_SCALE)
        pio.write_image(fig2, pdf_path, format="pdf", scale=STATIC_EXPORT_SCALE)
    except Exception:
        # Kaleido missing or export failed; skip silently (HTML still works).
        return

    meta = fig.layout.meta if hasattr(fig.layout, "meta") else None
    meta = meta if isinstance(meta, dict) else {}
    meta["export_svg"] = f"{STATIC_EXPORT_DIRNAME}/{svg_path.name}"
    meta["export_pdf"] = f"{STATIC_EXPORT_DIRNAME}/{pdf_path.name}"
    try:
        fig.update_layout(meta=meta)
    except Exception:
        # Last resort: don't fail the report for metadata.
        return

def build_one_session(session_dir: Path, *, overwrite: bool, export_static: bool = False) -> Path:
    minute = _read_csv_safely(session_dir / "features_minute.csv")
    session_id = str(minute["session_id"].iloc[0]) if "session_id" in minute.columns and len(minute) else session_dir.name
    participant_id = session_id.split("_")[0] if "_" in session_id else session_id

    qc = {}
    qc_path = session_dir / "qc_summary.json"
    if qc_path.exists():
        try:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
        except Exception:
            qc = {}

    n = len(minute)
    valid_n = int(pd.to_numeric(minute.get("valid_joint_minute", pd.Series([1]*n)), errors="coerce").fillna(0).sum()) if n else 0
    valid_pct = (100.0 * valid_n / n) if n else 0.0

    # time window if available
    t0 = _safe_text(minute["minute_utc"].iloc[0]) if "minute_utc" in minute.columns and n else ""
    t1 = _safe_text(minute["minute_utc"].iloc[-1]) if "minute_utc" in minute.columns and n else ""

    figs: List[Tuple[str, str, go.Figure, str]] = []
    fig_num = 1
    for col, lab, _unit in DEFAULT_METRICS:
        if col not in minute.columns:
            continue
        # choose validity mask per modality if available
        vcol = "valid_joint_minute"
        if col.startswith("hr_") and "valid_hr_minute" in minute.columns:
            vcol = "valid_hr_minute"
        elif col.startswith("eda_") and "valid_eda_minute" in minute.columns:
            vcol = "valid_eda_minute"
        elif col.startswith("temp_") and "valid_temp_minute" in minute.columns:
            vcol = "valid_temp_minute"
        elif col.startswith("acc_") and "valid_acc_minute" in minute.columns:
            vcol = "valid_acc_minute"

        desc = _subtitle_timeseries(minute, ycol=col, ylabel=lab, valid_col=vcol, session_id=session_id)
        fig = build_timeseries_figure(minute, session_id=session_id, ycol=col, ylabel=lab, valid_col=vcol, description=desc, show_individual=True)
        figs.append((f"fig_{col}", _pretty_figure_title(fig_num, col, kind="feature"), fig, desc))
        fig_num += 1

    # Phase distributions (boxplots) for all metrics (QC-passed minutes)
    phase_dist_fig = build_phase_distributions_boxgrid(minute, session_id=session_id, metrics=DEFAULT_METRICS, valid_col="valid_joint_minute")
    if phase_dist_fig and len(phase_dist_fig.data):
        figs.append(("fig_phase_box", _pretty_figure_title(fig_num, "phase_box", kind="dist"), phase_dist_fig, "Distribution across protocol phases using QC-passed minutes only (valid_joint_minute=1)."))
        fig_num += 1

    # phase-level correlation within session (valid_only=1, phase means)
    corr_fig = None
    phase_path = session_dir / "features_phase.csv"
    if phase_path.exists():
        phase = _read_csv_safely(phase_path)
        if "valid_only" in phase.columns:
            phase = phase.loc[pd.to_numeric(phase["valid_only"], errors="coerce").fillna(0).astype(int) == 1].copy()
        # keep only mean-of-minute columns -> those end with "__mean"
        metric_cols = [c for c in phase.columns if c.endswith("__mean")]
        if len(metric_cols) >= 3:
            M = phase[metric_cols].apply(pd.to_numeric, errors="coerce")
            corr = _spearman_corr(M)
            corr_fig = build_corr_heatmap(f"{session_id} — Phase-level Spearman correlation", corr)

    if corr_fig is not None:
        figs.append(("fig_corr", _pretty_figure_title(fig_num, "corr", kind="corr"), corr_fig, "Spearman correlation on phase-level means (valid minutes only)."))
        fig_num += 1


    # Correlation p-values (if exported by batch pipeline v2)
    pvals_path = session_dir / "phase_corr_spearman_pvalues.csv"
    if pvals_path.exists():
        pvals = _read_csv_safely(pvals_path, index_col=0)
        if len(pvals.columns) and len(pvals.index):
            pv_fig = build_pvalue_heatmap(f"{session_id} — Phase-level Spearman p-values", pvals)
            figs.append(("fig_corr_p", _pretty_figure_title(fig_num, "corr_pvalues", kind="corr_p"), pv_fig, "Spearman correlation p-values (phase-level means)."))
            fig_num += 1

    # Phase mean deltas vs baseline (uses exported phase means if available)
    phase_means_path = session_dir / "phase_means_valid.csv"
    phase_for_delta = None
    if phase_means_path.exists():
        phase_for_delta = _read_csv_safely(phase_means_path)
    elif (session_dir / "features_phase.csv").exists():
        phase_for_delta = _read_csv_safely(session_dir / "features_phase.csv")

    if phase_for_delta is not None and len(phase_for_delta):
        # choose __mean columns (phase-level averages)
        mean_cols = [c for c in phase_for_delta.columns if c.endswith("__mean")]
        mean_cols = [c for c in mean_cols if c not in {"minute_index__mean"}]
        mean_cols = mean_cols[:12]  # keep page weight sane
        if len(mean_cols) >= 3:
            delta_fig = build_phase_delta_boxgrid(phase_for_delta, session_id=session_id, metrics_mean_cols=mean_cols)
            if delta_fig and len(delta_fig.data):
                figs.append(("fig_phase_delta", _pretty_figure_title(fig_num, "phase_delta", kind="dist"), delta_fig, "Phase mean changes relative to the baseline phase (first protocol phase)."))
                fig_num += 1

    out_html = session_dir / f"{session_id}_interactive_report.html"
    if out_html.exists() and not overwrite:
        return out_html

    # Export static (SVG/PDF) copies for publication-grade downloads.
    # These are linked directly from the HTML (no browser-side PDF export tricks).
    for anchor, _title, fig, _desc in figs:
        _export_static_assets(fig, session_dir, stem=anchor, enabled=export_static)

    # HTML assembly
    parts = [_html_header(f"Wearable Physiology – {session_id} Report")]
    parts.append(f"<h1>Wearable Physiology Report – {session_id} Session Report</h1>")
    parts.append(f"<p class=\'sub\'>Participant {participant_id} • Session {session_id} • Time zone: UTC</p>")
    parts.append("<div class='card summary'>")
    parts.append("<h2 style='margin:0 0 8px 0;font-size:22px'>Session summary</h2>")
    parts.append("<div class='meta'>This report visualizes minute-aligned Empatica features with a protocol ribbon (blocks + phase boundaries). Hover for point-level context (block/phase/QC). Click legend items to toggle traces; individual traces can be shown/hidden. Export via the camera icon in each figure toolbar.</div>")
    parts.append("<div class='kpi'>")
    parts.append(f"<div class='box'><div class='big'>{n}</div><div class='lab'>total protocol minutes</div></div>")
    parts.append(f"<div class='box'><div class='big'>{valid_n} ({valid_pct:.1f}%)</div><div class='lab'>QC-passed minutes (valid_joint_minute)</div></div>")
    parts.append(f"<div class='box'><div class='big'>{(100-valid_pct):.1f}%</div><div class='lab'>minutes not QC-passed</div></div>")
    parts.append("</div></div>")

    # TOC
    parts.append("<div class='toc'><b>Contents</b><ul>")
    for anchor, title, _fig, _desc in figs:
        parts.append(f"<li><a href='#{anchor}'>{title}</a></li>")
    parts.append("</ul></div><hr/>")
    parts.append("<p class='meta'><b>Controls:</b> hover for details, drag to zoom, double-click to reset. Click legend items to hide/show traces. Export via the camera icon in the figure toolbar.</p>")

    # Figures
    for i, (anchor, title, fig, desc) in enumerate(figs, start=1):
        parts.append(_figure_card_html(anchor, title, desc, fig))

    parts.append("</body></html>")
    out_html.write_text("\n".join(parts), encoding="utf-8")
    return out_html

def build_combined(batch_out: Path, *, overwrite: bool, export_static: bool = False) -> Path:
    session_dirs = _find_session_dirs(batch_out)
    if not session_dirs:
        raise FileNotFoundError(f"No session folders with features_minute.csv found under: {batch_out}")

    minutes_all = []
    phases_all = []
    for sd in session_dirs:
        m = _read_csv_safely(sd / "features_minute.csv")
        if "session_id" not in m.columns:
            m["session_id"] = sd.name
        minutes_all.append(m)

        ph_path = sd / "features_phase.csv"
        if ph_path.exists():
            ph = _read_csv_safely(ph_path)
            if "session_id" not in ph.columns:
                ph["session_id"] = sd.name
            phases_all.append(ph)

    df_all = pd.concat(minutes_all, ignore_index=True)
    df_all["minute_index"] = _minute_axis(df_all)

    n_sessions = df_all["session_id"].nunique()
    n_rows = len(df_all)
    valid_rows = int(pd.to_numeric(df_all.get("valid_joint_minute", pd.Series([1]*n_rows)), errors="coerce").fillna(0).sum())
    valid_pct = (100.0 * valid_rows / n_rows) if n_rows else 0.0

    figs: List[Tuple[str, str, go.Figure, str]] = []
    fig_num = 1
    for col, lab, _unit in DEFAULT_METRICS:
        if col not in df_all.columns:
            continue
        desc = ("Cohort envelope for <b>{}</b>. Median curve across sessions computed per protocol minute using QC-passed minutes; shaded band is 10–90% across sessions. ""Legend toggles participants (Pxx): clicking Pxx shows/hides all sessions for that participant (scales to 80 sessions). ""Hover reports minute, value, and the cohort mode of protocol block/phase.").format(_safe_text(col))
        fig = build_envelope_figure(df_all, ycol=col, ylabel=lab, description=desc, valid_col="valid_joint_minute", show_individual_hidden=True)
        figs.append((f"fig_combined_{col}", _pretty_figure_title(fig_num, col, kind="feature"), fig, desc))
        fig_num += 1


    
    # --- Cohort correlation matrices (prefer precomputed files from v14 all_sessions folder) ---
    all_sessions_dir = (batch_out / "all_sessions")
    def _try_read_mat(name: str) -> Optional[pd.DataFrame]:
        p = all_sessions_dir / name
        if p.exists():
            return _read_csv_safely(p, index_col=0)
        return None

    # Phase-level Spearman (rho) + p-values (optional)
    mat_phase = _try_read_mat("all_sessions_phase_corr_spearman.csv")
    if mat_phase is None:
        mat_phase = _try_read_mat("phase_corr_spearman.csv")
    mat_p = _try_read_mat("all_sessions_phase_corr_spearman_pvalues.csv")
    if mat_p is None:
        mat_p = _try_read_mat("phase_corr_spearman_pvalues.csv")
    if mat_phase is not None:
        desc = "Phase-level Spearman correlation (ρ) computed on session×phase aggregates (QC-passed)."
        fig = build_matrix_heatmap(mat_phase, title=_pretty_figure_title(fig_num, "Phase-level Spearman correlation (ρ)", kind="analysis"), description=desc, zmin=-1, zmax=1)
        figs.append((f"fig_combined_phase_corr", _pretty_figure_title(fig_num, "corr", kind="corr"), fig, desc))
        fig_num += 1
    if mat_p is not None:
        desc = "Phase-level Spearman correlation p-values (approx; see methods)."
        fig = build_matrix_heatmap(mat_p, title=_pretty_figure_title(fig_num, "Phase-level correlation p-values", kind="analysis"), description=desc, zmin=0, zmax=1, fmt=".3f")
        figs.append((f"fig_combined_phase_p", _pretty_figure_title(fig_num, "corr_p", kind="corr_p"), fig, desc))
        fig_num += 1

    # Minute-level meta correlation (block bootstrap) and partial (control ENMO)
    mat_min = _try_read_mat("all_sessions_minute_corr_spearman_meta_mean.csv")
    if mat_min is not None:
        desc = "Minute-level within-session meta Spearman correlation mean (block bootstrap)."
        fig = build_matrix_heatmap(mat_min, title=_pretty_figure_title(fig_num, "Minute-level meta Spearman correlation (ρ)", kind="analysis"), description=desc, zmin=-1, zmax=1)
        figs.append((f"fig_combined_min_corr", _pretty_figure_title(fig_num, "Minute-level meta Spearman correlation (ρ)"), fig, desc))
        fig_num += 1

    mat_part = _try_read_mat("all_sessions_minute_partial_corr_spearman_meta_mean.csv")
    if mat_part is not None:
        desc = "Minute-level within-session meta partial Spearman correlation mean (control ENMO). Control variable is excluded from the matrix."
        fig = build_matrix_heatmap(mat_part, title=_pretty_figure_title(fig_num, "Minute-level partial meta correlation (ρ)", kind="analysis"), description=desc, zmin=-1, zmax=1)
        figs.append((f"fig_combined_min_part_corr", _pretty_figure_title(fig_num, "Minute-level partial meta correlation (ρ, control ENMO)"), fig, desc))
        fig_num += 1

    # Phase pattern heatmap (z-scored)
    zpat = _try_read_mat("all_sessions_phase_means_zscore.csv")
    if zpat is not None:
        desc = "Protocol phase × metric signature heatmap. Values are z-scored within session across phases, then averaged across sessions."
        # zpat may include protocol_phase column in first col when saved without index; handle both
        if "protocol_phase" not in zpat.columns and zpat.index.name != "protocol_phase":
            zpat = zpat.reset_index().rename(columns={zpat.columns[0]: "protocol_phase"})
        fig = build_phase_pattern_heatmap(zpat, title=_pretty_figure_title(fig_num, "Phase pattern heatmap (z-scored)", kind="analysis"), description=desc)
        figs.append((f"fig_combined_phase_pattern", _pretty_figure_title(fig_num, "Phase pattern heatmap (z-scored)"), fig, desc))
        fig_num += 1

# Cohort phase distributions (boxplots of per-session phase means)
    if phases_all:
        ph_all = pd.concat(phases_all, ignore_index=True)
        if "valid_only" in ph_all.columns:
            ph_all = ph_all.loc[pd.to_numeric(ph_all["valid_only"], errors="coerce").fillna(0).astype(int) == 1].copy()

        mean_cols = [c for c in ph_all.columns if c.endswith("__mean")]
        # keep a stable, readable subset (same as DEFAULT_METRICS where possible)
        preferred = []
        base_map = {m[0]: m[1] for m in DEFAULT_METRICS}  # minute metric -> label
        for minute_col, _lab, _u in DEFAULT_METRICS:
            c = f"{minute_col}__mean"
            if c in mean_cols:
                preferred.append(c)
        # fall back to the first N mean cols
        metric_cols = preferred if preferred else mean_cols
        metric_cols = metric_cols[:12]

        if "protocol_phase" in ph_all.columns and len(metric_cols) >= 3:
                        # Dropdown boxplot UI (one metric at a time) for readability
            phases = list(dict.fromkeys(ph_all["protocol_phase"].astype(str).tolist()))

            fig_pd = go.Figure()
            metric_items = []
            for c in metric_cols:
                base = c.replace("__mean", "")
                metric_items.append((c, _pub_label(base) + " — Mean"))

            for i, (c, lab) in enumerate(metric_items):
                tmp = ph_all[["protocol_phase", c]].copy()
                tmp["protocol_phase"] = tmp["protocol_phase"].astype(str)
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                tmp = tmp.dropna(subset=[c])

                fig_pd.add_trace(
                    go.Violin(
                        x=tmp["protocol_phase"],
                        y=tmp[c],
                        name=lab,
                        points="suspectedoutliers",
                        jitter=0.18,
                        scalemode="width",
                        spanmode="hard",
                        box=dict(visible=True, width=0.25),
                        meanline=dict(visible=True),
                        marker=dict(size=4, opacity=0.55),
                        line=dict(width=1.2),
                        visible=(i == 0),
                        hovertemplate="phase=%{x}<br>" + f"{lab}=%{{y:.3g}}<extra></extra>",
                        showlegend=False,
                    )
                )

            buttons = []
            for i, (_c, lab) in enumerate(metric_items):
                vis = [False] * len(metric_items)
                vis[i] = True
                buttons.append(
                    dict(
                        label=lab,
                        method="update",
                        args=[
                            {"visible": vis},
                            {"yaxis": {"title": {"text": lab}}},
                        ],
                    )
                )

            fig_pd.update_xaxes(
                title="protocol phase",
                type="category",
                categoryorder="array",
                categoryarray=phases,
                tickangle=-25,
                tickfont=dict(size=HEATMAP_TICK_SIZE),
                automargin=True,
            )
            fig_pd.update_yaxes(
                title=dict(text=metric_items[0][1], font=dict(size=AXIS_TITLE_SIZE)),
                tickfont=dict(size=TICK_SIZE),
                automargin=True,
                zeroline=False,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
            )
            fig_pd.update_layout(
                _plotly_base_layout("Combined cohort — Phase distributions (per-session phase means)"),
                updatemenus=[
                    dict(
                        type="dropdown",
                        x=0.0,
                        y=1.12,
                        xanchor="left",
                        yanchor="top",
                        buttons=buttons,
                        direction="down",
                        showactive=True,
                        bgcolor="rgba(255,255,255,0.92)",
                        bordercolor="rgba(0,0,0,0.18)",
                        borderwidth=1,
                        font=dict(size=14),
                    )
                ],
                margin=dict(l=65, r=30, t=70, b=60),
                height=720,
            )

            figs.append(("fig_combined_phase_box", _pretty_figure_title(fig_num, "combined_phase_box", kind="dist"), fig_pd,
                         "Per-session phase means (valid minutes only) summarized as boxplots across the cohort for each protocol phase."))
            fig_num += 1

            # Cohort deltas vs baseline (per session baseline = first phase)
            # compute deltas per session, then provide:
            #   (a) a clean cohort summary heatmap (median Δ) and
            #   (b) an optional distribution view (violin+box) via dropdown (one metric at a time)
            deltas = []
            for sid, g in ph_all.groupby("session_id", sort=False):
                gg = g.copy()
                # stable order within session
                if "protocol_block" in gg.columns:
                    gg = gg.sort_values(["protocol_block", "protocol_phase"], kind="stable")
                ph_order = list(dict.fromkeys(gg["protocol_phase"].astype(str).tolist()))
                if not ph_order:
                    continue
                base = ph_order[0]
                base_row = gg.loc[gg["protocol_phase"].astype(str) == base, metric_cols].apply(pd.to_numeric, errors="coerce")
                if len(base_row) == 0:
                    continue
                base_vals = base_row.iloc[0]
                for ph in ph_order:
                    row = gg.loc[gg["protocol_phase"].astype(str) == ph, metric_cols].apply(pd.to_numeric, errors="coerce")
                    if len(row) == 0:
                        continue
                    d = row.iloc[0] - base_vals
                    rec = {"session_id": sid, "protocol_phase": ph}
                    for c in metric_cols:
                        rec[c] = d.get(c)
                    deltas.append(rec)

            if deltas:
                ddf = pd.DataFrame(deltas)

                # ---------- (a) Cohort summary: median Δ heatmap ----------
                heat_metrics = [c for c in metric_cols if c in ddf.columns]
                heat_phases = phases  # from above (appearance order)

                med = np.full((len(heat_metrics), len(heat_phases)), np.nan, dtype=float)
                iqr = np.full_like(med, np.nan)
                nn  = np.zeros_like(med, dtype=int)

                for mi, c in enumerate(heat_metrics):
                    for pj, ph in enumerate(heat_phases):
                        vals = pd.to_numeric(ddf.loc[ddf["protocol_phase"].astype(str) == str(ph), c], errors="coerce").dropna()
                        nn[mi, pj] = int(len(vals))
                        if len(vals) == 0:
                            continue
                        med[mi, pj] = float(np.nanmedian(vals))
                        q1, q3 = np.nanpercentile(vals.to_numpy(dtype=float), [25, 75])
                        iqr[mi, pj] = float(q3 - q1)

                ylab = [_pub_label(c.replace("__mean", "")) + " — Mean" for c in heat_metrics]
                xlab = [str(ph) for ph in heat_phases]

                # For hover we want to show median, IQR, and n.
                custom = np.stack([iqr, nn.astype(float)], axis=-1)  # (..., 2)

                hm = go.Heatmap(
                    z=med,
                    x=xlab,
                    y=ylab,
                    zmid=0,
                    colorscale="RdBu",
                    reversescale=True,
                    xgap=1,
                    ygap=1,
                    colorbar=dict(
                        title=dict(text="Median Δ", font=dict(size=AXIS_TITLE_SIZE)),
                        tickfont=dict(size=TICK_SIZE),
                        len=0.85,
                    ),
                    customdata=custom,
                    hovertemplate=(
                        "y=%{y}<br>"
                        "x=%{x}<br>"
                        "median Δ=%{z:.3g}<br>"
                        "IQR=%{customdata[0]:.3g}<br>"
                        "n=%{customdata[1]:.0f}"
                        "<extra></extra>"
                    ),
                )

                fig_d_sum = go.Figure(data=[hm])
                fig_d_sum.update_layout(_plotly_base_layout("Combined cohort — Phase deltas vs baseline (median Δ heatmap)",
                                                           "Cohort summary of within-session changes relative to each session's first phase."))
                fig_d_sum.update_xaxes(tickangle=-25, tickfont=dict(size=HEATMAP_TICK_SIZE), automargin=True)
                fig_d_sum.update_yaxes(autorange="reversed", tickfont=dict(size=HEATMAP_TICK_SIZE), automargin=True, scaleanchor="x", scaleratio=1)
                fig_d_sum.update_layout(height=max(820, 240 + 55 * len(ylab)), width=max(980, 240 + 85 * len(xlab)))
                figs.append(("fig_combined_phase_delta_summary", _pretty_figure_title(fig_num, "combined_phase_delta_summary", kind="delta_summary"), fig_d_sum,
                             "Median within-session phase deltas (Δ vs each session's baseline phase). Hover shows median, IQR and n."))
                fig_num += 1

                # ---------- (b) Distribution view: dropdown per metric (violin + box) ----------
                fig_d = go.Figure()
                metric_items = []
                for c in heat_metrics:
                    metric_items.append((c, _pub_label(c.replace("__mean", "")) + " — Mean"))

                for i, (c, lab) in enumerate(metric_items):
                    tmp = ddf[["protocol_phase", c]].copy()
                    tmp["protocol_phase"] = tmp["protocol_phase"].astype(str)
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                    tmp = tmp.dropna(subset=[c])

                    fig_d.add_trace(
                        go.Violin(
                            x=tmp["protocol_phase"],
                            y=tmp[c],
                            name=lab,
                            points="suspectedoutliers",
                            jitter=0.18,
                            scalemode="width",
                            spanmode="hard",
                            box=dict(visible=True, width=0.25),
                            meanline=dict(visible=True),
                            marker=dict(size=4, opacity=0.55),
                            line=dict(width=1.2),
                            visible=(i == 0),
                            hovertemplate="phase=%{x}<br>" + f"Δ {lab}=%{{y:.3g}}<extra></extra>",
                            showlegend=False,
                        )
                    )

                buttons = []
                for i, (_c, lab) in enumerate(metric_items):
                    vis = [False] * len(metric_items)
                    vis[i] = True
                    buttons.append(dict(label=lab, method="update", args=[{"visible": vis}, {"yaxis": {"title": {"text": f"Δ {lab}"}}}]))

                fig_d.update_xaxes(
                    title="protocol phase",
                    type="category",
                    categoryorder="array",
                    categoryarray=heat_phases,
                    tickangle=-25,
                    tickfont=dict(size=HEATMAP_TICK_SIZE),
                    automargin=True,
                )
                fig_d.update_yaxes(
                    title=dict(text=f"Δ {metric_items[0][1]}", font=dict(size=AXIS_TITLE_SIZE)),
                    tickfont=dict(size=TICK_SIZE),
                    automargin=True,
                    zeroline=False,
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.08)",
                )
                fig_d.update_layout(
                    _plotly_base_layout("Combined cohort — Phase deltas vs baseline (distribution view)"),
                    updatemenus=[
                        dict(
                            type="dropdown",
                            x=0.0,
                            y=1.12,
                            xanchor="left",
                            yanchor="top",
                            buttons=buttons,
                            direction="down",
                            showactive=True,
                            bgcolor="rgba(255,255,255,0.92)",
                            bordercolor="rgba(0,0,0,0.18)",
                            borderwidth=1,
                            font=dict(size=14),
                        )
                    ],
                    margin=dict(l=65, r=30, t=70, b=60),
                    height=720,
                    shapes=[dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=0, y1=0,
                                 line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))],
                )

                figs.append(("fig_combined_phase_delta", _pretty_figure_title(fig_num, "combined_phase_delta", kind="dist"), fig_d,
                             "Within-session change relative to the first protocol phase, shown as cohort violin+box distributions per phase (dropdown metric)."))
                fig_num += 1

    # Combined correlation matrices exported by batch pipeline v2 (preferred for consistency)
    combined_corr_path = batch_out / "combined_phase_corr_spearman.csv"
    combined_p_path = batch_out / "combined_phase_corr_spearman_pvalues.csv"
    combined_means_path = batch_out / "combined_phase_means_valid.csv"

    if combined_p_path.exists():
        pvals = _read_csv_safely(combined_p_path, index_col=0)
        if len(pvals.columns) and len(pvals.index):
            pv_fig = build_pvalue_heatmap("Combined cohort — Phase-level Spearman p-values (pooled)", pvals)
            figs.append(("fig_combined_corr_p", _pretty_figure_title(fig_num, "combined_corr_pvalues", kind="combined_p"), pv_fig,
                         "Spearman correlation p-values pooled across sessions (phase-level means)."))
            fig_num += 1

    # Combined phase-level correlation (pooled over sessions)
    corr_fig = None
    if (batch_out / "combined_phase_corr_spearman.csv").exists():
        corr = _read_csv_safely(batch_out / "combined_phase_corr_spearman.csv", index_col=0)
        if len(corr.columns) and len(corr.index):
            corr_fig = build_corr_heatmap("Combined cohort — Phase-level Spearman correlation (pooled)", corr, colorscale=POOLED_CORR_COLORSCALE, reversescale=POOLED_CORR_REVERSE)
    elif phases_all:
        ph_all = pd.concat(phases_all, ignore_index=True)
        if "valid_only" in ph_all.columns:
            ph_all = ph_all.loc[pd.to_numeric(ph_all["valid_only"], errors="coerce").fillna(0).astype(int) == 1].copy()
        metric_cols = [c for c in ph_all.columns if c.endswith("__mean")]
        if len(metric_cols) >= 3 and len(ph_all) >= 3:
            M = ph_all[metric_cols].apply(pd.to_numeric, errors="coerce")
            corr = _spearman_corr(M)
            corr_fig = build_corr_heatmap("Combined cohort — Phase-level Spearman correlation (pooled)", corr, colorscale=POOLED_CORR_COLORSCALE, reversescale=POOLED_CORR_REVERSE)

    if corr_fig is not None:
        figs.append(("fig_combined_corr", _pretty_figure_title(fig_num, "combined_corr", kind="combined_corr"), corr_fig, "Spearman correlation on phase-level means pooled across all sessions (valid minutes only)."))
        fig_num += 1

    out_html = batch_out / "all_sessions" / "all_sessions_interactive_report.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    if out_html.exists() and not overwrite:
        return out_html

    # Export static (SVG/PDF) copies for publication-grade downloads.
    for anchor, _title, fig, _desc in figs:
        _export_static_assets(fig, out_html.parent, stem=anchor, enabled=export_static)

    parts = [_html_header("Wearable Physiology – All Sessions Report")]
    parts.append("<h1>Wearable Physiology – All Sessions Report</h1>")
    parts.append(f"<p class='sub'>sessions={n_sessions} • minute-rows={n_rows:,} • QC-passed minutes={valid_rows:,} ({valid_pct:.1f}%)</p>")
    parts.append("<div class='card summary'>")
    parts.append("<h2 style='margin:0 0 8px 0;font-size:22px'>Cohort summary</h2>")
    parts.append("<div class='meta'>This report aggregates all session outputs into cohort-level envelope plots aligned by protocol minute. Use the legend to reveal individual session traces (hidden by default). The correlation heatmap summarizes cross-modal coupling at the phase level pooled across sessions.</div>")
    
    # Hyperparameters (pulled from all_sessions JSON artifacts when available)
    hp = _collect_pipeline_hyperparameters(all_sessions_dir)
    if hp:
        parts.append("<div class='meta'><b>Hyperparameters</b></div>")
        parts.append("<div class='meta'><ul style='margin:6px 0 0 18px;padding:0;line-height:1.45'>")
        for k in sorted(hp.keys()):
            v = _safe_text(hp[k])
            parts.append(f"<li><code>{k}</code>: {v}</li>")
        parts.append("</ul></div>")

    parts.append("<div class='kpi'>")
    parts.append(f"<div class='box'><div class='big'>{n_sessions}</div><div class='lab'>sessions included</div></div>")
    parts.append(f"<div class='box'><div class='big'>{n_rows:,}</div><div class='lab'>minute-rows pooled</div></div>")
    parts.append(f"<div class='box'><div class='big'>{valid_rows:,} ({valid_pct:.1f}%)</div><div class='lab'>QC-passed minutes (valid_joint_minute)</div></div>")
    parts.append("</div></div>")

    parts.append("<div class='toc'><b>Contents</b><ul>")
    for anchor, title, _fig, _desc in figs:
        parts.append(f"<li><a href='#{anchor}'>{title}</a></li>")
    parts.append("</ul></div><hr/>")
    parts.append("<p class='meta'><b>Controls:</b> hover for details, drag to zoom, double-click to reset. Click legend items to hide/show traces. Export via the camera icon in the figure toolbar.</p>")

    for i, (anchor, title, fig, desc) in enumerate(figs, start=1):
        parts.append(_figure_card_html(anchor, title, desc, fig))

    parts.append("</body></html>")
    out_html.write_text("\n".join(parts), encoding="utf-8")
    return out_html

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build interactive Plotly HTML reports from empatica_batch_pipeline outputs.")
    ap.add_argument("--batch-out", required=True, type=Path, help="Output directory produced by empatica_batch_pipeline_v14.py")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing html outputs.")
    ap.add_argument("--build-combined", action="store_true", help="Build cohort-level combined interactive report in --batch-out/all_sessions.")
    ap.add_argument("--export-static", action="store_true", default=EXPORT_STATIC_DEFAULT, help="Also export each figure as SVG+PDF via Kaleido (no empty folders; skipped if Kaleido unavailable).")
    args = ap.parse_args()

    batch_out: Path = args.batch_out
    batch_out.mkdir(parents=True, exist_ok=True)

    session_dirs = _find_session_dirs(batch_out)
    if not session_dirs:
        raise SystemExit(f"No sessions found under {batch_out}. Expected subfolders containing features_minute.csv")

    for sd in session_dirs:
        out = build_one_session(sd, overwrite=args.overwrite, export_static=args.export_static)
        print(f"[OK] wrote {out}")

    if args.build_combined:
        outc = build_combined(batch_out, overwrite=args.overwrite, export_static=args.export_static)
        print(f"[OK] wrote {outc}")

if __name__ == "__main__":
    main()
