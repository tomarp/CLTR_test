import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from .utils import (
    _pretty_feature_title, _pretty_figure_title, _safe_text,
    is_all_sessions_id
)
from .io import (
    read_csv_safely, find_session_dirs,
    _read_square_csv_matrix
)

# Configuration
PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.3.0.min.js"
C_PRIMARY = "#0072B2"
C_ORANGE  = "#D55E00"
C_LIGHT   = "rgba(0,0,0,0.18)"
BLOCK_COLORS = ["rgba(0,0,0,0.06)", "rgba(0,0,0,0.10)"]
PHASE_GAP_COLOR = "rgba(255,255,255,1.0)"

FONT_FAMILY = "Arial"
TICK_SIZE = 16
AXIS_TITLE_SIZE = 18
HOVER_LABEL_SIZE = 14
HEATMAP_TICK_SIZE = 12
HEATMAP_TICK_ANGLE = -30
HEATMAP_WIDTH = 980
HEATMAP_HEIGHT = 760

DEFAULT_METRICS = [
    ("hr_bpm_mean", "Heart rate (bpm)", "bpm"),
    ("eda_tonic_mean", "EDA tonic (µS)", "µS"),
    ("eda_phasic_p95", "EDA phasic p95 (µS)", "µS"),
    ("temp_smooth_C_mean", "Skin temperature (°C)", "°C"),
    ("acc_enmo_g_p95", "ENMO p95 (g)", "g"),
    ("steps_count", "Steps (count/min)", "count/min"),
    ("scr_count", "SCR count (peaks/min)", "count/min"),
]

def _pub_label(name: str) -> str:
    return _pretty_feature_title(name).replace(" Mean", "")

def _dedupe_square_matrix(mat: pd.DataFrame, label_fn):
    if mat is None or mat.empty: return mat, []
    cols = [str(c) for c in mat.columns]
    idx = [str(i) for i in mat.index]
    common = [c for c in cols if c in idx]
    if len(common) >= 2 and (len(common) != len(cols) or len(common) != len(idx)):
        mat = mat.loc[common, common]
        cols = common
    disp = [label_fn(c) for c in cols]
    keep_pos, seen = [], set()
    for j, d in enumerate(disp):
        if d not in seen:
            keep_pos.append(j); seen.add(d)
    if len(keep_pos) == len(cols): return mat, disp
    kept_cols = [cols[j] for j in keep_pos]
    return mat.loc[kept_cols, kept_cols], [disp[j] for j in keep_pos]

def _plotly_base_layout(title: str, subtitle: Optional[str] = None) -> Dict:
    return dict(
        font=dict(family=FONT_FAMILY, size=TICK_SIZE, color="black"),
        margin=dict(l=55, r=30, t=28, b=45),
        paper_bgcolor="white", plot_bgcolor="white",
        hovermode="closest", showlegend=False,
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.85)", font=dict(size=HOVER_LABEL_SIZE))
    )

def _html_header(title: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'/><title>{title}</title>" \
           f"<script src='{PLOTLY_CDN}'></script>" \
           f"<style>body {{ font-family: sans-serif; margin: 28px; }} .card {{ border: 1px solid #ddd; padding: 16px; margin-bottom: 24px; }}</style></head><body>"

def _plot_div(fig: go.Figure, div_id: str) -> str:
    fig.update_layout(showlegend=False)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)

def build_timeseries_figure(df: pd.DataFrame, session_id: str, ycol: str, ylabel: str, valid_col: str = "valid_joint_minute") -> go.Figure:
    if df.empty or ycol not in df.columns: return go.Figure()
    x = df["minute_index"] if "minute_index" in df.columns else np.arange(len(df))
    y = pd.to_numeric(df[ycol], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="all minutes", line=dict(color=C_LIGHT)))
    if valid_col in df.columns:
        v = pd.to_numeric(df[valid_col], errors="coerce").fillna(0).astype(int)
        y_v = y.copy(); y_v[v != 1] = np.nan
        fig.add_trace(go.Scatter(x=x, y=y_v, mode="lines", name="QC-passed", line=dict(color=C_PRIMARY)))
    fig.update_layout(_plotly_base_layout(f"{session_id} — {ylabel}"))
    return fig

def build_envelope_figure(df_all: pd.DataFrame, ycol: str, ylabel: str) -> go.Figure:
    d = df_all.copy()
    xcol = "minute_index"
    if xcol not in d.columns: d[xcol] = np.arange(len(d))
    piv = d.pivot_table(index=xcol, columns="session_id", values=ycol, aggfunc="mean")
    x = piv.index.to_numpy()
    q10, q90, med = piv.quantile(0.10, axis=1), piv.quantile(0.90, axis=1), piv.median(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=q90.to_numpy(), mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=q10.to_numpy(), mode="lines", fill="tonexty", fillcolor="rgba(0,114,178,0.18)", line=dict(width=0), name="10-90% band"))
    fig.add_trace(go.Scatter(x=x, y=med.to_numpy(), mode="lines", line=dict(color=C_PRIMARY, width=3), name="median"))
    fig.update_layout(_plotly_base_layout(f"Combined — {ylabel}"))
    return fig

def build_one_session_html(session_dir: Path, output_path: Path):
    minute = pd.read_csv(session_dir / "features_minute.csv")
    session_id = session_dir.name
    parts = [_html_header(f"Report - {session_id}"), f"<h1>Session {session_id}</h1>"]
    for col, lab, _ in DEFAULT_METRICS:
        if col in minute.columns:
            fig = build_timeseries_figure(minute, session_id, col, lab)
            parts.append(f"<div class='card'><h2>{lab}</h2>{_plot_div(fig, f'fig_{col}')}</div>")
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")

def build_combined(batch_out: Path, overwrite: bool = True):
    session_dirs = find_session_dirs(batch_out)
    dfs = []
    for sd in session_dirs:
        f = sd / "features_minute.csv"
        if f.exists():
            m = pd.read_csv(f)
            m["session_id"] = sd.name
            dfs.append(m)
    if not dfs: return
    df_all = pd.concat(dfs, ignore_index=True)
    all_sessions_dir = batch_out / "all_sessions"
    all_sessions_dir.mkdir(parents=True, exist_ok=True)
    output_path = all_sessions_dir / "all_sessions_interactive_report.html"
    parts = [_html_header("All Sessions Report"), "<h1>All Sessions Combined Report</h1>"]
    for col, lab, _ in DEFAULT_METRICS:
        if col in df_all.columns:
            fig = build_envelope_figure(df_all, col, lab)
            parts.append(f"<div class='card'><h2>{lab}</h2>{_plot_div(fig, f'fig_combined_{col}')}</div>")
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
