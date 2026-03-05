import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyArrowPatch
import pandas as pd
import numpy as np
import math
from typing import List, Tuple, Optional, Any, Union, Dict
from .utils import format_plot_title, _wrap_block, is_all_sessions_id
from .analysis import compute_cohort_phase_deltas_across_sessions

PLOT_FONTS = {'TITLE': 13, 'AXIS_LABEL': 11, 'TICK': 10, 'LEGEND': 9, 'ANNOT': 9}
PRIMARY_BLUE = "#0072B2"
SECONDARY_ORANGE = "#D55E00"
CORR_FIGSIZE = (11, 8)
CORR_CELL_FONTSIZE = 8

def _prep_ax(ax: plt.Axes, *, session_id: str, title: str, enable_grid: bool = True):
    if enable_grid:
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(axis="both", labelsize=9)
    if str(session_id).startswith("Wearable Physiology –"):
        ax.set_title(f"{session_id} | {title}", fontsize=PLOT_FONTS['TITLE'], pad=8)
    else:
        ax.set_title(f"{session_id} — {title}", fontsize=PLOT_FONTS['TITLE'], pad=10)

def _draw_protocol_ribbon(ax: plt.Axes, df: pd.DataFrame, *, t: Union[pd.Series, np.ndarray], y0: float, y1: float, gap_minutes: float = 1.0):
    if df.empty: return
    x = t.to_numpy() if isinstance(t, pd.Series) else t
    is_dt = isinstance(x[0], (pd.Timestamp, np.datetime64))
    one_step = pd.Timedelta(minutes=1) if is_dt else 1.0
    d = df.sort_values("minute_utc")
    blocks = d["protocol_block"].astype(str).to_numpy()
    segs = []
    start = 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or blocks[i] != blocks[start]:
            segs.append((blocks[start], start, i))
            start = i
    palette = ["#dbeafe", "#fef3c7", "#fde2e4", "#dcfce7", "#ede9fe"]
    for j, (blk, s, e) in enumerate(segs):
        x0, x1 = x[s], x[e - 1] + one_step
        if is_dt:
            x0n, x1n = mdates.date2num(pd.to_datetime(x0)), mdates.date2num(pd.to_datetime(x1))
            rect = Rectangle((x0n, y0), width=(x1n - x0n), height=(y1 - y0), facecolor=palette[j % len(palette)], zorder=0.8)
        else:
            rect = Rectangle((float(x0), y0), width=float(x1) - float(x0), height=(y1 - y0), facecolor=palette[j % len(palette)], zorder=0.8)
        ax.add_patch(rect)
        xc = x0 + (x1 - x0) / 2 if is_dt else (float(x0) + float(x1)) / 2
        ax.text(xc, y0 + 0.55 * (y1 - y0), f"Block {blk}", ha="center", va="center", fontsize=9, color="0.20",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.6), zorder=5)

def plot_timeseries_qcaware(pdf, df, session_id, ycol, ylabel, valid_col, title, extra_hlines=None):
    if df.empty: return
    d = df.sort_values("minute_utc")
    t = pd.to_datetime(d["minute_utc"], utc=True).dt.tz_convert(None)
    y_all = pd.to_numeric(d[ycol], errors="coerce")
    mask_valid = (d[valid_col] == 1) & y_all.notna()
    mask_invalid = (d[valid_col] != 1) & y_all.notna()
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    y_v = y_all[mask_valid]
    lo, hi = (y_v.quantile(0.01), y_v.quantile(0.99)) if not y_v.dropna().empty else (0, 1)
    yr = hi - lo; pad = 0.1 * yr if yr > 0 else 1.0; head = 0.18 * yr if yr > 0 else 0.2
    y_band0 = hi + pad; y_band1 = y_band0 + head
    ax.set_ylim(lo - pad, y_band1)
    ax.axhspan(y_band0, y_band1, color="white", alpha=1.0, zorder=0.6)
    _draw_protocol_ribbon(ax, d, t=t, y0=y_band0, y1=y_band1)
    ax.plot(t, y_all, linewidth=0.5, alpha=0.18, color="0.25", zorder=2)
    ax.plot(t[mask_valid], y_all[mask_valid], linewidth=0.9, color=PRIMARY_BLUE, zorder=3)
    ax.scatter(t[mask_invalid], y_all[mask_invalid], s=18, marker="x", color="0.25", zorder=4)
    ax.set_ylabel(ylabel); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _prep_ax(ax, session_id=session_id, title=title)
    pdf.savefig(fig); plt.close(fig)

def plot_summary_page(pdf, qc, session_id):
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.axis("off")
    lines = [f"Session ID: {qc.get('session_id','')}", f"Participant: {qc.get('participant','')}",
             f"Minutes: total={qc.get('minutes_total',0)} | valid={qc.get('minutes_valid_joint',0)}"]
    ax.text(0.02, 0.98, _wrap_block(lines), va="top", ha="left", fontsize=11)
    ax.set_title(format_plot_title("session", session_id, "Session QC summary"), fontsize=12, pad=4)
    pdf.savefig(fig); plt.close(fig)

def plot_phase_distributions(pdf, df, session_id, metric, ylabel):
    d = df[df["valid_joint_minute"] == 1].copy()
    if d.empty or metric not in d.columns: return
    phases = list(dict.fromkeys(d["protocol_phase"].astype(str).tolist()))
    data = [pd.to_numeric(d[d["protocol_phase"].astype(str) == p][metric], errors="coerce").dropna().to_numpy() for p in phases]
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.violinplot(data, positions=range(1, len(phases)+1), showextrema=False)
    ax.boxplot(data, positions=range(1, len(phases)+1), showfliers=False)
    _prep_ax(ax, session_id=session_id, title=f"Phase distribution: {metric}")
    ax.set_xticks(range(1, len(phases)+1)); ax.set_xticklabels(phases, rotation=30, ha="right")
    pdf.savefig(fig); plt.close(fig)

def plot_phase_delta_from_baseline(pdf, df, session_id, metric, ylabel):
    d = df[df["valid_joint_minute"] == 1].copy()
    if d.empty or metric not in d.columns: return
    phases = list(dict.fromkeys(d["protocol_phase"].astype(str).tolist()))
    base_val = d[d["protocol_phase"].astype(str) == phases[0]][metric].mean()
    deltas = [d[d["protocol_phase"].astype(str) == p][metric].mean() - base_val for p in phases]
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.bar(phases, deltas, color=PRIMARY_BLUE)
    ax.axhline(0, color="black", linewidth=0.8)
    _prep_ax(ax, session_id=session_id, title=f"Phase delta vs baseline: {metric}")
    pdf.savefig(fig); plt.close(fig)

def plot_block_summary_table(pdf, df, session_id):
    v = df[df["valid_joint_minute"] == 1].copy()
    if v.empty: return
    g = v.groupby("protocol_block")["hr_bpm_mean"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300); ax.axis("off")
    ax.table(cellText=g.values, colLabels=g.columns, loc="center")
    _prep_ax(ax, session_id=session_id, title="Block-level summary")
    pdf.savefig(fig); plt.close(fig)

def plot_phase_correlation(pdf, df, session_id, metrics):
    d = df[df["valid_joint_minute"] == 1].copy()
    cols = [c for c in metrics if c in d.columns]
    if len(cols) < 3: return
    corr = d.groupby("protocol_phase")[cols].mean().corr(method="spearman")
    plot_corr_heatmap(pdf, corr, title=format_plot_title("session", session_id, "Phase-level Spearman correlation"))

def plot_phase_correlation_pvalues(pdf, df, session_id, metrics):
    d = df[df["valid_joint_minute"] == 1].copy()
    cols = [c for c in metrics if c in d.columns]
    if len(cols) < 3: return
    from scipy.stats import spearmanr
    g = d.groupby("protocol_phase")[cols].mean()
    pmat = g.corr(method=lambda x, y: spearmanr(x, y)[1])
    plot_corr_heatmap(pdf, pmat, title=format_plot_title("session", session_id, "Phase-level p-values"), vmin=0, vmax=0.1, cmap="viridis_r")

def plot_corr_heatmap(pdf, mat, title, vmin=-1.0, vmax=1.0, cmap="coolwarm"):
    if mat is None or mat.empty: return
    fig, ax = plt.subplots(figsize=CORR_FIGSIZE, dpi=300)
    im = ax.imshow(mat.to_numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax)
    pdf.savefig(fig); plt.close(fig)

def plot_cohort_phase_distributions(pdf, df, session_id, metric, ylabel, phase_order=None):
    d = df[df["valid_joint_minute"] == 1].copy()
    if d.empty or metric not in d.columns: return
    phases = phase_order if phase_order else list(dict.fromkeys(d["protocol_phase"].astype(str).tolist()))
    data = [pd.to_numeric(d[d["protocol_phase"].astype(str) == p][metric], errors="coerce").dropna().to_numpy() for p in phases]
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.violinplot(data, positions=range(1, len(phases)+1), showextrema=False)
    _prep_ax(ax, session_id=session_id, title=f"Cohort Phase distribution: {metric}")
    pdf.savefig(fig); plt.close(fig)

def plot_cohort_phase_delta_across_sessions(pdf, df, session_id, metric, ylabel, baseline_phase="acclimation", phase_order=None):
    deltas = compute_cohort_phase_deltas_across_sessions(df, metric, baseline_phase=baseline_phase)
    if deltas.empty: return
    phases = phase_order if phase_order else list(dict.fromkeys(deltas["protocol_phase"].astype(str).tolist()))
    data = [pd.to_numeric(deltas[deltas["protocol_phase"].astype(str) == p]["delta"], errors="coerce").dropna().to_numpy() for p in phases]
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300)
    ax.violinplot(data, positions=range(1, len(phases)+1), showextrema=False)
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    _prep_ax(ax, session_id=session_id, title=f"Cohort Phase delta: {metric}")
    pdf.savefig(fig); plt.close(fig)

def write_text_page(pdf, title, lines, wrap_width=112, fontsize=11):
    fig, ax = plt.subplots(figsize=(11.0, 6.2), dpi=300); ax.axis("off")
    ax.text(0.02, 0.95, _wrap_block(lines, width=wrap_width), va="top", ha="left", fontsize=fontsize)
    ax.set_title(title, fontsize=12)
    pdf.savefig(fig); plt.close(fig)
