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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from .processing import (
    ensure_datetime_utc, empirical_sampling_rate, eda_process, bvp_process,
    temp_process, acc_process, steps_process
)
from .analysis import (
    detect_gaps, summarize_signal, compute_phase_mean_matrices,
    estimate_block_length_minutes, meta_analyze_corr_matrices,
    spearman_corr_with_pvalues
)
from .io import (
    read_csv_safely, schema_from_dataframe, write_output_schema, find_session_dirs
)
from .pdf_report import (
    plot_summary_page, plot_timeseries_qcaware, plot_phase_distributions,
    plot_phase_delta_from_baseline, plot_block_summary_table, plot_phase_correlation,
    plot_phase_correlation_pvalues, plot_corr_heatmap, plot_cohort_phase_distributions,
    plot_cohort_phase_delta_across_sessions, write_text_page
)
from .utils import is_all_sessions_id, format_plot_title

# Constants
HIGH_MOTION_ENMO_P95_THRESHOLD_G = 0.30
GAP_DETECTION_THRESHOLD_SECONDS = 2.0
PHASE_CORR_METRICS = ["hr_bpm_mean", "eda_tonic_mean", "eda_phasic_p95", "temp_smooth_C_mean", "acc_enmo_g_p95", "steps_count", "scr_count"]
BOOTSTRAP_N = 300
BOOTSTRAP_BLOCK_MINUTES = "auto"
BOOTSTRAP_BLOCK_MINUTES_MIN = 5
BOOTSTRAP_BLOCK_MINUTES_MAX = 30
BOOTSTRAP_SEED = 42
PARTIAL_CONTROL_VAR = "acc_enmo_g_p95"

PLAUSIBILITY_RANGES = {
    "hr_bpm_mean": {"min": 35.0, "max": 180.0, "units": "bpm"},
    "hr_bpm_std": {"min": 0.0, "max": 40.0, "units": "bpm"},
    "eda_tonic_mean": {"min": 0.0, "max": 30.0, "units": "uS"},
    "eda_phasic_mean": {"min": -5.0, "max": 15.0, "units": "uS"},
    "eda_phasic_p95": {"min": 0.0, "max": 20.0, "units": "uS"},
    "scr_count": {"min": 0.0, "max": 60.0, "units": "count/min"},
    "temp_smooth_C_mean": {"min": 15.0, "max": 42.0, "units": "C"},
    "temp_slope_Cps_mean": {"min": -0.05, "max": 0.05, "units": "C/s"},
    "acc_enmo_g_mean": {"min": 0.0, "max": 3.0, "units": "g"},
    "acc_enmo_g_p95": {"min": 0.0, "max": 6.0, "units": "g"},
    "acc_activity_mean": {"min": 0.0, "max": 5.0, "units": "a.u."},
    "steps_count": {"min": 0.0, "max": 250.0, "units": "steps/min"},
}

def minute_floor_utc(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, utc=True).dt.floor("min")

def load_timeline(timeline_csv: Path, *, timeline_tz: str) -> pd.DataFrame:
    tl = pd.read_csv(timeline_csv)
    colmap = {c.lower().strip(): c for c in tl.columns}
    def require(aliases: List[str]) -> str:
        for a in aliases:
            if a in colmap: return colmap[a]
        raise ValueError(f"timeline missing one of {aliases}")

    c_datetime = require(["datetime", "minute_utc", "datetime_utc"])
    c_session = require(["session id", "session_id"])
    c_part = require(["participant id", "participant_id", "participant"])
    c_minute = require(["minute index", "minute_index", "protocol_minute"])
    c_block = require(["protocol block", "protocol_block"])
    c_phase = require(["protocol phase", "protocol_phase"])

    out = pd.DataFrame({
        "datetime_local": tl[c_datetime],
        "session_id": tl[c_session].astype(str),
        "participant": tl[c_part].astype(str),
        "minute_index": tl[c_minute],
        "protocol_block": tl[c_block],
        "protocol_phase": tl[c_phase],
    })
    dt_raw = pd.to_datetime(out["datetime_local"], errors="coerce")
    if getattr(dt_raw.dt, "tz", None) is None:
        dt_utc = dt_raw.dt.tz_localize(timeline_tz).dt.tz_convert("UTC")
    else:
        dt_utc = dt_raw.dt.tz_convert("UTC")
    out["minute_utc"] = dt_utc.dt.floor("min")
    return out

def group_minute(df: pd.DataFrame) -> pd.core.groupby.generic.DataFrameGroupBy:
    df = df.copy()
    df["minute_utc"] = minute_floor_utc(df["datetime_utc"])
    return df.groupby("minute_utc")

def minute_agg_stats(g: pd.core.groupby.generic.DataFrameGroupBy, col: str) -> pd.DataFrame:
    x = pd.to_numeric(g[col], errors="coerce")
    return pd.DataFrame({
        f"{col}_mean": x.mean(),
        f"{col}_std": x.std(ddof=1),
        f"{col}_p95": x.quantile(0.95),
        f"{col}_iqr": x.apply(lambda s: np.subtract(*np.nanpercentile(s.to_numpy(dtype=float), [75, 25])) if s.notna().sum() >= 5 else np.nan),
    })

def build_minute_features(
    session_id: str, timeline_session: pd.DataFrame,
    eda_df: pd.DataFrame, bvp_df: pd.DataFrame,
    temp_df: pd.DataFrame, acc_df: pd.DataFrame, steps_df: pd.DataFrame
) -> pd.DataFrame:
    tl = timeline_session.copy().sort_values("minute_utc")
    eda_g = group_minute(eda_df)
    eda_tonic = minute_agg_stats(eda_g, "eda_tonic")
    eda_phasic = minute_agg_stats(eda_g, "eda_phasic")
    scr_count = eda_g["scr_peaks"].sum().rename("scr_count").to_frame()
    bvp_g = group_minute(bvp_df)
    hr = minute_agg_stats(bvp_g, "hr_bpm")
    temp_g = group_minute(temp_df)
    temp_s = minute_agg_stats(temp_g, "temp_smooth_C")
    temp_sl = minute_agg_stats(temp_g, "temp_slope_Cps")
    acc_g = group_minute(acc_df)
    acc_act = minute_agg_stats(acc_g, "acc_activity")
    enmo = minute_agg_stats(acc_g, "acc_enmo_g")
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
    feat["scr_rate_per_min"] = feat["scr_count"]
    feat["steps_rate_per_min"] = feat["steps_count"]
    for base in ["eda_tonic_mean", "hr_bpm_mean", "temp_smooth_C_mean", "acc_activity_mean", "steps_count"]:
        feat[f"has_{base.split('_')[0] if base!='steps_count' else 'steps'}"] = feat[base].notna().astype(int)
    feat["high_motion"] = (feat["acc_enmo_g_p95"] > HIGH_MOTION_ENMO_P95_THRESHOLD_G).astype(int)
    return feat

def add_oob_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, meta in PLAUSIBILITY_RANGES.items():
        if col not in df.columns: continue
        lo, hi = meta["min"], meta["max"]
        df[f"{col}_oob"] = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
        df[f"{col}_oob"] = df[f"{col}_oob"].astype(int)
    return df

def add_validity_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = add_oob_flags(df)
    df["valid_hr_minute"] = (df["hr_bpm_mean"].notna() & (df.get("hr_bpm_mean_oob", 0) == 0)).astype(int)
    df["valid_eda_minute"] = (df["eda_tonic_mean"].notna() & (df.get("eda_tonic_mean_oob", 0) == 0)).astype(int)
    df["valid_temp_minute"] = (df["temp_smooth_C_mean"].notna() & (df.get("temp_smooth_C_mean_oob", 0) == 0)).astype(int)
    df["valid_acc_minute"] = (df["acc_enmo_g_mean"].notna() & (df.get("acc_enmo_g_mean_oob", 0) == 0)).astype(int)
    df["valid_joint_minute"] = ((df["valid_hr_minute"] == 1) & (df["valid_eda_minute"] == 1) &
                                (df["valid_temp_minute"] == 1) & (df["valid_acc_minute"] == 1) &
                                (df["high_motion"] == 0)).astype(int)
    return df

def phase_aggregate(df: pd.DataFrame, *, valid_only: bool) -> pd.DataFrame:
    d = df.copy()
    if valid_only: d = d.loc[d["valid_joint_minute"] == 1].copy()
    group_cols = ["session_id", "participant", "protocol_block", "protocol_phase"]
    metric_cols = [c for c in d.columns if c.endswith("_mean") or c in ["scr_count", "scr_rate_per_min", "steps_count", "steps_rate_per_min"]]
    aggs = {c: ["mean", "std", "median"] for c in metric_cols if c in d.columns}
    out = d.groupby(group_cols).agg(aggs)
    out.columns = [f"{a}__{b}" for a, b in out.columns]
    out = out.reset_index()
    cov = d.groupby(group_cols)["minute_utc"].count().rename("n_minutes").reset_index()
    out = out.merge(cov, on=group_cols, how="left")
    out["valid_only"] = int(valid_only)
    return out

def aggregate_phase_qcaware(df: pd.DataFrame) -> pd.DataFrame:
    allp = phase_aggregate(df, valid_only=False)
    valp = phase_aggregate(df, valid_only=True)
    return pd.concat([allp, valp], ignore_index=True)

def process_one_session(session_id: str, session_dir: Path, timeline_session: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = ensure_datetime_utc(read_csv_safely(session_dir / "accelerometer.csv"), ["unix_timestamp_us"])
    eda = ensure_datetime_utc(read_csv_safely(session_dir / "eda.csv"), ["unix_timestamp_us"])
    bvp = ensure_datetime_utc(read_csv_safely(session_dir / "bvp.csv"), ["unix_timestamp_us"])
    temp = ensure_datetime_utc(read_csv_safely(session_dir / "temperature.csv"), ["unix_timestamp_us"])
    steps = ensure_datetime_utc(read_csv_safely(session_dir / "steps.csv"), ["unix_timestamp_us"])
    acc_p, _ = acc_process(acc); eda_p, _ = eda_process(eda); bvp_p, _ = bvp_process(bvp)
    temp_p, _ = temp_process(temp); steps_p = steps_process(steps)
    minute_feat = build_minute_features(session_id, timeline_session, eda_p, bvp_p, temp_p, acc_p, steps_p)
    minute_feat = add_validity_flags(minute_feat); phase_feat = aggregate_phase_qcaware(minute_feat)
    minute_feat.to_csv(out_dir / "features_minute.csv", index=False)
    phase_feat.to_csv(out_dir / "features_phase.csv", index=False)
    qc = {"session_id": session_id, "participant": str(timeline_session["participant"].iloc[0]),
          "minutes_total": len(minute_feat), "minutes_valid_joint": int(minute_feat["valid_joint_minute"].sum()),
          "plausibility_ranges": PLAUSIBILITY_RANGES}
    with PdfPages(out_dir / f"{session_id}_report.pdf") as pdf:
        plot_summary_page(pdf, qc, session_id)
        for ycol, ylabel, vcol, title in [
            ("eda_tonic_mean","EDA tonic (µS)","valid_eda_minute","EDA tonic (minute mean)"),
            ("hr_bpm_mean","Heart rate (bpm)","valid_hr_minute","Heart rate (minute mean)"),
            ("temp_smooth_C_mean","Skin temperature (°C)","valid_temp_minute","Skin temperature (minute mean)"),
            ("acc_enmo_g_p95","ENMO p95 (g)","valid_acc_minute","Acceleration ENMO (minute p95)"),
        ]:
            if ycol in minute_feat.columns:
                plot_timeseries_qcaware(pdf, minute_feat, session_id=session_id, ycol=ycol, ylabel=ylabel, valid_col=vcol, title=title)
        for metric, ylabel in [("hr_bpm_mean", "Heart rate (bpm)"), ("eda_tonic_mean", "EDA tonic (µS)")]:
            if metric in minute_feat.columns:
                plot_phase_distributions(pdf, minute_feat, session_id=session_id, metric=metric, ylabel=ylabel)
                plot_phase_delta_from_baseline(pdf, minute_feat, session_id=session_id, metric=metric, ylabel=ylabel)
        plot_block_summary_table(pdf, minute_feat, session_id=session_id)
        plot_phase_correlation(pdf, minute_feat, session_id=session_id, metrics=PHASE_CORR_METRICS)
    return qc

def generate_combined_results(out_root: Path, sessions: List[Tuple[str, Path]], timeline: pd.DataFrame, tz_name: str):
    combined_dir = out_root / "all_sessions"
    combined_dir.mkdir(parents=True, exist_ok=True)
    dfs = []
    for sid, _ in sessions:
        f = out_root / sid / "features_minute.csv"
        if f.exists(): dfs.append(pd.read_csv(f))
    if not dfs: return
    all_min = pd.concat(dfs, ignore_index=True)
    all_min.to_csv(combined_dir / "all_sessions_features_minute_long.csv", index=False)
    with PdfPages(combined_dir / "all_sessions_report.pdf") as pdf:
        write_text_page(pdf, "All Sessions Overview", [f"Sessions: {len(sessions)}", f"Total Minutes: {len(all_min)}"])
        for col, lab in [("hr_bpm_mean", "Heart rate (bpm)"), ("eda_tonic_mean", "EDA tonic (µS)")]:
            if col in all_min.columns:
                plot_cohort_phase_distributions(pdf, all_min, "Wearable Physiology – All sessions", col, lab)

def run_batch(sessions_root: Path, outdir: Path, timeline_csv: Path, timeline_tz: str):
    timeline = load_timeline(timeline_csv, timeline_tz=timeline_tz)
    outdir.mkdir(parents=True, exist_ok=True)
    session_dirs = find_session_dirs(sessions_root)
    sessions = []
    for sdir in session_dirs:
        sid = sdir.name
        tl_s = timeline.loc[timeline["session_id"].astype(str) == sid].copy()
        if tl_s.empty: continue
        process_one_session(sid, sdir, tl_s, outdir / sid)
        sessions.append((sid, sdir))
    if len(sessions) > 1:
        generate_combined_results(outdir, sessions, timeline, timeline_tz)
