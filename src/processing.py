import numpy as np
import pandas as pd
from scipy import signal
from typing import List, Tuple, Optional, Dict, Any
import datetime

try:
    import neurokit2 as nk
except ImportError:
    nk = None

HAVE_NK = nk is not None

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def ensure_datetime_utc(df: pd.DataFrame, ts_col_candidates: List[str]) -> pd.DataFrame:
    df = df.copy()
    ts_col = pick_first_existing(df, ts_col_candidates + ["unix_timestamp_us", "timestamp", "datetime", "date"])
    if ts_col is None:
        raise ValueError(f"Timestamp column not found. Tried: {ts_col_candidates}")

    df = df.loc[df[ts_col].astype(str) != str(ts_col)].copy()
    s = df[ts_col]

    if not pd.api.types.is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().mean() > 0.95:
            s = s_num

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.dropna().empty:
            raise ValueError("All timestamps are NaN.")
        med = float(np.nanmedian(x))
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

def empirical_sampling_rate(df: pd.DataFrame) -> float:
    if "datetime_utc" not in df.columns or len(df) < 5:
        return float("nan")
    t = df["datetime_utc"].astype("int64").to_numpy(dtype=np.int64) / 1e9
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) < 3:
        return float("nan")
    med = float(np.median(dt))
    return float(1.0 / med) if med > 0 else float("nan")

def eda_process(eda: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not HAVE_NK: raise ImportError("NeuroKit2 required for EDA processing.")
    info: Dict[str, Any] = {"method": "neurokit2"}
    df = eda.copy()
    val_col = pick_first_existing(df, ["eda_uS", "eda", "EDA"])
    if val_col is None:
        raise ValueError("EDA value column not found.")
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
    if not HAVE_NK: raise ImportError("NeuroKit2 required for BVP processing.")
    info: Dict[str, Any] = {"method": "neurokit2"}
    df = bvp.copy()
    val_col = pick_first_existing(df, ["bvp", "BVP", "bvp_nW"])
    if val_col is None:
        raise ValueError("BVP value column not found.")
    fs = empirical_sampling_rate(df)
    info["fs_hz_empirical"] = fs
    x = pd.to_numeric(df[val_col], errors="coerce").to_numpy(dtype=float)
    info["n_missing_raw"] = int(np.isnan(x).sum())

    try:
        clean = nk.ppg_clean(x, sampling_rate=fs)
        signals, _ = nk.ppg_process(clean, sampling_rate=fs)
        df["bvp_clean"] = clean
        df["hr_bpm"] = signals["PPG_Rate"].to_numpy(dtype=float)
        if "PPG_Quality" in signals.columns:
            df["ppg_quality"] = signals["PPG_Quality"].to_numpy(dtype=float)
        info["nk_ok"] = True
        return df, info
    except Exception as e:
        info["nk_ok"] = False
        info["nk_error"] = str(e)
        raise ValueError(f"NeuroKit2 PPG processing failed: {e}")

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

    win = 11 if len(x) >= 11 else (len(x) // 2) * 2 + 1
    if win < 5:
        df["temp_smooth_C"] = x
        df["temp_slope_Cps"] = np.nan
    else:
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
        raise ValueError("Accelerometer columns missing.")
    fs = empirical_sampling_rate(df)
    info["fs_hz_empirical"] = fs
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=float)
    mag = np.sqrt(x*x + y*y + z*z)
    df["acc_mag_g"] = mag
    df["acc_enmo_g"] = np.maximum(0.0, mag - 1.0)

    if np.isfinite(fs) and fs > 1:
        hp = 0.25
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
