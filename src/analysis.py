import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import List, Tuple, Dict, Any

def spearman_corr_with_pvalues(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(df.columns)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    pval = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for i, ci in enumerate(cols):
        xi = pd.to_numeric(df[ci], errors="coerce").to_numpy(dtype=float)
        for j, cj in enumerate(cols):
            if j < i:
                corr.iat[i, j] = corr.iat[j, i]
                pval.iat[i, j] = pval.iat[j, i]
                continue
            xj = pd.to_numeric(df[cj], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 3:
                continue
            rho, pv = spearmanr(xi[mask], xj[mask])
            try:
                rho, pv = float(rho), float(pv)
            except Exception:
                rho, pv = np.nan, np.nan
            corr.iat[i, j] = corr.iat[j, i] = rho
            pval.iat[i, j] = pval.iat[j, i] = pv
    np.fill_diagonal(pval.values, 0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr, pval

def detect_gaps(df: pd.DataFrame, fs_hz: float, *, gap_s: float = 2.0) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    if not np.isfinite(fs_hz) or fs_hz <= 0 or "datetime_utc" not in df.columns or len(df) < 3:
        return []
    t = df["datetime_utc"].sort_values().to_numpy()
    dt = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
    gaps = np.where(dt > gap_s)[0]
    return [(pd.Timestamp(t[i]), pd.Timestamp(t[i+1]), float(dt[i])) for i in gaps]

def robust_iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 5: return float("nan")
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 - q1)

def summarize_signal(x: np.ndarray) -> Dict[str, float]:
    x = x.astype(float)
    x_f = x[np.isfinite(x)]
    if x_f.size == 0:
        return {"n": float(x.size), "n_finite": 0.0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "p05": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": float(x.size), "n_finite": float(x_f.size),
        "mean": float(np.mean(x_f)),
        "std": float(np.std(x_f, ddof=1)) if x_f.size > 1 else 0.0,
        "min": float(np.min(x_f)), "p05": float(np.percentile(x_f, 5)),
        "median": float(np.median(x_f)), "p95": float(np.percentile(x_f, 95)), "max": float(np.max(x_f)),
    }

def compute_phase_mean_matrices(
    df_minute: pd.DataFrame,
    metrics: List[str],
    *,
    valid_only: bool = True,
    clean_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df_minute.copy()
    if valid_only and ("valid_joint_minute" in d.columns):
        d = d.loc[d["valid_joint_minute"] == 1].copy()
    cols = [c for c in metrics if c in d.columns]
    if (not cols) or ("protocol_phase" not in d.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    g = d.groupby("protocol_phase")[cols].mean(numeric_only=True)
    if clean_labels:
        g = g.rename(columns={c: c.replace("_mean", "").strip("_") for c in g.columns})
    corr, pval = spearman_corr_with_pvalues(g)
    return g, corr, pval

def compute_cohort_phase_deltas_across_sessions(
    df: pd.DataFrame,
    metric: str,
    *,
    baseline_phase: str = "acclimation",
) -> pd.DataFrame:
    d = df.copy()
    if "valid_joint_minute" in d.columns:
        d = d.loc[d["valid_joint_minute"] == 1].copy()
    if d.empty or metric not in d.columns or "protocol_phase" not in d.columns or "session_id" not in d.columns:
        return pd.DataFrame()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])
    out_rows = []
    for sid, ds in d.groupby("session_id"):
        base = ds.loc[ds["protocol_phase"].astype(str) == str(baseline_phase), metric]
        if base.dropna().empty: continue
        base_mean = float(np.nanmean(base.to_numpy(dtype=float)))
        for ph, dph in ds.groupby(ds["protocol_phase"].astype(str)):
            ph_mean = float(np.nanmean(dph[metric].to_numpy(dtype=float)))
            out_rows.append({"session_id": str(sid), "protocol_phase": str(ph), "delta": ph_mean - base_mean})
    return pd.DataFrame(out_rows)

def estimate_iat_minutes(x: np.ndarray, *, max_lag: int = 60) -> float:
    x = x[np.isfinite(x)]
    if x.size < 8: return float("nan")
    x = x - np.mean(x)
    var = float(np.dot(x, x))
    if var <= 1e-12: return 1.0
    r = np.correlate(x, x, mode="full")[x.size - 1 : x.size + max_lag]
    acf = r / var
    s = 0.0
    for k in range(1, len(acf)):
        if acf[k] <= 0: break
        s += float(acf[k])
    return float(max(1.0, 1.0 + 2.0 * s))

def estimate_block_length_minutes(df_minute: pd.DataFrame, metrics: List[str], min_len: int = 5, max_len: int = 30) -> int:
    taus = []
    for m in [m for m in metrics if m in df_minute.columns]:
        x = pd.to_numeric(df_minute[m], errors="coerce").to_numpy(dtype=float)
        taus.append(estimate_iat_minutes(x))
    if not taus: return min_len
    tau_ref = float(np.nanmax(taus))
    return int(np.clip(int(np.ceil(2.0 * tau_ref)), min_len, max_len))

def meta_analyze_corr_matrices(mats: List[pd.DataFrame], weights: List[float] = None) -> pd.DataFrame:
    if not mats: return pd.DataFrame(dtype=float)
    idx, cols = mats[0].index, mats[0].columns
    if weights is None: weights = [1.0] * len(mats)
    Z, W = np.zeros((len(idx), len(cols))), np.zeros((len(idx), len(cols)))
    for mat, w in zip(mats, weights):
        a = mat.reindex(index=idx, columns=cols).to_numpy()
        z = np.arctanh(np.clip(a, -0.999999, 0.999999))
        mask = np.isfinite(z)
        Z[mask] += w * z[mask]; W[mask] += w
    with np.errstate(invalid="ignore", divide="ignore"): R = np.tanh(Z / W)
    return pd.DataFrame(R, index=idx, columns=cols)
