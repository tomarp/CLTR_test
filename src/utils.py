import re
from typing import List

# Shared constants or helpers can go here.

def format_plot_title(scope: str, session_id: str, fig_title: str) -> str:
    """One-line publication title used across Matplotlib.
    scope: 'session' or 'all'
    """
    ft = str(fig_title).strip()
    for prefix in ("Wearable Physiology –", "Wearable Physiology -", "Wearable Physiology"):
        if ft.startswith(prefix):
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

def is_all_sessions_id(session_id: str) -> bool:
    s = str(session_id or "").strip().lower()
    if not s:
        return True
    s = s.replace("–", "-").replace("—", "-")
    return (
        s.startswith("all")
        or "all sessions" in s
        or "all_sessions" in s
        or "combined" in s
        or "cohort" in s
        or "wearable physiology - all sessions" in s
        or "wearable physiology - all" in s
    )

def _wrap_block(lines: List[str], *, width: int = 110) -> str:
    import textwrap
    out: List[str] = []
    for ln in lines:
        if not ln:
            out.append("")
            continue
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

def _safe_text(x) -> str:
    import math
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)

def _pretty_feature_title(col: str) -> str:
    if not col:
        return "Physiological Feature"
    raw = str(col).strip()
    toks = [t for t in re.split(r"[_\-\s]+", raw) if t]
    MOD = {
        "hr": "Heart Rate", "bpm": "Heart Rate", "eda": "Electrodermal Activity (EDA)",
        "gsr": "Electrodermal Activity (EDA)", "temp": "Skin Temperature",
        "skintemp": "Skin Temperature", "temperature": "Skin Temperature",
        "bvp": "Blood Volume Pulse (BVP)", "ppg": "Blood Volume Pulse (BVP)",
        "acc": "Acceleration", "accel": "Acceleration", "ibi": "Inter‑beat Interval (IBI)",
        "rr": "Inter‑beat Interval (IBI)",
    }
    STAT = {
        "mean": "Mean", "avg": "Mean", "median": "Median", "std": "Standard deviation",
        "sd": "Standard deviation", "var": "Variance", "min": "Minimum", "max": "Maximum",
        "iqr": "Interquartile range", "mad": "Median absolute deviation",
    }
    UNIT = {
        "us": "µS", "µs": "µS", "degc": "°C", "c": "°C", "bpm": "bpm", "ms": "ms",
    }
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
            if m: stat = f"{int(m.group(1))}th percentile"
    if feature is None:
        feature = raw.replace("_", " ").replace("-", " ").strip().title()
    if stat and unit: return f"{feature} — {stat} ({unit})"
    if stat: return f"{feature} — {stat}"
    if unit: return f"{feature} ({unit})"
    return feature

def _pretty_figure_title(fig_num: int, col_or_name: str, kind: str = "feature") -> str:
    if kind == "corr": return f"Figure {fig_num}: Phase‑level Spearman correlation (ρ) of phase means"
    if kind == "corr_p": return f"Figure {fig_num}: Phase‑level Spearman correlation p-values of phase means"
    if kind == "combined_corr": return f"Figure {fig_num}: Pooled phase‑level Spearman correlation (ρ) across sessions"
    if kind == "combined_p": return f"Figure {fig_num}: Pooled phase‑level Spearman correlation p-values across sessions"
    if kind == "delta_summary": return f"Figure {fig_num}: Combined phase delta summary (baseline-referenced phase means)"
    if kind == "dist": return f"Figure {fig_num}: {_pretty_feature_title(col_or_name)}"
    return f"Figure {fig_num}: {_pretty_feature_title(col_or_name)}"
