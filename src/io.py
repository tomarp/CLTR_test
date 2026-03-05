import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

def read_csv_safely(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)

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
            f"# Auto-generated\n{prefix.upper()} = " + json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )

def _read_square_csv_matrix(path: Path):
    df = pd.read_csv(path, index_col=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def find_session_dirs(sessions_root: Path) -> List[Path]:
    out = []
    for p in sorted(sessions_root.iterdir()):
        if p.is_dir() and ((p / "eda.csv").exists() or (p / "features_minute.csv").exists()):
            out.append(p)
    return out

def _collect_pipeline_hyperparameters(all_sessions_dir: Path) -> Dict[str, str]:
    candidates = [
        all_sessions_dir / "qc_summary.json",
        all_sessions_dir / "qc_summary_all_sessions.json",
        all_sessions_dir / "report_meta.json",
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
    if isinstance(meta, dict):
        for key in ["tunable_parameters", "hyperparameters", "params", "config"]:
            if key in meta and isinstance(meta[key], dict):
                for kk, vv in meta[key].items():
                    hp[kk] = str(vv)
    return hp

_read_csv_safely = read_csv_safely
