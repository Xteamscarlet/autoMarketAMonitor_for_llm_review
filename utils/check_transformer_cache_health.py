# -*- coding: utf-8 -*-
"""Health check for cached transformer factors.

Usage:
  python utils/check_transformer_cache_health.py
  python utils/check_transformer_cache_health.py --fail-on-collapse
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_settings


@dataclass
class CacheFileStats:
    path: str
    code: str
    rows: int
    last_date: Optional[str]
    last_prob: float
    last_pred_ret: float
    last_unc: float
    prob_mean: float
    prob_std: float
    prob_min: float
    prob_max: float
    pred_ret_mean: float
    pred_ret_std: float
    unc_mean: float
    unc_std: float
    collapse_reasons: List[str]


def _safe_float(x: object) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=np.float64)
    series = pd.to_numeric(df[col], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _extract_cache_df(payload: object) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if isinstance(payload, pd.DataFrame):
        return payload, None
    if not isinstance(payload, dict):
        return None, None
    frame = payload.get("df")
    if isinstance(frame, pd.DataFrame):
        last_date = payload.get("last_date")
        return frame, None if last_date is None else str(last_date)
    return None, None


def _detect_code_from_path(path: str) -> str:
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    # transformer_factors_000001 -> 000001
    if "_" in stem:
        return stem.split("_")[-1]
    return stem


def _compute_file_stats(
    path: str,
    prob_std_floor: float,
    prob_range_floor: float,
    ret_std_floor: float,
) -> Optional[CacheFileStats]:
    try:
        with open(path, "rb") as file:
            payload = pickle.load(file)
    except Exception:
        return None

    df, last_date = _extract_cache_df(payload)
    if df is None or df.empty:
        return None

    prob = _to_numeric_series(df, "transformer_prob")
    if prob.empty:
        return None

    if "transformer_pred_ret_raw" in df.columns:
        pred_ret = _to_numeric_series(df, "transformer_pred_ret_raw")
    else:
        pred_ret = _to_numeric_series(df, "transformer_pred_ret")

    if "transformer_uncertainty" in df.columns:
        unc = _to_numeric_series(df, "transformer_uncertainty")
    elif "transformer_conf" in df.columns:
        conf = _to_numeric_series(df, "transformer_conf")
        unc = 1.0 - conf
    else:
        unc = pd.Series(dtype=np.float64)

    prob_std = _safe_float(prob.std(ddof=0))
    prob_min = _safe_float(prob.min())
    prob_max = _safe_float(prob.max())
    prob_range = prob_max - prob_min if np.isfinite(prob_min) and np.isfinite(prob_max) else float("nan")
    pred_ret_std = _safe_float(pred_ret.std(ddof=0))

    collapse_reasons: List[str] = []
    if np.isfinite(prob_std) and prob_std <= prob_std_floor:
        collapse_reasons.append(f"prob_std<= {prob_std_floor:.4f}")
    if np.isfinite(prob_range) and prob_range <= prob_range_floor:
        collapse_reasons.append(f"prob_range<= {prob_range_floor:.4f}")
    if np.isfinite(pred_ret_std) and pred_ret_std <= ret_std_floor:
        collapse_reasons.append(f"pred_ret_std<= {ret_std_floor:.5f}")

    return CacheFileStats(
        path=path,
        code=_detect_code_from_path(path),
        rows=int(len(df)),
        last_date=last_date,
        last_prob=_safe_float(prob.iloc[-1]),
        last_pred_ret=_safe_float(pred_ret.iloc[-1] if not pred_ret.empty else np.nan),
        last_unc=_safe_float(unc.iloc[-1] if not unc.empty else np.nan),
        prob_mean=_safe_float(prob.mean()),
        prob_std=prob_std,
        prob_min=prob_min,
        prob_max=prob_max,
        pred_ret_mean=_safe_float(pred_ret.mean() if not pred_ret.empty else np.nan),
        pred_ret_std=pred_ret_std,
        unc_mean=_safe_float(unc.mean() if not unc.empty else np.nan),
        unc_std=_safe_float(unc.std(ddof=0) if not unc.empty else np.nan),
        collapse_reasons=collapse_reasons,
    )


def _cross_section_summary(stats: List[CacheFileStats]) -> Dict[str, float]:
    latest_prob = np.asarray([s.last_prob for s in stats], dtype=np.float64)
    latest_ret = np.asarray([s.last_pred_ret for s in stats], dtype=np.float64)
    latest_unc = np.asarray([s.last_unc for s in stats], dtype=np.float64)
    mask_prob = np.isfinite(latest_prob)
    mask_ret = np.isfinite(latest_ret)
    mask_unc = np.isfinite(latest_unc)

    return {
        "files": len(stats),
        "latest_prob_std": float(np.std(latest_prob[mask_prob])) if mask_prob.any() else float("nan"),
        "latest_prob_min": float(np.min(latest_prob[mask_prob])) if mask_prob.any() else float("nan"),
        "latest_prob_max": float(np.max(latest_prob[mask_prob])) if mask_prob.any() else float("nan"),
        "latest_ret_std": float(np.std(latest_ret[mask_ret])) if mask_ret.any() else float("nan"),
        "latest_unc_std": float(np.std(latest_unc[mask_unc])) if mask_unc.any() else float("nan"),
    }


def _print_report(stats: List[CacheFileStats], cross: Dict[str, float], top_n: int) -> None:
    collapsed = [s for s in stats if s.collapse_reasons]
    print("=" * 88)
    print("Transformer Cache Health Report")
    print("=" * 88)
    print(f"files={len(stats)}, collapsed={len(collapsed)}")
    print(
        "cross_section: "
        f"latest_prob_std={cross['latest_prob_std']:.6f}, "
        f"latest_prob_range=({cross['latest_prob_min']:.6f}, {cross['latest_prob_max']:.6f}), "
        f"latest_ret_std={cross['latest_ret_std']:.6f}, latest_unc_std={cross['latest_unc_std']:.6f}"
    )
    print("-" * 88)

    if not collapsed:
        print("No collapsed cache files detected by configured thresholds.")
        return

    collapsed = sorted(collapsed, key=lambda x: (len(x.collapse_reasons), x.prob_std, x.pred_ret_std))
    print(f"Top {min(top_n, len(collapsed))} collapsed files:")
    for item in collapsed[:top_n]:
        reason_text = ", ".join(item.collapse_reasons)
        print(
            f"{item.code}: prob(last/mean/std)={item.last_prob:.6f}/{item.prob_mean:.6f}/{item.prob_std:.6f}, "
            f"pred_ret(last/std)={item.last_pred_ret:+.6f}/{item.pred_ret_std:.6f}, "
            f"unc(last/std)={item.last_unc:.6f}/{item.unc_std:.6f} | {reason_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check health of transformer factor cache files.")
    parser.add_argument("--cache-dir", type=str, default="", help="Cache directory. Default uses config.paths.cache_dir.")
    parser.add_argument("--glob", type=str, default="transformer_factors_*.pkl", help="File glob pattern.")
    parser.add_argument("--prob-std-floor", type=float, default=0.01, help="Collapse threshold for per-file prob std.")
    parser.add_argument("--prob-range-floor", type=float, default=0.02, help="Collapse threshold for per-file prob range.")
    parser.add_argument("--ret-std-floor", type=float, default=0.0005, help="Collapse threshold for per-file pred_ret std.")
    parser.add_argument("--top-n", type=int, default=20, help="Print at most top N collapsed files.")
    parser.add_argument("--json-out", type=str, default="", help="Optional output JSON file.")
    parser.add_argument("--fail-on-collapse", action="store_true", help="Return non-zero when collapsed files are detected.")
    args = parser.parse_args()

    cache_dir = args.cache_dir or get_settings().paths.cache_dir
    pattern = os.path.join(cache_dir, args.glob)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No cache files found: {pattern}")
        return 1

    stats: List[CacheFileStats] = []
    for path in files:
        item = _compute_file_stats(
            path=path,
            prob_std_floor=args.prob_std_floor,
            prob_range_floor=args.prob_range_floor,
            ret_std_floor=args.ret_std_floor,
        )
        if item is not None:
            stats.append(item)

    if not stats:
        print("No valid transformer cache payloads found.")
        return 1

    cross = _cross_section_summary(stats)
    _print_report(stats, cross, top_n=max(1, args.top_n))

    collapsed = [s for s in stats if s.collapse_reasons]
    if args.json_out:
        payload = {
            "summary": cross,
            "collapsed_count": len(collapsed),
            "files": [asdict(s) for s in stats],
        }
        with open(args.json_out, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print(f"JSON report saved to: {args.json_out}")

    if args.fail_on_collapse and collapsed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
