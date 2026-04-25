# -*- coding: utf-8 -*-
"""Rebuild transformer factor cache files.

Examples:
  python utils/rebuild_transformer_cache.py --force-refresh
  python utils/rebuild_transformer_cache.py --codes 600519 000333
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from config import STOCK_CODES, get_settings


def _resolve_device(name: str):
    if torch is None:
        return None
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_stocks_data():
    from data import check_and_clean_cache, download_stocks_data, load_pickle_cache

    settings = get_settings()
    if check_and_clean_cache(settings.paths.stock_cache_file):
        cached = load_pickle_cache(settings.paths.stock_cache_file)
        if isinstance(cached, dict) and isinstance(cached.get("stocks_data"), dict):
            return cached["stocks_data"]
    return download_stocks_data(STOCK_CODES)


def _resolve_targets(codes: List[str]) -> List[Tuple[str, str]]:
    if not codes:
        return [(name, code) for name, code in STOCK_CODES.items()]
    code_set = {str(code).zfill(6) for code in codes}
    targets = [(name, code) for name, code in STOCK_CODES.items() if code in code_set]
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild transformer factor caches.")
    parser.add_argument("--codes", nargs="+", default=None, help="Stock codes to rebuild, default all configured stocks.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--force-refresh", action="store_true", help="Delete existing cache file before recompute.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any code fails.")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    try:
        from data import calculate_orthogonal_factors
        from data.cache import get_transformer_cache_path
    except Exception as exc:
        print(f"Cannot import factor pipeline: {exc}")
        print("Hint: install missing deps first (e.g. talib, efinance, akshare).")
        return 1
    stocks_data = _load_stocks_data()
    if not isinstance(stocks_data, dict) or not stocks_data:
        print("No stock data available; cannot rebuild transformer caches.")
        return 1

    targets = _resolve_targets(args.codes or [])
    if not targets:
        print("No matched target codes found in STOCK_CODES.")
        return 1

    success = 0
    skipped = 0
    failed: List[str] = []

    print(f"Rebuilding transformer caches for {len(targets)} stocks (device={device}) ...")
    for name, code in targets:
        df = stocks_data.get(name)
        if df is None or df.empty:
            skipped += 1
            continue

        cache_path = get_transformer_cache_path(code)
        if args.force_refresh and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass

        try:
            calculate_orthogonal_factors(
                df.copy(),
                stock_code=code,
                device=device,
                allow_save_cache=True,
            )
            if os.path.exists(cache_path):
                success += 1
            else:
                failed.append(f"{code}: cache file not created")
        except Exception as exc:  # pragma: no cover
            failed.append(f"{code}: {exc}")

    print(
        f"Done. success={success}, skipped={skipped}, failed={len(failed)}, "
        f"force_refresh={args.force_refresh}"
    )
    for item in failed[:20]:
        print(f"[FAIL] {item}")

    if args.strict and failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
