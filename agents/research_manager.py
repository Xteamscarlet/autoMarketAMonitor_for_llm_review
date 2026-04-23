# -*- coding: utf-8 -*-
"""Research manager that combines fundamentals and technical analysis agents."""

from typing import Dict, Optional

import pandas as pd

from agents.fundamentals_agent import analyze_fundamentals
from agents.technicals_agent import analyze_technicals
from config import get_settings
from data import calculate_all_indicators, calculate_orthogonal_factors, get_single_stock_data


PRICE_COLUMN_ALIASES = {
    "Open": ("Open", "开盘"),
    "High": ("High", "最高"),
    "Low": ("Low", "最低"),
    "Close": ("Close", "收盘"),
    "Volume": ("Volume", "成交量"),
    "Turnover Rate": ("Turnover Rate", "换手率"),
}


def _find_first_column(df: pd.DataFrame, aliases) -> Optional[str]:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def prepare_research_dataframe(code: str) -> Optional[pd.DataFrame]:
    raw_df = get_single_stock_data(code)
    if raw_df is None or raw_df.empty:
        return None

    df = pd.DataFrame(index=raw_df.index)
    for target_col, aliases in PRICE_COLUMN_ALIASES.items():
        source_col = _find_first_column(raw_df, aliases)
        if source_col:
            df[target_col] = pd.to_numeric(raw_df[source_col], errors="coerce")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in required_cols):
        return None

    if "Turnover Rate" not in df.columns:
        df["Turnover Rate"] = 3.0

    df = df.dropna(subset=required_cols)
    if df.empty:
        return None

    df = calculate_all_indicators(df)
    df = calculate_orthogonal_factors(df, stock_code=code)
    return df


def analyze_stock(code: str, name: str, df: Optional[pd.DataFrame] = None) -> Dict:
    settings = get_settings()
    result = {
        "fundamentals": None,
        "technicals": None,
        "blended_score": None,
    }

    if settings.analysis.enable_fundamental_agent:
        result["fundamentals"] = analyze_fundamentals(code=code, name=name)

    if df is None and settings.analysis.enable_technical_agent:
        df = prepare_research_dataframe(code)

    if settings.analysis.enable_technical_agent and df is not None and not df.empty:
        result["technicals"] = analyze_technicals(df=df, code=code, name=name)

    blended = 0.0
    total_weight = 0.0
    if result["fundamentals"] and settings.analysis.blend_fundamental_score:
        blended += result["fundamentals"]["score"] * settings.analysis.fundamental_weight
        total_weight += settings.analysis.fundamental_weight
    if result["technicals"] and settings.analysis.blend_technical_score:
        blended += result["technicals"]["score"] * settings.analysis.technical_weight
        total_weight += settings.analysis.technical_weight

    result["blended_score"] = round(blended / total_weight, 4) if total_weight > 0 else None
    return result
