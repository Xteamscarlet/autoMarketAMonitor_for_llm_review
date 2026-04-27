# -*- coding: utf-8 -*-
"""数据层统一导出"""
from data.types import FEATURES, BASE_OHLCV_COLS, TRADITIONAL_FACTOR_COLS
from data.loader import download_market_data, download_stocks_data, get_single_stock_data
from data.normalize import normalize_stock_dataframe, normalize_market_dataframe
from data.cache import (
    check_and_clean_cache,
    load_pickle_cache,
    save_pickle_cache,
    get_transformer_cache_path,
    load_transformer_cache,
    save_transformer_cache,
)
from data.fundamentals import load_fundamental_history, get_fundamental_snapshot_asof

try:
    from data.regime import get_market_regime, get_market_regime_enhanced, RegimeInfo
except ImportError as exc:
    _REGIME_IMPORT_ERROR = exc

    class RegimeInfo:
        pass

    def get_market_regime(*args, **kwargs):
        raise ImportError("Market regime helpers require optional dependency talib") from _REGIME_IMPORT_ERROR

    def get_market_regime_enhanced(*args, **kwargs):
        raise ImportError("Market regime helpers require optional dependency talib") from _REGIME_IMPORT_ERROR

try:
    from data.indicators import calculate_all_indicators, calculate_orthogonal_factors
except ImportError as exc:
    _INDICATORS_IMPORT_ERROR = exc

    def calculate_all_indicators(*args, **kwargs):
        raise ImportError("Technical indicators require optional dependency talib") from _INDICATORS_IMPORT_ERROR

    def calculate_orthogonal_factors(*args, **kwargs):
        raise ImportError("Technical factors require optional dependency talib") from _INDICATORS_IMPORT_ERROR

__all__ = [
    "FEATURES",
    "BASE_OHLCV_COLS",
    "TRADITIONAL_FACTOR_COLS",
    "download_market_data",
    "download_stocks_data",
    "get_single_stock_data",
    "normalize_stock_dataframe",
    "normalize_market_dataframe",
    "check_and_clean_cache",
    "load_pickle_cache",
    "save_pickle_cache",
    "get_transformer_cache_path",
    "load_transformer_cache",
    "save_transformer_cache",
    "calculate_all_indicators",
    "calculate_orthogonal_factors",
    "load_fundamental_history",
    "get_fundamental_snapshot_asof",
    "get_market_regime",
    "get_market_regime_enhanced",
    "RegimeInfo",
]
