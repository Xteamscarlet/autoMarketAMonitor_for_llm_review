# -*- coding: utf-8 -*-
"""
技术指标计算模块（增强版）
统一封装 talib 指标计算，确保训练/回测/实盘使用完全一致的逻辑
新增：数据长度校验、NaN处理、异常捕获
"""
import logging
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import talib as ta

from data.types import FEATURES, BASE_OHLCV_COLS, TRADITIONAL_FACTOR_COLS, NON_FACTOR_COLS, AI_FACTOR_COLS

logger = logging.getLogger(__name__)

# 延迟导入 Transformer
_transformer_available = None

# 新增：指标计算所需的最小数据长度
MIN_DATA_LENGTH = {
    'MA5': 5,
    'MA10': 10,
    'MA20': 20,
    'MACD': 35,  # 26 + 9
    'KDJ': 9,
    'RSI': 14,
    'ADX': 14,
    'BBANDS': 20,
    'OBV': 1,
    'CCI': 20,
    'ATR': 14,
}


def _check_transformer_available():
    global _transformer_available
    if _transformer_available is None:
        try:
            from model.predictor import calculate_transformer_factor_series
            _transformer_available = True
        except ImportError:
            _transformer_available = False
            logger.warning("TransformerStock 模块未安装，Transformer因子将不可用")
    return _transformer_available


# ==================== 新增：安全的指标计算函数 ====================
def safe_sma(series: pd.Series, period: int) -> pd.Series:
    """安全的简单移动平均计算

    Args:
        series: 输入序列
        period: 周期

    Returns:
        移动平均序列，数据不足时返回NaN序列
    """
    if len(series) < period:
        logger.warning(f"数据长度 {len(series)} 不足以计算 {period} 日移动平均")
        return pd.Series(np.nan, index=series.index)

    if series.isna().all():
        logger.warning("输入序列全为NaN")
        return pd.Series(np.nan, index=series.index)

    result = ta.SMA(series.values, timeperiod=period)
    return pd.Series(result, index=series.index)


def safe_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[
    pd.Series, pd.Series, pd.Series]:
    """安全的MACD计算

    Returns:
        (MACD, Signal, Hist) 元组
    """
    min_len = slow + signal - 1
    if len(series) < min_len:
        logger.warning(f"数据长度 {len(series)} 不足以计算MACD (需要 {min_len})")
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series

    if series.isna().all():
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series

    try:
        macd, signal_line, hist = ta.MACD(series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.Series(macd, index=series.index), pd.Series(signal_line, index=series.index), pd.Series(hist,
                                                                                                          index=series.index)
    except Exception as e:
        logger.error(f"MACD计算失败: {e}")
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, nan_series, nan_series


def safe_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """安全的RSI计算"""
    if len(series) < period + 1:
        logger.warning(f"数据长度 {len(series)} 不足以计算 {period} 日RSI")
        return pd.Series(np.nan, index=series.index)

    if series.isna().all():
        return pd.Series(np.nan, index=series.index)

    try:
        result = ta.RSI(series.values, timeperiod=period)
        return pd.Series(result, index=series.index)
    except Exception as e:
        logger.error(f"RSI计算失败: {e}")
        return pd.Series(np.nan, index=series.index)


def check_indicator_result(result: pd.Series, indicator_name: str, code: str = "") -> bool:
    """检查指标计算结果是否有效

    Args:
        result: 指标计算结果
        indicator_name: 指标名称
        code: 股票代码

    Returns:
        结果是否有效
    """
    if result is None:
        logger.warning(f"[{code}] {indicator_name} 计算返回None")
        return False

    if isinstance(result, pd.Series):
        nan_ratio = result.isna().sum() / len(result)
        if nan_ratio > 0.5:
            logger.warning(f"[{code}] {indicator_name} NaN比例过高: {nan_ratio:.1%}")
            return False

    return True


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标（对应 FEATURES 列表）- 增强版

    在原始 OHLCV + Turnover Rate 基础上计算：
    MA5, MA10, MA20, MACD, KDJ, RSI, ADX, BBANDS, OBV, CCI

    Args:
        df: 必须包含 Open, High, Low, Close, Volume, Turnover Rate 列

    Returns:
        添加了技术指标列的 DataFrame
    """
    df = df.copy()

    # 新增：数据长度检查
    if len(df) < 35:  # MACD需要最多数据
        logger.warning(f"数据长度 {len(df)} 不足以计算所有指标，部分指标将为NaN")

    # 新增：NaN预处理
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].ffill().bfill()

    # 均线 - 使用安全计算
    df['MA5'] = safe_sma(df['Close'], 5)
    df['MA10'] = safe_sma(df['Close'], 10)
    df['MA20'] = safe_sma(df['Close'], 20)

    # MACD - 使用安全计算
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = safe_macd(df['Close'])

    # KDJ
    try:
        if len(df) >= 9:
            df['K'], df['D'] = ta.STOCH(
                df['High'].values, df['Low'].values, df['Close'].values,
                fastk_period=9, slowk_period=3, slowd_period=3
            )
            df['K'] = pd.Series(df['K'], index=df.index)
            df['D'] = pd.Series(df['D'], index=df.index)
            df['J'] = 3 * df['K'] - 2 * df['D']
        else:
            df['K'] = np.nan
            df['D'] = np.nan
            df['J'] = np.nan
    except Exception as e:
        logger.error(f"KDJ计算失败: {e}")
        df['K'] = np.nan
        df['D'] = np.nan
        df['J'] = np.nan

    # RSI - 使用安全计算
    df['RSI'] = safe_rsi(df['Close'], 14)

    # ADX
    try:
        if len(df) >= 14:
            df['ADX'] = pd.Series(ta.ADX(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14),
                                  index=df.index)
        else:
            df['ADX'] = np.nan
    except Exception as e:
        logger.error(f"ADX计算失败: {e}")
        df['ADX'] = np.nan

    # 布林带
    try:
        if len(df) >= 20:
            upper, middle, lower = ta.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['BB_Upper'] = pd.Series(upper, index=df.index)
            df['BB_Middle'] = pd.Series(middle, index=df.index)
            df['BB_Lower'] = pd.Series(lower, index=df.index)
        else:
            df['BB_Upper'] = np.nan
            df['BB_Middle'] = np.nan
            df['BB_Lower'] = np.nan
    except Exception as e:
        logger.error(f"布林带计算失败: {e}")
        df['BB_Upper'] = np.nan
        df['BB_Middle'] = np.nan
        df['BB_Lower'] = np.nan

    # OBV
    try:
        df['OBV'] = pd.Series(ta.OBV(df['Close'].values, df['Volume'].values), index=df.index)
    except Exception as e:
        logger.error(f"OBV计算失败: {e}")
        df['OBV'] = np.nan

    # CCI
    try:
        if len(df) >= 20:
            df['CCI'] = pd.Series(ta.CCI(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=20),
                                  index=df.index)
        else:
            df['CCI'] = np.nan
    except Exception as e:
        logger.error(f"CCI计算失败: {e}")
        df['CCI'] = np.nan

    # ★ 新增收益率特征
    for lag in [1, 3, 5, 10]:
        df[f'ret_{lag}'] = df['Close'].pct_change(lag)

    return df


def calculate_orthogonal_factors(
        df: pd.DataFrame,
        stock_code: str = "",
        device=None,
        allow_save_cache: bool = False,
) -> pd.DataFrame:
    """计算正交化因子（传统因子 + Transformer因子）- 增强版

    传统因子使用 rolling rank 标准化到 [0, 1]
    Transformer 因子保持模型原始输出

    Args:
        df: 包含 OHLCV 的 DataFrame
        stock_code: 股票代码
        device: Transformer 推理设备
        allow_save_cache: 是否允许保存 Transformer 缓存

    Returns:
        添加了因子列的 DataFrame
    """
    df = df.copy()

    # 前向填充缺失值
    if df['Close'].isna().any():
        df['Close'] = df['Close'].ffill().bfill()
    df = df.dropna(subset=['Close', 'Volume'])

    # 新增：数据长度检查
    if len(df) < 20:
        logger.warning(f"[{stock_code}] 数据长度 {len(df)} 不足以计算所有因子")

    # ========== 传统因子计算 ==========
    # 动量因子
    df['mom_10'] = df['Close'].pct_change(10)
    df['mom_20'] = df['Close'].pct_change(20)

    # ATR - 安全计算
    try:
        if len(df) >= 14:
            df['atr'] = pd.Series(ta.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14),
                                  index=df.index)
            df['atr_pct'] = df['atr'] / df['Close']
        else:
            df['atr'] = np.nan
            df['atr_pct'] = np.nan
    except Exception as e:
        logger.error(f"[{stock_code}] ATR计算失败: {e}")
        df['atr'] = np.nan
        df['atr_pct'] = np.nan

    # 量价相关性
    price_change = df['Close'].pct_change()
    vol_change = df['Volume'].pct_change()
    df['vol_price_res'] = (price_change * vol_change).rolling(5).mean()

    # RSI标准化
    df['rsi_norm'] = (safe_rsi(df['Close'], 14) - 50) / 50

    # MACD标准化
    _, _, macdhist = safe_macd(df['Close'])
    df['macd_hist_norm'] = macdhist / df['Close']

    # 乖离率
    ma20 = df['Close'].rolling(20).mean()
    df['bias_20'] = (df['Close'] - ma20) / ma20

    # 布林带宽度
    try:
        if len(df) >= 20:
            upper, middle, lower = ta.BBANDS(df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_width'] = (upper - lower) / middle
        else:
            df['bb_width'] = np.nan
    except Exception as e:
        logger.error(f"[{stock_code}] 布林带宽度计算失败: {e}")
        df['bb_width'] = np.nan

    # ========== Transformer 因子计算 ==========
    if _check_transformer_available():
        from data.cache import load_transformer_cache, save_transformer_cache
        from model.predictor import calculate_transformer_factor_series

        valid_dates = df['Close'].dropna().index
        current_last_date = valid_dates[-1] if len(valid_dates) > 0 else None

        cached_df = load_transformer_cache(stock_code, current_last_date)
        if cached_df is not None:
            trans_result = cached_df.reindex(df.index)
            trans_result['transformer_prob'] = trans_result['transformer_prob'].fillna(0.5)
            trans_result['transformer_pred_ret'] = trans_result['transformer_pred_ret'].fillna(0.0)
            trans_result['transformer_uncertainty'] = trans_result['transformer_uncertainty'].fillna(0.15)
        else:
            try:
                trans_result = calculate_transformer_factor_series(
                    df=df, code=stock_code, device=device
                )
                if allow_save_cache and current_last_date is not None:
                    save_transformer_cache(stock_code, current_last_date, trans_result)
            except Exception as e:
                logger.error(f"[{stock_code}] Transformer因子计算失败: {e}")
                trans_result = pd.DataFrame(index=df.index)
                trans_result['transformer_prob'] = 0.5
                trans_result['transformer_pred_ret'] = 0.0
                trans_result['transformer_uncertainty'] = 0.15
    else:
        trans_result = pd.DataFrame(index=df.index)
        trans_result['transformer_prob'] = 0.5
        trans_result['transformer_pred_ret'] = 0.0
        trans_result['transformer_uncertainty'] = 0.15

    df['transformer_prob'] = trans_result['transformer_prob']
    df['transformer_pred_ret'] = trans_result['transformer_pred_ret']
    df['transformer_conf'] = 1.0 - trans_result['transformer_uncertainty']

    # ========== 标准化 ==========
    # 传统因子: rolling rank 标准化到 [0, 1]
    for col in TRADITIONAL_FACTOR_COLS:
        if col in df.columns:
            df[col] = df[col].rolling(window=250, min_periods=20).rank(pct=True)
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0.5)

    # AI因子: transformer_prob 已经是 [0,1] 概率
    # transformer_pred_ret: tanh 压缩后映射到 [0, 1]
    if 'transformer_pred_ret' in df.columns:
        compressed = np.tanh(df['transformer_pred_ret'] * 10)
        df['transformer_pred_ret'] = (compressed + 1) / 2

    # 全局清理
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0.5, inplace=True)

    return df

def safe_ma(arr: np.ndarray, window: int) -> np.ndarray:
    """
    计算简单移动平均，返回与 arr 等长的 ndarray。
    前 window-1 个位置为 np.nan。
    """
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=np.float64)

    # 使用 pandas 的 rolling 计算，再转回 numpy
    s = pd.Series(arr)
    ma = s.rolling(window=window, min_periods=window).mean().to_numpy()
    return ma

def get_market_regime(
    close: np.ndarray,
    window: int = 20,
    threshold: float = 0.0,
    neutral_value: str = "neutral",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据价格与 MA20 判断市场状态：上涨/下跌/中性。

    Parameters
    ----------
    close : np.ndarray
        收盘价序列，1D。
    window : int
        均线窗口，默认 20。
    threshold : float
        价格距离均线超过该阈值才认为是上涨/下跌，否则为中性。
        例如 threshold=0.02 表示价格距离均线超过 2% 才有方向。
    neutral_value : str
        当无法计算 MA 或距离在阈值内时，返回的市场状态值。

    Returns
    -------
    regime : np.ndarray
        市场状态序列，取值：
        - "up"    : 价格 > MA * (1 + threshold)
        - "down"  : 价格 < MA * (1 - threshold)
        - neutral_value : 其它情况
    ma : np.ndarray
        移动平均序列，与 close 等长。
    """
    if close.ndim != 1:
        raise ValueError("close 必须是 1D array")

    ma = safe_ma(close, window)

    # 安全地检查 NaN：用 np.isnan，而不是 .isna()
    nan_mask = np.isnan(ma)

    # 初始化 regime
    regime = np.full_like(close, neutral_value, dtype=object)

    # 只在非 NaN 位置判断方向
    valid_mask = ~nan_mask

    price = close[valid_mask]
    ma_valid = ma[valid_mask]

    # 计算相对距离
    rel = (price - ma_valid) / ma_valid

    # 根据阈值划分状态
    up_mask = rel > threshold
    down_mask = rel < -threshold

    # 用布尔索引给 regime 赋值
    # 注意：先赋值 "down"，再 "up"，保证 up 优先于 down（如果同时满足）
    regime[valid_mask] = neutral_value
    regime[valid_mask][down_mask] = "down"
    regime[valid_mask][up_mask] = "up"

    return regime, ma


# 在 run_backtest_no_transformer_new.py 中添加收益率计算

def prepare_stock_data(stocks_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    准备股票数据：计算因子和收益率
    """
    for stock_name, df in stocks_data.items():
        # 2. 计算未来收益率（关键修复）
        df['future_return_1d'] = calculate_future_return(df['Close'])

        # 3. 更新数据
        stocks_data[stock_name] = df

    return stocks_data


def calculate_future_return(close_prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    计算未来收益率

    Args:
        close_prices: 收盘价序列
        periods: 未来期数，默认1天

    Returns:
        未来收益率序列
    """
    # 使用对数收益率
    future_return = np.log(close_prices.shift(-periods) / close_prices)

    # 或者使用简单收益率（根据您的策略选择）
    # future_return = close_prices.pct_change(periods).shift(-periods)

    return future_return

def calculate_orthogonal_factors_without_transformer(
    df: pd.DataFrame
) -> pd.DataFrame:
    """计算正交化因子（传统因子）

    传统因子使用 rolling rank 标准化到 [0, 1]
    Transformer 因子保持模型原始输出

    Args:
        df: 包含 OHLCV 的 DataFrame

    Returns:
        添加了因子列的 DataFrame
    """
    df = df.copy()

    # 前向填充缺失值
    if df['Close'].isna().any():
        df['Close'] = df['Close'].ffill().bfill()
    df = df.dropna(subset=['Close', 'Volume'])

    # ========== 传统因子计算 ==========
    df['mom_10'] = df['Close'].pct_change(10)
    df['mom_20'] = df['Close'].pct_change(20)

    df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['atr_pct'] = df['atr'] / df['Close']

    price_change = df['Close'].pct_change()
    vol_change = df['Volume'].pct_change()
    df['vol_price_res'] = (price_change * vol_change).rolling(5).mean()

    df['rsi_norm'] = (ta.RSI(df['Close'], 14) - 50) / 50

    _, _, macdhist = ta.MACD(df['Close'])
    df['macd_hist_norm'] = macdhist / df['Close']

    ma20 = df['Close'].rolling(20).mean()
    df['bias_20'] = (df['Close'] - ma20) / ma20

    upper, middle, lower = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['bb_width'] = (upper - lower) / middle

    # ========== 标准化 ==========
    # 传统因子: rolling rank 标准化到 [0, 1]
    for col in TRADITIONAL_FACTOR_COLS:
        if col in df.columns:
            df[col] = df[col].rolling(window=250, min_periods=20).rank(pct=True)
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0.5)


    # 全局清理
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0.5, inplace=True)

    return df
