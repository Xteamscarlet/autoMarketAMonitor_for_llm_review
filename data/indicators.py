# -*- coding: utf-8 -*-
"""
技术指标计算模块
统一封装 talib 指标计算，确保训练/回测/实盘使用完全一致的逻辑
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import talib as ta

from data.indicators_no_transformer import safe_sma
from data.types import FEATURES, BASE_OHLCV_COLS, TRADITIONAL_FACTOR_COLS, NON_FACTOR_COLS, AI_FACTOR_COLS

logger = logging.getLogger(__name__)

# 延迟导入 Transformer
_transformer_available = None


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


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标（对应 FEATURES 列表）

    在原始 OHLCV + Turnover Rate 基础上计算：
    MA5, MA10, MA20, MACD, KDJ, RSI, ADX, BBANDS, OBV, CCI

    Args:
        df: 必须包含 Open, High, Low, Close, Volume, Turnover Rate 列

    Returns:
        添加了技术指标列的 DataFrame
    """
    df = df.copy()

    # 均线
    df['MA5'] = safe_sma(df['Close'], period=5)
    df['MA10'] = safe_sma(df['Close'], period=10)
    df['MA20'] = safe_sma(df['Close'], period=20)

    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )

    # KDJ
    df['K'], df['D'] = ta.STOCH(
        df['High'], df['Low'], df['Close'],
        fastk_period=9, slowk_period=3, slowd_period=3
    )
    df['J'] = 3 * df['K'] - 2 * df['D']

    # RSI
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)

    # ADX
    df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 布林带
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = ta.BBANDS(
        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    # OBV
    df['OBV'] = ta.OBV(df['Close'], df['Volume'])

    # CCI
    df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)

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
    """计算正交化因子（传统因子 + Transformer因子）

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
        df['Close'] = df['Close'].ffill()
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
            trans_result = calculate_transformer_factor_series(
                df=df, code=stock_code, device=device
            )
            if allow_save_cache and current_last_date is not None:
                save_transformer_cache(stock_code, current_last_date, trans_result)
    else:
        trans_result = pd.DataFrame(index=df.index)
        trans_result['transformer_prob'] = 0.5
        trans_result['transformer_pred_ret'] = 0.0
        trans_result['transformer_uncertainty'] = 0.15

    df['transformer_prob'] = trans_result['transformer_prob']
    # 保留原始收益率预测，供仓位控制/风控使用。
    df['transformer_pred_ret_raw'] = trans_result['transformer_pred_ret']
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




from data.regime import get_market_regime

