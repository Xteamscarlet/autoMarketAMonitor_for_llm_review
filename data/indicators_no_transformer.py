# -*- coding: utf-8 -*-
"""
技术指标计算模块（无Transformer版本）
只计算传统因子，不依赖 Transformer 模型
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import talib as ta

from data.types import FEATURES, BASE_OHLCV_COLS, TRADITIONAL_FACTOR_COLS, NON_FACTOR_COLS

logger = logging.getLogger(__name__)


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



def safe_sma(series, period):
    """
    安全的简单移动平均计算
    修复：
    1. 兼容 numpy.ndarray / pd.Series / list 输入
    2. 消除同名递归调用，直接用 pandas rolling 计算
    """
    # ---- 类型归一化：统一转成 pd.Series ----
    if isinstance(series, np.ndarray):
        s = pd.Series(series)
    elif isinstance(series, (list, tuple)):
        s = pd.Series(series)
    elif isinstance(series, pd.Series):
        s = series.copy()
    else:
        logger.warning(f"safe_sma 收到不支持的类型: {type(series)}，尝试转成 Series")
        s = pd.Series(series)

    n = len(s)

    # ---- 长度不足 ----
    if n < period:
        logger.warning(f"数据长度 {n} 不足以计算 {period} 日移动平均")
        return pd.Series(np.nan, index=s.index)

    # ---- 全 NaN 检测（用 pd.Series 的 .isna()，不会再报错）----
    if s.isna().all():
        logger.warning("输入序列全为NaN")
        return pd.Series(np.nan, index=s.index)

    # ---- 直接用 pandas rolling 计算 SMA，不再递归调用自己 ----
    result = s.rolling(window=int(period), min_periods=period).mean()

    return result


def check_indicator_result(result, indicator_name, code=""):
    """检查指标计算结果是否有效"""
    if result is None:
        logger.warning(f"[{code}] {indicator_name} 计算返回None")
        return False

    nan_ratio = pd.isna(result).sum() / len(result)
    if nan_ratio > 0.5:
        logger.warning(f"[{code}] {indicator_name} NaN比例过高: {nan_ratio:.1%}")
        return False

    return True


def calculate_orthogonal_factors_no_transformer(
        df: pd.DataFrame,
        stock_code: str = '',
        n_components: int = 5,
) -> pd.DataFrame:
    """计算正交化因子（无Transformer版本）

    与原版本的区别：
    1. 不计算 Transformer 因子
    2. 不依赖模型文件
    3. 仅使用传统技术因子

    Args:
        df: 包含 OHLCV 数据的 DataFrame
        stock_code: 股票代码（用于日志）
        n_components: PCA 保留的主成分数量

    Returns:
        添加了正交化因子列的 DataFrame
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = df.copy()

    # 1. 计算基础技术指标
    df = calculate_all_indicators(df)

    # 2. 计算衍生因子
    # 动量因子
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

    # 波动率因子
    df['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std()
    df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()

    # 成交量因子
    df['Volume_Ratio_5'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # 价格位置因子
    df['Price_Position_20'] = (df['Close'] - df['Low'].rolling(window=20).min()) / (
            df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min() + 1e-8
    )

    # ATR
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 3. 定义因子列表（排除 Transformer 因子）
    base_cols = set(NON_FACTOR_COLS)
    transformer_cols = ['transformer_prob', 'transformer_pred_ret', 'transformer_conf',
                        'transformer_uncertainty']

    factor_cols = [col for col in df.columns
                   if col not in base_cols and col not in transformer_cols]

    # 4. 标准化因子
    for col in factor_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    # 5. PCA 正交化
    valid_factors = [col for col in factor_cols if col in df.columns and not df[col].isna().all()]

    if len(valid_factors) >= n_components:
        factor_data = df[valid_factors].values

        # 标准化
        scaler = StandardScaler()
        factor_scaled = scaler.fit_transform(factor_data)

        # PCA
        pca = PCA(n_components=min(n_components, len(valid_factors)))
        pca_factors = pca.fit_transform(factor_scaled)

        # 添加 PCA 因子列
        for i in range(pca_factors.shape[1]):
            df[f'PCA_{i + 1}'] = pca_factors[:, i]

        logger.info(f"[无Transformer] {stock_code} PCA 解释方差比: {pca.explained_variance_ratio_[:3]}")
    else:
        logger.warning(f"[无Transformer] {stock_code} 因子数量不足，跳过 PCA")

    # 6. 时序标准化（滚动排名）
    for col in factor_cols:
        if col in df.columns:
            df[col] = df[col].rolling(window=250, min_periods=20).rank(pct=True)
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0.5)

    # 全局清理
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0.5, inplace=True)

    logger.info(f"[无Transformer模式] {stock_code} 因子计算完成，仅使用传统因子")

    return df


def get_market_regime(market_data: Optional[pd.DataFrame], current_date) -> str:
    """市场状态判断 — 委托给增强版 regime 模块
    
    ★ 修复：从 3 状态 (strong/weak/neutral) 升级到 5 状态
    (strong_bull/bull/neutral/weak/bear)，与 engine.py 一致
    """
    from data.regime import get_market_regime as _get_regime_enhanced
    return _get_regime_enhanced(market_data, current_date)
