# -*- coding: utf-8 -*-
"""
类型定义和常量 V2
新增：5种市场状态枚举、板块映射
"""

# Transformer模型使用的特征（V2 新增收益率特征）
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover Rate',
    'MA5', 'MA10', 'MA20',
    'MACD', 'K', 'D', 'J', 'RSI', 'ADX',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'OBV', 'CCI',
    'ret_1', 'ret_3', 'ret_5', 'ret_10',  # ★ 新增收益率特征
]

BASE_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

TRADITIONAL_FACTOR_COLS = [
    'mom_10', 'mom_20', 'atr_pct', 'vol_price_res',
    'rsi_norm', 'macd_hist_norm', 'bias_20', 'bb_width',
]

AI_FACTOR_COLS = ['transformer_prob', 'transformer_pred_ret', 'transformer_conf']

ALL_FACTOR_COLS = TRADITIONAL_FACTOR_COLS + AI_FACTOR_COLS

NON_FACTOR_COLS = [
    'Close', 'Open', 'High', 'Low', 'Volume', 'MA20', 'Combined_Score',
    'Close_raw', 'transformer_pred_ret_raw', 'transformer_available',
]

# ★ 扩展市场状态（5种）
REGIME_STRONG_BULL = 'strong_bull'
REGIME_BULL = 'bull'
REGIME_NEUTRAL = 'neutral'
REGIME_WEAK = 'weak'
REGIME_BEAR = 'bear'
ALL_REGIMES = [REGIME_STRONG_BULL, REGIME_BULL, REGIME_NEUTRAL, REGIME_WEAK, REGIME_BEAR]

# 涨跌停比例
LIMIT_RATIO_DICT = {
    'main': 0.10,
    'gem': 0.20,
    'star': 0.20,
}


def get_limit_ratio(code: str) -> float:
    """根据股票代码获取涨跌停比例"""
    if code.startswith('300') or code.startswith('688'):
        return LIMIT_RATIO_DICT['gem']
    return LIMIT_RATIO_DICT['main']
