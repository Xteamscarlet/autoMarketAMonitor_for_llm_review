# -*- coding: utf-8 -*-
"""
增强版市场状态判断模块
从简单的 MA20 上下判断 -> 多维度市场状态评估
维度：
1. 趋势方向（MA 方向 + 价格 vs MA）
2. 波动率水平（高/中/低）
3. 趋势强度（方向一致性）
4. 量价配合度

状态细分：
- strong_bull: 强多头（趋势+低波+量价配合）-> 满仓操作
- bull: 多头（价格>MA20, MA20上升）-> 正常操作
- neutral: 中性震荡 -> 减仓操作
- weak: 弱势（价格<MA20, MA20下降）-> 限制性买入
- bear: 空头急跌（趋势+高波+量价背离）-> 仅允许防御性标的
"""
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.indicators_no_transformer import safe_sma

logger = logging.getLogger(__name__)


@dataclass
class RegimeInfo:
    """市场状态详细信息"""
    regime: str           # strong_bull / bull / neutral / weak / bear
    trend_direction: int  # 1=上升, 0=平, -1=下降
    volatility_level: str # high / medium / low
    trend_strength: float # 0~1
    volume_price_align: float  # -1~1, 正=量价齐升
    position_multiplier: float # 仓位乘数建议
    description: str

    @property
    def is_tradable(self) -> bool:
        return self.regime != 'bear'


def get_market_regime_enhanced(
    market_data: pd.DataFrame,
    date,
    ma_period: int = 20,
    vol_period: int = 60,
    trend_period: int = 20,
) -> RegimeInfo:
    """
    增强版市场状态判断

    Args:
        market_data: 大盘数据 DataFrame (需含 Close, Volume)
        date: 判断日期
        ma_period: MA 周期
        vol_period: 波动率计算周期
        trend_period: 趋势强度周期

    Returns:
        RegimeInfo 对象
    """
    if market_data is None or market_data.empty:
        return RegimeInfo('neutral', 0, 'medium', 0.5, 0.0, 0.5, '无数据,默认中性')

    try:
        idx = market_data.index.get_indexer([pd.Timestamp(date)], method='ffill')[0]
        if idx == -1 or idx < vol_period:
            return RegimeInfo('neutral', 0, 'medium', 0.5, 0.0, 0.5, '数据不足,默认中性')

        close = market_data['Close'].iloc[idx]
        if pd.isna(close) or close <= 0:
            return RegimeInfo('neutral', 0, 'medium', 0.5, 0.0, 0.5, '价格异常')

        # ========== 1. 趋势方向 ==========
        ma = safe_sma(market_data['Close'], period=ma_period)
        if ma is None or pd.isna(ma.iloc[idx]):
            return RegimeInfo('neutral', 0, 'medium', 0.5, 0.0, 0.5, 'MA计算失败')

        above_ma = close > ma.iloc[idx]

        # MA 方向
        if idx >= 5:
            ma_slope = (ma.iloc[idx] - ma.iloc[idx - 5]) / ma.iloc[idx - 5]
            ma_rising = ma_slope > 0.001  # >0.1% 5日涨幅才算上升
        else:
            ma_rising = above_ma

        # ========== 2. 波动率水平 ==========
        returns = market_data['Close'].pct_change().iloc[max(0, idx - vol_period):idx + 1]
        current_vol = returns.std() * np.sqrt(252) if len(returns) > 10 else 0.20
        # 用历史波动率的中位数做基准
        vol_series = market_data['Close'].pct_change().rolling(vol_period).std() * np.sqrt(252)
        vol_median = vol_series.median() if not vol_series.median() != vol_series.median() else 0.20

        vol_ratio = current_vol / vol_median if vol_median > 0 else 1.0

        if vol_ratio > 1.5:
            volatility_level = 'high'
        elif vol_ratio < 0.7:
            volatility_level = 'low'
        else:
            volatility_level = 'medium'

        # ========== 3. 趋势强度 ==========
        if idx >= trend_period:
            trend_returns = market_data['Close'].iloc[idx - trend_period:idx + 1].pct_change().dropna()
            if len(trend_returns) > 5:
                positive_ratio = (trend_returns > 0).mean()
                # 方向一致性 + 幅度
                mean_ret = trend_returns.mean()
                std_ret = trend_returns.std() + 1e-8
                trend_strength = min(1.0, abs(mean_ret / std_ret) * 0.5 + positive_ratio * 0.5)
            else:
                trend_strength = 0.5
        else:
            trend_strength = 0.5

        # ========== 4. 量价配合 ==========
        volume_price_align = 0.0
        if 'Volume' in market_data.columns and idx >= 10:
            recent_close = market_data['Close'].iloc[idx - 10:idx + 1].pct_change().dropna()
            recent_vol = market_data['Volume'].iloc[idx - 10:idx + 1].pct_change().dropna()
            if len(recent_close) >= 5 and len(recent_vol) >= 5:
                min_len = min(len(recent_close), len(recent_vol))
                recent_close = recent_close.iloc[-min_len:]
                recent_vol = recent_vol.iloc[-min_len:]
                # 同向变化 = 量价配合
                same_direction = ((recent_close > 0) & (recent_vol > 0)).sum() + \
                                 ((recent_close < 0) & (recent_vol < 0)).sum()
                volume_price_align = same_direction / min_len * 2 - 1  # 映射到 -1~1

        # ========== 综合判断 ==========
        if above_ma and ma_rising:
            trend_direction = 1
            if volatility_level == 'low' and trend_strength > 0.6 and volume_price_align > 0.2:
                regime = 'strong_bull'
                pos_mult = 1.0
                desc = f'强多头(低波+强趋势+量价配合)'
            else:
                regime = 'bull'
                pos_mult = 0.8
                desc = f'多头(MA上方+MA上升)'
        elif not above_ma and not ma_rising:
            trend_direction = -1
            if volatility_level == 'high' and volume_price_align < -0.2:
                regime = 'bear'
                pos_mult = 0.15
                desc = f'空头急跌(高波+量价背离)'
            else:
                regime = 'weak'
                pos_mult = 0.3
                desc = f'弱势(MA下方+MA下降)'
        else:
            trend_direction = 0
            regime = 'neutral'
            pos_mult = 0.5
            desc = f'中性震荡(MA与价格方向不一致)'

        return RegimeInfo(
            regime=regime,
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            trend_strength=trend_strength,
            volume_price_align=volume_price_align,
            position_multiplier=pos_mult,
            description=desc,
        )

    except Exception as e:
        logger.warning(f"get_market_regime_enhanced 异常: {e}")
        return RegimeInfo('neutral', 0, 'medium', 0.5, 0.0, 0.5, f'异常: {e}')


def get_market_regime(market_data: pd.DataFrame, date) -> str:
    """向后兼容的旧接口：返回 regime 字符串"""
    info = get_market_regime_enhanced(market_data, date)
    return info.regime
