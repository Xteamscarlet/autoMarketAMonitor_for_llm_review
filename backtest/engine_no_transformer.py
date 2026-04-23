# -*- coding: utf-8 -*-
"""
回测引擎（无Transformer版本）
执行单股回测循环，处理买卖信号、仓位管理、交易成本
移除了所有 Transformer 相关的买入过滤条件
"""
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest.optimizer import calculate_dynamic_weights
from data.types import get_limit_ratio, NON_FACTOR_COLS
from data.indicators_no_transformer import get_market_regime
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

from config import CommissionConfig, SlippageConfig

MAX_TRANSACTION_COST_RATIO = 0.20
MAX_RUNTIME_SLIPPAGE_RATE = 0.05


_commission_cache = None

def calculate_transaction_cost(
    price: float,
    shares: int,
    direction: str,
    code: str,
    commission_rate: Optional[float] = None,
    min_commission: Optional[float] = None,
    stamp_duty_rate: Optional[float] = None,
    transfer_fee_rate: Optional[float] = None,
) -> float:
    """
    计算交易成本（佣金 + 印花税 + 过户费）
    修复：缓存 CommissionConfig 避免循环内重复读取
    """
    global _commission_cache
    if _commission_cache is None:
        _commission_cache = CommissionConfig.from_env()
    _commission_cfg = _commission_cache

    if commission_rate is None:
        commission_rate = _commission_cfg.commission_rate
    if min_commission is None:
        min_commission = _commission_cfg.min_commission
    if stamp_duty_rate is None:
        stamp_duty_rate = _commission_cfg.stamp_duty_rate
    if transfer_fee_rate is None:
        transfer_fee_rate = _commission_cfg.transfer_fee_rate

    if shares <= 0 or price <= 0 or not np.isfinite(price):
        return 0.0

    amount = price * shares
    # 佣金（买卖都收）
    commission = max(amount * commission_rate, min_commission)

    # 印花税（仅卖出收取，千分之0.5）
    stamp_duty = amount * stamp_duty_rate if direction == 'sell' else 0

    # 过户费（沪深股票，买卖都收，十万分之一）
    transfer_fee = amount * transfer_fee_rate if code.startswith('6') else 0

    total_cost = commission + stamp_duty + transfer_fee
    max_allowed = amount * MAX_TRANSACTION_COST_RATIO
    if total_cost > max_allowed:
        logger.warning(
            "[%s] transaction cost %.4f capped to %.4f (%.1f%% of notional)",
            code,
            total_cost,
            max_allowed,
            MAX_TRANSACTION_COST_RATIO * 100,
        )
        total_cost = max_allowed
    return total_cost


def calculate_multi_timeframe_score_no_transformer(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """计算复合得分（仅使用传统因子，不包含Transformer因子）"""
    df = df.copy()

    base_cols = set(NON_FACTOR_COLS)
    # 排除 Transformer 相关列
    transformer_cols = ['transformer_prob', 'transformer_pred_ret', 'transformer_conf']
    factor_cols = [col for col in df.columns
                   if col not in base_cols and col not in transformer_cols]

    if weights is None or not weights:
        if factor_cols:
            default_weight = 1.0 / len(factor_cols)
            weights = {col: default_weight for col in factor_cols}
        else:
            df['Combined_Score'] = 0.5
            return df

    # 过滤有效因子
    valid_factors = [col for col in factor_cols if col in weights and col in df.columns]
    if not valid_factors:
        df['Combined_Score'] = 0.5
        return df

    score = sum(df[col] * weights.get(col, 0) for col in valid_factors)
    df['Combined_Score'] = score
    return df


def run_backtest_loop_no_transformer(
    df: pd.DataFrame,
    stock_code: str,
    market_data: Optional[pd.DataFrame],
    weights: Dict[str, float],
    params: Dict[str, Dict],
    regime: Optional[str] = None,
    stocks_data: Optional[Dict] = None,
    initial_capital: float = 100000.0,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], pd.DataFrame]:
    """回测引擎主循环（无Transformer版本）

    与原版本的区别：
    1. 移除了 transformer_prob 阈值检查
    2. 移除了 transformer_conf 置信度检查
    3. 放宽了成交量过滤条件（从 1.5 倍改为 0.8 倍）
    4. 仅依赖传统因子得分进行买卖决策

    Args:
        df: 包含因子和价格数据的 DataFrame
        stock_code: 股票代码
        market_data: 大盘数据
        weights: 因子权重
        params: 策略参数 {regime: {param: value}}
        regime: 固定市场状态（None 则动态判断）
        stocks_data: 所有股票数据（权重动态更新用）
        initial_capital: 初始资金

    Returns:
        (trades_df, stats, df_with_score)
    """
    df = df.copy()

    if 'Combined_Score' not in df.columns:
        df = calculate_multi_timeframe_score_no_transformer(df, weights=weights)

    limit_ratio = get_limit_ratio(stock_code)
    df['limit_up'] = df['Close'].shift(1) * (1 + limit_ratio)
    df['limit_down'] = df['Close'].shift(1) * (1 - limit_ratio)

    if 'atr' not in df.columns:
        import talib as ta
        df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # 滑点配置
    _slippage_cfg = SlippageConfig.from_env()
    buy_slippage_rate = float(np.clip(_slippage_cfg.buy_slippage_rate, 0.0, MAX_RUNTIME_SLIPPAGE_RATE))
    sell_slippage_rate = float(np.clip(_slippage_cfg.sell_slippage_rate, 0.0, MAX_RUNTIME_SLIPPAGE_RATE))
    if buy_slippage_rate != _slippage_cfg.buy_slippage_rate or sell_slippage_rate != _slippage_cfg.sell_slippage_rate:
        logger.warning(
            "[%s] slippage clipped at runtime: buy %.6f->%.6f, sell %.6f->%.6f",
            stock_code,
            _slippage_cfg.buy_slippage_rate,
            buy_slippage_rate,
            _slippage_cfg.sell_slippage_rate,
            sell_slippage_rate,
        )

    trades = []
    position = 0
    buy_price_raw = 0.0
    buy_date = None
    shares = 0
    actual_buy_cost = 0.0
    peak_price = 0.0

    # 检查是否有 Transformer 因子（应该没有，但做防御性检查）
    has_transformer_conf = 'transformer_conf' in df.columns and not df['transformer_conf'].isna().all()
    has_pred_ret = 'transformer_pred_ret' in df.columns and not df['transformer_pred_ret'].isna().all()

    for i in range(60, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        score = df['Combined_Score'].iloc[i]

        # 动态市场状态判断
        current_regime = regime if regime else get_market_regime(market_data, date)
        p = params.get(current_regime, params.get('neutral', params))

        # ========== 持仓时处理卖出逻辑 ==========
        if position > 0:
            # T+1 限制：买入当天不能卖出
            if buy_date is not None and date <= buy_date:
                # 更新 peak 但不执行卖出
                if price > peak_price:
                    peak_price = price
                continue

            unrealized_profit = (price - buy_price_raw) / buy_price_raw
            drawdown_from_peak = (peak_price - price) / peak_price

            sell_reason = None

            # 姝㈡崯
            if unrealized_profit <= p.get('stop_loss', -0.08):
                sell_reason = 'stop_loss'

            # 移动止损
            if sell_reason is None and unrealized_profit >= p.get('trailing_profit_level1', 0.06):
                if drawdown_from_peak >= p.get('trailing_drawdown_level1', 0.08):
                    sell_reason = 'trailing'

            if sell_reason is None and unrealized_profit >= p.get('trailing_profit_level2', 0.12):
                if drawdown_from_peak >= p.get('trailing_drawdown_level2', 0.04):
                    sell_reason = 'trailing'

            # 鏃堕棿姝㈡崯
            if sell_reason is None and (date - buy_date).days >= p.get('hold_days', 15):
                sell_reason = 'time_stop'

            # 信号衰减
            if sell_reason is None and score < p.get('sell_threshold', -0.2):
                sell_reason = 'signal_decay'

            # 动态止盈
            if sell_reason is None:
                atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else price * 0.02
                if unrealized_profit >= p.get('take_profit_multiplier', 3.0) * (atr / buy_price_raw):
                    sell_reason = 'take_profit'

            if sell_reason:
                if price <= df['limit_down'].iloc[i]:
                    continue

                sell_price_raw = price * (1 - sell_slippage_rate)
                sell_commission_total = calculate_transaction_cost(sell_price_raw, shares, 'sell', stock_code)
                actual_sell_proceeds = sell_price_raw * shares - sell_commission_total
                net_ret = (actual_sell_proceeds - actual_buy_cost) / actual_buy_cost

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': date,
                    'net_return': net_ret,
                    'reason': sell_reason,
                    'shares': shares,
                    'signal_strength': score,
                    'confidence': df['transformer_conf'].loc[buy_date] if has_transformer_conf else None,
                })
                position = 0

        # ========== 空仓时处理买入逻辑（无Transformer版本） ==========
        if position == 0:
            # 买入条件：综合得分超过阈值
            buy_threshold = p.get('buy_threshold', 0.6)
            if score < buy_threshold:
                continue

            # 涨跌停检查
            if price >= df['limit_up'].iloc[i]:
                continue
            if price <= df['limit_down'].iloc[i]:
                continue

            # 成交量过滤（放宽条件：只需超过20日均值的80%）
            if 'Volume' in df.columns:
                prev_vol = df['Volume'].iloc[i - 1]
                vol_ma20_prev = df['Volume'].iloc[i - 20: i].mean()
                if not pd.isna(vol_ma20_prev) and prev_vol < vol_ma20_prev * 0.8:
                    continue

            # ===== 无Transformer版本：跳过所有Transformer相关检查 =====
            # 原版本的 transformer_prob 和 transformer_conf 检查已移除

            # 执行买入
            buy_price_raw = price * (1 + buy_slippage_rate)
            shares = int(initial_capital / buy_price_raw / 100) * 100
            if shares <= 0:
                continue

            buy_commission = calculate_transaction_cost(buy_price_raw, shares, 'buy', stock_code)
            actual_buy_cost = buy_price_raw * shares + buy_commission

            position = shares
            buy_date = date
            peak_price = price

    if not trades:
        return None, None, df

    trades_df = pd.DataFrame(trades)
    stats = {
        'total_trades': len(trades_df),
        'win_rate': (trades_df['net_return'] > 0).mean() * 100,
        'avg_return': trades_df['net_return'].mean() * 100,
        'total_return': ((trades_df['net_return'] + 1).prod() - 1) * 100,
    }
    return trades_df, stats, df

