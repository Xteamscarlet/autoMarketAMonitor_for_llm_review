# -*- coding: utf-8 -*-
"""
回测引擎
执行单股回测循环，处理买卖信号、仓位管理、交易成本
从 backtest_market_v2.py 的 run_backtest_loop() 提取
"""
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest.optimizer import calculate_dynamic_weights
from data.types import get_limit_ratio, NON_FACTOR_COLS
from data.indicators import get_market_regime
from risk_manager import RiskManager

logger = logging.getLogger(__name__)


from config import CommissionConfig,SlippageConfig  # 在文件顶部导入

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
    未传入的费率参数使用 config.py 中的全局配置。
    """
    # engine.py（在 run_backtest_loop 函数内或模块级）
    _commission_cfg = CommissionConfig.from_env()  # 用 .env 里的配置

    # 使用全局配置作为默认值
    if commission_rate is None:
        commission_rate = _commission_cfg.commission_rate
    if min_commission is None:
        min_commission = _commission_cfg.min_commission
    if stamp_duty_rate is None:
        stamp_duty_rate = _commission_cfg.stamp_duty_rate
    if transfer_fee_rate is None:
        transfer_fee_rate = _commission_cfg.transfer_fee_rate

    amount = price * shares
    commission = max(amount * commission_rate, min_commission)
    stamp_duty = amount * stamp_duty_rate if direction == 'sell' else 0.0
    transfer_fee = amount * transfer_fee_rate if direction == 'sell' else 0.0

    return commission + stamp_duty + transfer_fee



def calculate_multi_timeframe_score(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """计算复合得分（自动适应因子数量）"""
    df = df.copy()

    base_cols = set(NON_FACTOR_COLS)
    factor_cols = [col for col in df.columns if col not in base_cols]

    if weights is None or not weights:
        if factor_cols:
            default_weight = 1.0 / len(factor_cols)
            weights = {col: default_weight for col in factor_cols}
        else:
            df['Combined_Score'] = 0.5
            return df

    score = pd.Series(0.5, index=df.index)
    for factor, weight in weights.items():
        if factor in df.columns and factor in factor_cols and weight != 0:
            factor_val = (df[factor] - 0.5) * 2
            score += factor_val * weight

    df['Combined_Score'] = score
    return df


def run_backtest_loop(
    df: pd.DataFrame,
    stock_code: str,
    market_data: Optional[pd.DataFrame],
    weights: Dict[str, float],
    params: Dict[str, Dict],
    regime: Optional[str] = None,
    stocks_data: Optional[Dict] = None,
    initial_capital: float = 100000.0,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], pd.DataFrame]:
    """回测引擎主循环

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
        df = calculate_multi_timeframe_score(df, weights=weights)

    limit_ratio = get_limit_ratio(stock_code)
    df['limit_up'] = df['Close'].shift(1) * (1 + limit_ratio)
    df['limit_down'] = df['Close'].shift(1) * (1 - limit_ratio)

    if 'atr' not in df.columns:
        import talib as ta
        df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    has_transformer_conf = 'transformer_conf' in df.columns
    has_pred_ret = 'transformer_pred_ret' in df.columns

    trades = []
    position = 0
    actual_buy_cost = 0
    shares = 0
    peak = 0
    buy_date = None

    current_weights = weights.copy()
    last_weight_update = 60
    # engine.py（在 run_backtest_loop 函数内或模块级）
    _slippage_cfg = SlippageConfig.from_env()  # 用 .env 里的配置

    # 然后用实例属性
    buy_slippage_rate = _slippage_cfg.buy_slippage_rate
    sell_slippage_rate = _slippage_cfg.sell_slippage_rate
    for i in range(60, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]

        transformer_prob = df['transformer_prob'].iloc[i] if 'transformer_prob' in df.columns else 0.5
        transformer_conf = df['transformer_conf'].iloc[i] if 'transformer_conf' in df.columns else 0.0

        # 权重动态更新
        if i - last_weight_update >= 60:
            hist_df = df.iloc[max(0, i - 250): i]
            factor_cols = [col for col in hist_df.columns if col not in NON_FACTOR_COLS]
            if factor_cols:
                new_weights = calculate_dynamic_weights(hist_df, factor_cols, ic_window_range=(20, 120), use_ewma=True)
                for col in factor_cols:
                    if col in current_weights and col in new_weights:
                        current_weights[col] = 0.7 * current_weights[col] + 0.3 * new_weights[col]
                sum_w = sum(current_weights.values())
                if sum_w > 0:
                    current_weights = {k: v / sum_w for k, v in current_weights.items()}
                last_weight_update = i
                df = calculate_multi_timeframe_score(df, weights=current_weights)

        score = df['Combined_Score'].iloc[i]
        curr_regime = get_market_regime(market_data, date) if regime is None else regime
        p = params.get(curr_regime, params.get('neutral', params))

        # ========== 买入逻辑 ==========
        if position == 0 and score > p.get('buy_threshold', 0.6):
            confidence_threshold = p.get('confidence_threshold', 0.5)
            if transformer_conf < confidence_threshold:
                continue

            transformer_threshold = p.get('transformer_buy_threshold', 0.6)
            if transformer_prob < transformer_threshold:
                continue

            if pd.isna(price) or price <= 0:
                continue

            if curr_regime == 'weak':
                continue

            if has_transformer_conf:
                confidence = df['transformer_conf'].iloc[i]
                if confidence < 0.6:
                    continue
            if price >= df['limit_up'].iloc[i] * 0.995: continue

            # 成交量过滤
            if 'Volume' in df.columns:
                prev_vol = df['Volume'].iloc[i - 1]
                vol_ma20_prev = df['Volume'].iloc[i - 20: i].mean()
                if not pd.isna(vol_ma20_prev) and prev_vol < vol_ma20_prev * 1.5:
                    continue

            # ATR 动态仓位
            atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else price * 0.02
            daily_vol = atr / price
            if daily_vol <= 0 or not np.isfinite(daily_vol):
                daily_vol = 0.02

            target_annual_vol = 0.10
            position_ratio = target_annual_vol / (daily_vol * np.sqrt(252) + 1e-6)
            position_ratio = min(max(position_ratio, 0.1), 1.0)

            if has_pred_ret:
                pred_ret = df['transformer_pred_ret'].iloc[i]
                signal_strength = max(0.5, min(1.5, 1 + pred_ret / 0.05))
                position_ratio *= signal_strength
                position_ratio = min(max(position_ratio, 0.1), 1.0)

            try:
                shares = max(100, int(initial_capital * position_ratio / price / 100) * 100)
                shares = min(shares, int(initial_capital / price / 100) * 100)
            except (ValueError, ZeroDivisionError):
                continue

            buy_price_raw = price * buy_slippage_rate
            buy_commission = calculate_transaction_cost(buy_price_raw, shares, 'buy', stock_code)
            actual_buy_cost = buy_price_raw * shares + buy_commission
            buy_date = date
            position = 1
            peak = buy_price_raw

        # ========== 卖出逻辑 ==========
        elif position == 1:
            if price > peak:
                peak = price

            sell_reason = None
            # ===== 新增 T+1 限制 =====
            if buy_date is not None and date <= buy_date:
                # 如果当前日期 <= 买入日期，说明是同一天，禁止卖出
                continue  # 或者 pass，取决于你的循环结构，确保不执行卖出
            # =========================
            # AI 强烈看空
            if transformer_prob < 0.3:
                sell_reason = 'ai_bearish'

            unrealized_profit = (price - buy_price_raw) / buy_price_raw
            drawdown_from_peak = (peak - price) / peak if peak > 0 else 0.0

            # 止损
            if unrealized_profit <= p['stop_loss']:
                sell_reason = 'stop_loss'

            # 移动止损
            if sell_reason is None:
                if (unrealized_profit > p['trailing_profit_level2']
                        and drawdown_from_peak >= p['trailing_drawdown_level2']):
                    sell_reason = 'trailing'
                elif (unrealized_profit > p['trailing_profit_level1']
                      and drawdown_from_peak >= p['trailing_drawdown_level1']):
                    sell_reason = 'trailing'

            # 时间止损
            if sell_reason is None and (date - buy_date).days >= p['hold_days']:
                sell_reason = 'time_stop'

            # 信号衰减
            if sell_reason is None and score < p['sell_threshold']:
                sell_reason = 'signal_decay'

            # 动态止盈
            if sell_reason is None:
                atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else price * 0.02
                if unrealized_profit >= p['take_profit_multiplier'] * (atr / buy_price_raw):
                    sell_reason = 'take_profit'

            if sell_reason:
                if price <= df['limit_down'].iloc[i]:
                    continue

                sell_price_raw = price * sell_slippage_rate
                sell_commission_total = calculate_transaction_cost(sell_price_raw, shares, 'sell', stock_code)
                actual_sell_proceeds = sell_price_raw * shares - sell_commission_total
                net_ret = (actual_sell_proceeds - actual_buy_cost) / actual_buy_cost

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': date,
                    'net_return': net_ret,
                    'reason': sell_reason,
                    'shares': shares,
                    'signal_strength': signal_strength if has_pred_ret else 1.0,
                    'confidence': df['transformer_conf'].loc[buy_date] if has_transformer_conf else None,
                })
                position = 0

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
