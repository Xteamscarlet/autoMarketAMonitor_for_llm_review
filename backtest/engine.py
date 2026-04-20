# -*- coding: utf-8 -*-
"""
回测引擎 V2 — 收益优化版
主要改进：
1. 缓存 CommissionConfig/SlippageConfig 避免循环内重复读取
2. 弱势市场不再一刀切阻断，而是降低仓位上限
3. 成交量过滤放宽（1.2x 替代 1.5x）
4. 新增量价背离检测作为卖出信号增强
5. 新增 RSI 超买超卖辅助判断
6. Walk-Forward 正确支持 6-tuple 划分
7. 改进止盈逻辑：分级止盈替代单纯乘数止盈
"""
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest.optimizer import calculate_dynamic_weights, build_factor_weights
from data.types import get_limit_ratio, NON_FACTOR_COLS
from data.regime import get_market_regime_enhanced, RegimeInfo
from risk_manager import RiskManager
from config import get_settings, CommissionConfig, SlippageConfig

logger = logging.getLogger(__name__)


# ==================== 模块级缓存 ====================
_commission_cfg: Optional[CommissionConfig] = None
_slippage_cfg: Optional[SlippageConfig] = None


def _get_commission_cfg() -> CommissionConfig:
    global _commission_cfg
    if _commission_cfg is None:
        _commission_cfg = CommissionConfig.from_env()
    return _commission_cfg


def _get_slippage_cfg() -> SlippageConfig:
    global _slippage_cfg
    if _slippage_cfg is None:
        _slippage_cfg = SlippageConfig.from_env()
    return _slippage_cfg


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
    """计算交易成本（佣金 + 印花税 + 过户费）"""
    cfg = _get_commission_cfg()
    if commission_rate is None:
        commission_rate = cfg.commission_rate
    if min_commission is None:
        min_commission = cfg.min_commission
    if stamp_duty_rate is None:
        stamp_duty_rate = cfg.stamp_duty_rate
    if transfer_fee_rate is None:
        transfer_fee_rate = cfg.transfer_fee_rate

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


def _compute_row_score(df: pd.DataFrame, idx: int, weights: Dict[str, float]) -> float:
    score = 0.5
    for factor, weight in weights.items():
        if factor not in df.columns or weight == 0:
            continue
        factor_value = df[factor].iloc[idx]
        if pd.isna(factor_value):
            continue
        score += (float(factor_value) - 0.5) * 2 * weight
    return float(score)


def _check_volume_divergence(df: pd.DataFrame, idx: int, lookback: int = 10) -> bool:
    """
    量价背离检测
    价格创新高但成交量递减 -> 顶背离（卖出信号）
    返回 True 表示存在顶背离
    """
    if idx < lookback or 'Volume' not in df.columns:
        return False

    recent = df.iloc[idx - lookback:idx + 1]
    prices = recent['Close']
    volumes = recent['Volume']

    if len(prices) < 5:
        return False

    # 价格是否在近期高点附近（近80%分位）
    price_percentile = (prices.rank(pct=True).iloc[-1])
    if price_percentile < 0.8:
        return False

    # 成交量是否递减
    vol_first_half = volumes.iloc[:len(volumes)//2].mean()
    vol_second_half = volumes.iloc[len(volumes)//2:].mean()

    if vol_first_half <= 0:
        return False

    # 量缩价涨 = 顶背离
    if vol_second_half / vol_first_half < 0.7:
        return True

    return False


def _check_rsi_extreme(df: pd.DataFrame, idx: int) -> Optional[str]:
    """RSI 极值检测，返回 'overbought' / 'oversold' / None"""
    if 'RSI' not in df.columns or idx < 14:
        return None

    rsi = df['RSI'].iloc[idx]
    if pd.isna(rsi):
        return None

    if rsi > 80:
        return 'overbought'
    elif rsi < 20:
        return 'oversold'
    return None


def _get_raw_transformer_pred_ret(df: pd.DataFrame, idx: int) -> float:
    """优先读取未压缩的收益率预测；兼容旧数据时再从 [0,1] 分数反推。"""
    if 'transformer_pred_ret_raw' in df.columns:
        val = df['transformer_pred_ret_raw'].iloc[idx]
        if pd.notna(val):
            return float(val)

    if 'transformer_pred_ret' in df.columns:
        val = df['transformer_pred_ret'].iloc[idx]
        if pd.notna(val):
            clipped = float(np.clip(val, 1e-6, 1 - 1e-6))
            return float(np.arctanh(2 * clipped - 1) / 10.0)

    return 0.0


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
    """回测引擎主循环 V2

    Args:
        df: 包含因子和价格数据的 DataFrame
        stock_code: 股票代码
        market_data: 大盘数据
        weights: 因子权重
        params: 策略参数 {regime: {param: value}}
        regime: 固定市场状态（None 则动态判断）
        stocks_data: 所有股票数据
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
    signal_strength = 1.0

    current_weights = weights.copy()
    last_weight_update = 60

    slip_cfg = _get_slippage_cfg()
    buy_slippage_rate = slip_cfg.buy_slippage_rate
    sell_slippage_rate = slip_cfg.sell_slippage_rate

    settings = get_settings()
    regime_cfg = settings.regime

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

        # 增强版市场状态判断
        if regime is None:
            regime_info = get_market_regime_enhanced(market_data, date)
            curr_regime = regime_info.regime
            regime_position_mult = regime_info.position_multiplier
        else:
            curr_regime = regime
            # 旧接口兼容
            regime_position_mult = {
                'strong_bull': 1.0, 'bull': 0.8,
                'neutral': 0.5, 'weak': 0.3, 'bear': 0.15
            }.get(regime, 0.5)

        p = params.get(curr_regime, params.get('neutral', params))
        effective_weights = build_factor_weights(
            df=df,
            base_weights=current_weights,
            transformer_weight=p.get('transformer_weight', 0.0),
        )
        score = _compute_row_score(df, i, effective_weights)

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

            # ★ 弱势市场不再一刀切阻断，而是限制仓位
            # bear 市场只允许极低仓位或跳过（除非得分极高）
            if curr_regime == 'bear' and score < 0.85:
                continue

            if has_transformer_conf:
                confidence = df['transformer_conf'].iloc[i]
                if confidence < 0.6:
                    continue

            if price >= df['limit_up'].iloc[i] * 0.995:
                continue

            # ★ 成交量过滤放宽：1.2x 替代 1.5x
            if 'Volume' in df.columns:
                prev_vol = df['Volume'].iloc[i - 1]
                vol_ma20_prev = df['Volume'].iloc[i - 20: i].mean()
                if not pd.isna(vol_ma20_prev) and prev_vol < vol_ma20_prev * 1.2:
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
                pred_ret = _get_raw_transformer_pred_ret(df, i)
                signal_strength = max(0.5, min(1.5, 1 + pred_ret / 0.05))
                position_ratio *= signal_strength

            # ★ 应用市场状态仓位乘数
            position_ratio *= regime_position_mult
            position_ratio = min(max(position_ratio, 0.05), 1.0)

            # RSI 超买区降低仓位
            rsi_status = _check_rsi_extreme(df, i)
            if rsi_status == 'overbought':
                position_ratio *= 0.5

            try:
                shares = max(100, int(initial_capital * position_ratio / price / 100) * 100)
                shares = min(shares, int(initial_capital / price / 100) * 100)
            except (ValueError, ZeroDivisionError):
                continue

            buy_price_raw = price * (1 + buy_slippage_rate)
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

            # T+1 限制
            if buy_date is not None and date <= buy_date:
                continue

            # AI 强烈看空
            if transformer_prob < p.get('transformer_sell_threshold', 0.3):
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
            if sell_reason is None and (date - buy_date).days >= p.get('hold_days', 15):
                sell_reason = 'time_stop'

            # 信号衰减
            if sell_reason is None and score < p.get('sell_threshold', -0.2):
                sell_reason = 'signal_decay'

            # ★ 量价顶背离检测 -> 额外卖出信号
            if sell_reason is None and _check_volume_divergence(df, i):
                if unrealized_profit > 0.02:  # 至少有2%利润才因背离卖出
                    sell_reason = 'volume_divergence'

            # ★ RSI 超卖区不卖出（可能反弹）
            # RSI 极度超卖时跳过部分卖出信号（保留止损和硬规则）
            rsi_status = _check_rsi_extreme(df, i)
            if sell_reason in ('signal_decay', 'time_stop') and rsi_status == 'oversold':
                sell_reason = None  # 超卖区暂缓衰减/时间止损

            # 动态止盈
            if sell_reason is None:
                atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else price * 0.02
                tp_mult = p.get('take_profit_multiplier', 3.0)
                if unrealized_profit >= tp_mult * (atr / buy_price_raw):
                    sell_reason = 'take_profit'

            # ★ 分级止盈（补充逻辑：盈利超过10%后，回撤超过3%即平仓一半）
            # 通过修改 hold_days / trailing 参数实现，这里用简化版
            if sell_reason is None and unrealized_profit > 0.10:
                if drawdown_from_peak > 0.03:
                    sell_reason = 'partial_profit_take'

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
                    'signal_strength': signal_strength if has_pred_ret else 1.0,
                    'confidence': df['transformer_conf'].loc[buy_date] if has_transformer_conf and buy_date in df['transformer_conf'].index else None,
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
