# -*- coding: utf-8 -*-
"""
回测引擎（无Transformer版本）- 增强版 V2
修复：
B4+M1: position=1.0 固定满仓 → ATR 动态仓位
B3: get_settings() 递归 → 直接导入
B7: 缺少 T+1 限制 → 添加
新增：弱势市场仓位乘数、量价背离检测（与 engine.py 一致）
"""
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from data.types import get_limit_ratio, TRADITIONAL_FACTOR_COLS
from data.regime import get_market_regime_enhanced
from config import CommissionConfig, SlippageConfig, get_settings as _get_settings

logger = logging.getLogger(__name__)

# ==================== 缓存配置 ====================
_commission_cache = None
MAX_TRANSACTION_COST_RATIO = 0.20
MAX_RUNTIME_SLIPPAGE_RATE = 0.05


def _get_commission_cfg() -> CommissionConfig:
    global _commission_cache
    if _commission_cache is None:
        _commission_cache = CommissionConfig.from_env()
    return _commission_cache


# ==================== 滑点 ====================
_slippage_cache = SlippageConfig.from_env()
BUY_SLIPPAGE_RATE = float(np.clip(_slippage_cache.buy_slippage_rate, 0.0, MAX_RUNTIME_SLIPPAGE_RATE))
SELL_SLIPPAGE_RATE = float(np.clip(_slippage_cache.sell_slippage_rate, 0.0, MAX_RUNTIME_SLIPPAGE_RATE))


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
    comm_cfg = _get_commission_cfg()

    commission_rate = commission_rate if commission_rate is not None else comm_cfg.commission_rate
    min_commission = min_commission if min_commission is not None else comm_cfg.min_commission
    stamp_duty_rate = stamp_duty_rate if stamp_duty_rate is not None else comm_cfg.stamp_duty_rate
    transfer_fee_rate = transfer_fee_rate if transfer_fee_rate is not None else comm_cfg.transfer_fee_rate

    if shares <= 0 or price <= 0 or not np.isfinite(price):
        return 0.0

    trade_value = price * shares
    commission = max(trade_value * commission_rate, min_commission)
    stamp_duty = trade_value * stamp_duty_rate if direction == 'sell' else 0.0
    transfer_fee = trade_value * transfer_fee_rate if code.startswith('6') else 0.0

    total_cost = commission + stamp_duty + transfer_fee
    max_allowed = trade_value * MAX_TRANSACTION_COST_RATIO
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


def apply_slippage(price: float, direction: str, slippage_rate: Optional[float] = None) -> float:
    """应用滑点"""
    if slippage_rate is None:
        slippage_rate = BUY_SLIPPAGE_RATE if direction == 'buy' else SELL_SLIPPAGE_RATE
    slippage_rate = float(np.clip(slippage_rate, 0.0, MAX_RUNTIME_SLIPPAGE_RATE))
    if direction == 'buy':
        return price * (1 + slippage_rate)
    else:
        return price * (1 - slippage_rate)


def _check_volume_divergence(df: pd.DataFrame, idx: int, lookback: int = 10) -> bool:
    """量价顶背离检测"""
    if idx < lookback or 'Volume' not in df.columns:
        return False
    recent = df.iloc[idx - lookback:idx + 1]
    if len(recent) < 5:
        return False
    price_percentile = recent['Close'].rank(pct=True).iloc[-1]
    if price_percentile < 0.8:
        return False
    vol_first = recent['Volume'].iloc[:len(recent)//2].mean()
    vol_second = recent['Volume'].iloc[len(recent)//2:].mean()
    if vol_first <= 0:
        return False
    return vol_second / vol_first < 0.7


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
    """单股回测主循环（无Transformer版本）- 增强版 V2"""
    df = df.copy()

    if len(df) < 60:
        logger.warning(f"[{stock_code}] 数据长度 {len(df)} 不足以进行回测")
        return None, None, df

    factor_cols = [col for col in df.columns if col in TRADITIONAL_FACTOR_COLS]
    if len(factor_cols) == 0:
        logger.warning(f"[{stock_code}] 无有效因子列")
        return None, None, df

    logger.info(f"[{stock_code}] 开始回测，数据长度: {len(df)}，因子数: {len(factor_cols)}")

    # ========== 1. 计算综合得分 ==========
    df['score'] = 0.0

    valid_weights = {k: v for k, v in weights.items() if k in df.columns}

    numeric_factor_cols = []
    for col in valid_weights:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
                numeric_factor_cols.append(col)
            except Exception:
                valid_weights[col] = 0.0
        else:
            numeric_factor_cols.append(col)

    weight_sum = sum(valid_weights.values())
    if weight_sum == 0:
        # 等权重回退
        equal_weight = 1.0 / len(factor_cols) if factor_cols else 0
        valid_weights = {col: equal_weight for col in factor_cols if col in df.columns}
        weight_sum = sum(valid_weights.values())
        if weight_sum == 0:
            df['score'] = 0.5
        else:
            for col, w in valid_weights.items():
                df['score'] += df[col] * (w / weight_sum)
        logger.info(f"[{stock_code}] 权重为0，使用等权重回退")
    else:
        for col, w in valid_weights.items():
            if col not in numeric_factor_cols:
                continue
            df['score'] += df[col] * (w / weight_sum)

    # ========== 2. 涨跌停/ATR ==========
    limit_ratio = get_limit_ratio(stock_code)
    df['limit_up'] = df['Close'].shift(1) * (1 + limit_ratio)
    df['limit_down'] = df['Close'].shift(1) * (1 - limit_ratio)

    if 'atr' not in df.columns:
        try:
            import talib as ta
            df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        except Exception:
            df['atr'] = df['Close'].pct_change().rolling(14).std() * df['Close']

    # ========== 3. 信号生成 ==========
    df['signal'] = 'hold'
    df['position'] = 0.0

    buy_signal_count = 0
    sell_signal_count = 0

    # 持仓状态
    position = 0  # ★ 修复：用 0/1 表示是否持仓，shares 单独跟踪
    shares = 0
    buy_price_raw = 0.0
    buy_date = None
    actual_buy_cost = 0.0
    peak_price = 0.0
    trailing_stop_level = 0

    trades = []

    for i in range(60, len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        score = df['score'].iloc[i]

        # 增强版市场状态
        if regime:
            current_regime = regime
        else:
            regime_info = get_market_regime_enhanced(market_data, current_date)
            current_regime = regime_info.regime
            regime_position_mult = regime_info.position_multiplier

        regime_params = params.get(current_regime, params.get('neutral', {}))

        buy_threshold = regime_params.get('buy_threshold', 0.6)
        sell_threshold = regime_params.get('sell_threshold', -0.2)
        stop_loss = regime_params.get('stop_loss', -0.08)
        hold_days = regime_params.get('hold_days', 15)
        trailing_profit_l1 = regime_params.get('trailing_profit_level1', 0.06)
        trailing_profit_l2 = regime_params.get('trailing_profit_level2', 0.12)
        trailing_dd_l1 = regime_params.get('trailing_drawdown_level1', 0.08)
        trailing_dd_l2 = regime_params.get('trailing_drawdown_level2', 0.04)
        take_profit_mult = regime_params.get('take_profit_multiplier', 3.0)

        if position > 0:
            # ★ T+1 限制
            if buy_date is not None and current_date <= buy_date:
                if current_price > peak_price:
                    peak_price = current_price
                continue

            peak_price = max(peak_price, current_price)
            pnl_pct = (current_price - buy_price_raw) / buy_price_raw
            drawdown_from_peak = (peak_price - current_price) / peak_price

            # 移动止损逻辑
            if pnl_pct > trailing_profit_l2:
                trailing_stop_level = 2
                stop_price = peak_price * (1 - trailing_dd_l2)
            elif pnl_pct > trailing_profit_l1:
                trailing_stop_level = max(trailing_stop_level, 1)
                stop_price = peak_price * (1 - trailing_dd_l1)
            else:
                stop_price = buy_price_raw * (1 + stop_loss)

            # 卖出条件
            sell_signal = False
            sell_reason = ""

            # 止损
            if current_price <= stop_price:
                sell_signal = True
                sell_reason = "stop_loss"
            # 移动止损触发
            elif trailing_stop_level > 0 and current_price < peak_price * (
                    1 - (trailing_dd_l1 if trailing_stop_level == 1 else trailing_dd_l2)):
                sell_signal = True
                sell_reason = "trailing_stop"
            # 得分卖出
            elif score < sell_threshold:
                sell_signal = True
                sell_reason = "signal_decay"
            # 时间止损
            elif buy_date is not None:
                days_held = (current_date - buy_date).days
                if days_held >= hold_days and pnl_pct < 0.02:
                    sell_signal = True
                    sell_reason = "time_stop"
            # 动态止盈
            elif pnl_pct >= take_profit_mult * (df['atr'].iloc[i] / buy_price_raw) if not pd.isna(df['atr'].iloc[i]) else False:
                sell_signal = True
                sell_reason = "take_profit"
            # 量价顶背离
            elif _check_volume_divergence(df, i) and pnl_pct > 0.02:
                sell_signal = True
                sell_reason = "volume_divergence"

            if sell_signal:
                if current_price <= df['limit_down'].iloc[i]:
                    continue

                if actual_buy_cost <= 0 or not np.isfinite(actual_buy_cost):
                    logger.warning(
                        "[%s] invalid actual_buy_cost %.4f before sell, reset position",
                        stock_code,
                        actual_buy_cost,
                    )
                    position = 0
                    shares = 0
                    actual_buy_cost = 0.0
                    buy_date = None
                    continue

                sell_price = apply_slippage(current_price, 'sell')
                cost = calculate_transaction_cost(sell_price, shares, 'sell', stock_code)
                actual_sell_proceeds = sell_price * shares - cost
                net_ret = (actual_sell_proceeds - actual_buy_cost) / actual_buy_cost

                trades.append({
                    'buy_date': buy_date,
                    'sell_date': current_date,
                    'net_return': net_ret,
                    'reason': sell_reason,
                    'shares': shares,
                    'signal_strength': score,
                })

                position = 0
                shares = 0
                buy_price_raw = 0.0
                buy_date = None
                actual_buy_cost = 0.0
                peak_price = 0.0
                trailing_stop_level = 0
                sell_signal_count += 1
                df.iloc[i, df.columns.get_loc('signal')] = 'sell'

        else:
            # 空仓 — 买入逻辑
            if score > buy_threshold:
                # 弱势市场限制
                if current_regime == 'bear' and score < 0.85:
                    continue

                # 涨跌停检查
                if current_price >= df['limit_up'].iloc[i] * 0.995:
                    continue

                # 成交量过滤（放宽：1.2x）
                if 'Volume' in df.columns:
                    prev_vol = df['Volume'].iloc[i - 1]
                    vol_ma20 = df['Volume'].iloc[i - 20:i].mean()
                    if not pd.isna(vol_ma20) and prev_vol < vol_ma20 * 1.2:
                        continue

                # ★ ATR 动态仓位（修复 B4+M1）
                atr = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else current_price * 0.02
                daily_vol = atr / current_price
                if daily_vol <= 0 or not np.isfinite(daily_vol):
                    daily_vol = 0.02
                target_annual_vol = 0.10
                position_ratio = target_annual_vol / (daily_vol * np.sqrt(252) + 1e-6)
                position_ratio = min(max(position_ratio, 0.1), 1.0)

                # 市场状态仓位乘数
                if not regime:
                    position_ratio *= regime_position_mult
                position_ratio = min(max(position_ratio, 0.05), 1.0)

                # 计算股数
                try:
                    target_shares = int(initial_capital * position_ratio / current_price / 100) * 100
                    max_affordable_shares = int(initial_capital / current_price / 100) * 100
                except (ValueError, ZeroDivisionError):
                    continue

                if max_affordable_shares < 100:
                    logger.debug(
                        "[%s] skip buy at %s: price %.2f exceeds one-lot budget %.2f",
                        stock_code,
                        current_date,
                        current_price,
                        initial_capital,
                    )
                    continue

                shares = min(max(target_shares, 100), max_affordable_shares)
                if shares < 100:
                    continue

                buy_price_raw = apply_slippage(current_price, 'buy')
                buy_commission = calculate_transaction_cost(buy_price_raw, shares, 'buy', stock_code)
                actual_buy_cost = buy_price_raw * shares + buy_commission
                if actual_buy_cost <= 0 or not np.isfinite(actual_buy_cost):
                    logger.warning(
                        "[%s] invalid buy cost %.4f (shares=%s, buy_price=%.4f), skip",
                        stock_code,
                        actual_buy_cost,
                        shares,
                        buy_price_raw,
                    )
                    continue

                position = 1
                buy_date = current_date
                peak_price = current_price
                trailing_stop_level = 0
                buy_signal_count += 1
                df.iloc[i, df.columns.get_loc('signal')] = 'buy'

        df.iloc[i, df.columns.get_loc('position')] = position

    logger.info(f"[{stock_code}] 信号统计: 买入={buy_signal_count}, 卖出={sell_signal_count}")

    if len(trades) == 0:
        logger.warning(f"[{stock_code}] 无交易记录")
        return None, None, df

    trades_df = pd.DataFrame(trades)
    stats = {
        'total_trades': len(trades_df),
        'win_rate': (trades_df['net_return'] > 0).mean() * 100,
        'avg_return': trades_df['net_return'].mean() * 100,
        'total_return': ((trades_df['net_return'] + 1).prod() - 1) * 100,
    }

    logger.info(
        f"[{stock_code}] 回测完成: 总交易={len(trades_df)}, "
        f"胜率={stats.get('win_rate', 0):.1f}%, "
        f"总收益={stats.get('total_return', 0):.2f}%")

    return trades_df, stats, df


def calculate_backtest_stats(
        trades_df: pd.DataFrame,
        initial_capital: float = 100000.0,
) -> Dict:
    """计算回测统计指标"""
    if trades_df is None or len(trades_df) == 0:
        return {}

    stats = {}
    stats['total_trades'] = len(trades_df)
    stats['win_rate'] = (trades_df['net_return'] > 0).mean() * 100
    stats['avg_return'] = trades_df['net_return'].mean() * 100
    stats['total_return'] = ((trades_df['net_return'] + 1).prod() - 1) * 100

    # 最大回撤
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    for _, trade in trades_df.iterrows():
        equity *= (1 + trade['net_return'])
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)
    stats['max_drawdown'] = max_dd * 100

    # 夏普
    returns = trades_df['net_return']
    if len(returns) > 1 and returns.std() > 0:
        stats['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252 / 15)
    else:
        stats['sharpe_ratio'] = 0.0

    # 利润因子
    wins = trades_df[trades_df['net_return'] > 0]['net_return'].sum()
    losses = abs(trades_df[trades_df['net_return'] < 0]['net_return'].sum())
    stats['profit_factor'] = wins / losses if losses > 0 else 0.0

    return stats


def build_equity_curve(
        trades_df: Optional[pd.DataFrame],
        initial_capital: float,
        date_index: pd.DatetimeIndex,
) -> pd.Series:
    """构建资金曲线"""
    if trades_df is None or len(trades_df) == 0:
        return pd.Series(initial_capital, index=date_index)

    try:
        equity = pd.Series(initial_capital, index=date_index)
        for _, trade in trades_df.iterrows():
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            net_return = trade['net_return']
            if buy_date in equity.index and sell_date in equity.index:
                sell_idx = equity.index.get_loc(sell_date)
                equity.iloc[sell_idx:] = equity.iloc[sell_idx] * (1 + net_return)
        return equity
    except Exception as e:
        logger.error(f"构建资金曲线失败: {e}")
        return pd.Series(initial_capital, index=date_index)
