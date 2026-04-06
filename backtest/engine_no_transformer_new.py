# -*- coding: utf-8 -*-
"""
回测引擎（无Transformer版本）- 增强版
执行单股回测循环，处理买卖信号、仓位管理、交易成本
移除了所有 Transformer 相关的买入过滤条件
新增：信号生成日志、滑点计算修正、资金曲线健壮性
"""
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest.optimizer import calculate_dynamic_weights
from data.types import get_limit_ratio, TRADITIONAL_FACTOR_COLS
from data.indicators_no_transformer import get_market_regime
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

from config import CommissionConfig, SlippageConfig

# ==================== 新增：滑点配置常量（明确定义） ====================
# 买入滑点：买入价 = 当前价 * (1 + BUY_SLIPPAGE_RATE)
# 例如：BUY_SLIPPAGE_RATE = 0.0015 表示买入时价格上涨0.15%
BUY_SLIPPAGE_RATE = 0.0015  # 买入滑点率

# 卖出滑点：卖出价 = 当前价 * (1 - SELL_SLIPPAGE_RATE)
# 例如：SELL_SLIPPAGE_RATE = 0.0015 表示卖出时价格下跌0.15%
SELL_SLIPPAGE_RATE = 0.0015  # 卖出滑点率


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
    计算交易成本（佣金 + 印花税 + 过户费）- 增强版
    未传入的费率参数使用 config.py 中的全局配置。

    Args:
        price: 交易价格
        shares: 交易股数
        direction: 'buy' 或 'sell'
        code: 股票代码
        commission_rate: 佣金费率
        min_commission: 最低佣金
        stamp_duty_rate: 印花税率（仅卖出）
        transfer_fee_rate: 过户费率

    Returns:
        总交易成本
    """
    settings = get_settings()
    comm_cfg: CommissionConfig = settings.commission

    # 使用传入参数或配置默认值
    commission_rate = commission_rate if commission_rate is not None else comm_cfg.commission_rate
    min_commission = min_commission if min_commission is not None else comm_cfg.min_commission
    stamp_duty_rate = stamp_duty_rate if stamp_duty_rate is not None else comm_cfg.stamp_duty_rate
    transfer_fee_rate = transfer_fee_rate if transfer_fee_rate is not None else comm_cfg.transfer_fee_rate

    trade_value = price * shares

    # 佣金（双向）
    commission = max(trade_value * commission_rate, min_commission)

    # 印花税（仅卖出）
    stamp_duty = trade_value * stamp_duty_rate if direction == 'sell' else 0.0

    # 过户费（双向，仅沪市）
    if code.startswith('6'):
        transfer_fee = trade_value * transfer_fee_rate
    else:
        transfer_fee = 0.0

    return commission + stamp_duty + transfer_fee


def apply_slippage(
        price: float,
        direction: str,
        slippage_rate: Optional[float] = None,
) -> float:
    """
    应用滑点 - 修正版

    Args:
        price: 原始价格
        direction: 'buy' 或 'sell'
        slippage_rate: 滑点率（可选，默认使用配置）

    Returns:
        调整后的价格
    """
    if slippage_rate is None:
        slippage_rate = BUY_SLIPPAGE_RATE if direction == 'buy' else SELL_SLIPPAGE_RATE

    if direction == 'buy':
        # 买入时价格上涨
        return price * (1 + slippage_rate)
    else:
        # 卖出时价格下跌
        return price * (1 - slippage_rate)


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
    """
    单股回测主循环（无Transformer版本）- 增强版

    Args:
        df: 包含所有因子列的 DataFrame
        stock_code: 股票代码
        market_data: 大盘数据（用于判断市场状态）
        weights: 因子权重字典
        params: 各市场状态下的策略参数
        regime: 强制指定的市场状态（None则自动判断）
        stocks_data: 其他股票数据（用于多周期评分）
        initial_capital: 初始资金

    Returns:
        (trades_df, stats, df)
        trades_df: 交易记录 DataFrame，无交易时返回 None
        stats: 统计指标字典，无交易时返回 None
        df: 添加了信号列的原始 DataFrame
    """
    df = df.copy()

    # 新增：数据长度检查
    if len(df) < 60:
        logger.warning(f"[{stock_code}] 数据长度 {len(df)} 不足以进行回测")
        return None, None, df

    # 新增：因子列检查
    factor_cols = [col for col in df.columns if col in TRADITIONAL_FACTOR_COLS ]
    if len(factor_cols) == 0:
        logger.warning(f"[{stock_code}] 无有效因子列")
        return None, None, df

    # 新增：日志记录
    logger.info(f"[{stock_code}] 开始回测，数据长度: {len(df)}，因子数: {len(factor_cols)}")

    # ========== 1. 计算综合得分 ==========
    df['score'] = 0.0
    valid_weights = {k: v for k, v in weights.items() if k in df.columns}
    # --- 新增：过滤掉非数值因子，避免字符串列参与运算 ---
    numeric_factor_cols = []
    for col in valid_weights:
        if col not in df.columns:
            continue
        # 如果是 object / string 类型，尝试转成 float，转不了就丢弃
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
                numeric_factor_cols.append(col)
            except Exception:
                logger.warning(
                    f"[{stock_code}] 因子列 {col} 为非数值类型，无法转换为 float，已剔除"
                )
                # 可选：把该因子权重置为 0，避免影响权重归一化
                valid_weights[col] = 0.0
        else:
            numeric_factor_cols.append(col)

    weight_sum = sum(valid_weights.values())
    if weight_sum <= 0:
        logger.warning(f"[{stock_code}] 所有权重为 0，回测终止")
        return None, None, df

    for col, w in valid_weights.items():
        # 只用数值列计算 score
        if col not in numeric_factor_cols:
            continue
        df['score'] += df[col] * (w / weight_sum)
    if len(valid_weights) == 0:
        logger.warning(f"[{stock_code}] 无有效权重，使用等权重")
        valid_weights = {col: 1.0 / len(factor_cols) for col in factor_cols}

    weight_sum = sum(valid_weights.values())
    for col, w in valid_weights.items():
        df['score'] += df[col] * (w / weight_sum)

    # ========== 2. 信号生成 ==========
    df['signal'] = 'hold'
    df['position'] = 0.0

    # 新增：信号统计
    buy_signal_count = 0
    sell_signal_count = 0

    # 获取参数
    regime_params = params.get(regime, params.get('neutral', {}))
    buy_threshold = regime_params.get('buy_threshold', 0.6)
    sell_threshold = regime_params.get('sell_threshold', -0.2)
    stop_loss = regime_params.get('stop_loss', -0.08)
    hold_days = regime_params.get('hold_days', 15)

    # 新增：参数日志
    logger.debug(
        f"[{stock_code}] 参数: buy_threshold={buy_threshold}, sell_threshold={sell_threshold}, stop_loss={stop_loss}")

    # 持仓状态
    position = 0.0
    buy_price = 0.0
    buy_date = None
    peak_price = 0.0
    trailing_stop_level = 0

    trades = []

    for i in range(20, len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        score = df['score'].iloc[i]

        # 获取市场状态
        current_regime = regime or get_market_regime(market_data, current_date)
        regime_params = params.get(current_regime, params.get('neutral', {}))

        # 更新参数
        buy_threshold = regime_params.get('buy_threshold', 0.6)
        sell_threshold = regime_params.get('sell_threshold', -0.2)
        stop_loss = regime_params.get('stop_loss', -0.08)
        hold_days = regime_params.get('hold_days', 15)
        trailing_profit_l1 = regime_params.get('trailing_profit_level1', 0.06)
        trailing_profit_l2 = regime_params.get('trailing_profit_level2', 0.12)
        trailing_dd_l1 = regime_params.get('trailing_drawdown_level1', 0.08)
        trailing_dd_l2 = regime_params.get('trailing_drawdown_level2', 0.04)

        if position > 0:
            # 持仓中
            peak_price = max(peak_price, current_price)
            pnl_pct = (current_price - buy_price) / buy_price

            # 移动止损逻辑
            if pnl_pct > trailing_profit_l2:
                trailing_stop_level = 2
                stop_price = peak_price * (1 - trailing_dd_l2)
            elif pnl_pct > trailing_profit_l1:
                trailing_stop_level = max(trailing_stop_level, 1)
                stop_price = peak_price * (1 - trailing_dd_l1)
            else:
                stop_price = buy_price * (1 + stop_loss)

            # 卖出条件
            sell_signal = False
            sell_reason = ""

            # 止损
            if current_price <= stop_price:
                sell_signal = True
                sell_reason = "stop_loss"
            # 止盈（移动止损触发）
            elif trailing_stop_level > 0 and current_price < peak_price * (
                    1 - (trailing_dd_l1 if trailing_stop_level == 1 else trailing_dd_l2)):
                sell_signal = True
                sell_reason = "trailing_stop"
            # 得分卖出
            elif score < sell_threshold:
                sell_signal = True
                sell_reason = "score_sell"
            # 时间止损
            elif buy_date is not None:
                days_held = (current_date - buy_date).days
                if days_held >= hold_days and pnl_pct < 0.02:
                    sell_signal = True
                    sell_reason = "time_stop"

            if sell_signal:
                # 新增：应用正确的滑点
                sell_price = apply_slippage(current_price, 'sell')

                # 计算交易成本
                cost = calculate_transaction_cost(sell_price, int(position * 100), 'sell', stock_code)

                # 记录交易
                trades.append({
                    'buy_date': buy_date,
                    'sell_date': current_date,
                    'buy_price': apply_slippage(buy_price, 'buy'),
                    'sell_price': sell_price,
                    'shares': int(position * 100),
                    'pnl_pct': pnl_pct,
                    'cost': cost,
                    'reason': sell_reason,
                })

                position = 0.0
                buy_price = 0.0
                buy_date = None
                peak_price = 0.0
                trailing_stop_level = 0
                sell_signal_count += 1
                df.iloc[i, df.columns.get_loc('signal')] = 'sell'

        else:
            # 空仓
            if score > buy_threshold:
                # 买入
                position = 1.0  # 满仓
                buy_price = current_price
                buy_date = current_date
                peak_price = current_price
                trailing_stop_level = 0
                buy_signal_count += 1
                df.iloc[i, df.columns.get_loc('signal')] = 'buy'

        df.iloc[i, df.columns.get_loc('position')] = position

    # 新增：信号统计日志
    logger.info(f"[{stock_code}] 信号统计: 买入={buy_signal_count}, 卖出={sell_signal_count}")

    # ========== 3. 构建交易记录 ==========
    if len(trades) == 0:
        logger.warning(f"[{stock_code}] 无交易记录")
        return None, None, df

    trades_df = pd.DataFrame(trades)

    # 计算净收益
    trades_df['net_return'] = trades_df['pnl_pct'] - trades_df['cost'] / (trades_df['buy_price'] * trades_df['shares'])

    # ========== 4. 计算统计指标 ==========
    stats = calculate_backtest_stats(trades_df, initial_capital)

    logger.info(
        f"[{stock_code}] 回测完成: 总交易={len(trades_df)}, 胜率={stats.get('win_rate', 0):.1f}%, 夏普={stats.get('sharpe_ratio', 0):.2f}")

    return trades_df, stats, df


def calculate_backtest_stats(
        trades_df: pd.DataFrame,
        initial_capital: float = 100000.0,
) -> Dict:
    """计算回测统计指标 - 增强版

    新增：边界检查、NaN处理
    """
    if trades_df is None or len(trades_df) == 0:
        return {}

    stats = {}

    # 基本统计
    stats['total_trades'] = len(trades_df)
    stats['win_rate'] = (trades_df['net_return'] > 0).mean() * 100 if len(trades_df) > 0 else 0
    stats['avg_return'] = trades_df['net_return'].mean() * 100 if len(trades_df) > 0 else 0

    # 总收益
    stats['total_return'] = ((trades_df['net_return'] + 1).prod() - 1) * 100 if len(trades_df) > 0 else 0

    # 最大回撤
    equity_curve = initial_capital
    peak = equity_curve
    max_dd = 0.0

    for _, trade in trades_df.iterrows():
        equity_curve *= (1 + trade['net_return'])
        peak = max(peak, equity_curve)
        dd = (equity_curve - peak) / peak
        max_dd = min(max_dd, dd)

    stats['max_drawdown'] = max_dd * 100

    # 夏普比率 - 安全计算
    returns = trades_df['net_return']
    if len(returns) > 1 and returns.std() > 0:
        stats['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252 / 15)  # 假设平均持仓15天
    else:
        stats['sharpe_ratio'] = 0.0

    # 利润因子 - 安全计算
    wins = trades_df[trades_df['net_return'] > 0]['net_return'].sum()
    losses = abs(trades_df[trades_df['net_return'] < 0]['net_return'].sum())
    stats['profit_factor'] = wins / losses if losses > 0 else 0.0

    return stats


def calculate_multi_timeframe_score_no_transformer(
        df: pd.DataFrame,
        stock_code: str,
        stocks_data: Optional[Dict[str, pd.DataFrame]],
        weights: Dict[str, float],
        current_date,
) -> float:
    """
    多周期评分（无Transformer版本）- 增强版

    Args:
        df: 当前股票数据
        stock_code: 股票代码
        stocks_data: 其他股票数据字典
        weights: 因子权重
        current_date: 当前日期

    Returns:
        综合得分
    """
    if current_date not in df.index:
        return 0.5

    try:
        # 基础得分
        factor_cols = [col for col in df.columns if col in TRADITIONAL_FACTOR_COLS]
        score = 0.0
        weight_sum = 0.0

        for col in factor_cols:
            if col in df.columns and col in weights:
                val = df.loc[current_date, col]
                if pd.notna(val):
                    score += val * weights[col]
                    weight_sum += weights[col]

        if weight_sum > 0:
            score /= weight_sum
        else:
            score = 0.5

        return np.clip(score, 0, 1)

    except Exception as e:
        logger.warning(f"[{stock_code}] 多周期评分计算失败: {e}")
        return 0.5


def build_equity_curve(
        trades_df: Optional[pd.DataFrame],
        initial_capital: float,
        date_index: pd.DatetimeIndex,
) -> pd.Series:
    """构建资金曲线 - 增强版

    处理交易记录为空的情况，返回初始资金序列

    Args:
        trades_df: 交易记录DataFrame
        initial_capital: 初始资金
        date_index: 日期索引

    Returns:
        资金曲线Series
    """
    # 新增：处理交易记录为空的情况
    if trades_df is None or len(trades_df) == 0:
        logger.warning("交易记录为空，返回初始资金序列")
        return pd.Series(initial_capital, index=date_index)

    try:
        # 构建资金曲线
        equity = pd.Series(initial_capital, index=date_index)

        for _, trade in trades_df.iterrows():
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            net_return = trade['net_return']

            if buy_date in equity.index and sell_date in equity.index:
                # 更新卖出日及之后的资金
                sell_idx = equity.index.get_loc(sell_date)
                equity.iloc[sell_idx:] = equity.iloc[sell_idx] * (1 + net_return)

        return equity

    except Exception as e:
        logger.error(f"构建资金曲线失败: {e}")
        return pd.Series(initial_capital, index=date_index)


# 新增：获取配置
def get_settings():
    from config import get_settings
    return get_settings()
