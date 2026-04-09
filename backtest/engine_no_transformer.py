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
    ★ 修复：缓存 CommissionConfig 避免循环内重复读取
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

    # 佣金（买卖都收）
    commission = max(price * shares * commission_rate, min_commission)

    # 印花税（仅卖出收取，千分之0.5）
    stamp_duty = price * shares * stamp_duty_rate if direction == 'sell' else 0

    # 过户费（沪市股票，买卖都收，十万分之一）
    transfer_fee = price * shares * transfer_fee_rate if code.startswith('6') else 0

    return commission + stamp_duty + transfer_fee


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
    buy_slippage_rate = _slippage_cfg.buy_slippage_rate
    sell_slippage_rate = _slippage_cfg.sell_slippage_rate

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
            # ★ T+1 限制：买入当天不能卖出
            if buy_date is not None and date <= buy_date:
                # 更新 peak 但不执行卖出
                if price > peak_price:
                    peak_price = price
                continue

            unrealized_profit = (price - buy_price_raw) / buy_price_raw
            drawdown_from_peak = (peak_price - price) / peak_price

            sell_reason = None

            # 止损
            if unrealized_profit <= p.get('stop_loss', -0.08):
                sell_reason = 'stop_loss'

            # 移动止损
            if sell_reason is None and unrealized_profit >= p.get('trailing_profit_level1', 0.06):
                if drawdown_from_peak >= p.get('trailing_drawdown_level1', 0.08):
                    sell_reason = 'trailing'

            if sell_reason is None and unrealized_profit >= p.get('trailing_profit_level2', 0.12):
                if drawdown_from_peak >= p.get('trailing_drawdown_level2', 0.04):
                    sell_reason = 'trailing'

            # 时间止损
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

# 在 engine.py 文件顶部导入新模块
import pandas as pd
from typing import List, Dict, Any, Optional
from .account import CashAccount
from .statistics import StrategyStatistics
# ... 其他原有导入 ...

class BacktestEngine:
    """
    修改后的回测引擎，使用 CashAccount 管理资金，并维护资金曲线。
    """
    def __init__(self, initial_capital: float, symbol: str, strategy_name: str, max_leverage: float = 1.0):
        # 初始化账户
        self.account = CashAccount(initial_capital)
        self.initial_capital = initial_capital
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.max_leverage = max_leverage  # 杠杆限制
        # 用于存储每笔交易的信号强度，用于后续分析
        self.signal_strengths = []
        # 用于存储模型预测收益（如果有）
        self.model_pred_returns = []

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        atr: pd.Series,
        transformer_pred_ret: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行回测主循环。
        :param data: 包含OHLCV的股票数据。
        :param signals: 交易信号Series，值为1（买入）、-1（卖出）、0（持有）。
        :param atr: ATR指标Series，用于仓位管理。
        :param transformer_pred_ret: 模型预测的收益率Series，用于信号强度调整。
        """
        # 重置账户（为了多次运行）
        self.account = CashAccount(self.initial_capital)
        self.signal_strengths = []
        self.model_pred_returns = []

        # 确保信号和数据索引对齐
        signals = signals.reindex(data.index).fillna(0)
        atr = atr.reindex(data.index)

        # 主循环
        position = 0  # 当前持仓股数
        for date, row in data.iterrows():
            # 1. 更新持仓市值（每日收盘价）
            if position > 0:
                self.account.update_position_value(position * row['close'])

            # 2. 处理交易信号
            signal = signals.loc[date]
            if signal == 0 and position == 0:
                continue  # 无信号且无持仓，跳过

            # 买入信号
            if signal == 1 and position == 0:
                # 计算仓位比例 (原逻辑，但增加了杠杆检查)
                daily_vol = atr.loc[date] / row['close']
                position_ratio = 0.1 / (daily_vol * np.sqrt(252) + 1e-6)  # 原始逻辑，目标年化波动率10%
                position_ratio = min(max(position_ratio, 0.1), 1.0)  # 限制在10%到100%之间

                # 应用杠杆限制
                position_ratio = min(position_ratio, self.max_leverage)

                # 计算购买股数
                max_shares_by_cash = int(self.account.cash / row['close'])  # 根据当前现金计算最大可买股数
                desired_shares = int(self.account.total_equity * position_ratio / row['close'])

                shares_to_buy = min(desired_shares, max_shares_by_cash)
                if shares_to_buy <= 0:
                    continue  # 现金不足一股，无法买入

                # 计算交易成本（你需要实现或引入成本计算函数）
                commission = self._calculate_commission(shares_to_buy * row['close'], 'buy')

                # 执行买入
                if self.account.buy(self.symbol, row['close'], shares_to_buy, commission, date):
                    position += shares_to_buy
                    # 记录信号强度
                    strength = 1.0  # 默认
                    if transformer_pred_ret is not None and date in transformer_pred_ret.index:
                        pred_ret = transformer_pred_ret.loc[date]
                        self.model_pred_returns.append(pred_ret)
                        # 原逻辑：根据预测收益调整信号强度
                        strength = max(0.5, min(1.5, 1 + pred_ret / 0.05))
                    self.signal_strengths.append(strength)

            # 卖出信号
            elif signal == -1 and position > 0:
                shares_to_sell = position
                commission = self._calculate_commission(shares_to_sell * row['close'], 'sell')
                self.account.sell(self.symbol, row['close'], shares_to_sell, commission, date)
                position = 0

        # 回测结束，如果还持仓，按最后一天收盘价清仓（模拟）
        if position > 0:
            last_date = data.index[-1]
            last_close = data.iloc[-1]['close']
            commission = self._calculate_commission(position * last_close, 'sell')
            self.account.sell(self.symbol, last_close, position, commission, last_date)
            position = 0

        # 生成统计结果
        equity_curve = self.account.get_equity_curve_series()
        stats_calc = StrategyStatistics(equity_curve)

        # 获取基础统计
        total_return = stats_calc.get_total_return()
        ann_return = stats_calc.get_annualized_return()
        ann_vol = stats_calc.get_annualized_volatility()
        sharpe = stats_calc.get_sharpe_ratio()
        max_dd, avg_dd, max_dd_duration = stats_calc.get_drawdown_stats()
        calmar = stats_calc.get_calmar_ratio()
        sortino = stats_calc.get_sortino_ratio()

        # 计算需要交易列表的指标
        profit_factor, win_rate, total_trades = self._calculate_trade_stats(self.account.transaction_history)

        results = {
            'strategy': self.strategy_name,
            'symbol': self.symbol,
            'initial_capital': self.initial_capital,
            'final_equity': equity_curve.iloc[-1],
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'max_drawdown_duration': max_dd_duration,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'equity_curve': equity_curve,  # 返回曲线用于绘图等
            'transactions': self.account.transaction_history,
            'status': 'COMPLETED'  # 可根据爆仓情况增加状态
        }
        return results

    def _calculate_commission(self, trade_value: float, action: str) -> float:
        """计算交易成本。你需要根据你的成本模型实现此方法。"""
        # 示例：万分之三的佣金，最低5元
        commission_rate = 0.0003
        min_commission = 5.0
        commission = max(trade_value * commission_rate, min_commission)
        # 如果是卖出，可能还需印花税等
        if action == 'sell':
            # 例如，千分之一的印花税
            stamp_duty = trade_value * 0.001
            commission += stamp_duty
        return commission

    def _calculate_trade_stats(self, transactions: List[Dict]) -> Tuple[float, float, int]:
        """从交易列表计算利润因子和胜率。"""
        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        total_trades = 0
        # 遍历交易对（买入后对应的卖出）
        # 注意：这里假设交易是配对的。需要更复杂的逻辑来处理未平仓交易。
        i = 0
        while i < len(transactions) - 1:
            buy_tx = transactions[i]
            sell_tx = transactions[i+1]
            if buy_tx['action'] == 'BUY' and sell_tx['action'] == 'SELL' and buy_tx['symbol'] == sell_tx['symbol']:
                # 计算这笔交易的盈亏
                buy_cost = buy_tx['price'] * buy_tx['shares'] + buy_tx['commission']
                sell_revenue = sell_tx['price'] * sell_tx['shares'] - sell_tx['commission']
                pnl = sell_revenue - buy_cost
                if pnl > 0:
                    gross_profit += pnl
                    wins += 1
                else:
                    gross_loss += abs(pnl)
                total_trades += 1
                i += 2  # 跳过已处理的交易对
            else:
                i += 1

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        return profit_factor, win_rate, total_trades
