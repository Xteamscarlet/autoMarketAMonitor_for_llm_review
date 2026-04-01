# -*- coding: utf-8 -*-
"""
回测引擎
执行单股回测循环，处理买卖信号、仓位管理、交易成本
从 backtest_market_v2.py 的 run_backtest_loop() 提取
"""
import logging
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from backtest.optimizer import calculate_dynamic_weights
from data.types import get_limit_ratio, NON_FACTOR_COLS
from data.indicators import get_market_regime
from risk_manager import apply_vectorized_risk_controls  # 步骤二新增的函数
from numba import njit

logger = logging.getLogger(__name__)


from config import CommissionConfig,SlippageConfig  # 在文件顶部导入

from data.indicators import get_market_regime  # 你原项目已有的

def build_regime_param_array(
    df: pd.DataFrame,
    market_data: pd.DataFrame | None,
    regime_fixed: str | None,
    params: dict[str, dict],
) -> pd.DataFrame:
    """
    给 df 新增列：
      - curr_regime: str（按 index 动态判断，或使用 regime_fixed）
      - param_dict:  逐行保存当天的参数字典（对象）
    注意：Series 里存 dict 会有一些开销，但回测本身很快；如果你在意可以改成只存需要的标量字段。
    """
    df = df.copy()
    dates = pd.to_datetime(df.index)

    # 1. 动态 regime
    if regime_fixed is not None:
        df["curr_regime"] = regime_fixed
    elif market_data is not None:
        # 这里用 .map 是为了尽量向量化；如果 get_market_regime 本身很慢，
        # 可以考虑先把 market_data 离散化到日级别再做 join
        df["curr_regime"] = dates.map(lambda d: get_market_regime(market_data, d))
    else:
        df["curr_regime"] = "neutral"

    # 2. 按 regime 取参数（和你逐行逻辑一致：regime -> neutral -> 整个 params）
    df["param_dict"] = df["curr_regime"].apply(
        lambda r: params.get(r, params.get("neutral", params))
    )

    return df


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
    use_vectorized_backtest: bool = False,  # 新增参数（默认 False）
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
    if use_vectorized_backtest:
        # 走新的向量化回测
        return run_backtest_loop_compatible(
            df=df,
            stock_code=stock_code,
            market_data=market_data,
            weights=weights,
            params=params,
            regime=regime,
            stocks_data=stocks_data,
            initial_capital=initial_capital,
            use_vectorized_backtest=True,
        )

    # ========== 以下是原有的逐行回测逻辑（保持不变） ==========
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
                    'confidence': df['transformer_conf'].iloc[buy_date] if has_transformer_conf else None,
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

# engine.py 内新增

_commission_cfg_cache = None
_slippage_cfg_cache = None


def _get_commission_cfg() -> CommissionConfig:
    global _commission_cfg_cache
    if _commission_cfg_cache is None:
        _commission_cfg_cache = CommissionConfig.from_env()
    return _commission_cfg_cache


def _get_slippage_cfg() -> SlippageConfig:
    global _slippage_cfg_cache
    if _slippage_cfg_cache is None:
        _slippage_cfg_cache = SlippageConfig.from_env()
    return _slippage_cfg_cache

# engine.py 新增

def generate_order_array(
    target_pos_ratio: np.ndarray,   # (n,) 目标仓位比例序列（0~1）
    current_pos_ratio: np.ndarray,  # (n,) 当前仓位比例序列（通常 0）
) -> np.ndarray:
    """
    计算每一期的“目标交易比例”（可正可负）。
    - trade_ratio > 0：加仓；< 0：减仓；≈0：不操作。
    - 无 Numba 依赖，纯 Numpy 向量化。

    注意：current_pos_ratio 在大多数场景下是全 0（初始空仓），
    若你希望保留“已有仓位”的场景，可以在外层把上期实际仓位填入即可。
    """
    # 安全转成 float64（Numba 更喜欢）
    target = np.asarray(target_pos_ratio, dtype=np.float64)
    current = np.asarray(current_pos_ratio, dtype=np.float64)

    # 截断到 [0, 1]
    target = np.clip(target, 0.0, 1.0)
    current = np.clip(current, 0.0, 1.0)

    return target - current  # (n,)

# engine.py 新增

def calculate_transaction_cost_vectorized(
    prices: np.ndarray,            # (n,) 成交价序列（通常为收盘价）
    trade_shares: np.ndarray,      # (n,) 实际交易股数（可正可负）
    direction: np.ndarray,         # (n,) 方向：1=买入，-1=卖出
    commission_rate: float,
    min_commission: float,
    stamp_duty_rate: float,
    transfer_fee_rate: float,
) -> np.ndarray:
    """
    向量化交易成本（与 calculate_transaction_cost 保持逻辑一致）：
    - 佣金：max(amount * commission_rate, min_commission)
    - 印花税：amount * stamp_duty_rate（仅卖出时生效）
    - 过户费：amount * transfer_fee_rate（仅卖出时生效）

    参数：
    - prices: (n,) 成交价序列（通常使用 close）
    - trade_shares: (n,) 实际成交股数（带符号：买入为正，卖出为负）
    - direction: (n,) 方向标识：1=买入，-1=卖出
    - commission_rate, min_commission, stamp_duty_rate, transfer_fee_rate：费率参数

    返回：
    - costs: (n,) 每期交易成本（非负）
    """
    prices = np.asarray(prices, dtype=np.float64)
    trade_shares = np.asarray(trade_shares, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    # 成交金额（带符号，与 trade_shares 同号）
    amounts = prices * trade_shares  # (n,)

    # 佣金：max(|amount| * commission_rate, min_commission)
    commission = np.maximum(np.abs(amounts) * commission_rate, min_commission)

    # 印花税 & 过户费：仅卖出（direction < 0）时收取
    is_sell = direction < 0
    stamp_duty = np.where(is_sell, np.abs(amounts) * stamp_duty_rate, 0.0)
    transfer_fee = np.where(is_sell, np.abs(amounts) * transfer_fee_rate, 0.0)

    return commission + stamp_duty + transfer_fee  # (n,)

# engine.py 新增

def calculate_transaction_cost_vectorized(
    prices: np.ndarray,            # (n,) 成交价序列（通常为收盘价）
    trade_shares: np.ndarray,      # (n,) 实际交易股数（可正可负）
    direction: np.ndarray,         # (n,) 方向：1=买入，-1=卖出
    commission_rate: float,
    min_commission: float,
    stamp_duty_rate: float,
    transfer_fee_rate: float,
) -> np.ndarray:
    """
    向量化交易成本（与 calculate_transaction_cost 保持逻辑一致）：
    - 佣金：max(amount * commission_rate, min_commission)
    - 印花税：amount * stamp_duty_rate（仅卖出时生效）
    - 过户费：amount * transfer_fee_rate（仅卖出时生效）

    参数：
    - prices: (n,) 成交价序列（通常使用 close）
    - trade_shares: (n,) 实际成交股数（带符号：买入为正，卖出为负）
    - direction: (n,) 方向标识：1=买入，-1=卖出
    - commission_rate, min_commission, stamp_duty_rate, transfer_fee_rate：费率参数

    返回：
    - costs: (n,) 每期交易成本（非负）
    """
    prices = np.asarray(prices, dtype=np.float64)
    trade_shares = np.asarray(trade_shares, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    # 成交金额（带符号，与 trade_shares 同号）
    amounts = prices * trade_shares  # (n,)

    # 佣金：max(|amount| * commission_rate, min_commission)
    commission = np.maximum(np.abs(amounts) * commission_rate, min_commission)

    # 印花税 & 过户费：仅卖出（direction < 0）时收取
    is_sell = direction < 0
    stamp_duty = np.where(is_sell, np.abs(amounts) * stamp_duty_rate, 0.0)
    transfer_fee = np.where(is_sell, np.abs(amounts) * transfer_fee_rate, 0.0)

    return commission + stamp_duty + transfer_fee  # (n,)

# engine.py 新增

@njit
def run_numba_backtest_engine(
    close_prices: np.ndarray,      # (n,) 收盘价序列
    trade_ratios: np.ndarray,      # (n,) 每期目标交易比例（可正可负）
    cost_array: np.ndarray,        # (n,) 每期交易成本（非负）
    initial_cash: float,
    is_buy: np.ndarray,            # (n,) 布尔数组：True=本期净买入，False=净卖出/不操作
    # 可选：滑点与最小交易单位
    slippage_bps: float = 0.0,     # 滑点（基点）；0 表示无滑点
    min_trade_value: float = 0.0,  # 最小成交金额（0 表示不限制）
) -> tuple:
    """
    Numba JIT 状态机：逐日更新 cash/持仓/权益。
    - 输入全为 Numpy 数组，返回四条曲线（与输入等长）。
    - slippage_bps：基点滑点（如 15=0.15%）。在买入时向上、卖出时向下打滑。
    - min_trade_value：单笔最小成交金额，低于此值视为不成交。

    注意：
    - 此函数虽然看起来像 for 循环，但被 @njit 编译为机器码，几万行数据几毫秒级别完成。
    """
    n = len(close_prices)
    # 预分配输出数组
    out_cash = np.empty(n, dtype=np.float64)
    out_pos_shares = np.empty(n, dtype=np.float64)
    out_equity = np.empty(n, dtype=np.float64)
    out_actual_trade_shares = np.empty(n, dtype=np.float64)

    cash = initial_cash
    pos_shares = 0.0

    # 滑点因子（1 + slippage_bps/10000）
    slippage_factor = 1.0 + slippage_bps / 10000.0

    for i in range(n):
        price = close_prices[i]
        trade_ratio = trade_ratios[i]
        cost = cost_array[i]
        buy_flag = is_buy[i]

        # 1) 当前总权益
        equity = cash + pos_shares * price

        # 2) 目标交易金额（基于权益）
        target_value = equity * trade_ratio       # 带符号：>0 买入，<0 卖出
        if not buy_flag and target_value > 0.0:
            # 若本应卖出（由 trade_ratio 符号判断），但方向标志为非买入，
            # 此时可能因为资金不足导致目标由买转卖。直接取目标值的负数作为卖出。
            target_value = -target_value

        # 3) 拆分为股数（考虑滑点）
        if buy_flag:
            # 买入：价格向上滑
            exec_price = price * slippage_factor
        else:
            # 卖出：价格向下滑
            exec_price = price / slippage_factor

        # 避免除零
        if exec_price <= 0.0:
            exec_price = 1e-12

        delta_shares = target_value / exec_price  # 带符号的股数差

        # 4) 若为买入方向，检查可用资金
        if buy_flag:
            # 需要的额外现金 = 买入金额 + 成本
            required_cash = delta_shares * exec_price + cost
            if required_cash > cash:
                # 资金不足，按最大可买调整
                # cash >= delta_shares * exec_price + cost
                # → delta_shares <= (cash - cost) / exec_price
                if exec_price > 0.0:
                    delta_shares = (cash - cost) / exec_price
                else:
                    delta_shares = 0.0

                # 确保不为负
                if delta_shares < 0.0:
                    delta_shares = 0.0

                # 再次计算对应的成本（简化处理：仍用原始 cost）
                # 严格做法是在循环外再跑一次成本函数；此处为性能折衷
        # 若为卖出方向，理论上不受资金约束，但需要确保不会卖出超过持仓
        else:
            # delta_shares < 0
            if -delta_shares > pos_shares:
                # 最多清仓
                delta_shares = -pos_shares

        # 5) 最小成交金额过滤（可选）
        trade_value = abs(delta_shares * exec_price)
        if min_trade_value > 0.0 and trade_value < min_trade_value:
            delta_shares = 0.0

        # 6) 更新持仓与现金
        cash -= delta_shares * exec_price + cost
        pos_shares += delta_shares

        # 7) 记录到输出数组
        out_cash[i] = cash
        out_pos_shares[i] = pos_shares
        out_equity[i] = cash + pos_shares * price
        out_actual_trade_shares[i] = delta_shares

    return (out_cash, out_pos_shares, out_equity, out_actual_trade_shares)

# engine.py 新增

def run_backtest_loop_vectorized(
    df: pd.DataFrame,
    strategy_instances: list,
    risk_cfg: Optional[Any] = None,           # RiskConfig
    commission_cfg: Optional[CommissionConfig] = None,
    slippage_cfg: Optional[SlippageConfig] = None,
    initial_cash: float = 1_000_000.0,
    slippage_bps: float = 0.0,
    min_trade_value: float = 0.0,
    # 可选：是否使用基类提供的 generate_signals_vectorized（若策略未覆写则逐行回退）
    use_vectorized_signals: bool = True,
) -> pd.DataFrame:
    """
    向量化回测主调度（步骤 4 的具体实现）：
    - 步骤 1：调用 strategy.generate_signals_vectorized()（逐行回退亦可）；
    - 步骤 2：调用 risk_manager.apply_vectorized_risk_controls()；
    - 步骤 3：Numpy 准备数组 → 订单数组 → 成本数组；
    - 步骤 4：调用 Numba JIT 状态机；
    - 步骤 5：将结果写回 df 并返回。

    说明：
    - 保持与原 run_backtest_loop 类似的输入输出（返回 df，带 equity_curve/backtest_cash/backtest_position 列）。
    - 建议在 run_backtest.py 中根据开关选择是否使用此函数。
    """
    out_df = df.copy()

    # ---------------------------
    # 0) 配置对象
    # ---------------------------
    if commission_cfg is None:
        commission_cfg = _get_commission_cfg()
    if slippage_cfg is None:
        slippage_cfg = _get_slippage_cfg()

    # ---------------------------
    # 1) 策略信号向量化
    # ---------------------------
    if use_vectorized_signals:
        for strategy in strategy_instances:
            # 直接调用 generate_signals_vectorized（基类默认实现会逐行回退）
            out_df = strategy.generate_signals_vectorized(
                out_df,
                params=getattr(strategy, "params", None),
            )
            # 若你有多策略融合逻辑，可在此处对 target_position_ratio 做加权平均/投票等

    # ---------------------------
    # 2) 风控向量化过滤
    # ---------------------------
    out_df = apply_vectorized_risk_controls(out_df, risk_cfg=risk_cfg)

    # ---------------------------
    # 3) 准备 Numpy 数组
    # ---------------------------
    # 确保 target_position_ratio 列存在且合法
    if "target_position_ratio" not in out_df.columns:
        raise KeyError(
            "apply_vectorized_risk_controls 必须在 df 中生成 'target_position_ratio' 列。"
            "请检查步骤一/二是否正确执行。"
        )

    target_ratio = out_df["target_position_ratio"].to_numpy(dtype=np.float64)
    n = len(target_ratio)

    # 使用收盘价作为成交价（根据需要可以切换为 open/next_open）
    close_col = "close" if "close" in out_df.columns else "Close"
    close = out_df[close_col].to_numpy(dtype=np.float64)

    # 初始仓位比率为 0（空仓）
    current_ratio = np.zeros(n, dtype=np.float64)

    # ---------------------------
    # 4) 订单数组（向量化）
    # ---------------------------
    trade_ratios = generate_order_array(target_ratio, current_ratio)  # (n,)

    # ---------------------------
    # 5) 成本数组（向量化）
    # ---------------------------
    # 5.1) 先按“目标股数”估算成本（简化：不考虑资金约束）
    target_values = np.empty(n, dtype=np.float64)
    for i in range(n):
        equity = initial_cash  # 近似值；真实 equity 在 Numba 循环中更新
        target_values[i] = equity * trade_ratios[i]

    # 避免除零
    safe_close = np.where(close > 0.0, close, 1e-12)
    target_shares_est = target_values / safe_close  # 带符号的股数

    # 方向：>0 为买入，<0 为卖出
    direction = np.where(target_shares_est >= 0, 1.0, -1.0)
    is_buy = target_shares_est >= 0

    # 5.2) 调用向量化成本函数
    cost_array = calculate_transaction_cost_vectorized(
        prices=close,
        trade_shares=target_shares_est,
        direction=direction,
        commission_rate=commission_cfg.commission_rate,
        min_commission=commission_cfg.min_commission,
        stamp_duty_rate=commission_cfg.stamp_duty_rate,
        transfer_fee_rate=commission_cfg.transfer_fee_rate,
    )

    # ---------------------------
    # 6) Numba JIT 状态机
    # ---------------------------
    cash_arr, pos_arr, equity_arr, trades_arr = run_numba_backtest_engine(
        close_prices=close,
        trade_ratios=trade_ratios,
        cost_array=cost_array,
        initial_cash=initial_cash,
        is_buy=is_buy,
        slippage_bps=slippage_bps,
        min_trade_value=min_trade_value,
    )

    # ---------------------------
    # 7) 结果写回 DataFrame
    # ---------------------------
    out_df = out_df.copy()
    out_df["equity_curve"] = equity_arr
    out_df["backtest_cash"] = cash_arr
    out_df["backtest_position"] = pos_arr
    out_df["backtest_trade_shares"] = trades_arr

    return out_df


def run_backtest_loop_compatible(
    df: pd.DataFrame,
    stock_code: str,
    market_data: Optional[pd.DataFrame],
    weights: Dict[str, float],
    params: Dict[str, Dict],
    regime: Optional[str] = None,
    stocks_data: Optional[Dict] = None,
    initial_capital: float = 100000.0,
    use_vectorized_backtest: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], pd.DataFrame]:
    if not use_vectorized_backtest:
        return run_backtest_loop(
            df=df,
            stock_code=stock_code,
            market_data=market_data,
            weights=weights,
            params=params,
            regime=regime,
            stocks_data=stocks_data,
            initial_capital=initial_capital,
        )

    logger.info("使用向量化回测引擎...")

    # ---------------- 新增：按 regime 把参数“注入到每一行” ----------------
    df = build_regime_param_array(df, market_data, regime, params)
    # 现在 df 里有：
    #   - "curr_regime"
    #   - "param_dict"  (逐行 dict，和逐行回测的 p 完全一致)

    # --------- 原来的 weights 也可以保持你动态更新逻辑（可选） ----------
    # 如果你希望和逐行一样“每 60 日滚动更新 current_weights”，
    # 可以在向量化引擎里做“分段常量权重”的窗口化；目前先用输入的 weights 即可。

    # --------- 策略实例：不再写死 TransformerStrategy ----------
    # 方案 A：最简单，直接复用你现有的因子 + weights 生成 target_position_ratio
    from strategies.base import BaseStrategy
    default_strategy = BaseStrategy()
    # 策略参数从哪来？你已经把 params 按 regime 放到 df["param_dict"] 里了，
    # 如果策略需要“某个 regime 下的参数”，可以在向量化函数里直接用 df["param_dict"]
    # 或者在这里挑一个默认策略参数 dict（比如 neutral）：
    default_strategy.params = params.get("neutral", params)
    strategy_instances = [default_strategy]

    # --------- 风控/成本配置保持不变 ----------
    risk_cfg = None
    try:
        from config import get_settings
        risk_cfg = get_settings().risk
    except Exception as e:
        logger.warning(f"无法获取风控配置: {e}")

    commission_cfg = _get_commission_cfg()
    slippage_cfg = _get_slippage_cfg()

    # --------- 调用向量化引擎 ----------
    try:
        result_df = run_backtest_loop_vectorized(
            df=df,                           # 这里已经包含 curr_regime / param_dict
            strategy_instances=strategy_instances,
            risk_cfg=risk_cfg,
            commission_cfg=commission_cfg,
            slippage_cfg=slippage_cfg,
            initial_cash=initial_capital,
            slippage_bps=0.0,
            min_trade_value=0.0,
            use_vectorized_signals=True,
        )
    except Exception as e:
        logger.error(f"向量化回测失败，回退到逐行回测: {e}")
        return run_backtest_loop(
            df=df,
            stock_code=stock_code,
            market_data=market_data,
            weights=weights,
            params=params,
            regime=regime,
            stocks_data=stocks_data,
            initial_capital=initial_capital,
        )

    # --------- 原有的统计/输出逻辑保持不变 ----------
    trades_df = None

    stats = {
        "initial_capital": initial_capital,
        "final_equity": float(result_df["equity_curve"].iloc[-1]) if len(result_df) > 0 else initial_capital,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
    }

    if len(result_df) > 0 and "equity_curve" in result_df.columns:
        equity = result_df["equity_curve"]
        stats["final_equity"] = float(equity.iloc[-1])
        stats["total_return"] = (equity.iloc[-1] / initial_capital - 1.0) * 100

        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        stats["max_drawdown"] = float(drawdown.min()) * 100

        # 简化版 Sharpe（用 Close 的日收益率）
        if "Close" in result_df.columns:
            returns = result_df["Close"].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                stats["sharpe_ratio"] = float(returns.mean() / returns.std() * np.sqrt(252))

    # 后续可以从 result_df 里还原交易记录（backtest_trade_shares 非零的日期）
    if "backtest_trade_shares" in result_df.columns:
        trade_mask = result_df["backtest_trade_shares"] != 0
        if trade_mask.any():
            trades_df = result_df[trade_mask].copy()

            # 1. 还原基础信息
            trades_df["date"] = trades_df.index
            trades_df["stock_code"] = stock_code

            # 2. 交易方向：正数为买入，负数为卖出
            trades_df["action"] = np.where(trades_df["backtest_trade_shares"] > 0, "buy", "sell")

            # 3. 交易价格（兼容大小写 Close/close）
            price_col = "Close" if "Close" in trades_df.columns else "close"
            trades_df["price"] = trades_df[price_col]

            # 4. 交易股数和交易金额
            trades_df["shares"] = trades_df["backtest_trade_shares"]
            trades_df["value"] = trades_df["shares"].abs() * trades_df["price"]

            # 5. 交易成本
            # 【注意】这里优先取向量化引擎输出的真实成本。
            # 如果你之前的 run_backtest_loop_vectorized 里没有把成本存进 df，这里会走简单估算。
            if "backtest_trade_cost" in trades_df.columns:
                trades_df["cost"] = trades_df["backtest_trade_cost"]
            else:
                # 容错：如果没有真实成本列，暂时填 0.0（建议看下方提示补充真实成本）
                trades_df["cost"] = 0.0

            # 6. 附加信息（可选：如果 df 里保留了当时的信号得分和市场状态，一并带出）
            if "Combined_Score" in trades_df.columns:
                trades_df["signal_score"] = trades_df["Combined_Score"]
            if "curr_regime" in trades_df.columns:
                trades_df["regime"] = trades_df["curr_regime"]

            # 7. 整理最终输出列（保持和你原来逐行回测输出的 trades_df 列名一致）
            output_cols = ["date", "stock_code", "action", "price", "shares", "value", "cost"]
            for extra_col in ["signal_score", "regime", "target_position_ratio"]:
                if extra_col in trades_df.columns:
                    output_cols.append(extra_col)

            trades_df = trades_df[output_cols].reset_index(drop=True)
        else:
            # 没有任何交易发生
            trades_df = pd.DataFrame(columns=["date", "stock_code", "action", "price", "shares", "value", "cost"])
    else:
        trades_df = None

    return trades_df, stats, result_df
