# -*- coding: utf-8 -*-
"""
统一评估指标模块
计算 9 个核心指标：收益率、胜率、交易次数、平均收益率、
最大回撤、夏普比率、利润因子、Sortino比率、Calmar比率
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# backtest/evaluator.py
import numpy as np
import pandas as pd
from typing import Dict, Optional

def calculate_comprehensive_stats(
    trades_df: pd.DataFrame,
    equity_curve: Optional[pd.Series] = None,
    benchmark_curve: Optional[pd.Series] = None,
    initial_cash: float = 100_000.0,
    commissions: float = 0.0,
) -> Dict[str, float]:
    """
    更完整的回测统计（尽量覆盖常见报告字段）
    - trades_df: 必须有 net_return 列；如有 buy_date/sell_date 会自动算交易时长
    - equity_curve: 可选，资金曲线（含初始资金），用于计算年化波动、回撤持续期等
    - benchmark_curve: 可选，基准净值，用于计算 Alpha/Beta
    """
    if trades_df is None or len(trades_df) == 0:
        return {}

    net_returns = trades_df["net_return"]
    n_trades = len(net_returns)

    # ---- 基础收益 ----
    total_return = ((1 + net_returns).prod() - 1) * 100
    win_rate = (net_returns > 0).mean() * 100
    avg_return = net_returns.mean() * 100
    best_trade = net_returns.max() * 100
    worst_trade = net_returns.min() * 100
    equity_final = initial_cash * (1 + net_returns).prod()  # ★ 修复：用复合收益替代简单求和
    equity_peak = initial_cash * (1 + net_returns.cumsum().max())

    # ---- 交易时长（如果存在 buy_date/sell_date） ----
    def _avg_or_max_durations(key):
        try:
            dur = (trades_df["sell_date"] - trades_df["buy_date"]).dt.total_seconds() / 86400
            if dur.empty:
                return None
            return {
                "max_trade_duration_days": dur.max(),
                "avg_trade_duration_days": dur.mean(),
            }
        except Exception:
            return {}

    durations = _avg_or_max_durations(None)

    # ---- 利润因子 ----
    gross_profit = net_returns[net_returns > 0].sum()
    gross_loss = abs(net_returns[net_returns < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-12)

    # ---- 期望值 ----
    expectancy = net_returns.mean() * 100  # 简单平均单笔收益
    # 或者用胜率*平均盈利 - 败率*平均亏损
    avg_win = net_returns[net_returns > 0].mean() if (net_returns > 0).any() else 0
    avg_loss = abs(net_returns[net_returns < 0].mean()) if (net_returns < 0).any() else 1e-12
    expectancy_pw = win_rate / 100 * avg_win * 100 - (1 - win_rate / 100) * avg_loss * 100

    # ---- 最大回撤 & 平均回撤（基于资金曲线） ----
    def _drawdown_stats(curve: pd.Series):
        cum = (1 + curve).cumprod() if not (curve + 1).max() <= 0 else pd.Series([1.0])
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min() * 100
        avg_dd = dd[dd < 0].mean() * 100 if (dd < 0).any() else 0.0
        # 回撤持续期
        in_drawdown = dd < 0
        if not in_drawdown.any():
            return max_dd, avg_dd, None, None
        groups = (~in_drawdown).cumsum()
        dd_durations = in_drawdown.groupby(groups).sum()
        max_dd_dur = dd_durations.max() if not dd_durations.empty else 0
        avg_dd_dur = dd_durations.mean() if not dd_durations.empty else 0
        return max_dd, avg_dd, max_dd_dur, avg_dd_dur

    max_drawdown, avg_drawdown, max_dd_duration, avg_dd_duration = _drawdown_stats(net_returns)

    # ---- 年化与波动率 ----
    # ★ 修复：基于交易持续天数计算年化，而非简单用 252/n_trades
    total_days = 252  # 默认1年
    if 'buy_date' in trades_df.columns and 'sell_date' in trades_df.columns:
        try:
            first_date = trades_df['buy_date'].min()
            last_date = trades_df['sell_date'].max()
            total_days = max((last_date - first_date).days, 1)
        except Exception:
            pass
    years = total_days / 252.0
    ann_factor = 252
    ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 and total_return > -100 else 0.0
    ann_vol = net_returns.std() * np.sqrt(ann_factor) * 100

    # 夏普 / Sortino / Calmar
    sharpe = (net_returns.mean() / (net_returns.std() + 1e-12)) * np.sqrt(ann_factor)
    downside = net_returns[net_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-12
    sortino = (net_returns.mean() / (downside_std + 1e-12)) * np.sqrt(ann_factor)
    calmar = ann_return / (abs(max_drawdown) + 1e-12)

    # CAGR
    # 如果有 equity_curve 且是日频，可以用日期计算实际年数
    cagr = ann_return  # 简化版；如果 equity_curve 带日期索引可以精确算

    # Alpha / Beta（需要基准曲线）
    alpha = None
    beta = None
    if benchmark_curve is not None and len(benchmark_curve) == len(net_returns):
        bench = benchmark_curve.pct_change().dropna().values
        strat = net_returns.values
        covm = np.cov(strat, bench)
        if covm.shape == (2, 2):
            beta = float(covm[0, 1] / (covm[1, 1] + 1e-12))
            alpha = (np.mean(strat) - beta * np.mean(bench)) * ann_factor * 100

    # SQN（System Quality Number）
    sqn = (net_returns.mean() / (net_returns.std() + 1e-12)) * np.sqrt(n_trades)

    # Kelly Criterion（简化）
    p = win_rate / 100
    b = avg_win / (avg_loss + 1e-12)
    kelly = p - (1 - p) / (b + 1e-12) if b > 0 else 0.0

    # Exposure Time
    # 如果你用的是 equity_curve（日频），可以计算持仓天数占比
    exposure_time = None
    if equity_curve is not None:
        # 假设 0 表示空仓，非0表示持仓
        exposure = (equity_curve != 0).mean() * 100 if equity_curve.dtype in [np.float64, np.int64] else None
        exposure_time = exposure

    # ---- 汇总 ----
    stats: Dict[str, float] = {
        # 基础
        "total_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_return, 4),
        "best_trade": round(best_trade, 4),
        "worst_trade": round(worst_trade, 4),
        # 资金
        "equity_final": round(equity_final, 2),
        "equity_peak": round(equity_peak, 2),
        "commissions": round(commissions, 2),
        # 收益
        "total_return": round(total_return, 4),
        "ann_return": round(ann_return, 4),
        "cagr": round(cagr, 4),
        # 风险
        "max_drawdown": round(max_drawdown, 4),
        "avg_drawdown": round(avg_drawdown, 4),
        "max_drawdown_duration": round(max_dd_duration, 2) if max_dd_duration is not None else None,
        "avg_drawdown_duration": round(avg_dd_duration, 2) if avg_dd_duration is not None else None,
        "ann_volatility": round(ann_vol, 4),
        # 风险调整
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        # 资产定价
        "alpha": round(alpha, 4) if alpha is not None else None,
        "beta": round(beta, 4) if beta is not None else None,
        # 交易质量
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy_pw, 4),
        "sqn": round(sqn, 4),
        "kelly_criterion": round(kelly, 4),
        "exposure_time": round(exposure_time, 4) if exposure_time is not None else None,
    }
    # 交易持续期（可能为空）
    if durations:
        stats.update({k: round(v, 2) for k, v in durations.items()})
    return stats

# 在 evaluator.py 中
from typing import Dict, Any

def evaluate_strategy(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    评价策略绩效，直接使用引擎已计算好的指标。
    可以在此添加额外的评价逻辑或风险检查。
    """
    # 简单地将引擎返回的结果作为评价结果
    # 在实际中，可以在此添加更多衍生指标或风险评分
    return backtest_results

def check_risk_limits(performance: Dict[str, Any], risk_limits: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    检查绩效是否通过风控阈值。
    :param performance: 包含绩效指标的字典。
    :param risk_limits: 风控阈值字典，例如 {'max_drawdown': -0.2, 'min_sharpe': 0.5, 'min_trades': 15}
    :return: (是否通过, 未通过的原因列表)
    """
    passed = True
    reasons = []

    if performance['max_drawdown'] < risk_limits.get('max_drawdown', -1.0):
        passed = False
        reasons.append(f"最大回撤 {performance['max_drawdown']:.2%} 超过限制 {risk_limits['max_drawdown']:.2%}")

    if performance['sharpe_ratio'] < risk_limits.get('min_sharpe', -999):
        passed = False
        reasons.append(f"夏普比率 {performance['sharpe_ratio']:.2f} 低于限制 {risk_limits['min_sharpe']}")

    if performance['total_trades'] < risk_limits.get('min_trades', 0):
        passed = False
        reasons.append(f"交易次数 {performance['total_trades']} 少于最小限制 {risk_limits['min_trades']}")

    if performance['win_rate'] < risk_limits.get('min_win_rate', 0.0):
        passed = False
        reasons.append(f"胜率 {performance['win_rate']:.2%} 低于限制 {risk_limits['min_win_rate']:.2%}")

    if performance['profit_factor'] < risk_limits.get('min_profit_factor', 0.0):
        passed = False
        reasons.append(f"利润因子 {performance['profit_factor']:.2f} 低于限制 {risk_limits['min_profit_factor']}")

    return passed, reasons

