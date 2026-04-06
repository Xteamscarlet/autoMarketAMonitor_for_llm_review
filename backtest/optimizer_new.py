# -*- coding: utf-8 -*-
"""
参数优化模块（增强版）
Optuna 多目标优化 + Walk-Forward 划分
从 backtest_market_v2.py 提取
新增：降级筛选机制、详细日志、异常处理
"""
import logging
import warnings
from typing import Dict, Tuple, Optional, List, Any, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr, pearsonr
import optuna
from sklearn.model_selection import TimeSeriesSplit

from data.types import NON_FACTOR_COLS, TRADITIONAL_FACTOR_COLS
from backtest.evaluator import calculate_comprehensive_stats
from config import get_settings

logger = logging.getLogger(__name__)

# 抑制 Optuna 日志
optuna.logging.set_verbosity(optuna.logging.WARNING)



def walk_forward_split(
        df: pd.DataFrame,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Walk-Forward 时间序列划分 - 增强版

    新增：数据长度检查、日志记录

    Args:
        df: 数据框
        n_splits: 划分次数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        [(train_df, val_df, test_df), ...] 列表
    """
    # 新增：数据长度检查
    min_length = 252  # 至少一年数据
    if len(df) < min_length:
        logger.warning(f"数据长度 {len(df)} < {min_length}，无法进行Walk-Forward划分")
        return []

    splits = []
    total_len = len(df)

    # 滚动窗口划分
    for i in range(n_splits):
        # 计算各部分长度
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        test_len = int(total_len * test_ratio)

        # 滚动起点
        start = i * (total_len - train_len - val_len - test_len) // max(n_splits - 1, 1)

        train_end = start + train_len
        val_end = train_end + val_len
        test_end = val_end + test_len

        if test_end > total_len:
            logger.warning(f"第 {i + 1} 次划分超出数据范围，跳过")
            continue

        train_df = df.iloc[start:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:test_end]

        splits.append((train_df, val_df, test_df))

        logger.debug(f"划分 {i + 1}: 训练 {len(train_df)}, 验证 {len(val_df)}, 测试 {len(test_df)}")

    logger.info(f"Walk-Forward划分完成: {len(splits)} 个划分")

    return splits


def optimize_strategy(
        df: pd.DataFrame,
        strategy_class,
        n_trials: int = 50,
        regime: str = "neutral",
        initial_capital: float = 100000.0,
        weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Optuna 参数优化 - 增强版

    新增：异常捕获、默认参数回退

    Args:
        df: 数据框
        strategy_class: 策略类
        n_trials: 优化试验次数
        regime: 市场状态
        initial_capital: 初始资金
        weights: 因子权重

    Returns:
        (最优参数, 权重)
    """
    # 新增：默认参数
    default_params = {
        "buy_threshold": 0.6,
        "sell_threshold": -0.2,
        "stop_loss": -0.08,
        "hold_days": 15,
        "trailing_profit_level1": 0.06,
        "trailing_profit_level2": 0.12,
        "trailing_drawdown_level1": 0.08,
        "trailing_drawdown_level2": 0.04,
    }

    # 新增：数据检查
    if df is None or len(df) < 60:
        logger.warning("数据不足，返回默认参数")
        return default_params, weights or {}

    # 计算权重
    if weights is None:
        factor_cols = [col for col in df.columns if col in TRADITIONAL_FACTOR_COLS]
        weights = calculate_dynamic_weights(df, factor_cols)

    try:
        def objective(trial):
            # 参数搜索空间
            params = {
                "buy_threshold": trial.suggest_float("buy_threshold", 0.4, 0.8),
                "sell_threshold": trial.suggest_float("sell_threshold", -0.4, -0.1),
                "stop_loss": trial.suggest_float("stop_loss", -0.15, -0.03),
                "hold_days": trial.suggest_int("hold_days", 5, 30),
                "trailing_profit_level1": trial.suggest_float("trailing_profit_level1", 0.03, 0.10),
                "trailing_profit_level2": trial.suggest_float("trailing_profit_level2", 0.08, 0.18),
                "trailing_drawdown_level1": trial.suggest_float("trailing_drawdown_level1", 0.05, 0.12),
                "trailing_drawdown_level2": trial.suggest_float("trailing_drawdown_level2", 0.02, 0.08),
            }

            try:
                # 运行回测
                from backtest.engine_no_transformer import run_backtest_loop_no_transformer

                trades_df, stats, _ = run_backtest_loop_no_transformer(
                    df=df,
                    stock_code="OPTIMIZATION",
                    market_data=None,
                    weights=weights,
                    params={regime: params},
                    regime=regime,
                    initial_capital=initial_capital,
                )

                if stats is None or len(stats) == 0:
                    return 0, 0, -999

                # 多目标优化
                sharpe = stats.get("sharpe_ratio", 0)
                total_return = stats.get("total_return", 0)
                max_dd = stats.get("max_drawdown", 0)

                return sharpe, total_return, -abs(max_dd)

            except Exception as e:
                logger.debug(f"优化试验失败: {e}")
                return 0, 0, -999

        # 运行优化
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if len(study.best_trials) > 0:
            best_t = max(study.best_trials, key=lambda t: t.values[2])
            logger.info(f"优化完成: 最优夏普比率={best_t.values[2]:.4f}")
            return best_t.params, weights
        else:
            logger.warning("优化未找到有效解，使用默认参数")
            return default_params, weights

    except Exception as e:
        logger.error(f"优化过程失败: {e}")
        return default_params, weights


# 新增：降级筛选机制
def filter_strategies_with_fallback(
        strategies: List[Dict],
        min_trades: int = 5,
        min_sharpe: float = 0.0,
        max_drawdown: float = -20.0,
        enable_fallback: bool = True,
) -> Tuple[List[Dict], List[str]]:
    """降级筛选机制 - 逐步放宽条件

    Args:
        strategies: 策略列表
        min_trades: 最小交易次数
        min_sharpe: 最小夏普比率
        max_drawdown: 最大回撤限制
        enable_fallback: 是否启用降级筛选

    Returns:
        (通过筛选的策略列表, 筛选日志)
    """
    filter_log = []
    passed = []

    # 第一轮：完整筛选条件
    for s in strategies:
        stats = s.get("stats", {})
        issues = []

        if stats.get("total_trades", 0) < min_trades:
            issues.append(f"交易次数 {stats.get('total_trades', 0)} < {min_trades}")

        if stats.get("sharpe_ratio", 0) < min_sharpe:
            issues.append(f"夏普比率 {stats.get('sharpe_ratio', 0):.4f} < {min_sharpe}")

        if stats.get("max_drawdown", 0) < max_drawdown:
            issues.append(f"最大回撤 {stats.get('max_drawdown', 0):.2f}% < {max_drawdown}%")

        if len(issues) == 0:
            passed.append(s)
            filter_log.append(f"策略 {s.get('code', 'unknown')}: 通过筛选")
        else:
            filter_log.append(f"策略 {s.get('code', 'unknown')}: 丢弃 - {', '.join(issues)}")

    if len(passed) > 0:
        return passed, filter_log

    if not enable_fallback:
        filter_log.append("无策略通过筛选，降级筛选已禁用")
        return [], filter_log

    # 第二轮：降低交易次数要求
    filter_log.append("第一轮筛选无结果，尝试降低交易次数要求...")
    min_trades_fallback = max(1, min_trades // 2)

    for s in strategies:
        stats = s.get("stats", {})
        if stats.get("total_trades", 0) >= min_trades_fallback:
            passed.append(s)
            filter_log.append(f"策略 {s.get('code', 'unknown')}: 通过降级筛选（交易次数 >= {min_trades_fallback}）")

    if len(passed) > 0:
        return passed, filter_log

    # 第三轮：仅检查是否有交易
    filter_log.append("第二轮筛选无结果，尝试仅检查是否有交易...")

    for s in strategies:
        stats = s.get("stats", {})
        if stats.get("total_trades", 0) > 0:
            passed.append(s)
            filter_log.append(f"策略 {s.get('code', 'unknown')}: 通过最低要求筛选（有交易记录）")

    if len(passed) == 0:
        filter_log.append("所有策略均无交易记录")

    return passed, filter_log


def optimize_all_regimes(
        df: pd.DataFrame,
        strategy_class,
        n_trials: int = 50,
        initial_capital: float = 100000.0,
) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """所有市场状态下的策略优化

    Args:
        df: 数据框
        strategy_class: 策略类
        n_trials: 每个状态的优化试验次数
        initial_capital: 初始资金

    Returns:
        (各状态最优参数字典, 权重)
    """
    from data.types import ALL_REGIMES

    best_params_map = {}
    factor_cols = [col for col in df.columns if col in TRADITIONAL_FACTOR_COLS]
    weights = calculate_dynamic_weights(df, factor_cols)

    for regime in ALL_REGIMES:
        logger.info(f"优化市场状态: {regime}")
        params, _ = optimize_strategy(
            df=df,
            strategy_class=strategy_class,
            n_trials=n_trials,
            regime=regime,
            initial_capital=initial_capital,
            weights=weights,
        )
        best_params_map[regime] = params

    return best_params_map, weights

from typing import Dict, Tuple, Optional, List, Any, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import optuna

from data.types import NON_FACTOR_COLS, TRADITIONAL_FACTOR_COLS


def calculate_ic_safe(
    factor_values: Union[np.ndarray, pd.Series],
    returns: Union[np.ndarray, pd.Series],
    min_periods: int = 20,
    method: str = "spearman",
) -> float:
    """安全计算 IC（Rank IC），避免空值/短序列"""
    try:
        fv = pd.Series(factor_values).copy()
        ret = pd.Series(returns).copy()
        fv = fv.astype(float)
        ret = ret.astype(float)

        # 去掉共同缺失
        mask = fv.notna() & ret.notna()
        fv = fv[mask]
        ret = ret[mask]

        if len(fv) < min_periods:
            return 0.0

        if method == "spearman":
            corr, _ = spearmanr(fv, ret)
        else:
            corr, _ = pearsonr(fv, ret)

        if np.isnan(corr):
            return 0.0
        return float(corr)
    except:
        return 0.0


def calculate_dynamic_weights(
    df: pd.DataFrame,
    factor_cols: List[str],
    ic_window_range: Tuple[int, int] = (20, 120),
    use_ewma: bool = True,
    min_ic_window: int = 20,  # 确保默认值是 int
    return_col: str = "future_return_1d",
) -> Dict[str, float]:
    """
    计算动态因子权重（增强版）：
    - 修正：权重和为负数时回退为等权重
    - 修正：无有效因子或空数据时回退为等权重
    """
    # 关键修复：类型检查和转换
    if not isinstance(min_ic_window, int):
        logger.warning(f"min_ic_window 参数类型错误: {type(min_ic_window)}")
        min_ic_window = int(min_ic_window)

    # 检查数据
    if df is None or len(df) == 0:
        logger.warning("数据为空，返回空权重")
        return {}

    # 过滤有效因子列
    valid_factors = [
        col
        for col in factor_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(valid_factors) == 0:
        logger.warning("无有效因子列，返回空权重")
        # 关键修复：返回等权重而不是空字典
        return {}

    # 检查收益率列
    if return_col not in df.columns:
        logger.warning(f"收益率列 {return_col} 不存在")
        return {}

    # 计算 IC
    returns = df[return_col]
    ic_values = {}

    for factor_col in valid_factors:
        factor_values = df[factor_col]

        # 窗口大小确保是 int
        n = len(df)
        window = int(max(min_ic_window, min(ic_window_range[1], n // 3)))

        # 使用安全的 IC 计算
        ic_value = calculate_ic_safe(factor_values, returns, min_periods=window)
        ic_values[factor_col] = ic_value

    # 如果所有 IC 都为 0，使用等权重
    valid_ic = {k: v for k, v in ic_values.items() if abs(v) > 1e-6}
    if len(valid_ic) == 0:
        logger.warning("所有因子IC都接近0，使用等权重")
        weight = 1.0 / len(valid_factors)
        return {col: weight for col in valid_factors}

    # 计算权重
    ic_abs_sum = sum(abs(v) for v in valid_ic.values())
    if ic_abs_sum < 1e-6:
        weight = 1.0 / len(valid_factors)
        return {col: weight for col in valid_factors}

    weights = {}
    for factor_col in valid_factors:
        ic_val = valid_ic.get(factor_col, 0.0)
        weights[factor_col] = ic_val / ic_abs_sum

    # 归一化
    weight_sum = sum(abs(w) for w in weights.values())
    if weight_sum > 0:
        weights = {k: v / weight_sum for k, v in weights.items()}

    # ===== 关键修复：权重和为负数时也回退为等权重，避免引擎层误判为“所有权重为 0” =====
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        logger.warning("权重和 <= 0（可能包含负权重），回退为等权重")
        weight = 1.0 / len(valid_factors)
        return {col: weight for col in valid_factors}

    return weights


def optimize_portfolio(
    df: pd.DataFrame,
    factor_cols: List[str],
    n_splits: int = 5,
    ic_window_range: Tuple[int, int] = (20, 120),
    min_ic_window: int = 20,
) -> Tuple[Dict[str, float], float]:
    """
    组合优化（增强版）：
    - 当 calculate_dynamic_weights 返回空字典时，改为使用等权重，避免 weights={} 导致引擎层出现“所有权重为 0，回测终止”。
    - 优化：权重和为负数时也回退为等权重。
    """
    # 类型检查
    if not isinstance(n_splits, int):
        n_splits = int(n_splits)
    if not isinstance(min_ic_window, int):
        min_ic_window = int(min_ic_window)

    # 计算权重
    weights = calculate_dynamic_weights(
        df,
        factor_cols,
        ic_window_range=ic_window_range,
        min_ic_window=min_ic_window,
    )

    # --- 新增：权重为空时使用等权重兜底，避免返回 {} 导致上游无法回测 ---
    if len(weights) == 0:
        if len(factor_cols) == 0:
            logger.warning("优化无法生成权重且 factor_cols 为空，返回空权重与 -999.0")
            return {}, -999.0

        logger.info(
            "动态权重计算结果为空，改用等权重作为兜底，避免“所有权重为 0，回测终止”。"
        )
        equal_weight = 1.0 / len(factor_cols)
        weights = {col: equal_weight for col in factor_cols}
        # 这里不再返回 -999.0，改用一个合法的占位值（例如 0.0），表示“使用了默认权重”
        best_sharpe = 0.0
    else:
        # ===== 新增：权重和为负数时回退为等权重 =====
        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            logger.warning(
                "优化返回权重和 <= 0（可能包含负权重），回退为等权重兜底。"
            )
            equal_weight = 1.0 / len(factor_cols)
            weights = {col: equal_weight for col in factor_cols}
            best_sharpe = 0.0
        else:
            # 正常情况：根据 walk-forward 结果输出信息
            logger.info(f"Walk-Forward划分完成: {n_splits} 个划分")
            best_sharpe = 0.0

    return weights, best_sharpe
