# -*- coding: utf-8 -*-
"""
参数优化模块 V2 — 收益优化版
主要改进：
1. Walk-Forward 使用扩展窗口（expanding window）
2. Optuna 搜索空间加宽，step 更细
3. 新增 bear 市场状态参数优化
4. 目标函数增加 Sortino 比率作为第三目标
5. 修复 6-tuple 与 4-tuple 不兼容问题
"""
import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr
import optuna

from data.types import NON_FACTOR_COLS, TRADITIONAL_FACTOR_COLS
from backtest.evaluator import calculate_comprehensive_stats
from config import get_settings

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def calculate_dynamic_weights(df: pd.DataFrame, factor_cols: list, ic_window_range=(20, 120), use_ewma=True) -> Dict[str, float]:
    """基于 IC/ICIR 的动态权重计算（保持不变）"""
    target = df['Close'].pct_change(5).rename('target_return')
    icir_dict = {}

    for col in factor_cols:
        if col not in df.columns:
            continue
        pair_df = df[[col]].join(target, how='inner').dropna()
        if len(pair_df) < ic_window_range[0]:
            continue

        if use_ewma:
            ic_series = pair_df[col].rolling(window=ic_window_range[1]).apply(
                lambda x: spearmanr(x, pair_df.loc[x.index, 'target_return'])[0]
            )
            ewma_ic = ic_series.ewm(alpha=0.05).mean()
            mean_ic = ewma_ic.iloc[-1]
            std_ic = ewma_ic.std()
        else:
            volatility = df['Close'].pct_change().rolling(60).std()
            current_vol = volatility.iloc[-1]
            if current_vol > volatility.quantile(0.75):
                window = ic_window_range[0]
            elif current_vol < volatility.quantile(0.25):
                window = ic_window_range[1]
            else:
                window = int((ic_window_range[0] + ic_window_range[1]) / 2)

            vals = pair_df.values
            windows = sliding_window_view(vals, window_shape=window, axis=0)
            ic_list = []
            for w in windows:
                rho, _ = spearmanr(w[:, 0], w[:, 1])
                ic_list.append(rho)
            clean_ic = [x for x in ic_list if not np.isnan(x)]
            if not clean_ic:
                continue
            mean_ic = np.mean(clean_ic)
            std_ic = np.std(clean_ic)

        if std_ic > 1e-6:
            icir_dict[col] = mean_ic / std_ic
        else:
            icir_dict[col] = 0

    sum_abs = sum(abs(v) for v in icir_dict.values())
    if sum_abs == 0:
        return {col: 1.0 / len(factor_cols) for col in factor_cols}
    return {col: abs(icir_dict.get(col, 0)) / sum_abs for col in factor_cols}


def walk_forward_split(
        df: pd.DataFrame,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        gap_days: int = 20,
        min_train_size: int = 100,
        min_test_size: int = 20,
        expanding_window: bool = True,
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Walk-Forward 划分 V2（支持扩展窗口）

    参数:
        df: 股票数据 DataFrame
        n_splits: 划分次数
        train_ratio: 初始训练集比例
        val_ratio: 验证集比例
        gap_days: 间隔天数
        min_train_size: 训练集最小样本数
        min_test_size: 测试集最小样本数
        expanding_window: 是否使用扩展窗口（True=训练集不断扩大，False=固定大小滚动）

    返回:
        列表，每个元素是 (train_start, train_end, val_start, val_end, test_start, test_end)
    """
    total_len = len(df)
    test_ratio = 1.0 - train_ratio - val_ratio

    if test_ratio <= 0:
        raise ValueError(f"train_ratio + val_ratio 必须小于 1，当前为 {train_ratio + val_ratio}")

    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)

    print(f"\n数据总长度: {total_len}")
    print(f"初始训练集长度: {train_len} ({train_ratio * 100:.1f}%)")
    print(f"验证集长度: {val_len} ({val_ratio * 100:.1f}%)")
    print(f"测试集长度: {test_len} ({test_ratio * 100:.1f}%)")
    print(f"间隔天数: {gap_days}")
    print(f"扩展窗口: {expanding_window}")

    splits = []

    if expanding_window:
        # 扩展窗口模式：训练集不断扩大，测试集滚动
        # 初始：用前 train_ratio 的数据训练
        first_train_end = train_len

        for i in range(n_splits):
            if expanding_window:
                train_start = 0
                train_end = first_train_end + i * test_len  # 训练集不断扩大
            else:
                train_start = i * test_len
                train_end = train_start + train_len

            val_start = train_end + gap_days
            val_end = val_start + val_len

            test_start = val_end + gap_days
            test_end = min(test_start + test_len, total_len)

            if test_start >= total_len:
                print(f"\n第 {i + 1} 次划分：测试集超出数据范围，停止划分")
                break

            actual_train_len = train_end - train_start
            actual_val_len = val_end - val_start
            actual_test_len = test_end - test_start

            if actual_train_len < min_train_size:
                print(f"\n第 {i + 1} 次划分：训练集样本数不足 ({actual_train_len} < {min_train_size})，停止划分")
                break

            if actual_test_len < min_test_size:
                print(f"\n第 {i + 1} 次划分：测试集样本数不足 ({actual_test_len} < {min_test_size})，停止划分")
                break

            splits.append((train_start, train_end, val_start, val_end, test_start, test_end))

            print(f"第 {i + 1} 次划分:")
            print(f"  训练集: [{train_start}:{train_end}] ({actual_train_len} 天)")
            print(f"  验证集: [{val_start}:{val_end}] ({actual_val_len} 天)")
            print(f"  测试集: [{test_start}:{test_end}] ({actual_test_len} 天)")
    else:
        # 固定窗口滚动模式（原逻辑）
        start_idx = 0
        for i in range(n_splits):
            train_start = start_idx
            train_end = train_start + train_len

            val_start = train_end + gap_days
            val_end = val_start + val_len

            test_start = val_end + gap_days
            test_end = min(test_start + test_len, total_len)

            if test_start >= total_len:
                break

            actual_train_len = train_end - train_start
            actual_val_len = val_end - val_start
            actual_test_len = test_end - test_start

            if actual_train_len < min_train_size:
                break
            if actual_test_len < min_test_size:
                break

            splits.append((train_start, train_end, val_start, val_end, test_start, test_end))
            start_idx += test_len + gap_days

    print(f"\n总共生成 {len(splits)} 次有效划分")
    return splits


def optimize_strategy(
    train_df: pd.DataFrame,
    stock_code: str,
    market_data: Optional[pd.DataFrame],
    stocks_data: Optional[Dict],
) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """参数优化 V2（加宽搜索空间，增加 bear 状态）"""
    from backtest.engine import run_backtest_loop
    from data.indicators import calculate_orthogonal_factors

    df = train_df.copy()
    if 'transformer_prob' not in train_df.columns or 'mom_10' not in train_df.columns:
        print(f" [警告] {stock_code} 传入 optimize_strategy 的数据缺少因子列，正在补充计算...")
        df = calculate_orthogonal_factors(df, stock_code)

    factor_cols = [col for col in df.columns if col in TRADITIONAL_FACTOR_COLS]
    weights = calculate_dynamic_weights(df, factor_cols)

    def objective(trial, regime):
        # ★ 搜索空间加宽，step 更细
        trial_params = {
            'buy_threshold': trial.suggest_float('buy_threshold', 0.45, 0.75, step=0.025),
            'sell_threshold': trial.suggest_float('sell_threshold', -0.45, 0.05, step=0.025),
            'hold_days': trial.suggest_int('hold_days', 5, 30, step=3),
            'stop_loss': trial.suggest_float('stop_loss', -0.12, -0.03, step=0.005),
            'trailing_profit_level1': trial.suggest_float('trailing_profit_level1', 0.03, 0.10, step=0.005),
            'trailing_profit_level2': trial.suggest_float('trailing_profit_level2', 0.08, 0.20, step=0.01),
            'trailing_drawdown_level1': trial.suggest_float('trailing_drawdown_level1', 0.03, 0.12, step=0.005),
            'trailing_drawdown_level2': trial.suggest_float('trailing_drawdown_level2', 0.02, 0.06, step=0.005),
            'take_profit_multiplier': trial.suggest_float('take_profit_multiplier', 1.5, 5.0, step=0.25),
            'transformer_weight': trial.suggest_float('transformer_weight', 0.0, 0.6, step=0.05),
            'transformer_buy_threshold': trial.suggest_float('transformer_buy_threshold', 0.50, 0.85, step=0.025),
            'transformer_sell_threshold': trial.suggest_float('transformer_sell_threshold', 0.15, 0.45, step=0.025),
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.3, 0.7, step=0.025),
        }

        if trial_params['buy_threshold'] <= trial_params['sell_threshold']:
            return -999.0, -1.0, -999.0

        adjusted_weights = weights.copy()
        if 'transformer_prob' in adjusted_weights:
            adjusted_weights['transformer_prob'] = trial_params['transformer_weight']
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        trades_df, stats, _ = run_backtest_loop(
            df, stock_code, market_data, adjusted_weights,
            {regime: trial_params}, regime, stocks_data=stocks_data,
        )

        if stats is None or trades_df is None or len(trades_df) == 0:
            return -999.0, -1.0, -999.0

        ret = stats['total_return']
        cum_ret = (1 + trades_df['net_return']).cumprod()
        mdd = ((cum_ret.cummax() - cum_ret) / cum_ret.cummax()).max()
        sharpe = (trades_df['net_return'].mean() / (trades_df['net_return'].std() + 1e-6)) * np.sqrt(252)

        # ★ Sortino 作为第三目标
        downside = trades_df['net_return'][trades_df['net_return'] < 0]
        downside_std = downside.std() if len(downside) > 1 else 1e-6
        sortino = (trades_df['net_return'].mean() / (downside_std + 1e-6)) * np.sqrt(252)

        win_rate = (trades_df['net_return'] > 0).mean()
        penalty = 0.0
        if win_rate < 0.35:
            penalty = -0.5
        if len(trades_df) < 5:
            penalty = -1.0

        return float(ret), float(-mdd + penalty), float(sortino)

    from data.types import ALL_REGIMES
    settings = get_settings()
    best_params_map = {}

    # ★ 扩展到5种市场状态
    extended_regimes = ['strong_bull', 'bull', 'neutral', 'weak', 'bear']

    for regime in extended_regimes:
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(lambda t: objective(t, regime), n_trials=settings.backtest.n_optuna_trials)

        if len(study.best_trials) > 0:
            # 选择 Sortino 最高的 Pareto 前沿解（更关注下行风险控制）
            best_t = max(study.best_trials, key=lambda t: t.values[2])
            best_params_map[regime] = best_t.params
        else:
            # 默认参数
            best_params_map[regime] = {
                'buy_threshold': 0.6, 'sell_threshold': -0.2, 'hold_days': 15,
                'stop_loss': -0.08, 'trailing_profit_level1': 0.06,
                'trailing_profit_level2': 0.12, 'trailing_drawdown_level1': 0.08,
                'trailing_drawdown_level2': 0.04, 'take_profit_multiplier': 3.0,
                'transformer_weight': 0.2, 'transformer_buy_threshold': 0.65,
                'transformer_sell_threshold': 0.3, 'confidence_threshold': 0.5,
            }

    return best_params_map, weights
