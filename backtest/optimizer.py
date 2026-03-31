# -*- coding: utf-8 -*-
"""
参数优化模块
Optuna 多目标优化 + Walk-Forward 划分
从 backtest_market_v2.py 提取
"""
import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr
import optuna
from sklearn.model_selection import TimeSeriesSplit

from data.types import NON_FACTOR_COLS
from backtest.engine import run_backtest_loop
from backtest.evaluator import calculate_comprehensive_stats
from config import get_settings

logger = logging.getLogger(__name__)

# 抑制 Optuna 日志
optuna.logging.set_verbosity(optuna.logging.WARNING)


def calculate_dynamic_weights(df: pd.DataFrame, factor_cols: list, ic_window_range=(20, 120), use_ewma=True) -> Dict[str, float]:
    """基于 IC/ICIR 的动态权重计算"""
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
    train_size: float = 0.7,
    test_size: float = 0.3,
    n_splits: int = 5,
    gap_days: int = 20,
) -> list:
    """Walk-Forward 划分（带 gap 防止数据泄露）"""
    total_len = len(df)
    train_len = int(total_len * train_size)
    test_len = int(total_len * test_size)
    splits = []
    start_idx = 0

    for i in range(n_splits):
        train_start = start_idx
        train_end = train_start + train_len
        test_start = train_end + gap_days
        test_end = min(test_start + test_len, total_len)

        if test_start >= total_len or test_end - test_start < 20 or train_end - train_start < 100:
            break

        splits.append((train_start, train_end, test_start, test_end))
        start_idx += test_len + gap_days

    return splits


def optimize_strategy(
    train_df: pd.DataFrame,
    stock_code: str,
    market_data: Optional[pd.DataFrame],
    stocks_data: Optional[Dict],
) -> Tuple[Dict[str, dict], Dict[str, float]]:
    """参数优化（单次，无交叉验证）"""
    from data.indicators import calculate_orthogonal_factors
    # ✅ 防御性检查：如果外面忘记算因子了，这里兜底算一下
    # 正常情况下，process_single_stock 传进来的 train_df 已经有因子了，不会走这里
    if 'transformer_prob' not in train_df.columns or 'mom_10' not in train_df.columns:
        print(f" [警告] {stock_code} 传入 optimize_strategy 的数据缺少因子列，正在补充计算...")
        df = calculate_orthogonal_factors(train_df, stock_code)

    factor_cols = [col for col in df.columns if col not in NON_FACTOR_COLS]
    weights = calculate_dynamic_weights(df, factor_cols)

    def objective(trial, regime):
        trial_params = {
            'buy_threshold': trial.suggest_float('buy_threshold', 0.5, 0.7, step=0.05),
            'sell_threshold': trial.suggest_float('sell_threshold', -0.4, 0.0, step=0.05),
            'hold_days': trial.suggest_int('hold_days', 5, 25, step=5),
            'stop_loss': trial.suggest_float('stop_loss', -0.10, -0.05, step=0.01),
            'trailing_profit_level1': trial.suggest_float('trailing_profit_level1', 0.05, 0.08, step=0.01),
            'trailing_profit_level2': trial.suggest_float('trailing_profit_level2', 0.10, 0.15, step=0.01),
            'trailing_drawdown_level1': trial.suggest_float('trailing_drawdown_level1', 0.05, 0.10, step=0.01),
            'trailing_drawdown_level2': trial.suggest_float('trailing_drawdown_level2', 0.03, 0.05, step=0.01),
            'take_profit_multiplier': trial.suggest_float('take_profit_multiplier', 2.0, 4.0, step=0.5),
            'transformer_weight': trial.suggest_float('transformer_weight', 0.0, 0.5, step=0.05),
            'transformer_buy_threshold': trial.suggest_float('transformer_buy_threshold', 0.6, 0.8, step=0.05),
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.4, 0.7, step=0.05),
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

        win_rate = (trades_df['net_return'] > 0).mean()
        penalty = 0.0
        if win_rate < 0.4:
            penalty = -0.5
        if len(trades_df) < 5:
            penalty = -1.0

        return float(ret), float(-mdd + penalty), float(sharpe)

    from data.types import ALL_REGIMES
    settings = get_settings()
    best_params_map = {}

    for regime in ALL_REGIMES:
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(lambda t: objective(t, regime), n_trials=settings.backtest.n_optuna_trials)

        if len(study.best_trials) > 0:
            best_t = max(study.best_trials, key=lambda t: t.values[2])
            best_params_map[regime] = best_t.params
        else:
            best_params_map[regime] = {
                'buy_threshold': 0.6, 'sell_threshold': -0.2, 'hold_days': 15,
                'stop_loss': -0.08, 'trailing_profit_level1': 0.06,
                'trailing_profit_level2': 0.12, 'trailing_drawdown_level1': 0.08,
                'trailing_drawdown_level2': 0.04, 'take_profit_multiplier': 3.0,
                'transformer_weight': 0.2, 'transformer_buy_threshold': 0.65,
                'confidence_threshold': 0.5,
            }

    return best_params_map, weights
