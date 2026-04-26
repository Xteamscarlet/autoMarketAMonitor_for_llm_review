# -*- coding: utf-8 -*-
"""
回测入口脚本 V2
修复：
1. walk_forward_split 返回6元组，process_single_stock 正确解包
2. 支持5种市场状态
3. 使用增强版市场状态判断
"""
import os
import json
import logging
import traceback
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import pandas as pd
import numpy as np

from config import get_settings, STOCK_CODES
from data import (
    download_market_data, download_stocks_data,
    check_and_clean_cache, load_pickle_cache,
    calculate_orthogonal_factors, save_pickle_cache,
)
from data.cache import load_transformer_cache
from data.types import NON_FACTOR_COLS
from backtest.engine import run_backtest_loop, calculate_multi_timeframe_score
from backtest.optimizer import (
    optimize_strategy, optimize_and_validate,
    walk_forward_split, calculate_dynamic_weights,
)
from backtest.evaluator import calculate_comprehensive_stats
from backtest.visualizer import visualize_backtest_with_split
from backtest.report import print_stock_backtest_report
from risk_manager import RiskManager
from utils.stock_filter import filter_codes_by_name, should_intercept_stock

_worker_market_data = None
_worker_stocks_data = None
_transformer_metadata_cache = None


def init_worker(m_data, s_data, preload_models: bool = False):
    global _worker_market_data, _worker_stocks_data
    _worker_market_data = m_data
    _worker_stocks_data = s_data
    if preload_models:
        try:
            import torch
            from model.predictor import _load_ensemble_models
            _load_ensemble_models(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception:
            pass


def _load_transformer_training_metadata(settings):
    global _transformer_metadata_cache
    if _transformer_metadata_cache is not None:
        return _transformer_metadata_cache

    path = getattr(settings.paths, "model_metadata_path", "model_training_metadata.json")
    if not os.path.exists(path):
        _transformer_metadata_cache = {}
        return _transformer_metadata_cache

    try:
        with open(path, "r", encoding="utf-8") as f:
            _transformer_metadata_cache = json.load(f)
    except Exception as exc:
        logging.getLogger(__name__).warning("无法读取 Transformer 训练边界元数据 %s: %s", path, exc)
        _transformer_metadata_cache = {}
    return _transformer_metadata_cache


def _get_transformer_train_end(metadata: dict, stock_code: str):
    if not metadata:
        return None
    by_code = metadata.get("train_end_by_code") or {}
    raw = by_code.get(str(stock_code)) or metadata.get("model_train_end_date")
    if not raw:
        return None
    parsed = pd.to_datetime(raw, errors="coerce")
    return parsed if pd.notna(parsed) else None


def _is_transformer_valid_for_split(metadata: dict, stock_code: str, train_df: pd.DataFrame) -> tuple[bool, str]:
    train_end = _get_transformer_train_end(metadata, stock_code)
    if train_end is None:
        return False, "missing_model_training_metadata"
    if train_df is None or train_df.empty:
        return False, "empty_train_split"

    split_train_end = pd.to_datetime(train_df.index[-1], errors="coerce")
    if pd.isna(split_train_end):
        return False, "invalid_split_train_end"

    if train_end > split_train_end:
        return False, f"model_train_end={train_end.date()} > split_train_end={split_train_end.date()}"
    return True, f"model_train_end={train_end.date()} <= split_train_end={split_train_end.date()}"


def _set_transformer_availability(df: pd.DataFrame, available: bool) -> pd.DataFrame:
    df = df.copy()
    df["transformer_available"] = bool(available)
    if not available:
        if "transformer_prob" in df.columns:
            df["transformer_prob"] = 0.5
        if "transformer_pred_ret" in df.columns:
            df["transformer_pred_ret"] = 0.5
        if "transformer_pred_ret_raw" in df.columns:
            df["transformer_pred_ret_raw"] = 0.0
        if "transformer_conf" in df.columns:
            df["transformer_conf"] = 0.0
        if "transformer_uncertainty" in df.columns:
            df["transformer_uncertainty"] = 1.0
    return df


def precompute_transformer_caches(stocks_data, settings) -> None:
    if not settings.backtest.precompute_transformer_cache:
        print("[Transformer预计算] 已关闭，直接进入回测。")
        return

    metadata = _load_transformer_training_metadata(settings)
    if not metadata:
        print("[Transformer预计算] 未找到训练边界元数据，跳过预计算；回测中将禁用 Transformer 因子。")
        return

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = None

    cache_hits = 0
    cache_misses = 0
    failures = []

    print("\n[2.5/3] 统一预计算 Transformer 因子缓存...")
    if device is not None:
        print(f"[Transformer预计算] 设备: {device}")

    for stock_name, stock_code in STOCK_CODES.items():
        df = stocks_data.get(stock_name)
        if df is None or df.empty:
            continue

        valid_dates = df["Close"].dropna().index if "Close" in df.columns else df.index
        current_last_date = valid_dates[-1] if len(valid_dates) > 0 else None
        if current_last_date is None:
            continue

        cached_df = load_transformer_cache(stock_code, current_last_date)
        if cached_df is not None:
            cache_hits += 1
            continue

        try:
            calculate_orthogonal_factors(
                df.copy(),
                stock_code=stock_code,
                device=device,
                allow_save_cache=True,
            )
            cache_misses += 1
        except Exception as exc:
            failures.append(f"{stock_name}({stock_code}): {exc}")

    print(
        f"[Transformer预计算] 完成 | 缓存命中={cache_hits} | 新生成={cache_misses} | 失败={len(failures)}"
    )
    for item in failures[:5]:
        print(f"[Transformer预计算][失败] {item}")


def _build_equity_curve(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_cash: float = 100000.0,
) -> pd.Series:
    """构建持仓策略的 equity 曲线（按日累乘）。

    ★ 修复2: 原实现用单步乘法 `equity[mask] = equity[mask] * (1 + daily)`，
    每天都从 initial_cash 出发，没有累积效应，导致 max_drawdown/sharpe/total_return 全部失真。
    新实现：
      1) 构造每日持仓收益序列 (持仓 * 当日涨跌)
      2) cumprod 得到净值因子
      3) 乘以初始资金得到 equity 曲线
    """
    if trades_df is None or len(trades_df) == 0:
        return pd.Series(initial_cash, index=df.index)

    stock_daily_ret = df['Close'].pct_change().fillna(0)
    position_status = pd.Series(0.0, index=df.index)

    for t in trades_df.itertuples():
        try:
            # Buy is executed on buy_date close; daily return exposure starts next trading day.
            holding_dates = df.index[(df.index > t.buy_date) & (df.index <= t.sell_date)]
            position_status.loc[holding_dates] = 1.0
        except KeyError:
            pass

    daily_portfolio_ret = position_status * stock_daily_ret
    equity = (1.0 + daily_portfolio_ret).cumprod() * initial_cash
    return equity


def _build_account_equity_curve(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_cash: float = 100000.0,
) -> pd.Series:
    """Build account-level equity curve from cash flows + mark-to-market."""
    if trades_df is None or len(trades_df) == 0:
        return pd.Series(initial_cash, index=df.index)

    df = df.sort_index()
    trades = trades_df.sort_values(["buy_date", "sell_date"]).reset_index(drop=True).copy()

    required_cols = {"buy_date", "sell_date", "shares", "actual_buy_cost", "actual_sell_proceeds"}
    if not required_cols.issubset(set(trades.columns)):
        return _build_equity_curve(df=df, trades_df=trades_df, initial_cash=initial_cash)

    equity = pd.Series(index=df.index, dtype=float)
    current_cash = float(initial_cash)
    in_position = False
    shares_held = 0
    sell_date = None
    current_trade = None
    trade_ptr = 0

    for date in df.index:
        close_price = float(df.at[date, "Close"]) if pd.notna(df.at[date, "Close"]) else np.nan

        if not in_position:
            while trade_ptr < len(trades):
                candidate_buy_date = pd.Timestamp(trades.at[trade_ptr, "buy_date"])
                if candidate_buy_date < date:
                    trade_ptr += 1
                    continue
                if candidate_buy_date > date:
                    break

                row = trades.iloc[trade_ptr]
                shares_held = int(row.get("shares", 0) or 0)
                buy_cost = float(row.get("actual_buy_cost", 0.0) or 0.0)
                if shares_held >= 1 and buy_cost > 0 and buy_cost <= current_cash + 1e-6:
                    current_cash -= buy_cost
                    current_trade = row
                    sell_date = pd.Timestamp(row["sell_date"])
                    in_position = True
                else:
                    trade_ptr += 1
                    current_trade = None
                    sell_date = None
                    shares_held = 0
                break

        if in_position:
            mark_to_market = current_cash
            if np.isfinite(close_price) and shares_held > 0:
                mark_to_market = current_cash + shares_held * close_price

            if date == sell_date:
                sell_proceeds = float(current_trade.get("actual_sell_proceeds", 0.0) or 0.0)
                if sell_proceeds > 0:
                    current_cash += sell_proceeds
                mark_to_market = current_cash

                in_position = False
                current_trade = None
                sell_date = None
                shares_held = 0
                trade_ptr += 1

            equity.loc[date] = mark_to_market
        else:
            equity.loc[date] = current_cash

    return equity.ffill().fillna(initial_cash)


def _build_benchmark_returns(market_data, df: pd.DataFrame) -> pd.Series:
    if market_data is None:
        return None
    try:
        bench = market_data['Close'].reindex(df.index).pct_change()
        bench = bench.fillna(0)
        if len(bench) == len(df):
            return bench
    except Exception:
        pass
    return None


def _build_test_dataframe_from_splits(df: pd.DataFrame, splits) -> pd.DataFrame:
    """Return the concatenated walk-forward test ranges for one stock."""
    if df is None or not splits:
        return pd.DataFrame()

    test_segments = []
    for split in splits:
        if len(split) < 6:
            continue
        seg_start, seg_end = split[4], split[5]
        seg_df = df.iloc[seg_start:seg_end]
        if len(seg_df) > 0:
            test_segments.append(seg_df)

    if not test_segments:
        return pd.DataFrame()

    test_df = pd.concat(test_segments).sort_index()
    return test_df[~test_df.index.duplicated(keep="first")]


def _calculate_daily_portfolio_stats(port_daily: pd.Series) -> dict:
    """Calculate portfolio metrics from a daily return curve, not fake trades."""
    returns = pd.Series(port_daily).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if returns.empty:
        return {}

    equity = (1.0 + returns).cumprod()
    total_return = (equity.iloc[-1] - 1.0) * 100
    years = max(len(returns) / 252.0, 1 / 252.0)
    ann_return = ((1.0 + total_return / 100.0) ** (1.0 / years) - 1.0) * 100 if total_return > -100 else -100.0

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = drawdown.min() * 100

    std = returns.std()
    sharpe = (returns.mean() / std) * np.sqrt(252) if std > 1e-12 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = (returns.mean() / downside_std) * np.sqrt(252) if len(downside) > 1 and downside_std > 1e-12 else 0.0

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / (gross_loss + 1e-12)
    calmar = ann_return / (abs(max_drawdown) + 1e-12)

    active_returns = returns[returns != 0]
    active_std = active_returns.std()
    sqn = (
        (active_returns.mean() / active_std) * np.sqrt(len(active_returns))
        if len(active_returns) > 1 and active_std > 1e-12 else 0.0
    )

    return {
        "total_return": float(total_return),
        "ann_return": float(ann_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "profit_factor": float(profit_factor),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "sqn": float(sqn),
    }


def process_single_stock(args):
    """
    处理单只股票的完整回测流程
    ★ 修复：walk_forward_split 返回6元组 (train_start, train_end, val_start, val_end, test_start, test_end)
    """
    t_start = time.time()

    try:
        stock_name, stock_data, stock_code = args
        skip, reason = should_intercept_stock(stock_code, stock_name, stock_data)
        if skip:
            print(f"[拦截-回测] 跳过 {stock_name} ({stock_code}): {reason}")
            return stock_code, None, None, None, None, None, None

        settings = get_settings()
        risk_mgr = RiskManager(settings.risk)
        transformer_metadata = _load_transformer_training_metadata(settings)

        df = stock_data.copy()
        if len(df) < 150:
            return stock_code, None, None, None, None, None, None

        # 1. 因子计算
        df = calculate_orthogonal_factors(
            df,
            stock_code,
            allow_save_cache=True,
            enable_transformer=bool(transformer_metadata),
        )

        # 2. Walk-Forward 划分
        splits = walk_forward_split(
            df,
            n_splits=settings.backtest.n_splits,
            train_ratio=settings.backtest.train_ratio,
            val_ratio=settings.backtest.val_ratio,
            gap_days=settings.backtest.gap_days,
            expanding_window=settings.backtest.expanding_window,
        )
        if not splits:
            return stock_code, None, None, None, None, None, None

        # ★ 修复：正确处理6元组 (train_start, train_end, val_start, val_end, test_start, test_end)
        validated_splits = []
        for s in splits:
            # s 是 6-tuple: (train_start, train_end, val_start, val_end, test_start, test_end)
            train_start, train_end, val_start, val_end, test_start, test_end = s
            if test_end <= len(df):
                validated_splits.append(s)

        if not validated_splits:
            return stock_code, None, None, None, None, None, None

        # 3. 多折优化和测试
        all_trades = []
        best_params_list = []
        best_weights_list = []
        transformer_ok_list = []
        total_commissions = 0.0
        rolling_capital = float(settings.backtest.initial_capital)

        for split_idx, split in enumerate(validated_splits):
            train_start, train_end, val_start, val_end, test_start, test_end = split

            train_df = df.iloc[train_start:train_end]
            val_df = df.iloc[val_start:val_end]
            test_df = df.iloc[test_start:test_end]

            if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 20:
                continue

            transformer_ok, transformer_reason = _is_transformer_valid_for_split(
                transformer_metadata, stock_code, train_df
            )
            if not transformer_ok:
                print(
                    f"  [Transformer-Disabled] {stock_code} split={split_idx + 1} "
                    f"reason={transformer_reason}"
                )
            train_df = _set_transformer_availability(train_df, transformer_ok)
            val_df = _set_transformer_availability(val_df, transformer_ok)
            test_df = _set_transformer_availability(test_df, transformer_ok)

            try:
                if settings.backtest.enable_val_validation:
                    best_params_map, best_weights = optimize_and_validate(
                        train_df, val_df, stock_code,
                        _worker_market_data, _worker_stocks_data,
                    )
                else:
                    best_params_map, best_weights = optimize_strategy(
                        train_df, stock_code, _worker_market_data, _worker_stocks_data,
                    )
            except Exception as e:
                print(f" [优化失败] {stock_name}: {e}")
                continue

            if not best_weights:
                factor_cols = [c for c in train_df.columns if c not in NON_FACTOR_COLS]
                best_weights = {c: 1.0 / len(factor_cols) for c in factor_cols}

            test_df = calculate_multi_timeframe_score(test_df, weights=best_weights)

            trades_df, stats, _ = run_backtest_loop(
                test_df, stock_code, _worker_market_data,
                best_weights, best_params_map,
                stocks_data=_worker_stocks_data,
                initial_capital=rolling_capital,
            )

            if trades_df is None or len(trades_df) == 0:
                continue

            if 'commission' in trades_df.columns:
                total_commissions += trades_df['commission'].sum()

            all_trades.append(trades_df)
            best_params_list.append(best_params_map)
            best_weights_list.append(best_weights)
            transformer_ok_list.append(transformer_ok)
            if stats is not None and stats.get("final_capital") is not None:
                rolling_capital = float(stats["final_capital"])

        if not all_trades:
            return stock_code, None, None, None, None, None, None

        combined_trades = pd.concat(all_trades, ignore_index=True)

        combined_test_df = _build_test_dataframe_from_splits(df, validated_splits)
        if combined_test_df.empty:
            combined_test_df = df.iloc[validated_splits[-1][4]:validated_splits[-1][5]]

        equity_curve = _build_account_equity_curve(
            combined_test_df,
            combined_trades,
            initial_cash=settings.backtest.initial_capital,
        )
        benchmark_returns = _build_benchmark_returns(_worker_market_data, combined_test_df)

        # 给报告打印用：保留最后一个 split 的起止日期作为"测试区间"展示
        last_split = validated_splits[-1]
        test_df_last = df.iloc[last_split[4]:last_split[5]]

        full_stats = calculate_comprehensive_stats(
            trades_df=combined_trades,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_returns,
            initial_cash=settings.backtest.initial_capital,
            commissions=total_commissions,
        )

        risk_result = risk_mgr.evaluate_soft_targets(full_stats)

        t_elapsed = time.time() - t_start
        print_stock_backtest_report(
            stock_name=stock_name,
            stock_code=stock_code,
            start_date=test_df_last.index[0] if len(test_df_last) > 0 else df.index[0],
            end_date=test_df_last.index[-1] if len(test_df_last) > 0 else df.index[-1],
            elapsed_seconds=t_elapsed,
            stats=full_stats,
            risk_result=risk_result,
        )

        if risk_result["discard"]:
            print(f" [DISCARD] {stock_name} - 核心风控指标未通过，策略丢弃")
            return stock_code, None, None, None, None, None, None

        if (full_stats.get('total_return', 0) <= 0
                or full_stats.get('win_rate', 0) < settings.risk.min_win_rate
                or full_stats.get('total_trades', 0) < settings.risk.min_trades
                or full_stats.get('total_trades', 0) > settings.risk.max_trades):
            print(f" [FILTER] {stock_name} - 额外筛选未通过")
            return stock_code, None, None, None, None, None, None

        print(f" [KEEP] {stock_name} - 通过全部检查")

        transformer_train_end = _get_transformer_train_end(transformer_metadata, stock_code)
        strategy_dict = {
            'name': stock_name,
            'params': best_params_list[0],
            'weights': best_weights_list[0],
            'transformer_boundary_ok': bool(any(transformer_ok_list)),
            'transformer_training_boundary': {
                'mode': 'per_split_train_end_guard',
                'train_end': str(transformer_train_end.date()) if transformer_train_end is not None else None,
            },
        }

        final_weights = best_weights_list[-1] if best_weights_list else {}
        df = calculate_multi_timeframe_score(df, weights=final_weights)

        metadata = {
            'processed_len': len(df),
            'validated_splits': validated_splits,
            'test_start_idx': validated_splits[-1][4] if validated_splits else int(len(df) * 0.7),
            'final_capital': rolling_capital,
        }

        return stock_code, strategy_dict, full_stats, df, combined_trades, validated_splits, metadata

    except Exception as e:
        print(f" [错误] 处理异常: {e}")
        traceback.print_exc()
        return args[2], None, None, None, None, None, None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    settings = get_settings()
    print("\n" + "=" * 80)
    print("增强版策略回测系统 V3 (5状态市场+批量推理+相关性风控)")
    print("=" * 80)

    # 1. 大盘数据
    print("\n[1/3] 检查大盘数据...")
    if check_and_clean_cache(settings.paths.market_cache_file):
        market_data = load_pickle_cache(settings.paths.market_cache_file)['market_data']
    else:
        market_data = download_market_data()
        if market_data is not None:
            save_pickle_cache(settings.paths.market_cache_file, {
                'market_data': market_data,
                'last_date': market_data.index[-1].strftime('%Y-%m-%d'),
            })
    if market_data is None:
        exit()

    # 2. 个股数据
    print("\n[2/3] 检查个股数据...")
    if check_and_clean_cache(settings.paths.stock_cache_file):
        stocks_data = load_pickle_cache(settings.paths.stock_cache_file)['stocks_data']
    else:
        stocks_data = download_stocks_data(STOCK_CODES)
        if stocks_data:
            last_dates = [df.index[-1] for df in stocks_data.values() if not df.empty]
            last_date = max(last_dates).strftime('%Y-%m-%d') if last_dates else None
            save_pickle_cache(settings.paths.stock_cache_file, {
                'stocks_data': stocks_data,
                'last_date': last_date,
            })
    if not stocks_data:
        exit()

    precompute_transformer_caches(stocks_data, settings)

    # 3. 并行回测
    print("\n[3/3] 开始策略优化与回测...")

    name_to_code_map = {name: STOCK_CODES.get(name) for name in stocks_data.keys() if STOCK_CODES.get(name)}
    clean_map = filter_codes_by_name(name_to_code_map)

    stock_list = [
        (name, stocks_data[name], code)
        for name, code in clean_map.items()
    ]

    regime_count = 5
    estimated_trials = len(stock_list) * settings.backtest.n_optuna_trials * regime_count
    print(
        f"[优化规模] 股票数={len(stock_list)} | regime数={regime_count} | "
        f"每个regime trials={settings.backtest.n_optuna_trials} | "
        f"估算总trial数={estimated_trials}"
    )
    print(
        f"[优化模式] 验证集复核={'开启' if settings.backtest.enable_val_validation else '关闭'} | "
        f"expanding_window={settings.backtest.expanding_window}"
    )

    # ★ 次要修复: GPU 推理多进程会让每个 worker 都把 ensemble 模型加载到同一张卡，
    # 12GB 显存会爆。GPU 时强制最多 2 进程；CPU 推理才放开多进程
    try:
        import torch as _torch
        gpu_available = _torch.cuda.is_available()
    except Exception:
        gpu_available = False

    reserved_cores = 2
    use_processes = max(1, cpu_count() - reserved_cores)
    if gpu_available:
        print(f"[多进程] 检测到 GPU，使用 {use_processes} 进程（总核心数 - {reserved_cores}）")
    else:
        print(f"[多进程] CPU 模式，使用 {use_processes} 进程（总核心数 - {reserved_cores}）")

    with Pool(
        processes=use_processes,
        initializer=init_worker,
        initargs=(market_data, stocks_data, False),
    ) as pool:
        raw_results = list(
            tqdm(
                pool.imap_unordered(process_single_stock, stock_list),
                total=len(stock_list),
                desc="进度",
            )
        )

    # 4. 结果汇总
    results = [r for r in raw_results if len(r) == 7 and r[1] is not None]
    all_strategies = {}
    all_stats = {}

    for code, strat, stat, df, trades, splits, metadata in results:
        if strat:
            all_strategies[code] = strat
        if stat:
            all_stats[strat['name']] = stat

    # 汇总表格
    print("\n" + "=" * 80)
    print("测试集汇总报告")
    print("=" * 80)

    sorted_stats = sorted(all_stats.items(), key=lambda x: x[1].get('total_return', 0), reverse=True)

    header = (
        f"{'名称':<10} {'收益%':>8} {'胜率%':>7} {'交易':>5} "
        f"{'夏普':>7} {'最大回撤%':>10} {'利润因子':>8} "
        f"{'Sortino':>8} {'Calmar':>8} {'SQN':>6} {'Kelly':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, s in sorted_stats:
        print(
            f"{name:<10} "
            f"{s.get('total_return', 0):>8.2f} "
            f"{s.get('win_rate', 0):>7.1f} "
            f"{s.get('total_trades', 0):>5} "
            f"{s.get('sharpe_ratio', 0):>7.2f} "
            f"{s.get('max_drawdown', 0):>10.2f} "
            f"{s.get('profit_factor', 0):>8.2f} "
            f"{s.get('sortino_ratio', 0):>8.2f} "
            f"{s.get('calmar_ratio', 0):>8.2f} "
            f"{s.get('sqn', 0):>6.2f} "
            f"{s.get('kelly_criterion', 0):>6.3f}"
        )

    # 组合回测
    print("\n" + "=" * 80)
    print("【组合回测】等权组合测试集总收益")
    print("=" * 80)

    portfolio_test_frames = []
    for _, _, _, df, _, splits, _ in results:
        test_df = _build_test_dataframe_from_splits(df, splits)
        if not test_df.empty:
            portfolio_test_frames.append(test_df)

    all_dates = sorted(set().union(*[set(df.index) for df in portfolio_test_frames])) if portfolio_test_frames else []
    portfolio = pd.DataFrame(index=all_dates)
    portfolio['return'] = 0.0

    n_valid = len(portfolio_test_frames)
    if n_valid == 0:
        print("警告: 没有有效策略，无法计算组合收益")
    else:
        for code, strat, stat, df, trades, splits, metadata in results:
            if strat is None or df is None or trades is None:
                continue
            test_df = _build_test_dataframe_from_splits(df, splits)
            if test_df.empty:
                continue

            stock_equity = _build_account_equity_curve(
                df=test_df,
                trades_df=trades,
                initial_cash=settings.backtest.initial_capital,
            )
            strategy_daily_ret = stock_equity.pct_change().reindex(portfolio.index).fillna(0.0)
            portfolio['return'] += strategy_daily_ret / n_valid

        portfolio['cum_ret'] = (1 + portfolio['return'].fillna(0)).cumprod()
        total_ret = (portfolio['cum_ret'].iloc[-1] - 1) * 100
        print(f"组合总收益: {total_ret:.2f}%")

        port_daily = portfolio['return'].fillna(0)
        port_stats = _calculate_daily_portfolio_stats(port_daily)
        if port_stats:
            print(
                f"组合夏普: {port_stats.get('sharpe_ratio', 0):.2f} | "
                f"最大回撤: {port_stats.get('max_drawdown', 0):.2f}% | "
                f"利润因子: {port_stats.get('profit_factor', 0):.2f} | "
                f"Sortino: {port_stats.get('sortino_ratio', 0):.2f} | "
                f"Calmar: {port_stats.get('calmar_ratio', 0):.2f} | "
                f"SQN: {port_stats.get('sqn', 0):.2f}"
            )

    # 保存策略参数
    with open(settings.paths.strategy_file, 'w', encoding='utf-8') as f:
        json.dump(all_strategies, f, ensure_ascii=False, indent=4)
    print(f"\n✓ 策略参数已写入: {settings.paths.strategy_file}")
    print(f"  保留策略数: {len(all_strategies)} / 总股票数: {len(stock_list)}")

    # 可视化
    print("\n" + "=" * 80)
    print("开始生成可视化图表...")
    print("=" * 80)

    viz_dir = os.path.join(settings.paths.result_dir, 'backtest_charts')
    os.makedirs(viz_dir, exist_ok=True)

    for code, strat, stat, df, trades, splits, metadata in results:
        if strat is None or trades is None or len(trades) == 0:
            continue

        stock_name = strat['name']
        chart_path = os.path.join(viz_dir, f'{stock_name}_{code}_backtest.png')

        try:
            actual_len = len(df)
            split_idx = int(actual_len * 0.7)

            if metadata and 'test_start_idx' in metadata:
                idx_from_meta = metadata['test_start_idx']
                if 0 < idx_from_meta < actual_len:
                    split_idx = idx_from_meta
            elif splits and len(splits) > 0:
                # ★ 修复：6元组，test_start 是 index 4
                test_start = splits[-1][4]
                if 0 < test_start < actual_len:
                    split_idx = test_start

            split_idx = max(1, min(split_idx, actual_len - 1))

            visualize_backtest_with_split(
                df=df, trades_df=trades, stock_name=stock_name,
                split_idx=split_idx, market_data=market_data,
                save_path=chart_path, strat=strat,
            )
            print(f" ✓ {stock_name} 图表已保存")
        except Exception as e:
            print(f" ✗ {stock_name} 图表生成失败: {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"所有可视化图表生成完成！目录: {viz_dir}")
    print("=" * 80)
