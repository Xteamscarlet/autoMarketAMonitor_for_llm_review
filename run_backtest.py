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


def init_worker(m_data, s_data):
    global _worker_market_data, _worker_stocks_data
    _worker_market_data = m_data
    _worker_stocks_data = s_data
    try:
        import torch
        from model.predictor import _load_ensemble_models
        _load_ensemble_models(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception:
        pass


def _build_equity_curve(df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.Series:
    """构建持仓策略的 equity 曲线（按日累乘）。

    ★ 修复2: 原实现用单步乘法 `equity[mask] = equity[mask] * (1 + daily)`，
    每天都从 initial_cash 出发，没有累积效应，导致 max_drawdown/sharpe/total_return 全部失真。
    新实现：
      1) 构造每日持仓收益序列 (持仓 * 当日涨跌)
      2) cumprod 得到净值因子
      3) 乘以初始资金得到 equity 曲线
    """
    initial_cash = 100000.0
    if trades_df is None or len(trades_df) == 0:
        return pd.Series(initial_cash, index=df.index)

    stock_daily_ret = df['Close'].pct_change().fillna(0)
    position_status = pd.Series(0.0, index=df.index)

    for t in trades_df.itertuples():
        try:
            holding_dates = df.loc[t.buy_date: t.sell_date].index
            position_status.loc[holding_dates] = 1.0
        except KeyError:
            pass

    daily_portfolio_ret = position_status * stock_daily_ret
    equity = (1.0 + daily_portfolio_ret).cumprod() * initial_cash
    return equity


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

        df = stock_data.copy()
        if len(df) < 150:
            return stock_code, None, None, None, None, None, None

        # 1. 因子计算
        df = calculate_orthogonal_factors(df, stock_code, allow_save_cache=True)

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
        total_commissions = 0.0

        for split_idx, split in enumerate(validated_splits):
            train_start, train_end, val_start, val_end, test_start, test_end = split

            train_df = df.iloc[train_start:train_end]
            val_df = df.iloc[val_start:val_end]
            test_df = df.iloc[test_start:test_end]

            if len(train_df) < 100 or len(val_df) < 50 or len(test_df) < 20:
                continue

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
                best_weights, best_params_map, stocks_data=_worker_stocks_data
            )

            if trades_df is None or len(trades_df) == 0:
                continue

            if 'commission' in trades_df.columns:
                total_commissions += trades_df['commission'].sum()

            all_trades.append(trades_df)
            best_params_list.append(best_params_map)
            best_weights_list.append(best_weights)

        if not all_trades:
            return stock_code, None, None, None, None, None, None

        combined_trades = pd.concat(all_trades, ignore_index=True)

        # ★ 修复2: equity_curve 应覆盖所有 split 的 test 区间，而非只用最后一个 split
        # 否则 trades(全 split) 和 equity(单 split) 时间不匹配，stats 全错
        test_segments = []
        for s in validated_splits:
            seg_start, seg_end = s[4], s[5]
            seg_df = df.iloc[seg_start:seg_end]
            if len(seg_df) > 0:
                test_segments.append(seg_df)

        if test_segments:
            combined_test_df = pd.concat(test_segments)
            # 去重相邻 split 重叠日期（理论上 walk-forward + gap_days 不会重叠，但保险）
            combined_test_df = combined_test_df[~combined_test_df.index.duplicated(keep='first')]
        else:
            combined_test_df = df.iloc[validated_splits[-1][4]:validated_splits[-1][5]]

        equity_curve = _build_equity_curve(combined_test_df, combined_trades)
        benchmark_returns = _build_benchmark_returns(_worker_market_data, combined_test_df)

        # 给报告打印用：保留最后一个 split 的起止日期作为"测试区间"展示
        last_split = validated_splits[-1]
        test_df_last = df.iloc[last_split[4]:last_split[5]]

        full_stats = calculate_comprehensive_stats(
            trades_df=combined_trades,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_returns,
            initial_cash=100000.0,
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

        strategy_dict = {
            'name': stock_name,
            'params': best_params_list[0],
            'weights': best_weights_list[0],
        }

        final_weights = best_weights_list[-1] if best_weights_list else {}
        df = calculate_multi_timeframe_score(df, weights=final_weights)

        metadata = {
            'processed_len': len(df),
            'validated_splits': validated_splits,
            'test_start_idx': validated_splits[-1][4] if validated_splits else int(len(df) * 0.7),
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

    # 3. 并行回测
    print("\n[3/3] 开始策略优化与回测...")

    name_to_code_map = {name: STOCK_CODES.get(name) for name in stocks_data.keys() if STOCK_CODES.get(name)}
    clean_map = filter_codes_by_name(name_to_code_map)

    stock_list = [
        (name, stocks_data[name], code)
        for name, code in clean_map.items()
    ]

    # ★ 次要修复: GPU 推理多进程会让每个 worker 都把 ensemble 模型加载到同一张卡，
    # 12GB 显存会爆。GPU 时强制最多 2 进程；CPU 推理才放开多进程
    try:
        import torch as _torch
        gpu_available = _torch.cuda.is_available()
    except Exception:
        gpu_available = False

    if gpu_available:
        use_processes = min(2, max(1, cpu_count() - 1))
        print(f"[多进程] 检测到 GPU，限制为 {use_processes} 进程以避免显存爆炸")
    else:
        use_processes = max(1, cpu_count() - 1)
        print(f"[多进程] CPU 模式，使用 {use_processes} 进程")

    with Pool(processes=use_processes, initializer=init_worker, initargs=(market_data, stocks_data)) as pool:
        raw_results = list(tqdm(pool.imap(process_single_stock, stock_list), total=len(stock_list), desc="进度"))

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

    all_dates = sorted(
        set().union(*[set(df.index) for _, _, _, df, _, _, _ in results if df is not None])
    )
    portfolio = pd.DataFrame(index=all_dates)
    portfolio['return'] = 0.0

    n_valid = len(results)
    if n_valid == 0:
        print("警告: 没有有效策略，无法计算组合收益")
    else:
        for code, strat, stat, df, trades, splits, metadata in results:
            if strat is None or df is None or trades is None:
                continue

            stock_daily_ret = df['Close'].pct_change()
            position_status = pd.Series(0, index=df.index)

            for t in trades.itertuples():
                try:
                    holding_dates = df.loc[t.buy_date: t.sell_date].index
                    position_status.loc[holding_dates] = 1
                except KeyError:
                    pass

            strategy_daily_ret = position_status * stock_daily_ret
            portfolio['return'] += strategy_daily_ret / n_valid

        portfolio['cum_ret'] = (1 + portfolio['return'].fillna(0)).cumprod()
        total_ret = (portfolio['cum_ret'].iloc[-1] - 1) * 100
        print(f"组合总收益: {total_ret:.2f}%")

        port_daily = portfolio['return'].fillna(0)
        port_trades = []
        in_trade = False
        buy_val = 1.0
        for date_val, ret in port_daily.items():
            if ret != 0 and not in_trade:
                in_trade = True
                buy_val = 1.0
            if in_trade:
                buy_val *= (1 + ret)
            if ret == 0 and in_trade:
                port_trades.append({'net_return': buy_val - 1})
                in_trade = False
        if in_trade:
            port_trades.append({'net_return': buy_val - 1})

        if port_trades:
            port_stats = calculate_comprehensive_stats(pd.DataFrame(port_trades))
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
