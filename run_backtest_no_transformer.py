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
    download_market_data,
    download_stocks_data,
    check_and_clean_cache,
    load_pickle_cache,
    save_pickle_cache,
)
from data.types import NON_FACTOR_COLS
from backtest.engine_no_transformer import run_backtest_loop_no_transformer, \
    calculate_multi_timeframe_score_no_transformer
from backtest.optimizer import optimize_strategy, walk_forward_split, calculate_dynamic_weights
from backtest.evaluator import calculate_comprehensive_stats
from backtest.visualizer import visualize_backtest_with_split
from backtest.report import print_stock_backtest_report
from risk_manager import RiskManager
from utils.stock_filter import filter_codes_by_name, should_intercept_stock

# ==================== 多进程全局变量 ====================
_worker_market_data = None
_worker_stocks_data = None


def init_worker(m_data, s_data):
    global _worker_market_data, _worker_stocks_data
    _worker_market_data = m_data
    _worker_stocks_data = s_data


def _build_equity_curve(df: pd.DataFrame, trades_df: pd.DataFrame, initial_cash: float = 100000.0):
    """构建资金曲线 - 修改版：确保资金不会为负"""
    if trades_df is None or len(trades_df) == 0:
        return None

    equity = initial_cash
    equity_curve = {df.index[0]: equity}

    for t in trades_df.itertuples():
        # 确保不会出现负资金
        if equity <= 0:
            break

        # 计算新权益，确保不会为负
        new_equity = equity * (1 + t.net_return)
        if new_equity < 0:
            new_equity = 0  # 爆仓处理

        equity = new_equity
        equity_curve[t.sell_date] = equity

    return pd.Series(equity_curve).sort_index()


def _build_benchmark_returns(market_data: pd.DataFrame, df: pd.DataFrame):
    """构建与 df 对齐的基准日收益率（大盘涨跌幅）"""
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


def process_single_stock_no_transformer(args):
    """ 子进程处理单只股票的完整流程（无Transformer版本）：
    1. 因子计算（仅传统因子）
    2. Walk-Forward 划分
    3. 多折优化 + 测试
    4. 风控评估
    5. 打印报告
    6. 决定保留/丢弃
    """
    t_start = time.time()
    try:
        stock_name, stock_data, stock_code = args
        # 统一拦截
        skip, reason = should_intercept_stock(stock_code, stock_name, stock_data)
        if skip:
            print(f"[拦截-回测] 跳过 {stock_name} ({stock_code}): {reason}")
            return stock_code, None, None, None, None, None, None

        settings = get_settings()
        risk_mgr = RiskManager(settings.risk)
        df = stock_data.copy()
        if len(df) < 150:
            return stock_code, None, None, None, None, None, None

        # ========== 使用无Transformer版本的因子计算 ==========
        from data.indicators_no_transformer import calculate_orthogonal_factors_no_transformer
        df = calculate_orthogonal_factors_no_transformer(df, stock_code=stock_code)

        # Walk-Forward 划分
        splits = walk_forward_split(df, n_splits=settings.backtest.n_splits, train_ratio=settings.backtest.train_ratio,
                                    val_ratio=settings.backtest.val_ratio)
        if not splits:
            return stock_code, None, None, None, None, None, None

        all_trades = []
        best_params_list = []
        best_weights_list = []
        validated_splits = []
        total_commissions = 0.0

        # 获取因子列（排除Transformer因子）
        base_cols = set(NON_FACTOR_COLS)
        transformer_cols = ['transformer_prob', 'transformer_pred_ret', 'transformer_conf']
        factor_cols = [col for col in df.columns if col not in base_cols and col not in transformer_cols]

        # ✅ 修改：使用6个变量解包
        for train_start, train_end, val_start, val_end, test_start, test_end in splits:
            train_df = df.iloc[train_start:train_end]
            val_df = df.iloc[val_start:val_end]
            # 新增验证集
            test_df = df.iloc[test_start:test_end]

            if len(train_df) < 100 or len(test_df) < 20:
                continue

            # 参数优化（无Transformer版本）
            best_params_map, best_weights = optimize_strategy_no_transformer(
                train_df, factor_cols, settings.backtest.n_optuna_trials
            )
            if not best_params_map:
                continue

            # 硬限制检查
            try:
                RiskManager.check_hard_limits(best_params_map)
            except ValueError as e:
                print(f"[硬限制] {stock_name} 参数不合法: {e}")
                continue

            # ✅ 修改：保存6个元素
            validated_splits.append((train_start, train_end, val_start, val_end, test_start, test_end))

            # 计算得分
            test_df = calculate_multi_timeframe_score_no_transformer(test_df, weights=best_weights)

            # 回测（无Transformer版本）
            trades_df, stats, _ = run_backtest_loop_no_transformer(
                test_df, stock_code, _worker_market_data, best_weights, best_params_map, stocks_data=_worker_stocks_data
            )
            if trades_df is None or len(trades_df) == 0:
                continue

            if 'commission' in trades_df.columns:
                total_commissions += trades_df['commission'].sum()

            all_trades.append(trades_df)
            best_params_list.append(best_params_map)
            best_weights_list.append(best_weights)

        # ========== 修改结束 ==========

        if not all_trades:
            return stock_code, None, None, None, None, None, None

        combined_trades = pd.concat(all_trades, ignore_index=True)

        # 构建资金曲线和基准 - 使用修改后的函数
        # ★ 修复：6元组 (train_start, train_end, val_start, val_end, test_start, test_end)
        # test 起止是 [4] 和 [5]，不是 [2] 和 [3]
        last_split = validated_splits[-1]
        test_df_last = df.iloc[last_split[4]:last_split[5]]
        equity_curve = _build_equity_curve(test_df_last, combined_trades)
        benchmark_returns = _build_benchmark_returns(_worker_market_data, test_df_last)

        # 计算完整统计
        full_stats = calculate_comprehensive_stats(
            trades_df=combined_trades,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_returns,
            initial_cash=100000.0,
            commissions=total_commissions,
        )

        # 风控评估
        risk_result = risk_mgr.evaluate_soft_targets(full_stats)

        # 打印报告
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

        # 根据风控结论决定保留/丢弃
        if risk_result["discard"]:
            print(f" [DISCARD] {stock_name} - 核心风控指标未通过，策略丢弃")
            return stock_code, None, None, None, None, None, None

        # 额外筛选：收益和交易次数
        if (full_stats.get('total_return', 0) <= 0 or full_stats.get('win_rate',
                                                                     0) < settings.risk.min_win_rate or full_stats.get(
                'total_trades', 0) < 3):
            print(f" [DISCARD] {stock_name} - 收益/胜率/交易次数不达标")
            return stock_code, None, None, None, None, None, None

        # 构建策略参数
        strategy = {
            'code': stock_code,
            'name': stock_name,
            'params': best_params_list[-1] if best_params_list else {},
            'weights': best_weights_list[-1] if best_weights_list else {},
            'stats': full_stats,
        }
        # ★ 修复：6元组中 test_start 是 [4]
        metadata = {
            'test_start_idx': last_split[4],
            'n_splits': len(validated_splits),
        }
        print(f" [KEEP] {stock_name} - 策略通过所有检查")
        return stock_code, strategy, full_stats, df, combined_trades, validated_splits, metadata

    except Exception as e:
        print(f"[ERROR] 处理股票失败: {e}")
        traceback.print_exc()
        return args[2], None, None, None, None, None, None


def optimize_strategy_no_transformer(df: pd.DataFrame, factor_cols: list, n_trials: int = 50):
    """参数优化（无Transformer版本）
    与原版本的区别：
    1. 不优化 transformer_weight
    2. 不优化 transformer_buy_threshold
    3. 不优化 confidence_threshold
    """
    import optuna
    from data.types import ALL_REGIMES
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 计算动态权重
    weights = calculate_dynamic_weights(df, factor_cols)

    def objective(trial, regime: str):
        params = {
            'buy_threshold': trial.suggest_float('buy_threshold', 0.4, 0.8),
            'sell_threshold': trial.suggest_float('sell_threshold', -0.3, 0.0),
            'stop_loss': trial.suggest_float('stop_loss', -0.15, -0.03),
            'hold_days': trial.suggest_int('hold_days', 5, 30),
            'trailing_profit_level1': trial.suggest_float('trailing_profit_level1', 0.03, 0.10),
            'trailing_profit_level2': trial.suggest_float('trailing_profit_level2', 0.08, 0.20),
            'trailing_drawdown_level1': trial.suggest_float('trailing_drawdown_level1', 0.05, 0.12),
            'trailing_drawdown_level2': trial.suggest_float('trailing_drawdown_level2', 0.02, 0.06),
            'take_profit_multiplier': trial.suggest_float('take_profit_multiplier', 2.0, 5.0),
            # 无Transformer版本：移除以下参数
            # 'transformer_weight': trial.suggest_float('transformer_weight', 0.0, 0.5),
            # 'transformer_buy_threshold': trial.suggest_float('transformer_buy_threshold', 0.5, 0.8),
            # 'confidence_threshold': trial.suggest_float('confidence_threshold', 0.4, 0.7),
        }

        test_df = calculate_multi_timeframe_score_no_transformer(df.copy(), weights=weights)
        trades_df, stats, _ = run_backtest_loop_no_transformer(
            test_df, 'test', None, weights, {'neutral': params}
        )
        if trades_df is None or len(trades_df) < 3:
            return -100.0, -100.0, -100.0

        ret = stats['total_return']
        mdd = full_stats.get('max_drawdown', 0) if (full_stats := calculate_comprehensive_stats(trades_df)) else 0
        sharpe = full_stats.get('sharpe_ratio', 0) if full_stats else 0
        penalty = -1.0 if mdd < -20 else 0
        return float(ret), float(-mdd + penalty), float(sharpe)

    best_params_map = {}
    for regime in ALL_REGIMES:
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(lambda t: objective(t, regime), n_trials=n_trials)

        if len(study.best_trials) > 0:
            best_t = max(study.best_trials, key=lambda t: t.values[2])
            best_params_map[regime] = best_t.params
        else:
            best_params_map[regime] = {
                'buy_threshold': 0.55,
                'sell_threshold': -0.15,
                'hold_days': 15,
                'stop_loss': -0.08,
                'trailing_profit_level1': 0.06,
                'trailing_profit_level2': 0.12,
                'trailing_drawdown_level1': 0.08,
                'trailing_drawdown_level2': 0.04,
                'take_profit_multiplier': 3.0,
            }
    return best_params_map, weights


def main():
    """主函数"""
    print("=" * 80)
    print("增强版策略回测系统 V3 (无Transformer版本)")
    print("=" * 80)
    settings = get_settings()

    # ========== 结果保存到独立目录 ==========
    # 使用不同的输出目录，不覆盖原有结果
    output_dir = "./stock_cache/no_transformer_results"
    os.makedirs(output_dir, exist_ok=True)

    # 图表目录
    viz_dir = os.path.join(output_dir, "backtest_charts")
    os.makedirs(viz_dir, exist_ok=True)

    # 1. 检查大盘数据
    print("\n[1/3] 检查大盘数据...")
    # ========== 修改1：使用独立的缓存路径 ==========
    market_cache_file = "./stock_cache/no_transformer_market_data.pkl"
    try:
        if not check_and_clean_cache(market_cache_file):
            market_data = download_market_data()
            save_pickle_cache(market_cache_file, market_data)
        else:
            market_data = load_pickle_cache(market_cache_file)
    except Exception as e:
        print(f"警告: 大盘数据加载失败: {e}")
        print("尝试使用本地缓存...")
        if os.path.exists(market_cache_file):
            market_data = load_pickle_cache(market_cache_file)
        else:
            print("错误: 没有可用的大盘数据")
            return

    # ========== 修改2：使用独立的缓存路径 ==========
    print("\n[2/3] 检查个股数据...")
    # ========== 统一使用 check_and_clean_cache ==========
    stocks_cache_file = "./stock_cache/no_transformer_stocks_data.pkl"
    if not check_and_clean_cache(stocks_cache_file):
        print("下载股票数据...")
        stocks_data = download_stocks_data(STOCK_CODES)
        save_pickle_cache(stocks_cache_file, stocks_data)
    else:
        print("使用缓存的股票数据...")
        stocks_data = load_pickle_cache(stocks_cache_file)

    # 数据验证
    if not stocks_data:
        print("错误: 无法获取个股数据")
        return
    print(f"stocks_data 类型: {type(stocks_data)}")
    print(f"stocks_data 长度: {len(stocks_data)}")
    print(f"stocks_data 键示例: {list(stocks_data.keys())[:3]}")

    # 3. 验证数据结构
    print(f"\n数据验证:")
    print(f" 类型: {type(stocks_data)}")
    print(f" 长度: {len(stocks_data) if stocks_data else 0}")
    if stocks_data and len(stocks_data) > 0:
        print(f" 前3个键: {list(stocks_data.keys())[:3]}")

    # 检查是否是错误的结构
    first_key = list(stocks_data.keys())[0]
    if first_key in ['stocks_data', 'last_date']:
        print("\n检测到错误的数据结构，正在修复...")
        if isinstance(stocks_data, dict) and 'stocks_data' in stocks_data:
            stocks_data = stocks_data['stocks_data']
        print(f"修复后键示例: {list(stocks_data.keys())[:3]}")

    if not stocks_data or len(stocks_data) == 0:
        print("\n错误: 无法获取有效的股票数据")
        return

    # 4. 保存到独立缓存
    save_pickle_cache(stocks_cache_file, stocks_data)
    print(f"\n✓ 缓存已保存到: {stocks_cache_file}")

    # 5. 过滤股票
    print("\n过滤股票...")
    valid_stocks = filter_codes_by_name(STOCK_CODES, verbose=True)
    print(f"过滤后股票数: {len(valid_stocks)}")

    # 6. 构建参数列表
    args_list = []
    matched = 0
    for name, code in valid_stocks.items():
        if name in stocks_data:
            df = stocks_data[name]
            args_list.append((name, df, code))
            matched += 1
        else:
            print(f"警告: 股票 {name} ({code}) 在列表中但缺少数据")

    print(f"\n匹配结果:")
    print(f" 过滤后股票数: {len(valid_stocks)}")
    print(f" 成功匹配数: {matched}")
    print(f" 最终有效股票数: {len(args_list)}")

    if len(args_list) == 0:
        print("\n❌ 错误: 没有有效的股票数据")
        return

    # 7. 多进程回测
    print(f"\n[3/3] 开始回测 {len(args_list)} 只股票...")
    results = []
    n_workers = min(cpu_count(), 4)
    with Pool(n_workers, initializer=init_worker, initargs=(market_data, stocks_data)) as pool:
        for result in tqdm(pool.imap(process_single_stock_no_transformer, args_list), total=len(args_list)):
            if result[1] is not None:
                results.append(result)

    # 4. 汇总报告
    print("\n" + "=" * 80)
    print("测试集汇总报告（无Transformer版本）")
    print("=" * 80)
    print(f"{'名称':<12} {'收益%':>10} {'胜率%':>8} {'交易':>6} {'夏普':>8} {'最大回撤%':>10} {'利润因子':>8}")
    print("-" * 80)
    for code, strat, stat, df, trades, splits, metadata in results:
        if stat is None:
            continue
        s = stat
        print(
            f"{strat['name']:<12} "
            f"{s.get('total_return', 0):>10.2f} "
            f"{s.get('win_rate', 0):>8.2f} "
            f"{s.get('total_trades', 0):>6} "
            f"{s.get('sharpe_ratio', 0):>8.2f} "
            f"{s.get('max_drawdown', 0):>10.2f} "
            f"{s.get('profit_factor', 0):>8.2f}"
        )

    # 5. 组合回测
    print("\n" + "=" * 80)
    print("【组合回测】等权组合测试集总收益（无Transformer版本）")
    print("=" * 80)
    n_valid = len(results)
    if n_valid == 0:
        print("警告: 没有有效策略，无法计算组合收益")
    else:
        all_dates = sorted(
            set().union(*[set(df.index) for _, _, _, df, _, _, _ in results if df is not None])
        )
        portfolio = pd.DataFrame(index=all_dates)
        portfolio['return'] = 0.0
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

    # 6. 保存策略参数到独立文件
    output_file = os.path.join(output_dir, "optimized_strategies_no_transformer.json")
    strategies_to_save = [
        {
            'code': s['code'],
            'name': s['name'],
            'params': s['params'],
            'weights': {k: float(v) for k, v in s['weights'].items()} if s['weights'] else {},
            'stats': s['stats'],
        }
        for _, s, _, _, _, _, _ in results if s is not None
    ]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(strategies_to_save, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ 策略参数已写入: {output_file}")
    print(f" 保留策略数: {len(strategies_to_save)} / 总股票数: {len(args_list)}")

    # 7. 生成可视化图表
    print("\n" + "=" * 80)
    print("开始生成可视化图表...")
    print("=" * 80)
    for code, strat, stat, df, trades, splits, metadata in results:
        if strat is None or trades is None or len(trades) == 0:
            continue
        stock_name = strat['name']
        chart_path = os.path.join(viz_dir, f'{stock_name}_{code}_backtest_no_transformer.png')
        try:
            actual_len = len(df)
            split_idx = int(actual_len * 0.7)
            if metadata and 'test_start_idx' in metadata:
                idx_from_meta = metadata['test_start_idx']
                if 0 < idx_from_meta < actual_len:
                    split_idx = idx_from_meta
            elif splits and len(splits) > 0:
                # ★ 修复：6元组中 test_start 是 [4]
                test_start = splits[-1][4]
                if 0 < test_start < actual_len:
                    split_idx = test_start
            split_idx = max(1, min(split_idx, actual_len - 1))

            visualize_backtest_with_split(
                df=df,
                trades_df=trades,
                stock_name=stock_name,
                split_idx=split_idx,
                market_data=market_data,
                save_path=chart_path,
                strat=strat,
            )
            print(f" ✓ {stock_name} 图表已保存")
        except Exception as e:
            print(f" ✗ {stock_name} 图表生成失败: {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"所有可视化图表生成完成！目录: {viz_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
