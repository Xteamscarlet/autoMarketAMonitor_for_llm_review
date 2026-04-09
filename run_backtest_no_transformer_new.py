# -*- coding: utf-8 -*-
"""
回测入口脚本（无Transformer版本）- 增强版
执行 Walk-Forward 回测、参数优化、风控过滤、报告输出、可视化
结果保存到独立目录，不覆盖原有Transformer版本的结果
新增：详细日志、异常处理、降级筛选机制
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
from data import check_and_clean_cache, save_pickle_cache, load_pickle_cache
from data.indicators_new import prepare_stock_data,calculate_orthogonal_factors_without_transformer
from data.loader_new import download_market_data, download_stocks_data
# from data import (
#     download_market_data, download_stocks_data,
#     check_and_clean_cache, load_pickle_cache,
#     save_pickle_cache,
# )
from data.types import NON_FACTOR_COLS, TRADITIONAL_FACTOR_COLS
from backtest.engine_no_transformer_new import run_backtest_loop_no_transformer, \
    calculate_multi_timeframe_score_no_transformer
from backtest.optimizer_new import optimize_strategy, walk_forward_split, calculate_dynamic_weights
from backtest.evaluator import calculate_comprehensive_stats
from backtest.visualizer import visualize_backtest_with_split
from backtest.report import print_stock_backtest_report
from risk_manager import RiskManager
from utils.stock_filter_new import filter_codes_by_name, should_intercept_stock

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_no_transformer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 多进程全局变量 ====================
_worker_market_data = None
_worker_stocks_data = None

# ========== 结果保存到独立目录 ==========
# 使用不同的输出目录，不覆盖原有结果
output_dir = "./stock_cache/no_transformer_results"
os.makedirs(output_dir, exist_ok=True)

# 图表目录
viz_dir = os.path.join(output_dir, "backtest_charts")
os.makedirs(viz_dir, exist_ok=True)


# ==================== 新增：数据完整性检查函数 ====================
def validate_data_for_backtest(
        df: pd.DataFrame,
        stock_code: str,
        min_days: int = 60,
) -> tuple:
    """验证数据是否适合回测

    Args:
        df: 数据框
        stock_code: 股票代码
        min_days: 最小数据天数

    Returns:
        (是否有效, 问题列表)
    """
    issues = []

    if df is None or len(df) == 0:
        issues.append("数据为空")
        return False, issues

    if len(df) < min_days:
        issues.append(f"数据天数 {len(df)} < {min_days}")

    # 检查必需列
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"缺失列: {missing}")

    # 检查缺失值
    for col in required_cols:
        if col in df.columns:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio > 0.1:
                issues.append(f"{col} 缺失比例 {nan_ratio:.1%}")

    # 检查价格有效性
    if 'Close' in df.columns:
        invalid = (df['Close'] <= 0).sum()
        if invalid > 0:
            issues.append(f"存在 {invalid} 个非正价格")

    return len(issues) == 0, issues


# ==================== 新增：降级筛选机制 ====================
def filter_strategies_with_fallback(
        strategies: list,
        risk_manager: RiskManager,
        min_trades: int = 5,
        min_sharpe: float = 0.0,
        max_drawdown: float = -20.0,
) -> tuple:
    """降级筛选机制 - 逐步放宽条件

    Args:
        strategies: 策略列表
        risk_manager: 风控管理器
        min_trades: 最小交易次数
        min_sharpe: 最小夏普比率
        max_drawdown: 最大回撤限制

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


def _init_worker(market_data, stocks_data):
    """初始化多进程工作进程"""
    global _worker_market_data, _worker_stocks_data
    _worker_market_data = market_data
    _worker_stocks_data = stocks_data


def _process_single_stock(stock_code: str) -> tuple:
    """处理单只股票的回测（多进程工作函数）- 增强版

    返回 7 元组: (code, strategy, stats, df, trades, splits, metadata)
    失败时返回: (code, None, None, None, None, None, None)
    """
    try:
        # 获取数据
        df = _worker_stocks_data.get(stock_code)
        if df is None or len(df) == 0:
            logger.warning(f"股票 {stock_code} 数据为空，跳过")
            return stock_code, None, None, None, None, None, None

        # 新增：数据验证
        is_valid, issues = validate_data_for_backtest(df, stock_code)
        if not is_valid:
            logger.warning(f"股票 {stock_code} 数据验证失败: {issues}")
            return stock_code, None, None, None, None, None, None
        df = calculate_orthogonal_factors_without_transformer(df)

        # 计算因子权重
        factor_cols = [col for col in df.columns if col in TRADITIONAL_FACTOR_COLS]
        best_weights = calculate_dynamic_weights(df, factor_cols)

        # 参数优化
        best_params_map, _ = optimize_strategy(
            df=df,
            strategy_class=None,  # 使用默认策略
            n_trials=30,
            initial_capital=100000.0,
            weights=best_weights,
        )

        # Walk-Forward 划分
        splits = walk_forward_split(df)
        if not splits:
            logger.warning(f"股票 {stock_code} 无法进行Walk-Forward划分")
            return stock_code, None, None, None, None, None, None

        # 使用最后一个划分进行回测
        train_df, val_df, test_df = splits[-1]

        # 执行回测
        trades_df, stats, result_df = run_backtest_loop_no_transformer(
            df=test_df,
            stock_code=stock_code,
            market_data=_worker_market_data,
            weights=best_weights,
            params=best_params_map,
            stocks_data=_worker_stocks_data,
        )

        # 检查结果
        if stats is None or len(stats) == 0:
            logger.warning(f"股票 {stock_code} 回测结果为空")
            return stock_code, None, None, None, None, None, None

        # 记录信号统计
        if trades_df is not None and len(trades_df) > 0:
            logger.info(f"股票 {stock_code}: 交易次数 {len(trades_df)}")
        else:
            logger.warning(f"股票 {stock_code}: 无交易记录")
            return stock_code, None, None, None, None, None, None
        # stock_code 实际上是股票名称，需要从 STOCK_CODES 取实际代码
        actual_code = STOCK_CODES.get(stock_code, stock_code)
        # ===== 构造 main() 需要的 7 元组字段（不改变上面的任何逻辑） =====
        strategy = {
            'code': actual_code,
            'name': stock_code,
            'params': best_params_map if best_params_map else {},
            'weights': best_weights if best_weights else {},
            'stats': stats,
        }

        # 计算 test_df 在原始 df 中的起始索引（给 main() 的 metadata 用）
        test_start_idx = len(df) - len(test_df) if test_df is not None else int(len(df) * 0.7)

        metadata = {
            'test_start_idx': test_start_idx,
            'n_splits': len(splits),
        }

        # ★ 修复：构造6元组格式兼容 (train_start, train_end, val_start, val_end, test_start, test_end)
        validated_splits = [(0, test_start_idx, test_start_idx, test_start_idx, test_start_idx, len(df))]
        # ===== 构造结束 =====

        return stock_code, strategy, stats, df, trades_df, validated_splits, metadata

    except Exception as e:
        logger.error(f"处理股票 {stock_code} 时出错: {e}")
        logger.error(traceback.format_exc())
        return stock_code, None, None, None, None, None, None


def main():
    """主函数 - 增强版"""
    logger.info("=" * 80)
    logger.info("开始回测（无Transformer版本）")
    logger.info("=" * 80)

    settings = get_settings()

    # 1. 准备数据
    logger.info("步骤1: 准备数据...")

    # 下载大盘数据
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
    if market_data is None or len(market_data) == 0:
        logger.error("大盘数据下载失败")
        return

    # 下载股票数据
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
    # ===== 关键修复：添加收益率计算 =====
    logger.info("步骤1.5: 计算收益率...")
    stocks_data = prepare_stock_data(stocks_data)
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


    # 过滤股票
    valid_codes = filter_codes_by_name(mapping=STOCK_CODES)
    logger.info(f"有效股票数量: {len(valid_codes)}")

    # 2. 多进程回测
    logger.info("步骤2: 执行回测...")

    results = []
    n_workers = min(cpu_count(), 4)

    with Pool(n_workers, initializer=_init_worker, initargs=(market_data, stocks_data)) as pool:
        for result in tqdm(
                pool.imap(_process_single_stock, valid_codes),
                total=len(valid_codes),
                desc="回测进度"
        ):
            results.append(result)

    # 3. 统计结果
    logger.info("步骤3: 统计结果...")
    # 成功 = 第2个元素 strategy 不为 None
    success_count = sum(1 for r in results if r[1] is not None)
    logger.info(f"回测成功: {success_count}/{len(results)}")

    # 4. 筛选策略
    logger.info("步骤4: 筛选策略...")
    successful_results = [r for r in results if r[1] is not None]

    if not successful_results:
        logger.error("所有股票回测均失败，请检查数据和策略参数")
        return

    # 为 filter_strategies_with_fallback 构造它需要的 dict 格式
    # (不改 filter 函数本身，只在调用前做格式转换)
    filter_input = []
    for code, strat, stat, df, trades, splits, metadata in successful_results:
        filter_input.append({
            "code": code,
            "stats": stat,
            "trades_count": len(trades) if trades is not None else 0,
            "strategy": strat,
        })

    # 使用降级筛选机制
    risk_manager = RiskManager()
    passed_strategies, filter_log = filter_strategies_with_fallback(
        strategies=filter_input,
        risk_manager=risk_manager,
    )

    # 输出筛选日志
    for log_entry in filter_log:
        logger.info(log_entry)

    # 5. 生成报告
    logger.info("步骤5: 生成报告...")
    if passed_strategies:
        logger.info(f"最终保留 {len(passed_strategies)} 个策略")

        # 保存结果
        output_dir = "./stock_cache/no_transformer_results"
        os.makedirs(output_dir, exist_ok=True)

        result_file = os.path.join(output_dir, "backtest_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            # 转换为可序列化格式
            serializable_results = []
            for r in passed_strategies:
                serializable_results.append({
                    "code": r.get("code"),
                    "stats": r.get("stats"),
                    "trades_count": r.get("trades_count"),
                })
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存到: {result_file}")


    else:
        logger.error("无策略通过筛选")

    # 6. 筛选出通过策略的完整结果（给可视化用）
    passed_codes = {r.get("code") for r in passed_strategies}
    passed_results = [
        r for r in results
        if r[0] in passed_codes and r[1] is not None
    ]
    # 【修改处】：使用 passed_results 提取完整参数，而不是只保存 stats
    backtest_results = []
    # 保存结果
    output_dir = "./stock_cache/no_transformer_results"
    os.makedirs(output_dir, exist_ok=True)
    # passed_results 包含 (code, strategy, stats, df, trades, splits, metadata)
    for code, strat, stat, df, trades, splits, metadata in passed_results:
        if strat is None: continue

        backtest_results.append({
            "code": strat.get('code'),
            "name": strat.get('name', code),
            "params": strat.get('params'),  # 保存优化后的参数
            "weights": strat.get('weights'),  # 保存优化后的因子权重
            "stats": strat.get('stats'),  # 保存回测表现
        })

    result_file = os.path.join(output_dir, "optimized_strategies_no_transformer.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(backtest_results, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存到: {result_file}")
    # 7. 生成可视化图表
    print("\n" + "=" * 80)
    print("开始生成可视化图表...")
    print("=" * 80)
    for code, strat, stat, df, trades, splits, metadata in passed_results:
        if strat is None or trades is None or len(trades) == 0:
            continue
        stock_name = strat.get('name', code)
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
    logger.info("=" * 80)
    logger.info("回测完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
