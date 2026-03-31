# -*- coding: utf-8 -*-
"""
实盘决策辅助主模块
从 run_market_bot_v2.py 重构，集成信号置信度分级和组合风控
"""
import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from config import get_settings, STOCK_CODES
from data import (
    download_market_data, download_stocks_data,
    check_and_clean_cache, load_pickle_cache,
    calculate_orthogonal_factors,
)
from data.indicators import get_market_regime
from data.types import NON_FACTOR_COLS
from backtest.engine import calculate_multi_timeframe_score, calculate_transaction_cost
from live.signal_filter import classify_signal_confidence, filter_by_microstructure
from live.portfolio_risk import check_portfolio_limits

logger = logging.getLogger(__name__)


def init_portfolio_file():
    """初始化持仓文件"""
    settings = get_settings()
    if not os.path.exists(settings.paths.portfolio_file):
        template = {}
        for name, code in STOCK_CODES.items():
            template[code] = {"name": name, "buy_price": 0.0, "buy_date": ""}
        with open(settings.paths.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=4)


def load_strategies() -> Optional[Dict]:
    """加载策略参数文件"""
    settings = get_settings()
    if not os.path.exists(settings.paths.strategy_file):
        logger.error(f"策略文件不存在: {settings.paths.strategy_file}")
        return None
    with open(settings.paths.strategy_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_advisor():
    """实盘决策辅助主函数"""
    settings = get_settings()
    init_portfolio_file()

    print("\n" + "=" * 60)
    print(f"个性化决策助手 V3 (AI增强+风控版) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取大盘数据
    if check_and_clean_cache(settings.paths.market_cache_file):
        market_data = load_pickle_cache(settings.paths.market_cache_file)['market_data']
    else:
        market_data = download_market_data()
        if market_data is None:
            print("❌ 无法获取大盘数据，程序退出。")
            return

    # 2. 获取个股数据
    if check_and_clean_cache(settings.paths.stock_cache_file):
        stocks_data = load_pickle_cache(settings.paths.stock_cache_file)['stocks_data']
    else:
        stocks_data = download_stocks_data(STOCK_CODES)

    # 3. 市场状态
    last_date = market_data.index[-1]
    regime = get_market_regime(market_data, last_date)
    print(f"📢 当前市场环境: 【{regime}】 (日期: {last_date.strftime('%Y-%m-%d')})")

    all_strategies = load_strategies()
    if not all_strategies:
        return

    with open(settings.paths.portfolio_file, 'r', encoding='utf-8') as f:
        portfolio_data = json.load(f)

    sell_candidates = []
    buy_candidates = []
    current_positions = []

    for code, pos_info in portfolio_data.items():
        name = pos_info['name']
        buy_price = float(pos_info.get('buy_price', 0))
        buy_date_str = pos_info.get('buy_date', '')
        stock_config = all_strategies.get(code)

        if not stock_config:
            continue

        params = stock_config['params'].get(regime)
        weights = stock_config['weights']
        if not params:
            params = stock_config['params'].get('neutral')
        if not params:
            continue
        if params['buy_threshold'] <= params['sell_threshold']:
            params['buy_threshold'] = params['sell_threshold'] + 0.05

        try:
            if name not in stocks_data:
                continue

            df = stocks_data[name].copy()
            if len(df) < 150:
                continue

            df = df.sort_index()
            df = calculate_orthogonal_factors(df, code)
            df = calculate_multi_timeframe_score(df, weights)

            latest = df.iloc[-1]
            current_price = latest['Close']
            current_score = latest['Combined_Score']

            # ========== 持仓判断 ==========
            if buy_price > 0:
                buy_date = datetime.strptime(buy_date_str, '%Y-%m-%d') if buy_date_str else datetime.now()
                hold_days = (datetime.now() - buy_date).days
                # ===== 新增 T+1 限制 开始 =====
                if hold_days < 1:
                    logger.info(f"{name} ({code}) 持仓不足1天，受T+1限制，暂不建议卖出")
                    continue  # 跳过本次循环，不生成卖出建议
                # ===== 新增 T+1 限制 结束 =====
                profit_pct = (current_price - buy_price) / buy_price

                # 记录当前持仓
                current_positions.append({
                    'code': code, 'name': name,
                    'ratio': profit_pct, 'sector': 'unknown',
                })

                reasons = []

                # 硬止损
                if profit_pct <= params['stop_loss']:
                    reasons.append(f"🩸 触发止损 ({profit_pct * 100:.1f}%)")

                # 移动止损
                df_hold = df[df.index >= buy_date]
                peak_price = df_hold['Close'].max() if not df_hold.empty else buy_price
                drawdown = (peak_price - current_price) / peak_price if peak_price > 0 else 0.0

                if (profit_pct > params['trailing_profit_level2']
                        and drawdown >= params['trailing_drawdown_level2']):
                    reasons.append("🛡️ 移动止损 (Level2)")
                elif (profit_pct > params['trailing_profit_level1']
                      and drawdown >= params['trailing_drawdown_level1']):
                    reasons.append("🛡️ 移动止损 (Level1)")

                # 动态止盈
                atr = latest.get('atr', current_price * 0.02)
                if not pd.isna(atr):
                    tp_ratio = params['take_profit_multiplier'] * (atr / buy_price)
                    if profit_pct >= tp_ratio:
                        reasons.append("🚀 动态止盈")

                # 时间止损
                if hold_days >= params['hold_days']:
                    reasons.append(f"⏳ 持仓到期 ({hold_days}天)")

                # 信号衰减
                if current_score < params['sell_threshold']:
                    reasons.append(f"📉 信号衰减 (得分:{current_score:.2f})")

                if reasons:
                    sell_candidates.append({
                        'name': name, 'code': code, 'price': current_price,
                        'profit': profit_pct * 100, 'reasons': reasons,
                    })

            # ========== 空仓判断 ==========
            else:
                if regime == 'weak':
                    continue

                if current_score > params['buy_threshold']:
                    # 信号置信度分级
                    level, pos_ratio = classify_signal_confidence(current_score, params['buy_threshold'])

                    if level == 'none':
                        continue

                    # 微观结构过滤
                    prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
                    allowed, micro_reason = filter_by_microstructure(code, current_price, prev_close)
                    if not allowed:
                        logger.info(f"{name}: {micro_reason}")
                        continue

                    # ATR 动态仓位
                    atr = latest.get('atr', current_price * 0.02)
                    if pd.isna(atr) or atr <= 0:
                        atr = current_price * 0.02
                    daily_vol = atr / current_price
                    target_annual_vol = 0.10
                    base_ratio = target_annual_vol / (daily_vol * np.sqrt(252) + 1e-6)
                    base_ratio = min(max(base_ratio, 0.1), 1.0)

                    final_ratio = base_ratio * pos_ratio

                    capital = 100000
                    shares = max(100, int(capital * final_ratio / current_price / 100) * 100)
                    shares = min(shares, int(capital / current_price / 100) * 100)

                    buy_candidates.append({
                        'name': name, 'code': code, 'price': current_price,
                        'score': current_score, 'threshold': params['buy_threshold'],
                        'level': level, 'position_ratio': final_ratio,
                        'recommended_shares': shares,
                        'suggested_capital': capital * final_ratio,
                        'transformer_score': latest.get('transformer_prob', 0.5),
                    })

        except Exception as e:
            logger.error(f"✗ {name} 计算出错: {e}")

    # ========== 组合风控过滤 ==========
    filtered_buy, warnings = check_portfolio_limits(current_positions, buy_candidates)
    for w in warnings:
        logger.warning(f"组合风控: {w}")

    # ========== 输出结果 ==========
    print("\n" + "-" * 20 + "【卖出监控】" + "-" * 20)
    if not sell_candidates:
        print(" ✅ 无需操作，持仓表现正常。")
    else:
        for item in sell_candidates:
            print(f"\n🚨 {item['name']} ({item['code']})")
            print(f" 现价: {item['price']:.2f} | 收益: {item['profit']:.2f}%")
            for r in item['reasons']:
                print(f" 原因: {r}")

    print("\n" + "-" * 20 + "【买入机会】" + "-" * 20)
    if not filtered_buy:
        if regime == 'weak':
            print(" 😴 今日大盘走势不佳，取消买入")
        else:
            print(" 😴 今日无符合条件的买入机会")
    else:
        filtered_buy.sort(key=lambda x: x['score'], reverse=True)
        for idx, item in enumerate(filtered_buy, 1):
            level_cn = {'strong': '🟢 强信号', 'medium': '🟡 中信号', 'weak': '🟠 弱信号'}
            print(f"\n{idx}. {level_cn.get(item['level'], '')} {item['name']} ({item['code']})")
            print(f" 现价: {item['price']:.2f} | 综合得分: {item['score']:.3f} (阈值: {item['threshold']:.2f})")
            print(f" AI观点: {item['transformer_score']:.2f}")
            print(f" 📊 建议仓位: {item['position_ratio'] * 100:.1f}% | 建议买入 {item['recommended_shares']:,} 股")
