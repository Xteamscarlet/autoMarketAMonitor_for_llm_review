# -*- coding: utf-8 -*-
"""
实盘决策辅助主模块 V2
改进：
1. 使用增强版市场状态（5种状态+仓位乘数）
2. 弱势市场允许限制性买入（不再一刀切）
3. 板块集中度检查
4. 相关性过滤
"""
import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from config import get_settings, STOCK_CODES, RebalanceFreq, SECTOR_MAP
from data import (
    download_market_data, download_stocks_data,
    check_and_clean_cache, load_pickle_cache,
    calculate_orthogonal_factors, save_pickle_cache,
)
from data.regime import get_market_regime_enhanced, RegimeInfo
from data.types import NON_FACTOR_COLS
from backtest.engine import calculate_multi_timeframe_score, calculate_transaction_cost
from live.signal_filter import classify_signal_confidence, filter_by_microstructure
from live.portfolio_risk import check_portfolio_limits, check_correlation_limit
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)


def init_portfolio_file():
    settings = get_settings()
    if not os.path.exists(settings.paths.portfolio_file):
        template = {}
        for name, code in STOCK_CODES.items():
            template[code] = {"name": name, "buy_price": 0.0, "buy_date": ""}
        with open(settings.paths.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=4)


def load_strategies() -> Optional[Dict]:
    settings = get_settings()
    if not os.path.exists(settings.paths.strategy_file):
        logger.error(f"策略文件不存在: {settings.paths.strategy_file}")
        return None
    with open(settings.paths.strategy_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def should_rebalance_today(settings) -> bool:
    cfg = settings.scheduler
    now = date.today()
    weekday = now.weekday()

    if weekday != cfg.rebalance_anchor_weekday:
        return False

    if cfg.rebalance_freq == RebalanceFreq.WEEKLY:
        return True

    anchor = cfg.rebalance_anchor_date
    if anchor:
        try:
            d0 = datetime.strptime(anchor, "%Y-%m-%d").date()
        except Exception:
            return _biweekly_even_week(now)
        monday0 = d0 - timedelta(days=d0.weekday())
        monday_now = now - timedelta(days=weekday)
        diff_days = (monday_now - monday0).days
        return diff_days % 14 == 0
    else:
        return _biweekly_even_week(now)


def _biweekly_even_week(d: date) -> bool:
    iso = d.isocalendar()
    return iso[1] % 2 == 0


def _dry_run_without_rebalance(settings):
    logger.info("非调仓日：仅刷新数据缓存（不做调仓）。")
    if not check_and_clean_cache(settings.paths.market_cache_file):
        download_market_data()
    if not check_and_clean_cache(settings.paths.stock_cache_file):
        download_stocks_data(STOCK_CODES)


def run_advisor():
    """实盘决策辅助主函数 V2"""
    settings = get_settings()
    print("\n" + "=" * 60)
    print("实盘决策助手 V3 (增强版)")
    print("=" * 60)
    print(f" 策略文件: {settings.paths.portfolio_file}")
    print(f" 模型路径: {settings.paths.model_path}")
    print(f" 风控 - 最大回撤: {settings.risk.max_drawdown_limit}%")
    print(f" 风控 - 单只最大仓位: {settings.risk.max_position_ratio:.0%}")
    print(f" 调仓频率: {settings.scheduler.rebalance_freq.value}")
    print("=" * 60)

    if not should_rebalance_today(settings):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] 非调仓日，本次跳过。")
        _dry_run_without_rebalance(settings)
        return

    init_portfolio_file()

    print("\n" + "=" * 60)
    print(f"增强决策助手 V3 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 大盘数据
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

    # 3. ★ 增强版市场状态
    last_date = market_data.index[-1]
    regime_info = get_market_regime_enhanced(market_data, last_date)
    print(f"📢 当前市场环境: 【{regime_info.regime}】")
    print(f"   趋势方向: {'↑' if regime_info.trend_direction > 0 else '↓' if regime_info.trend_direction < 0 else '→'}")
    print(f"   波动率水平: {regime_info.volatility_level}")
    print(f"   趋势强度: {regime_info.trend_strength:.2f}")
    print(f"   量价配合: {regime_info.volume_price_align:.2f}")
    print(f"   建议仓位乘数: {regime_info.position_multiplier:.0%}")
    print(f"   {regime_info.description}")

    all_strategies = load_strategies()
    if not all_strategies:
        return

    with open(settings.paths.portfolio_file, 'r', encoding='utf-8') as f:
        portfolio_data = json.load(f)

    sell_candidates = []
    buy_candidates = []
    current_positions = []

    # 收集已有持仓代码（用于相关性检查）
    existing_codes = [code for code, pos in portfolio_data.items() if float(pos.get('buy_price', 0)) > 0]

    for code, pos_info in portfolio_data.items():
        name = pos_info['name']
        buy_price = float(pos_info.get('buy_price', 0))
        buy_date_str = pos_info.get('buy_date', '')
        stock_config = all_strategies.get(code)

        if not stock_config:
            continue

        # ★ 查找匹配的 regime 参数（支持5种状态）
        params = None
        for r in [regime_info.regime, 'neutral']:
            params = stock_config['params'].get(r)
            if params:
                break
        weights = stock_config['weights']
        if not params:
            continue
        if params.get('buy_threshold', 0.6) <= params.get('sell_threshold', -0.2):
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
                if hold_days < 1:
                    logger.info(f"{name} ({code}) 持仓不足1天，受T+1限制，暂不建议卖出")
                    continue
                profit_pct = (current_price - buy_price) / buy_price

                current_positions.append({
                    'code': code, 'name': name,
                    'ratio': profit_pct, 'sector': SECTOR_MAP.get(code, 'unknown'),
                })

                reasons = []

                if profit_pct <= params.get('stop_loss', -0.08):
                    reasons.append(f"🩸 触发止损 ({profit_pct * 100:.1f}%)")

                df_hold = df[df.index >= buy_date]
                peak_price = df_hold['Close'].max() if not df_hold.empty else buy_price
                drawdown = (peak_price - current_price) / peak_price if peak_price > 0 else 0.0

                tp1 = params.get('trailing_profit_level1', 0.06)
                tp2 = params.get('trailing_profit_level2', 0.12)
                td1 = params.get('trailing_drawdown_level1', 0.08)
                td2 = params.get('trailing_drawdown_level2', 0.04)

                if profit_pct > tp2 and drawdown >= td2:
                    reasons.append("🛡️ 移动止损 (Level2)")
                elif profit_pct > tp1 and drawdown >= td1:
                    reasons.append("🛡️ 移动止损 (Level1)")

                atr = latest.get('atr', current_price * 0.02)
                if not pd.isna(atr):
                    tp_ratio = params.get('take_profit_multiplier', 3.0) * (atr / buy_price)
                    if profit_pct >= tp_ratio:
                        reasons.append("🚀 动态止盈")

                if hold_days >= params.get('hold_days', 15):
                    reasons.append(f"⏳ 持仓到期 ({hold_days}天)")

                if current_score < params.get('sell_threshold', -0.2):
                    reasons.append(f"📉 信号衰减 (得分:{current_score:.2f})")

                if reasons:
                    sell_candidates.append({
                        'name': name, 'code': code, 'price': current_price,
                        'profit': profit_pct * 100, 'reasons': reasons,
                    })

            # ========== 空仓判断 ==========
            else:
                # ★ 弱势市场不再一刀切，而是限制仓位
                if regime_info.regime == 'bear' and current_score < 0.85:
                    continue  # bear 市场只允许极强信号

                if current_score > params.get('buy_threshold', 0.6):
                    level, pos_ratio = classify_signal_confidence(
                        current_score, params.get('buy_threshold', 0.6)
                    )

                    if level == 'none':
                        continue

                    prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
                    allowed, micro_reason = filter_by_microstructure(code, current_price, prev_close)
                    if not allowed:
                        logger.info(f"{name}: {micro_reason}")
                        continue

                    # ★ 相关性检查
                    corr_ok, max_corr = check_correlation_limit(
                        code, existing_codes, stocks_data,
                        max_correlation=settings.risk.max_correlation,
                    )
                    if not corr_ok:
                        logger.info(f"{name}: 与已有持仓相关性过高({max_corr:.2f})，跳过")
                        continue

                    # ATR 动态仓位
                    atr = latest.get('atr', current_price * 0.02)
                    if pd.isna(atr) or atr <= 0:
                        atr = current_price * 0.02
                    daily_vol = atr / current_price
                    target_annual_vol = 0.10
                    base_ratio = target_annual_vol / (daily_vol * np.sqrt(252) + 1e-6)
                    base_ratio = min(max(base_ratio, 0.1), 1.0)

                    # ★ 应用市场状态仓位乘数
                    final_ratio = base_ratio * pos_ratio * regime_info.position_multiplier
                    final_ratio = min(max(final_ratio, 0.05), 1.0)

                    capital = 100000
                    shares = max(100, int(capital * final_ratio / current_price / 100) * 100)
                    shares = min(shares, int(capital / current_price / 100) * 100)

                    buy_candidates.append({
                        'name': name, 'code': code, 'price': current_price,
                        'score': current_score,
                        'threshold': params.get('buy_threshold', 0.6),
                        'level': level, 'position_ratio': final_ratio,
                        'recommended_shares': shares,
                        'suggested_capital': capital * final_ratio,
                        'transformer_score': latest.get('transformer_prob', 0.5),
                        'sector': SECTOR_MAP.get(code, 'unknown'),
                    })

        except Exception as e:
            logger.error(f"✗ {name} 计算出错: {e}")

    # 组合风控过滤
    filtered_buy, warnings = check_portfolio_limits(current_positions, buy_candidates)
    for w in warnings:
        logger.warning(f"组合风控: {w}")

    # 输出结果
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
        if regime_info.regime in ('bear', 'weak'):
            print(f" 😴 当前市场偏弱({regime_info.regime})，减少买入")
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
