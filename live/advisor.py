# -*- coding: utf-8 -*-
"""Live trading advisor with AI fundamentals and technical research integration."""

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agents.research_manager import analyze_stock
from backtest.engine import calculate_multi_timeframe_score
from config import RebalanceFreq, SECTOR_MAP, STOCK_CODES, get_settings
from data import (
    calculate_orthogonal_factors,
    check_and_clean_cache,
    download_market_data,
    download_stocks_data,
    load_pickle_cache,
    save_pickle_cache,
)
from data.regime import get_market_regime_enhanced
from live.portfolio_risk import check_correlation_limit, check_portfolio_limits
from live.signal_filter import classify_signal_confidence, filter_by_microstructure

logger = logging.getLogger(__name__)


def init_portfolio_file() -> None:
    settings = get_settings()
    if os.path.exists(settings.paths.portfolio_file):
        return

    template = {}
    for name, code in STOCK_CODES.items():
        template[code] = {"name": name, "buy_price": 0.0, "buy_date": "", "position_ratio": 0.0}

    with open(settings.paths.portfolio_file, "w", encoding="utf-8") as file:
        json.dump(template, file, ensure_ascii=False, indent=2)


def load_strategies() -> Optional[Dict]:
    settings = get_settings()
    if not os.path.exists(settings.paths.strategy_file):
        logger.error("策略文件不存在: %s", settings.paths.strategy_file)
        return None
    with open(settings.paths.strategy_file, "r", encoding="utf-8") as file:
        return json.load(file)


def should_rebalance_today(settings) -> bool:
    cfg = settings.scheduler
    today = date.today()
    weekday = today.weekday()

    if weekday != cfg.rebalance_anchor_weekday:
        return False

    if cfg.rebalance_freq == RebalanceFreq.WEEKLY:
        return True

    anchor = cfg.rebalance_anchor_date
    if anchor:
        try:
            start_date = datetime.strptime(anchor, "%Y-%m-%d").date()
        except Exception:
            return _is_biweekly_even_week(today)
        monday0 = start_date - timedelta(days=start_date.weekday())
        monday_today = today - timedelta(days=weekday)
        return (monday_today - monday0).days % 14 == 0

    return _is_biweekly_even_week(today)


def _is_biweekly_even_week(current_day: date) -> bool:
    return current_day.isocalendar()[1] % 2 == 0


def _dry_run_without_rebalance(settings) -> None:
    logger.info("非调仓日：仅刷新数据缓存，不做交易建议。")
    if not check_and_clean_cache(settings.paths.market_cache_file):
        download_market_data()
    if not check_and_clean_cache(settings.paths.stock_cache_file):
        download_stocks_data(STOCK_CODES)


def _load_market_data():
    settings = get_settings()
    if check_and_clean_cache(settings.paths.market_cache_file):
        cached = load_pickle_cache(settings.paths.market_cache_file)
        return cached.get("market_data")

    market_data = download_market_data()
    if market_data is not None:
        save_pickle_cache(
            settings.paths.market_cache_file,
            {
                "market_data": market_data,
                "last_date": market_data.index[-1].strftime("%Y-%m-%d"),
            },
        )
    return market_data


def _load_stocks_data():
    settings = get_settings()
    if check_and_clean_cache(settings.paths.stock_cache_file):
        cached = load_pickle_cache(settings.paths.stock_cache_file)
        return cached.get("stocks_data")

    stocks_data = download_stocks_data(STOCK_CODES)
    if stocks_data:
        last_dates = [df.index[-1] for df in stocks_data.values() if not df.empty]
        last_date = max(last_dates).strftime("%Y-%m-%d") if last_dates else None
        save_pickle_cache(
            settings.paths.stock_cache_file,
            {
                "stocks_data": stocks_data,
                "last_date": last_date,
            },
        )
    return stocks_data


def _select_regime_params(stock_config: Dict, regime_name: str) -> Optional[Dict]:
    params_block = stock_config.get("params", {})
    for candidate in (regime_name, "neutral"):
        params = params_block.get(candidate)
        if params:
            params = dict(params)
            if params.get("buy_threshold", 0.6) <= params.get("sell_threshold", -0.2):
                params["buy_threshold"] = params.get("sell_threshold", -0.2) + 0.05
            return params
    return None


def _estimate_existing_position_ratio(pos_info: Dict, settings) -> float:
    explicit_ratio = pos_info.get("position_ratio")
    try:
        ratio = float(explicit_ratio)
        if ratio > 0:
            return min(ratio, settings.risk.max_position_ratio)
    except (TypeError, ValueError):
        pass
    return settings.risk.max_position_ratio


def _compute_research_bonus(settings, research: Dict) -> float:
    fundamentals = research.get("fundamentals") or {}
    technicals = research.get("technicals") or {}

    bonus = 0.0
    if fundamentals and settings.analysis.blend_fundamental_score:
        bonus += settings.analysis.fundamental_weight * (float(fundamentals.get("score", 0.5)) - 0.5)
    if technicals and settings.analysis.blend_technical_score:
        bonus += settings.analysis.technical_weight * (float(technicals.get("score", 0.5)) - 0.5)
    return float(bonus)


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return num


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    parsed = _safe_float(raw)
    return default if parsed is None else float(parsed)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _compute_transformer_contribution(latest: pd.Series, weights: Dict) -> float:
    contrib = 0.0
    for factor, weight in (weights or {}).items():
        if not str(factor).startswith("transformer_"):
            continue
        weight_f = _safe_float(weight)
        if weight_f is None or weight_f == 0:
            continue
        factor_value = latest.get(factor)
        value_f = _safe_float(factor_value)
        if value_f is None:
            continue
        contrib += (value_f - 0.5) * 2.0 * weight_f
    return float(contrib)


def _detect_transformer_collapse(prepared_rows: List[Dict], std_floor: float, min_count: int) -> Dict:
    probs = [row["transformer_prob"] for row in prepared_rows if row.get("transformer_prob") is not None]
    count = len(probs)
    if count == 0:
        return {"active": False, "count": 0, "std": None, "min": None, "max": None}

    arr = np.asarray(probs, dtype=float)
    std = float(np.std(arr))
    min_prob = float(np.min(arr))
    max_prob = float(np.max(arr))
    active = bool(count >= min_count and np.isfinite(std) and std < std_floor)
    return {"active": active, "count": count, "std": std, "min": min_prob, "max": max_prob}


def run_advisor(force_rebalance: bool = False) -> None:
    settings = get_settings()

    print("\n" + "=" * 60)
    print("实盘决策助手 V4")
    print("=" * 60)
    print(f"策略文件: {settings.paths.strategy_file}")
    print(f"持仓文件: {settings.paths.portfolio_file}")
    print(f"模型路径: {settings.paths.model_path}")
    print(f"调仓频率: {settings.scheduler.rebalance_freq.value}")
    print(f"基本面 Agent: {'开启' if settings.analysis.enable_fundamental_agent else '关闭'}")
    print(f"技术面 Agent: {'开启' if settings.analysis.enable_technical_agent else '关闭'}")
    print("=" * 60)

    if not force_rebalance and not should_rebalance_today(settings):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] 非调仓日，本次跳过。")
        _dry_run_without_rebalance(settings)
        return
    if force_rebalance:
        logger.warning("已启用强制调仓模式：跳过调仓日检查。")

    init_portfolio_file()

    market_data = _load_market_data()
    if market_data is None or market_data.empty:
        logger.error("无法获取大盘数据")
        return

    stocks_data = _load_stocks_data()
    if not stocks_data:
        logger.error("无法获取个股数据")
        return

    regime_info = get_market_regime_enhanced(market_data, market_data.index[-1])
    print(f"当前市场环境: {regime_info.regime}")
    print(f"趋势强度: {regime_info.trend_strength:.2f}")
    print(f"波动水平: {regime_info.volatility_level}")
    print(f"建议仓位乘数: {regime_info.position_multiplier:.0%}")
    print(f"说明: {regime_info.description}")

    strategies = load_strategies()
    if not strategies:
        return

    with open(settings.paths.portfolio_file, "r", encoding="utf-8") as file:
        portfolio_data = json.load(file)

    sell_candidates: List[Dict] = []
    buy_candidates: List[Dict] = []
    current_positions: List[Dict] = []
    existing_codes = [code for code, pos in portfolio_data.items() if float(pos.get("buy_price", 0)) > 0]
    transformer_std_floor = max(_env_float("ADVISOR_TRANSFORMER_STD_FLOOR", 0.01), 0.0)
    transformer_min_count = max(_env_int("ADVISOR_TRANSFORMER_MIN_COUNT", 8), 1)
    transformer_drop_ratio = float(np.clip(_env_float("ADVISOR_TRANSFORMER_DROP_RATIO", 1.0), 0.0, 1.0))

    factor_cache: Dict[str, Dict] = {}
    transformer_rows: List[Dict] = []
    for code, pos_info in portfolio_data.items():
        name = pos_info.get("name", "")
        stock_config = strategies.get(code)
        if not stock_config or name not in stocks_data:
            continue

        params = _select_regime_params(stock_config, regime_info.regime)
        weights = stock_config.get("weights", {})
        if not params:
            continue

        try:
            df = stocks_data[name].copy().sort_index()
            if len(df) < 150:
                continue

            df = calculate_orthogonal_factors(df, code)
            df = calculate_multi_timeframe_score(df, weights)
            latest = df.iloc[-1]
            base_score_raw = float(latest.get("Combined_Score", 0.5))
            current_price = _safe_float(latest.get("Close"))
            if current_price is None or current_price <= 0:
                continue
            transformer_prob = _safe_float(latest.get("transformer_prob", 0.5))
            if transformer_prob is None:
                transformer_prob = 0.5

            factor_cache[code] = {
                "df": df,
                "latest": latest,
                "current_price": float(current_price),
                "base_score_raw": base_score_raw,
                "transformer_prob": transformer_prob,
                "transformer_contrib": _compute_transformer_contribution(latest, weights),
            }
            transformer_rows.append({"transformer_prob": transformer_prob})
        except Exception as exc:
            logger.exception("%s (%s) factor precompute failed: %s", name, code, exc)

    transformer_stats = _detect_transformer_collapse(transformer_rows, transformer_std_floor, transformer_min_count)
    transformer_guardrail_active = bool(transformer_stats.get("active")) and transformer_drop_ratio > 0
    if transformer_guardrail_active:
        std_val = transformer_stats.get("std", 0.0)
        logger.warning(
            "Transformer guardrail active: std=%.6f < %.6f, n=%d, drop_ratio=%.2f",
            float(std_val),
            transformer_std_floor,
            int(transformer_stats.get("count", 0)),
            transformer_drop_ratio,
        )
        print(
            f"[Guardrail] transformer_prob std={float(std_val):.6f} (< {transformer_std_floor:.4f}), "
            f"降权比例={transformer_drop_ratio:.0%}"
        )

    for code, pos_info in portfolio_data.items():
        name = pos_info.get("name", "")
        buy_price = float(pos_info.get("buy_price", 0) or 0)
        buy_date_str = pos_info.get("buy_date", "")
        stock_config = strategies.get(code)

        if not stock_config or name not in stocks_data:
            continue

        params = _select_regime_params(stock_config, regime_info.regime)
        if not params:
            continue
        cached = factor_cache.get(code)
        if not cached:
            continue

        try:
            df = cached["df"].copy()
            latest = cached["latest"]
            current_price = float(cached["current_price"])
            base_score_raw = float(cached["base_score_raw"])
            transformer_adjustment = 0.0
            if transformer_guardrail_active:
                transformer_adjustment = float(cached.get("transformer_contrib", 0.0)) * transformer_drop_ratio
            base_score = base_score_raw - transformer_adjustment

            if buy_price > 0:
                buy_date = datetime.strptime(buy_date_str, "%Y-%m-%d") if buy_date_str else datetime.now()
                hold_days = (datetime.now() - buy_date).days
                if hold_days < 1:
                    logger.info("%s (%s) 持仓不足 1 天，受 T+1 限制，暂不建议卖出", name, code)
                    continue

                profit_pct = (current_price - buy_price) / buy_price if buy_price else 0.0
                current_positions.append(
                    {
                        "code": code,
                        "name": name,
                        "ratio": _estimate_existing_position_ratio(pos_info, settings),
                        "sector": SECTOR_MAP.get(code, "unknown"),
                    }
                )

                reasons: List[str] = []
                if profit_pct <= params.get("stop_loss", -0.08):
                    reasons.append(f"触发止损，当前收益 {profit_pct * 100:.1f}%")

                df_hold = df[df.index >= buy_date]
                peak_price = float(df_hold["Close"].max()) if not df_hold.empty else buy_price
                drawdown = (peak_price - current_price) / peak_price if peak_price > 0 else 0.0

                tp1 = params.get("trailing_profit_level1", 0.06)
                tp2 = params.get("trailing_profit_level2", 0.12)
                td1 = params.get("trailing_drawdown_level1", 0.08)
                td2 = params.get("trailing_drawdown_level2", 0.04)

                if profit_pct > tp2 and drawdown >= td2:
                    reasons.append("达到移动止盈 Level2")
                elif profit_pct > tp1 and drawdown >= td1:
                    reasons.append("达到移动止盈 Level1")

                atr = latest.get("atr", current_price * 0.02)
                if pd.notna(atr):
                    tp_ratio = params.get("take_profit_multiplier", 3.0) * (float(atr) / buy_price)
                    if profit_pct >= tp_ratio:
                        reasons.append("达到动态止盈阈值")

                if hold_days >= params.get("hold_days", 15):
                    reasons.append(f"持仓到期，已持有 {hold_days} 天")

                if base_score < params.get("sell_threshold", -0.2):
                    reasons.append(f"综合得分回落至 {base_score:.3f}")

                if reasons:
                    sell_candidates.append(
                        {
                            "name": name,
                            "code": code,
                            "price": current_price,
                            "profit": profit_pct * 100,
                            "reasons": reasons,
                        }
                    )
                continue

            if regime_info.regime == "bear" and base_score < 0.85:
                continue

            research = analyze_stock(code=code, name=name, df=df)
            research_bonus = _compute_research_bonus(settings, research)
            enhanced_score = base_score + research_bonus
            fundamentals = research.get("fundamentals") or {}
            technicals = research.get("technicals") or {}
            fundamental_score = fundamentals.get("score")

            if fundamental_score is not None and float(fundamental_score) < settings.analysis.min_fundamental_score:
                logger.info(
                    "%s (%s) 基本面得分 %.3f 低于阈值 %.3f，跳过",
                    name,
                    code,
                    float(fundamental_score),
                    settings.analysis.min_fundamental_score,
                )
                continue

            if enhanced_score <= params.get("buy_threshold", 0.6):
                continue

            level, signal_ratio = classify_signal_confidence(enhanced_score, params.get("buy_threshold", 0.6))
            if level == "none":
                continue

            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else current_price
            allowed, micro_reason = filter_by_microstructure(code, current_price, prev_close)
            if not allowed:
                logger.info("%s (%s) 微观结构过滤: %s", name, code, micro_reason)
                continue

            corr_ok, max_corr = check_correlation_limit(
                code,
                existing_codes,
                stocks_data,
                max_correlation=settings.risk.max_correlation,
            )
            if not corr_ok:
                logger.info("%s (%s) 与持仓相关性过高 %.2f，跳过", name, code, max_corr)
                continue

            atr = latest.get("atr", current_price * 0.02)
            if pd.isna(atr) or float(atr) <= 0:
                atr = current_price * 0.02
            daily_vol = float(atr) / current_price if current_price else 0.02
            target_annual_vol = 0.10
            base_ratio = target_annual_vol / (daily_vol * np.sqrt(252) + 1e-6)
            base_ratio = min(max(base_ratio, 0.1), 1.0)
            final_ratio = base_ratio * signal_ratio * regime_info.position_multiplier
            final_ratio = min(max(final_ratio, 0.05), settings.risk.max_position_ratio)

            capital = 100000
            recommended_shares = max(100, int(capital * final_ratio / current_price / 100) * 100)
            recommended_shares = min(recommended_shares, int(capital / current_price / 100) * 100)
            if recommended_shares <= 0:
                continue

            buy_candidates.append(
                {
                    "name": name,
                    "code": code,
                    "price": current_price,
                    "score": round(enhanced_score, 4),
                    "base_score": round(base_score, 4),
                    "base_score_raw": round(base_score_raw, 4),
                    "transformer_adjustment": round(transformer_adjustment, 4),
                    "research_bonus": round(research_bonus, 4),
                    "threshold": params.get("buy_threshold", 0.6),
                    "level": level,
                    "ratio": final_ratio,
                    "position_ratio": final_ratio,
                    "recommended_shares": recommended_shares,
                    "suggested_capital": capital * final_ratio,
                    "fundamental_score": fundamentals.get("score"),
                    "fundamental_summary": fundamentals.get("summary"),
                    "technical_score_ai": technicals.get("score"),
                    "technical_summary_ai": technicals.get("summary"),
                    "transformer_score": latest.get("transformer_prob", 0.5),
                    "transformer_conf": latest.get("transformer_conf", 0.0),
                    "transformer_pred_ret_raw": latest.get("transformer_pred_ret_raw", latest.get("transformer_pred_ret", 0.0)),
                    "sector": SECTOR_MAP.get(code, "unknown"),
                }
            )
        except Exception as exc:
            logger.exception("%s (%s) 生成建议时出错: %s", name, code, exc)

    filtered_buy, warnings = check_portfolio_limits(current_positions, buy_candidates)
    for warning in warnings:
        logger.warning("组合风控: %s", warning)

    print("\n" + "-" * 20 + "【卖出监控】" + "-" * 20)
    if not sell_candidates:
        print("无需要操作的卖出信号。")
    else:
        for item in sell_candidates:
            print(f"\n{item['name']} ({item['code']})")
            print(f"现价: {item['price']:.2f} | 收益: {item['profit']:.2f}%")
            for reason in item["reasons"]:
                print(f"原因: {reason}")

    print("\n" + "-" * 20 + "【买入机会】" + "-" * 20)
    if not filtered_buy:
        if regime_info.regime in ("bear", "weak"):
            print(f"当前市场偏弱({regime_info.regime})，本次不主动扩张仓位。")
        else:
            print("今日无满足条件的买入机会。")
        return

    filtered_buy.sort(key=lambda item: item["score"], reverse=True)
    level_name = {"strong": "强信号", "medium": "中信号", "weak": "弱信号"}
    for index, item in enumerate(filtered_buy, 1):
        print(f"\n{index}. {level_name.get(item['level'], '信号')} {item['name']} ({item['code']})")
        print(
            f"现价: {item['price']:.2f} | 增强得分: {item['score']:.3f} "
            f"(基础: {item['base_score']:.3f}, research_bonus: {item['research_bonus']:+.3f}, 阈值: {item['threshold']:.2f})"
        )
        if abs(float(item.get("transformer_adjustment", 0.0))) > 1e-8:
            print(
                f"Transformer adjustment: {float(item['transformer_adjustment']):+.4f} "
                f"(raw: {float(item.get('base_score_raw', item['base_score'])):.3f} -> used: {item['base_score']:.3f})"
            )
        if item.get("fundamental_score") is not None:
            print(f"基本面: {float(item['fundamental_score']):.2f} | {item.get('fundamental_summary') or '无'}")
        if item.get("technical_score_ai") is not None:
            print(f"技术面AI: {float(item['technical_score_ai']):.2f} | {item.get('technical_summary_ai') or '无'}")
        print(
            f"建议仓位: {item['position_ratio'] * 100:.1f}% | "
            f"建议买入: {item['recommended_shares']:,} 股 | "
            f"Transformer 概率: {float(item['transformer_score']):.4f} | "
            f"Conf: {float(item.get('transformer_conf', 0.0)):.4f} | "
            f"PredRet: {float(item.get('transformer_pred_ret_raw', 0.0)) * 100:+.3f}%"
        )
