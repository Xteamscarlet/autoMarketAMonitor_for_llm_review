# -*- coding: utf-8 -*-
"""
组合层面风控模块 V2
新增：相关性检查、板块集中度（基于 SECTOR_MAP）
"""
import logging
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from risk_manager import RiskManager
from config import get_settings, SECTOR_MAP

logger = logging.getLogger(__name__)


def check_portfolio_limits(
    current_positions: List[Dict],
    new_candidates: List[Dict],
    max_total_ratio: float = 0.8,
) -> Tuple[List[Dict], List[str]]:
    """组合仓位风控 V2（含板块集中度检查）"""
    settings = get_settings()
    rm = RiskManager(settings.risk)

    # ★ 注入板块信息和相关性检查
    enriched_positions = []
    for pos in current_positions:
        code = pos.get('code', '')
        pos_with_sector = {**pos, 'sector': SECTOR_MAP.get(code, 'unknown')}
        enriched_positions.append(pos_with_sector)

    enriched_candidates = []
    for cand in new_candidates:
        code = cand.get('code', '')
        cand_with_sector = {**cand, 'sector': SECTOR_MAP.get(code, 'unknown')}
        enriched_candidates.append(cand_with_sector)

    filtered, warnings = rm.check_portfolio_risk(
        current_positions=enriched_positions,
        new_candidates=enriched_candidates,
        max_total_ratio=max_total_ratio,
        max_single_ratio=settings.risk.max_position_ratio,
        max_same_sector_ratio=settings.risk.max_sector_ratio,
    )

    return filtered, warnings


def check_daily_loss_limit(
    daily_pnl: float,
    total_capital: float,
    max_daily_loss_ratio: float = 0.03,
) -> Tuple[bool, str]:
    """单日最大亏损检查"""
    loss_ratio = abs(daily_pnl) / total_capital if total_capital > 0 else 0
    if daily_pnl < 0 and loss_ratio > max_daily_loss_ratio:
        return False, f"单日亏损({loss_ratio:.1%})超过限制({max_daily_loss_ratio:.1%})，暂停交易"
    return True, ""


def check_correlation_limit(
    new_code: str,
    existing_codes: List[str],
    stocks_data: Dict[str, pd.DataFrame],
    max_correlation: float = 0.7,
    lookback: int = 60,
) -> Tuple[bool, float]:
    """
    ★ 新增：持仓相关性检查
    计算新候选股票与已有持仓的收益相关性，过高则拒绝

    Returns:
        (是否通过, 最大相关系数)
    """
    if not existing_codes:
        return True, 0.0

    new_name = None
    for name, code in SECTOR_MAP.items():
        if code == new_code:
            new_name = name
            break

    if new_name is None or new_name not in stocks_data:
        return True, 0.0

    new_df = stocks_data[new_name]
    if len(new_df) < lookback:
        return True, 0.0

    new_ret = new_df['Close'].pct_change().iloc[-lookback:]
    max_corr = 0.0

    for existing_code in existing_codes:
        existing_name = None
        for name, code in SECTOR_MAP.items():
            if code == existing_code:
                existing_name = name
                break

        if existing_name is None or existing_name not in stocks_data:
            continue

        existing_df = stocks_data[existing_name]
        if len(existing_df) < lookback:
            continue

        existing_ret = existing_df['Close'].pct_change().iloc[-lookback:]

        # 对齐日期
        common_idx = new_ret.index.intersection(existing_ret.index)
        if len(common_idx) < 30:
            continue

        corr = new_ret.loc[common_idx].corr(existing_ret.loc[common_idx])
        if abs(corr) > max_corr:
            max_corr = abs(corr)

    if max_corr > max_correlation:
        return False, max_corr

    return True, max_corr
