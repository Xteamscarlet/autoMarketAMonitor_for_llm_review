# -*- coding: utf-8 -*-
"""Fundamentals analyst agent with coverage-aware scoring."""

import json
import logging
import os
from typing import Any, Dict, List

from config import get_settings
from data.fundamentals import load_fundamental_snapshot
from llm import LLMUnavailableError, OpenAICompatibleClient

logger = logging.getLogger(__name__)


def _metric_value(metrics: Dict[str, Dict[str, Any]], name: str) -> Any:
    return (metrics.get(name) or {}).get("value")


def _rule_based_score(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    metrics = snapshot.get("metrics", {})
    coverage_ratio = float(snapshot.get("coverage_ratio", 0.0))
    positives: List[str] = []
    risks: List[str] = []
    score = 0.5

    roe = _metric_value(metrics, "roe")
    if roe is not None:
        if roe >= 15:
            score += 0.10
            positives.append("ROE 较高")
        elif roe < 8:
            score -= 0.08
            risks.append("ROE 偏弱")

    net_margin = _metric_value(metrics, "net_margin")
    if net_margin is not None:
        if net_margin >= 10:
            score += 0.06
            positives.append("净利率较好")
        elif net_margin < 3:
            score -= 0.05
            risks.append("净利率偏低")

    gross_margin = _metric_value(metrics, "gross_margin")
    if gross_margin is not None and gross_margin >= 20:
        score += 0.03
        positives.append("毛利率较稳")

    asset_liability_ratio = _metric_value(metrics, "asset_liability_ratio")
    if asset_liability_ratio is not None:
        if asset_liability_ratio > 70:
            score -= 0.08
            risks.append("资产负债率偏高")
        elif asset_liability_ratio < 50:
            score += 0.04
            positives.append("资产负债结构稳健")

    current_ratio = _metric_value(metrics, "current_ratio")
    if current_ratio is not None:
        if current_ratio >= 1.5:
            score += 0.04
            positives.append("流动比率健康")
        elif current_ratio < 1.0:
            score -= 0.06
            risks.append("短期偿债压力偏高")

    net_profit_yoy = _metric_value(metrics, "net_profit_yoy")
    if net_profit_yoy is not None:
        if net_profit_yoy >= 20:
            score += 0.08
            positives.append("利润增速较强")
        elif net_profit_yoy < 0:
            score -= 0.08
            risks.append("利润同比下滑")

    operating_cashflow = _metric_value(metrics, "operating_cashflow")
    if operating_cashflow is not None and operating_cashflow < 0:
        score -= 0.08
        risks.append("经营现金流为负")

    free_cashflow = _metric_value(metrics, "free_cashflow")
    if free_cashflow is not None and free_cashflow > 0:
        score += 0.03
        positives.append("自由现金流为正")

    rd_ratio = _metric_value(metrics, "rd_ratio")
    if rd_ratio is not None and rd_ratio >= 5:
        score += 0.03
        positives.append("研发投入占比不低")

    if coverage_ratio < 0.35:
        score = min(score, 0.48)
        risks.append("财报覆盖率偏低")
    elif coverage_ratio < 0.55:
        score -= 0.03
        risks.append("部分关键财报字段缺失")

    score = max(0.0, min(1.0, score))
    confidence = min(0.90, 0.20 + coverage_ratio * 0.70)

    if positives or risks:
        summary_parts = positives[:2] + risks[:2]
        summary = "；".join(summary_parts)
    else:
        summary = "财报数据不足，采用保守判断"

    return {
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "summary": summary,
        "positives": positives,
        "risks": risks,
        "tags": positives[:2] + risks[:2],
        "source": "rule_based",
        "latest_period": snapshot.get("latest_period"),
        "coverage_ratio": round(coverage_ratio, 4),
        "filled_metrics": snapshot.get("filled_metrics", []),
        "missing_metrics": snapshot.get("missing_metrics", []),
    }


def _llm_score(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAICompatibleClient()
    system_prompt = (
        "你是 A 股基本面分析师。"
        "请基于提供的财报指标、覆盖率和最近报告期，输出严格 JSON。"
        "如果数据缺失较多，不要脑补，应明确说明信息不足并降低置信度。"
    )
    user_prompt = (
        f"股票: {snapshot.get('name', '')}({snapshot.get('code', '')})\n"
        f"最新报告期: {snapshot.get('latest_period')}\n"
        f"覆盖率: {snapshot.get('coverage_ratio')}\n"
        f"已命中指标: {snapshot.get('filled_metrics', [])}\n"
        f"缺失指标: {snapshot.get('missing_metrics', [])}\n"
        f"财报指标: {json.dumps(snapshot.get('metrics', {}), ensure_ascii=False)}\n\n"
        "请输出 JSON 对象，字段必须包含: "
        "score(0到1), confidence(0到1), summary, positives(array), risks(array), tags(array)。"
    )
    response = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
    response["source"] = "llm"
    response["latest_period"] = snapshot.get("latest_period")
    response["coverage_ratio"] = snapshot.get("coverage_ratio")
    response["filled_metrics"] = snapshot.get("filled_metrics", [])
    response["missing_metrics"] = snapshot.get("missing_metrics", [])
    return response


def analyze_fundamentals(code: str, name: str = "", refresh: bool = False) -> Dict[str, Any]:
    settings = get_settings()
    cache_path = os.path.join(settings.paths.ai_analysis_cache_dir, f"{code}_fundamental_analysis.json")
    if os.path.exists(cache_path) and not refresh:
        try:
            with open(cache_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            logger.warning("Failed to load cached fundamental analysis for %s", code)

    snapshot = load_fundamental_snapshot(code=code, name=name, refresh=refresh)
    try:
        result = _llm_score(snapshot) if settings.llm.enabled else _rule_based_score(snapshot)
    except (LLMUnavailableError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Fundamental LLM analysis failed for %s: %s", code, exc)
        result = _rule_based_score(snapshot)

    result.update(
        {
            "code": code,
            "name": name,
            "snapshot_available": snapshot.get("raw_available", False),
            "source_details": snapshot.get("source_details", {}),
        }
    )
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result
