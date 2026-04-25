# -*- coding: utf-8 -*-
"""Technical analyst agent inspired by multi-agent hedge fund workflows."""

import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd

from config import get_settings
from llm import OpenAICompatibleClient

logger = logging.getLogger(__name__)


def _build_snapshot(df: pd.DataFrame, code: str, name: str = "") -> Dict[str, Any]:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    close = float(latest.get("Close", 0.0))
    prev_close = float(prev.get("Close", close)) or close
    volume = float(latest.get("Volume", 0.0))
    volume_mean = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else volume

    return {
        "code": code,
        "name": name,
        "date": str(df.index[-1]),
        "close": close,
        "change_pct": float((close / prev_close - 1.0) * 100) if prev_close else 0.0,
        "ma5": float(latest.get("MA5", close)),
        "ma10": float(latest.get("MA10", close)),
        "ma20": float(latest.get("MA20", close)),
        "rsi": float(latest.get("RSI", 50.0)),
        "adx": float(latest.get("ADX", 20.0)),
        "macd": float(latest.get("MACD", 0.0)),
        "macd_hist": float(latest.get("MACD_Hist", 0.0)),
        "bb_upper": float(latest.get("BB_Upper", close)),
        "bb_lower": float(latest.get("BB_Lower", close)),
        "atr": float(latest.get("atr", 0.0)),
        "volume_ratio": float(volume / volume_mean) if volume_mean else 1.0,
        "transformer_prob": float(latest.get("transformer_prob", 0.5)),
        "transformer_conf": float(latest.get("transformer_conf", 0.5)),
        "transformer_pred_ret": float(latest.get("transformer_pred_ret_raw", latest.get("transformer_pred_ret", 0.0))),
        "combined_score": float(latest.get("Combined_Score", 0.5)),
    }


def _rule_based_analysis(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    score = 0.5
    positives: List[str] = []
    risks: List[str] = []

    if snapshot["close"] > snapshot["ma20"] > 0 and snapshot["ma5"] >= snapshot["ma10"] >= snapshot["ma20"]:
        score += 0.12
        positives.append("均线呈多头排列")
    elif snapshot["close"] < snapshot["ma20"]:
        score -= 0.10
        risks.append("价格位于 MA20 下方")

    if snapshot["macd_hist"] > 0:
        score += 0.05
        positives.append("MACD 动能为正")
    else:
        score -= 0.04
        risks.append("MACD 动能偏弱")

    if 45 <= snapshot["rsi"] <= 70:
        score += 0.04
        positives.append("RSI 处于健康区间")
    elif snapshot["rsi"] > 80:
        score -= 0.06
        risks.append("RSI 过热")
    elif snapshot["rsi"] < 25:
        score += 0.02
        positives.append("RSI 超卖，存在反弹可能")

    if snapshot["volume_ratio"] >= 1.2:
        score += 0.04
        positives.append("成交量放大")

    score += (snapshot["transformer_prob"] - 0.5) * 0.20
    score += max(min(snapshot["transformer_pred_ret"], 0.10), -0.10)

    if snapshot["transformer_conf"] < 0.45:
        score -= 0.04
        risks.append("AI 置信度偏低")

    score = max(0.0, min(1.0, score))
    stance = "bullish" if score >= 0.62 else "bearish" if score <= 0.38 else "neutral"

    return {
        "score": round(score, 4),
        "confidence": round(max(0.30, min(0.85, abs(score - 0.5) * 2 + 0.35)), 4),
        "stance": stance,
        "summary": "；".join((positives[:2] + risks[:2]) or ["技术形态中性"]),
        "positives": positives,
        "risks": risks,
        "key_levels": {
            "support": round(snapshot["bb_lower"], 2),
            "resistance": round(snapshot["bb_upper"], 2),
        },
        "source": "rule_based",
    }


def _llm_analysis(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAICompatibleClient()
    system_prompt = (
        "你是 A 股技术分析师，风格参考多 agent research workflow。"
        "请综合趋势、动量、波动、量价配合以及 AI 概率信号，输出严格 JSON。"
    )
    user_prompt = (
        f"技术快照: {json.dumps(snapshot, ensure_ascii=False)}\n"
        "请输出 JSON 对象，字段必须包含: "
        "score(0到1), confidence(0到1), stance, summary, positives(array), risks(array), "
        "key_levels(object, 包含 support 和 resistance)。"
    )
    response = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
    response["source"] = "llm"
    return response


def analyze_technicals(df: pd.DataFrame, code: str, name: str = "", refresh: bool = False) -> Dict[str, Any]:
    settings = get_settings()
    trade_date = str(df.index[-1]).split(" ")[0]
    cache_path = os.path.join(settings.paths.ai_analysis_cache_dir, f"{code}_{trade_date}_technical_analysis.json")
    if os.path.exists(cache_path) and not refresh:
        try:
            with open(cache_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            logger.warning("Failed to load cached technical analysis for %s", code)

    snapshot = _build_snapshot(df.tail(settings.analysis.max_recent_days_for_technical), code=code, name=name)
    try:
        result = _llm_analysis(snapshot) if settings.llm.enabled else _rule_based_analysis(snapshot)
    except Exception as exc:
        logger.warning("Technical LLM analysis failed for %s: %s", code, exc)
        result = _rule_based_analysis(snapshot)

    result.update({"code": code, "name": name, "trade_date": trade_date})
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result
