# -*- coding: utf-8 -*-
"""
独立预测入口脚本
对指定股票池进行集成预测，并可叠加财报 / 技术分析 AI 研究结果。
"""
import os
import json
import logging
import argparse
import time

import pandas as pd

from utils.stock_filter import filter_codes_by_name
from config import get_settings, STOCK_CODES
from agents.research_manager import analyze_stock


def main():
    parser = argparse.ArgumentParser(description="集成预测")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码列表")
    parser.add_argument("--pool", action="store_true", help="使用 stock_pool.json 中的股票池")
    parser.add_argument("--top", type=int, default=30, help="输出前 N 只股票")
    parser.add_argument("--output", type=str, default="stock_predictions.json", help="输出文件路径")
    parser.add_argument("--notify", action="store_true", help="发送企业微信通知")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    settings = get_settings()

    print("\n" + "=" * 60)
    print(f"集成预测 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    target_codes = args.codes
    if args.pool:
        pool_file = settings.paths.stock_pool_file
        if os.path.exists(pool_file):
            with open(pool_file, "r", encoding="utf-8") as f:
                pool_data = json.load(f)
            pool_codes = list(pool_data.get("default_pool", {}).keys())
            target_codes = list(set(target_codes or []) | set(pool_codes)) if target_codes else pool_codes
            print(f"✓ 从股票池加载 {len(pool_codes)} 只股票")
        else:
            print(f"✗ 股票池文件不存在: {pool_file}")

    if not target_codes:
        name_to_code_map = {n: c for n, c in STOCK_CODES.items()}
        clean_map = filter_codes_by_name(name_to_code_map)
        target_codes = list(clean_map.values())
        print(f"✓ 使用默认股票池: {len(target_codes)} 只")

    print(f"  目标股票数: {len(target_codes)}")
    print(f"  输出前 {args.top} 只")
    print("=" * 60)

    from model.predictor import predict_stocks

    results = predict_stocks(target_codes)
    if results.empty:
        print("没有成功预测任何股票")
        return

    code_to_name = {v: k for k, v in STOCK_CODES.items()}
    enriched_rows = []
    for _, row in results.iterrows():
        row_dict = row.to_dict()
        code = row_dict["code"]
        name = code_to_name.get(code, "")
        research = analyze_stock(code=code, name=name)
        fundamentals = research.get("fundamentals") or {}
        technicals = research.get("technicals") or {}

        research_bonus = 0.0
        if settings.analysis.blend_fundamental_score and fundamentals:
            research_bonus += settings.analysis.fundamental_weight * (fundamentals.get("score", 0.5) - 0.5)
        if settings.analysis.blend_technical_score and technicals:
            research_bonus += settings.analysis.technical_weight * (technicals.get("score", 0.5) - 0.5)

        row_dict["name"] = name
        row_dict["fundamental_score"] = fundamentals.get("score")
        row_dict["fundamental_summary"] = fundamentals.get("summary")
        row_dict["technical_score_ai"] = technicals.get("score")
        row_dict["technical_summary_ai"] = technicals.get("summary")
        row_dict["research_bonus"] = round(float(research_bonus), 4)
        row_dict["enhanced_score"] = round(float(row_dict["expected_score"] + research_bonus), 4)
        enriched_rows.append(row_dict)

    results = pd.DataFrame(enriched_rows)
    results = results.sort_values(by="enhanced_score", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    top_results = results.head(args.top)
    print(
        f"\n{'排名':>4} {'代码':<8} {'趋势':<4} {'概率':>6} {'预测收益':>8} "
        f"{'增强得分':>10} {'基本面':>8} {'技术AI':>8}"
    )
    print("-" * 80)
    for _, row in top_results.iterrows():
        fundamental_score = row["fundamental_score"] if pd.notna(row["fundamental_score"]) else 0.5
        technical_score = row["technical_score_ai"] if pd.notna(row["technical_score_ai"]) else 0.5
        print(
            f"{int(row['rank']):>4} {row['code']:<8} {row['trend']:<4} "
            f"{row['probability']:>6.1%} {row['predicted_ret'] * 100:>+7.2f}% "
            f"{row['enhanced_score']:>+10.4f} {fundamental_score:>8.2f} {technical_score:>8.2f}"
        )

    results.to_json(args.output, orient="records", force_ascii=False, indent=2)
    print(f"\n✓ 预测结果已保存: {args.output} ({len(results)} 只股票)")

    if args.notify and settings.paths.wechat_webhook:
        try:
            import requests

            summary = f"📊 集成预测完成\n共 {len(results)} 只股票\n\nTOP 10:\n"
            for _, row in results.head(10).iterrows():
                summary += (
                    f"  {row['code']} {row['trend']} 概率{row['probability']:.0%} "
                    f"预测{row['predicted_ret'] * 100:+.2f}% 增强分{row['enhanced_score']:+.3f}\n"
                )

            headers = {"Content-Type": "application/json"}
            data = {"msgtype": "text", "text": {"content": summary}}
            requests.post(settings.paths.wechat_webhook, headers=headers, data=json.dumps(data), timeout=5)

            if settings.paths.wechat_upload_url and os.path.exists(args.output):
                send_file_to_wechat(args.output, settings.paths.wechat_webhook, settings.paths.wechat_upload_url)
        except Exception as exc:
            print(f"通知发送失败: {exc}")


def send_file_to_wechat(file_path, webhook_url, upload_url):
    """发送文件到企业微信"""
    try:
        import requests

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            upload_resp = requests.post(upload_url, files=files, params={"type": "file"}, timeout=30)
            media_id = upload_resp.json().get("media_id")
            if media_id:
                headers = {"Content-Type": "application/json"}
                data = {"msgtype": "file", "file": {"media_id": media_id}}
                requests.post(webhook_url, headers=headers, data=json.dumps(data), timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
