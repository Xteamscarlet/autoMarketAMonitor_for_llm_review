# -*- coding: utf-8 -*-
"""
独立预测入口脚本
对指定股票池进行集成预测并输出排序结果
"""
import os
import json
import logging
import argparse
import time

from config import get_settings


def main():
    parser = argparse.ArgumentParser(description="集成预测")
    parser.add_argument('--codes', nargs='+', default=None, help='指定股票代码列表')
    parser.add_argument('--pool', action='store_true', help='使用 stock_pool.json 中的股票池')
    parser.add_argument('--top', type=int, default=30, help='输出前N只股票')
    parser.add_argument('--output', type=str, default='stock_predictions.json', help='输出文件路径')
    parser.add_argument('--notify', action='store_true', help='发送企业微信通知')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('predict')
    settings = get_settings()

    print("\n" + "=" * 60)
    print(f"集成预测 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 确定目标代码
    target_codes = args.codes

    if args.pool:
        pool_file = settings.paths.stock_pool_file
        if os.path.exists(pool_file):
            with open(pool_file, 'r') as f:
                pool_data = json.load(f)
            pool_codes = list(pool_data.get('default_pool', {}).keys())
            if target_codes:
                target_codes = list(set(target_codes) | set(pool_codes))
            else:
                target_codes = pool_codes
            print(f"✓ 从股票池加载 {len(pool_codes)} 只股票")
        else:
            print(f"✗ 股票池文件不存在: {pool_file}")

    if not target_codes:
        # 使用默认列表
        from config import STOCK_CODES
        target_codes = list(STOCK_CODES.values())
        print(f"✓ 使用默认股票池: {len(target_codes)} 只")

    print(f"  目标股票数: {len(target_codes)}")
    print(f"  输出前 {args.top} 只")
    print("=" * 60)

    # 执行预测
    from model.predictor import predict_stocks
    results = predict_stocks(target_codes)

    if results.empty:
        print("没有成功预测任何股票")
        return

    # 输出结果
    top_results = results.head(args.top)
    print(f"\n{'排名':>4} {'代码':<8} {'趋势':<4} {'概率':>6} {'预测收益':>8} {'综合得分':>8} {'分歧度':>6} {'模型数':>4} {'Scaler':<6}")
    print("-" * 70)
    for _, row in top_results.iterrows():
        print(f"{int(row['rank']):>4} {row['code']:<8} {row['trend']:<4} "
              f"{row['probability']:>6.1%} {row['predicted_ret']:>+7.2f}% "
              f"{row['expected_score']:>+8.4f} {row['uncertainty']:>6.4f} "
              f"{int(row['ensemble_size']):>4} {row['scaler_type']:<6}")

    # 保存
    results.to_json(args.output, orient='records', force_ascii=False, indent=2)
    print(f"\n✓ 预测结果已保存: {args.output} ({len(results)} 只股票)")

    # 通知
    if args.notify and settings.paths.wechat_webhook:
        try:
            import requests
            summary = f"📊 集成预测完成\n共 {len(results)} 只股票\n\n"
            summary += "TOP 10:\n"
            for _, row in results.head(10).iterrows():
                summary += f"  {row['code']} {row['trend']} 概率{row['probability']:.0%} 预测{row['predicted_ret']:+.2f}%\n"

            headers = {'Content-Type': 'application/json'}
            data = {"msgtype": "text", "text": {"content": summary}}
            requests.post(settings.paths.wechat_webhook, headers=headers, data=json.dumps(data), timeout=5)

            # 发送文件
            if settings.paths.wechat_upload_url and os.path.exists(args.output):
                send_file_to_wechat(args.output, settings.paths.wechat_webhook, settings.paths.wechat_upload_url)
        except Exception as e:
            print(f"通知发送失败: {e}")


def send_file_to_wechat(file_path, webhook_url, upload_url):
    """发送文件到企业微信"""
    try:
        import requests
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            upload_resp = requests.post(upload_url, files=files, params={'type': 'file'}, timeout=30)
            media_id = upload_resp.json().get('media_id')
            if media_id:
                headers = {'Content-Type': 'application/json'}
                data = {"msgtype": "file", "file": {"media_id": media_id}}
                requests.post(webhook_url, headers=headers, data=json.dumps(data), timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
