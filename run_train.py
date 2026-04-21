# -*- coding: utf-8 -*-
"""
模型训练入口脚本
负责数据检查、调用训练流程、以及训练后通知。
"""
import argparse
import json
import logging
import os
import sys
import time

import requests

from config import get_settings


def console_print(message: str = "") -> None:
    """Print with a fallback for Windows consoles that are not UTF-8."""
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = (
            message.replace("✓", "[OK]")
            .replace("✗", "[X]")
            .replace("⚠️", "[WARN]")
            .replace("❌", "[ERR]")
        )
        print(safe_message)


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class ChatLogger(logging.Handler):
    """企业微信日志推送 Handler。"""

    def __init__(self, webhook_url: str = ""):
        super().__init__()
        self.setLevel(logging.WARNING)
        self.original_stdout = None
        self.webhook_url = webhook_url

    def set_stdout(self, stdout):
        self.original_stdout = stdout

    def emit(self, record):
        if self.original_stdout:
            self.original_stdout.write(self.format(record) + "\n")
        if self.webhook_url:
            try:
                headers = {"Content-Type": "application/json"}
                data = {"msgtype": "text", "text": {"content": self.format(record).strip()}}
                requests.post(self.webhook_url, headers=headers, data=json.dumps(data), timeout=5)
            except Exception:
                pass


def send_file_to_wechat(file_path: str, webhook_url: str = "", upload_url: str = "") -> bool:
    """上传文件到企业微信并发送。"""
    if not webhook_url or not upload_url:
        return False
    try:
        if not os.path.exists(file_path):
            return False
        if os.path.getsize(file_path) > 20 * 1024 * 1024:
            return False

        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            upload_resp = requests.post(upload_url, files=files, params={"type": "file"}, timeout=30)
            upload_result = upload_resp.json()
            if upload_result.get("errcode") != 0:
                return False

            media_id = upload_result.get("media_id")
            headers = {"Content-Type": "application/json"}
            data = {"msgtype": "file", "file": {"media_id": media_id}}
            send_resp = requests.post(webhook_url, headers=headers, data=json.dumps(data), timeout=10)
            return send_resp.json().get("errcode") == 0
    except Exception:
        return False


def prepare_training_data(settings) -> bool:
    """检查训练数据文件是否存在。"""
    if os.path.exists(settings.paths.stock_data_file):
        console_print(f"✓ 训练数据已存在: {settings.paths.stock_data_file}")
        return True

    console_print(f"✗ 训练数据文件不存在: {settings.paths.stock_data_file}")
    console_print("  请先使用 TransformerStock.py 中的 get_all_stock_data() 下载数据")
    return False


def cleanup_stale_checkpoints(keep_last: int = 0) -> None:
    """★ 修复1: 清理旧 epoch checkpoint，避免 trainer.py 错误 resume 旧 lr/state。

    Args:
        keep_last: 保留最近 N 个 ckpt（默认 0 = 全部清理）
    """
    import glob as _glob

    pattern = "model_epoch_*.pth"
    files = sorted(_glob.glob(pattern), key=os.path.getctime)
    if not files:
        return
    to_remove = files[:-keep_last] if keep_last > 0 else files
    for f in to_remove:
        try:
            os.remove(f)
            console_print(f"  ✓ 已清理旧 ckpt: {f}")
        except OSError as e:
            console_print(f"  ✗ 清理失败 {f}: {e}")


def main():
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Transformer 模型训练")
    parser.add_argument("--dry-run", action="store_true", help="仅检查环境，不执行训练")
    parser.add_argument("--notify", action="store_true", help="训练完成后发送企业微信通知")
    parser.add_argument(
        "--resume", action="store_true",
        help="不清理旧 epoch ckpt，允许 trainer.py 从最近一个 ckpt 恢复",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，多次训练时用不同 seed 形成 ensemble 多样性",
    )
    args = parser.parse_args()

    # ★ 修复4: 设置随机种子，便于多 seed 训练形成真正的 ensemble
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch as _torch
        _torch.manual_seed(args.seed)
        _torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    settings = get_settings()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger("train")
    if args.notify and settings.paths.wechat_webhook:
        chat_handler = ChatLogger(settings.paths.wechat_webhook)
        chat_handler.set_stdout(__import__("sys").stdout)
        logger.addHandler(chat_handler)

    console_print("\n" + "=" * 60)
    console_print(f"Transformer 模型训练 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    console_print("=" * 60)
    console_print(f"  Lookback: {settings.model.lookback_days} 天")
    console_print(f"  Batch Size: {settings.model.batch_size}")
    console_print(f"  Epochs: {settings.model.epochs}")
    console_print(f"  Learning Rate: {settings.model.learning_rate}")
    console_print(f"  Weight Decay: {settings.model.weight_decay}")
    console_print(f"  Layers: {settings.model.num_layers} x {settings.model.dim_feedforward}")
    console_print(f"  Heads: {settings.model.num_heads}")
    console_print(f"  Dropout: {settings.model.dropout}")
    console_print(f"  Accumulation: {settings.model.accumulation_steps} steps")
    console_print(f"  EMA Decay: {settings.model.ema_decay}")
    console_print("=" * 60)

    if args.dry_run:
        console_print("\n[DRY RUN] 仅检查环境，不执行训练")
        data_ok = prepare_training_data(settings)
        if data_ok:
            console_print("✓ 环境检查通过，可以开始训练")
        else:
            console_print("✗ 环境检查未通过，请先准备数据")
        return

    if not prepare_training_data(settings):
        return

    # ★ 修复1: 默认清理旧 epoch ckpt（避免污染本次 lr/optimizer state）
    if not args.resume:
        console_print("\n清理旧 epoch checkpoint（使用 --resume 可跳过）...")
        cleanup_stale_checkpoints(keep_last=0)

    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        console_print(f"\n✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        console_print("\n⚠️ 未检测到 GPU，将使用 CPU 训练（速度较慢）")

    console_print("\n开始训练...\n")
    start_time = time.time()

    try:
        from model.trainer import train_model

        train_model(settings)
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback

        traceback.print_exc()
        if args.notify and settings.paths.wechat_webhook:
            try:
                headers = {"Content-Type": "application/json"}
                data = {"msgtype": "text", "text": {"content": f"❌ 训练失败: {e}"}}
                requests.post(settings.paths.wechat_webhook, headers=headers, data=json.dumps(data), timeout=5)
            except Exception:
                pass
        return

    elapsed = time.time() - start_time
    console_print(f"\n✓ 训练完成！耗时: {elapsed / 60:.1f} 分钟")

    if args.notify:
        model_files = []
        for p in [settings.paths.model_path, settings.paths.swa_model_path]:
            if os.path.exists(p):
                size_mb = os.path.getsize(p) / 1024 / 1024
                model_files.append(f"{os.path.basename(p)} ({size_mb:.1f}MB)")

        topk_dir = settings.paths.topk_checkpoint_dir
        if os.path.exists(topk_dir):
            for f in os.listdir(topk_dir):
                if f.endswith(".pth"):
                    size_mb = os.path.getsize(os.path.join(topk_dir, f)) / 1024 / 1024
                    model_files.append(f"{f} ({size_mb:.1f}MB)")

        msg = f"✓ 训练完成！\n耗时: {elapsed / 60:.1f}分钟\n模型文件:\n" + "\n".join(model_files)

        if settings.paths.wechat_webhook:
            try:
                headers = {"Content-Type": "application/json"}
                data = {"msgtype": "text", "text": {"content": msg}}
                requests.post(settings.paths.wechat_webhook, headers=headers, data=json.dumps(data), timeout=5)
            except Exception:
                pass

        prediction_file = "stock_predictions.json"
        if os.path.exists(prediction_file) and settings.paths.wechat_upload_url:
            send_file_to_wechat(
                prediction_file,
                settings.paths.wechat_webhook,
                settings.paths.wechat_upload_url,
            )

    console_print("\n模型文件:")
    for p in [
        settings.paths.model_path,
        settings.paths.swa_model_path,
        settings.paths.scaler_path,
        settings.paths.global_scaler_path,
    ]:
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1024 / 1024
            console_print(f"  ✓ {p} ({size_mb:.1f} MB)")
        else:
            console_print(f"  ✗ {p} (不存在)")


if __name__ == "__main__":
    main()
