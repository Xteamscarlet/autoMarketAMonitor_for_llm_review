# -*- coding: utf-8 -*-
"""实盘决策入口脚本。"""

import argparse
import logging
import os
import sys

from config import get_settings


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="实盘决策助手")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细日志")
    parser.add_argument("--dry-run", action="store_true", help="仅检查环境和配置，不执行决策")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("advisor")
    settings = get_settings()

    print("\n" + "=" * 60)
    print("实盘决策助手启动")
    print("=" * 60)
    print(f"  策略文件: {settings.paths.strategy_file}")
    print(f"  持仓文件: {settings.paths.portfolio_file}")
    print(f"  模型路径: {settings.paths.model_path}")
    print(f"  风控 - 最大回撤: {settings.risk.max_drawdown_limit}%")
    print(f"  风控 - 最小利润因子: {settings.risk.min_profit_factor}")
    print(f"  风控 - 单只最大仓位: {settings.risk.max_position_ratio:.0%}")
    print(f"  风控 - 最小胜率: {settings.risk.min_win_rate}%")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] 环境检查:")

        if os.path.exists(settings.paths.strategy_file):
            import json

            with open(settings.paths.strategy_file, "r", encoding="utf-8") as file:
                strategies = json.load(file)
            print(f"  [OK] 策略文件: {len(strategies)} 只股票的策略参数")
        else:
            print("  [MISSING] 策略文件不存在，请先运行 run_backtest.py")

        model_exists = os.path.exists(settings.paths.model_path)
        swa_exists = os.path.exists(settings.paths.swa_model_path)
        print(f"  {'[OK]' if model_exists else '[MISSING]'} EMA 模型: {settings.paths.model_path}")
        print(f"  {'[OK]' if swa_exists else '[MISSING]'} SWA 模型: {settings.paths.swa_model_path}")

        scaler_exists = os.path.exists(settings.paths.scaler_path)
        global_scaler_exists = os.path.exists(settings.paths.global_scaler_path)
        print(f"  {'[OK]' if scaler_exists else '[MISSING]'} 专用 Scaler: {settings.paths.scaler_path}")
        print(f"  {'[OK]' if global_scaler_exists else '[MISSING]'} 全局 Scaler: {settings.paths.global_scaler_path}")

        portfolio_exists = os.path.exists(settings.paths.portfolio_file)
        print(f"  {'[OK]' if portfolio_exists else '[MISSING]'} 持仓文件: {settings.paths.portfolio_file}")

        print(
            f"  [INFO] 基本面 Agent: {'开启' if settings.analysis.enable_fundamental_agent else '关闭'} | "
            f"技术面 Agent: {'开启' if settings.analysis.enable_technical_agent else '关闭'}"
        )
        print(
            f"  [INFO] LLM: {'开启' if settings.llm.enabled else '关闭'} | "
            f"Model: {settings.llm.model or '未配置'} | Base URL: {settings.llm.base_url or '未配置'}"
        )

        try:
            import torch

            if torch.cuda.is_available():
                print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("  [INFO] 未检测到 GPU，使用 CPU 推理")
        except ImportError:
            print("  [MISSING] PyTorch 未安装")

        return

    try:
        from live.advisor import run_advisor

        run_advisor()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as exc:
        logger.error("决策执行失败: %s", exc)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
