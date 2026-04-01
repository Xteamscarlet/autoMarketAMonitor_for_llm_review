# -*- coding: utf-8 -*-
"""
双层风控管理器
- 硬限制：回测前参数合法性校验，违反则直接阻断
- 软目标：回测后全局风险评估，不达标则策略被标记为 discard
- 新增：向量化风控（apply_vectorized_risk_controls）
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import get_settings, RiskConfig


class RiskManager:
    """双层风控管理器"""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or get_settings().risk

    # ==================== 硬限制检查 ====================
    @staticmethod
    def check_hard_limits(params: dict, regime: str = "neutral") -> bool:
        """回测前硬限制检查
        检查项：
        1. 止损不能太宽（<-15%）
        2. 买入阈值不能太低（<0.4）
        3. 买入阈值必须大于卖出阈值
        4. 持仓天数合理（1-60天）
        5. 移动止损参数逻辑合理
        Raises:
            ValueError: 参数不合法时抛出
        """
        regime_params = params.get(regime, params.get("neutral", params))

        stop_loss = regime_params.get("stop_loss", -0.08)
        if stop_loss < -0.15:
            raise ValueError(f"止损过宽: {stop_loss:.2%} < -15%")

        buy_threshold = regime_params.get("buy_threshold", 0.6)
        if buy_threshold < 0.4:
            raise ValueError(f"买入阈值过低: {buy_threshold:.2f} < 0.4")

        sell_threshold = regime_params.get("sell_threshold", -0.2)
        if buy_threshold <= sell_threshold:
            raise ValueError(
                f"买入阈值({buy_threshold:.2f})必须大于卖出阈值({sell_threshold:.2f})"
            )

        hold_days = regime_params.get("hold_days", 15)
        if not (1 <= hold_days <= 60):
            raise ValueError(f"持仓天数不合理: {hold_days}，应在[1, 60]范围内")

        # 移动止损逻辑检查
        tp1 = regime_params.get("trailing_profit_level1", 0.06)
        tp2 = regime_params.get("trailing_profit_level2", 0.12)
        td1 = regime_params.get("trailing_drawdown_level1", 0.08)
        td2 = regime_params.get("trailing_drawdown_level2", 0.04)

        if tp1 <= 0 or tp2 <= 0:
            raise ValueError(f"移动止损触发利润必须为正: level1={tp1}, level2={tp2}")

        if td1 <= 0 or td2 <= 0:
            raise ValueError(f"移动止损回撤幅度必须为正: level1={td1}, level2={td2}")

        if tp2 <= tp1:
            raise ValueError(f"Level2触发利润({tp2})应大于Level1({tp1})")

        return True

    # ==================== 软目标评估 ====================
    def evaluate_soft_targets(self, stats: dict) -> dict:
        """回测后软目标评估
        Returns:
            dict: {
                'passed': bool,
                'violations': List[str],
                'discard': bool,
                'details': {
                    'max_drawdown_pct': float,
                    'max_drawdown_limit': float,
                    'max_drawdown_ok': bool,
                    'profit_factor': float,
                    'min_profit_factor': float,
                    'profit_factor_ok': bool,
                    'sharpe_ratio': float,
                    'min_sharpe_ratio': float,
                    'sharpe_ok': bool,
                    'win_rate': float,
                    'min_win_rate': float,
                    'win_rate_ok': bool,
                    'total_trades': int,
                    'min_trades': int,
                    'max_trades': int,
                    'trades_ok': bool,
                }
            }
        """
        violations = []
        details = {}

        # ---- 最大回撤（核心指标，不通过直接丢弃） ----
        max_dd = stats.get("max_drawdown", 0)
        dd_limit = self.config.max_drawdown_limit
        dd_ok = max_dd >= dd_limit
        details["max_drawdown_pct"] = max_dd
        details["max_drawdown_limit"] = dd_limit
        details["max_drawdown_ok"] = dd_ok
        if not dd_ok:
            violations.append(f"max_drawdown({max_dd:.2f}%) < {dd_limit:.2f}%")

        # ---- 利润因子（核心指标，不通过直接丢弃） ----
        pf = stats.get("profit_factor", 0)
        pf_min = self.config.min_profit_factor
        pf_ok = pf >= pf_min
        details["profit_factor"] = pf
        details["min_profit_factor"] = pf_min
        details["profit_factor_ok"] = pf_ok
        if not pf_ok:
            violations.append(f"profit_factor({pf:.2f}) < {pf_min:.2f}")

        # ---- 夏普比率（软指标） ----
        sharpe = stats.get("sharpe_ratio", 0)
        sharpe_min = self.config.min_sharpe_ratio
        sharpe_ok = sharpe >= sharpe_min
        details["sharpe_ratio"] = sharpe
        details["min_sharpe_ratio"] = sharpe_min
        details["sharpe_ok"] = sharpe_ok
        if not sharpe_ok:
            violations.append(f"sharpe_ratio({sharpe:.2f}) < {sharpe_min:.2f}")

        # ---- 胜率（软指标） ----
        wr = stats.get("win_rate", 0)
        wr_min = self.config.min_win_rate
        wr_ok = wr >= wr_min
        details["win_rate"] = wr
        details["min_win_rate"] = wr_min
        details["win_rate_ok"] = wr_ok
        if not wr_ok:
            violations.append(f"win_rate({wr:.1f}%) < {wr_min:.1f}%")

        # ---- 交易次数（软指标） ----
        nt = stats.get("total_trades", 0)
        nt_ok = self.config.min_trades <= nt <= self.config.max_trades
        details["total_trades"] = nt
        details["min_trades"] = self.config.min_trades
        details["max_trades"] = self.config.max_trades
        details["trades_ok"] = nt_ok
        if nt < self.config.min_trades:
            violations.append(f"total_trades({nt}) < {self.config.min_trades}")
        if nt > self.config.max_trades:
            violations.append(f"total_trades({nt}) > {self.config.max_trades} (过度交易)")

        passed = len(violations) == 0

        # 核心指标不通过 → discard
        critical_failed = not dd_ok or not pf_ok

        discard = critical_failed

        return {
            "passed": passed,
            "violations": violations,
            "discard": discard,
            "details": details,
        }

    # ==================== 组合层面风控 ====================
    @staticmethod
    def check_portfolio_risk(
        current_positions: List[dict],
        new_candidates: List[dict],
        max_total_ratio: float = 0.8,
        max_single_ratio: float = 0.3,
        max_same_sector_ratio: float = 0.4,
    ) -> Tuple[List[dict], List[str]]:
        """组合层面风控过滤
        Args:
            current_positions: 当前持仓列表 [{'code': str, 'ratio': float, 'sector': str}]
            new_candidates: 新买入候选列表 [{'code': str, 'ratio': float, 'sector': str, 'score': float}]
            max_total_ratio: 最大总仓位比例
            max_single_ratio: 单只最大仓位比例
            max_same_sector_ratio: 同板块最大仓位比例
        Returns:
            (filtered_candidates, warnings): 过滤后的候选和警告信息
        """
        warnings = []

        # 当前总仓位
        current_total = sum(p.get("ratio", 0) for p in current_positions)

        remaining_ratio = max_total_ratio - current_total
        if remaining_ratio <= 0:
            warnings.append(f"总仓位已满({current_total:.1%})，拒绝所有新买入")
            return [], warnings

        # 按得分排序
        sorted_candidates = sorted(new_candidates, key=lambda x: x.get("score", 0), reverse=True)
        filtered = []
        used_ratio = 0
        sector_exposure = {}

        # 统计当前板块暴露
        for p in current_positions:
            sector = p.get("sector", "unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + p.get("ratio", 0)

        for cand in sorted_candidates:
            cand_ratio = cand.get("ratio", 0)

            # 单只仓位检查
            if cand_ratio > max_single_ratio:
                warnings.append(f"{cand['code']} 仓位({cand_ratio:.1%})超限，降至{max_single_ratio:.1%}")
                cand_ratio = max_single_ratio

            # 总仓位检查
            if used_ratio + cand_ratio > remaining_ratio:
                available = remaining_ratio - used_ratio
                if available <= 0.01:
                    warnings.append(f"剩余仓位不足，跳过 {cand['code']}")
                    break

                warnings.append(f"{cand['code']} 仓位被截断至{available:.1%}")
                cand_ratio = available

            # 板块集中度检查
            sector = cand.get("sector", "unknown")
            sector_current = sector_exposure.get(sector, 0)
            if sector_current + cand_ratio > max_same_sector_ratio:
                allowed = max(0, max_same_sector_ratio - sector_current)
                if allowed <= 0.01:
                    warnings.append(f"{sector}板块已满，跳过 {cand['code']}")
                    continue
                warnings.append(f"{cand['code']} 板块({sector})仓位被截断至{allowed:.1%}")
                cand_ratio = allowed

            # 通过所有检查
            filtered.append({**cand, "ratio": cand_ratio})
            used_ratio += cand_ratio
            sector_exposure[sector] = sector_exposure.get(sector, 0) + cand_ratio

        return filtered, warnings

    # ==================== 向量化风控（新增） ====================

    # - 市场状态动态缩放
    def _apply_market_regime_scaling(
        self,
        target_ratio: np.ndarray,
        regime: np.ndarray,
        regime_scaling: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        根据 market_regime 列对 target_ratio 进行动态缩放。
        - 若 regime_scaling 为 None，直接返回原 target_ratio。
        - 若某 regime 不在 regime_scaling 中，保持原值。
        - 支持常见的 regime 枚举（示例：bull/bear/sideways）。
        """
        if regime_scaling is None:
            return target_ratio

        # 复制一份，避免原地改输入（调用侧会直接赋值回 df 的列）
        out = target_ratio.copy()

        # 遍历所有 regime 分段，使用布尔掩码进行向量化乘法
        for reg_label, scale_factor in regime_scaling.items():
            mask = regime == reg_label
            if not np.any(mask):
                continue

            # 限制 scale_factor 合理范围（负数会被 clamp 到 0）
            scale_factor_clamped = max(scale_factor, 0.0)
            out[mask] = out[mask] * scale_factor_clamped

        # 确保仍在 [0, 1] 范围（由于浮点误差可能略超边界）
        np.clip(out, 0.0, 1.0, out=out)
        return out

    # - 涨跌停不可交易过滤
    def _apply_limit_price_filter(
        self,
        df: pd.DataFrame,
        target_ratio: np.ndarray,
        signal_action: np.ndarray,
        # 控制开关（可外传参数；此处直接从 self.config 可见性不好，先写死）
        limit_up_col: str = "limit_up",
        limit_down_col: str = "limit_down",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        涨跌停日禁止交易（视为不可成交），适用于 A 股场景。
        - 如果 df 中没有 limit_up/limit_down 列，则跳过过滤。
        - 涨停日：禁止买入（只清零买入信号对应的目标仓位；不影响已持仓）。
        - 跌停日：禁止卖出（只清零卖出信号对应的目标仓位；不影响已持仓）。
        说明：
          - 买入/卖出由 signal_action 决定；对于已持仓，target_ratio 不为 0。
          - 这里仅把“不可成交”那天的“方向性目标调整”清零，后续引擎会自动保持当前仓位。
        """
        has_limit_up = limit_up_col in df.columns
        has_limit_down = limit_down_col in df.columns

        if not (has_limit_up or has_limit_down):
            return target_ratio, signal_action

        # 读取为 Numpy 数组加速（缺失列已在上面排除）
        if has_limit_up:
            limit_up_arr = df[limit_up_col].to_numpy(dtype=bool)
        else:
            limit_up_arr = np.zeros(len(df), dtype=bool)

        if has_limit_down:
            limit_down_arr = df[limit_down_col].to_numpy(dtype=bool)
        else:
            limit_down_arr = np.zeros(len(df), dtype=bool)

        # 涨停日：禁止买入（signal_action == 1 且 limit_up）
        mask_limit_up_block = (signal_action == 1) & limit_up_arr
        target_ratio[mask_limit_up_block] = 0.0
        signal_action[mask_limit_up_block] = 0

        # 跌停日：禁止卖出（signal_action == -1 且 limit_down）
        mask_limit_down_block = (signal_action == -1) & limit_down_arr
        target_ratio[mask_limit_down_block] = 0.0
        signal_action[mask_limit_down_block] = 0

        # 注意：若当日既涨停又跌停（极端异常），上面会把买卖都清零（保守策略）
        return target_ratio, signal_action

    # - 向量化风控主入口
    def apply_vectorized_risk_controls(
        self,
        df: pd.DataFrame,
        # 可选：市场状态标签列名（默认：market_regime）
        regime_col: str = "market_regime",
        # 可选：涨跌停标签列名（默认：limit_up / limit_down）
        limit_up_col: str = "limit_up",
        limit_down_col: str = "limit_down",
        # 可选：市场状态对应的仓位缩放系数（默认示例；可从外部传入）
        regime_scaling: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        向量化风控：在进入底层回测引擎前，对策略生成的“目标仓位序列”进行过滤/缩放。
        - 输入 df 必须包含：
            * target_position_ratio：建议仓位（0~1）
            * signal_action：1 买 / -1 卖 / 0 持有
        - 可选列（有则生效，无则跳过）：
            * market_regime：市场状态标签（例如：bull/bear/sideways/neutral 等）
            * limit_up/limit_down：当日是否涨停/跌停（布尔值或可转换为布尔）
        - 基础逻辑：
            1) 单股最大仓位上限（使用 RiskConfig.max_position_ratio）；
            2) 市场状态动态缩放（如熊市目标仓位乘以 0.3）；
            3) 涨跌停日不可交易过滤（禁止对应方向的订单）。
        - 所有操作尽量向量化（numpy/pandas），避免 Python for 循环。
        - 就地修改 df['target_position_ratio']、df['signal_action']（如需要），并返回 df 以便链式调用。

        Args:
            df: 策略输出的 DataFrame（包含 target_position_ratio、signal_action 等）
            regime_col: 市场状态列名（默认 "market_regime"）
            limit_up_col: 涨停标志列名（默认 "limit_up"）
            limit_down_col: 跌停标志列名（默认 "limit_down"）
            regime_scaling: 市场状态 -> 仓位缩放系数；若为 None，则不启用市场状态缩放

        Returns:
            pd.DataFrame: 经过风控调整后的 df（target_position_ratio 与 signal_action 可能被修改）
        """
        # ---- 前置校验 ----
        if not isinstance(df, pd.DataFrame):
            raise TypeError("apply_vectorized_risk_controls 要求输入为 pd.DataFrame")

        required_cols = ["target_position_ratio", "signal_action"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"DataFrame 缺少必要列: {missing_cols}")

        # 复制目标列（避免直接改原列，最后再统一写回；这里为了简化，直接就地修改并返回 df）
        # 若你想完全不可变，可先 df = df.copy() 再操作。
        target_ratio = df["target_position_ratio"].to_numpy(dtype=np.float64)
        signal_action = df["signal_action"].to_numpy(dtype=np.int8)

        # ---- 1) 单股最大仓位 cap（来自 RiskConfig.max_position_ratio） ----
        max_pos_ratio = getattr(self.config, "max_position_ratio", None)
        if max_pos_ratio is not None:
            # 确保在 [0, 1] 范围（配置异常保护）
            max_pos_ratio_clamped = min(max(max_pos_ratio, 0.0), 1.0)
            np.clip(target_ratio, 0.0, max_pos_ratio_clamped, out=target_ratio)

        # ---- 2) 市场状态动态缩放 ----
        if regime_scaling is not None:
            if regime_col in df.columns:
                regime_arr = df[regime_col].to_numpy(dtype=str)
                target_ratio = self._apply_market_regime_scaling(
                    target_ratio,
                    regime_arr,
                    regime_scaling=regime_scaling,
                )
            else:
                # 没有市场状态列，但 regime_scaling 不为 None 时只打一条日志提示
                import warnings
                warnings.warn(
                    f"启用了 regime_scaling，但 df 中缺少 '{regime_col}' 列，跳过市场状态缩放。",
                    RuntimeWarning,
                )

        # ---- 3) 涨跌停不可交易过滤 ----
        # 若 df 中有 limit_up/limit_down 列，则应用过滤；否则跳过。
        target_ratio, signal_action = self._apply_limit_price_filter(
            df,
            target_ratio,
            signal_action,
            limit_up_col=limit_up_col,
            limit_down_col=limit_down_col,
        )

        # ---- 4) 最终安全 clip：保证在 [0, 1] 区间，并清理极小误差 ----
        np.clip(target_ratio, 0.0, 1.0, out=target_ratio)
        # 把 signal_action 限制在 {-1, 0, 1}
        # 方法：先设 0/1/-1 的合法掩码，其余位置置为 0（保守）
        valid_action_mask = (signal_action == -1) | (signal_action == 0) | (signal_action == 1)
        signal_action[~valid_action_mask] = 0

        # ---- 5) 写回 df 列（返回同一份 df） ----
        df["target_position_ratio"] = target_ratio
        df["signal_action"] = signal_action

        return df
