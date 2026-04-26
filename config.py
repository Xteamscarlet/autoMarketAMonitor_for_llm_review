# -*- coding: utf-8 -*-
"""
集中配置管理模块
使用 dataclass + .env 实现统一配置源，避免参数散落在各文件�?
"""
import os
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 加载 .env 文件
load_dotenv()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in ("true", "1", "yes")


_MAX_SLIPPAGE_RATE = 0.05
_MAX_COMMISSION_RATE = 0.02
_MAX_STAMP_DUTY_RATE = 0.02
_MAX_TRANSFER_FEE_RATE = 0.01
_MAX_MIN_COMMISSION = 200.0


def _clip_with_warning(name: str, value: float, lower: float, upper: float) -> float:
    clipped = min(max(value, lower), upper)
    if clipped != value:
        logger.warning("[%s] clipped from %.6f to %.6f", name, value, clipped)
    return clipped


def _safe_finite_float(value: float, default: float, name: str) -> float:
    try:
        v = float(value)
    except (ValueError, TypeError):
        logger.warning("[%s] invalid value %r, fallback to default %.6f", name, value, default)
        return default
    if not math.isfinite(v):
        logger.warning("[%s] non-finite value %r, fallback to default %.6f", name, value, default)
        return default
    return v


def _normalize_rate_from_env(key: str, default: float, upper: float) -> float:
    """Normalize rate configs with hard constraints.

    Supports common percent-style input:
    - 0.001 means 0.1%
    - 0.1 means 10%
    - 10 means 10% (converted to 0.10)
    """
    raw = _safe_finite_float(_env_float(key, default), default, key)
    if raw >= 1.0:
        if raw <= 100.0:
            raw = raw / 100.0
            logger.warning("[%s] interpreted as percent input, normalized to %.6f", key, raw)
        else:
            logger.warning("[%s] too large %.6f, fallback to default %.6f", key, raw, default)
            raw = default

    raw = _clip_with_warning(key, raw, 0.0, upper)
    if raw < 0:
        raw = 0.0
    if not math.isfinite(raw):
        raw = default
    return raw


def _normalize_slippage_rate(value: float) -> float:
    """Normalize common slippage misconfigurations into a safe small rate."""
    v = _safe_finite_float(value, 0.001, "SLIPPAGE_RATE")
    original = v
    if v < 0:
        v = 0.0
    elif v >= 1.0:
        if v <= 100.0:
            v = v / 100.0
        else:
            v = 0.001

    # Users sometimes fill price multipliers (e.g. 0.9985) instead of slippage (0.0015).
    if v > 0.2:
        normalized = 1.0 - v
        if 0.0 <= normalized <= 0.2:
            v = normalized

    v = _clip_with_warning("SLIPPAGE_RATE", v, 0.0, _MAX_SLIPPAGE_RATE)
    if v != original:
        logger.warning("[SLIPPAGE_RATE] normalized from %.6f to %.6f", original, v)
    return v


@dataclass
class ModelConfig:
    """深度学习模型超参数"""
    lookback_days: int = 120
    batch_size: int = 256
    epochs: int = 30                 # ★ 修复1: 7 → 30，避免训练不足
    learning_rate: float = 3e-5
    weight_decay: float = 0.10       # ★ 修复1: 0.05 → 0.10，加强正则
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.50            # ★ 修复1: 0.45 → 0.50，加强正则
    num_classes: int = 4

    # 训练高级配置
    accumulation_steps: int = 4
    ema_decay: float = 0.9995        # ★ 修复1: 0.999 → 0.9995，更平滑
    warmup_epochs: int = 2           # ★ 修复1: 3 → 2，减少 warmup 占比
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    cycle_amplitude: float = 0.2
    cycle_length: int = 5
    topk_save_count: int = 3
    grad_clip_norm: float = 0.3
    label_smoothing: float = 0.1
    time_decay_rate: float = 0.001
    # ★ 修复1: 早停参数迁入 config 集中管理
    early_stop_patience: int = 6     # 7 → 30 epoch 配 6 patience
    early_stop_min_delta: float = 0.0005
    convergence_patience: int = 5
    convergence_min_delta: float =0.003
    swa_start_ratio: float = 0.5     # ★ 修复1: 0.6 → 0.5，让 SWA 至少累积一半 epoch
    # ★ 修复5: 收益率标准化系数（rets 除以该值后送入 SmoothL1Loss）
    ret_target_scale: float = 0.05

    # 推理配置
    mc_forward_train: int = 10     # 训练/实盘 MC Dropout 采样次数
    mc_forward_backtest: int = 3   # 回测时降低采样次数加�?
    inference_batch_size: int = 64 # 批量推理 batch size
    focal_gamma: float = 1.5
    cls_loss_weight_initial: float = 1.3
    cls_loss_weight_final: float = 1.0
    ret_loss_weight_initial: float = 0.05
    ret_loss_weight_final: float = 0.20
    cls_time_weight_power: float = 0.5
    ret_time_weight_power: float = 1.0
    use_balanced_sampler: bool = True
    sampler_class_power: float = 1.0
    sampler_time_power: float = 0.3
    sampler_min_weight: float = 0.05
    sampler_max_weight: float = 20.0
    sampler_replacement: bool = True
    head_diag_enabled: bool = True
    head_diag_interval: int = 1
    head_diag_max_batches: int = 32
    head_diag_cls_dominance_warn: float = 0.85
    head_diag_prob_std_warn: float = 0.02
    head_diag_ret_std_warn: float = 0.003

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            lookback_days=_env_int("LOOKBACK_DAYS", 120),
            batch_size=_env_int("BATCH_SIZE", 256),
            epochs=_env_int("EPOCHS", 30),
            learning_rate=_env_float("LEARNING_RATE", 3e-5),
            weight_decay=_env_float("WEIGHT_DECAY", 0.10),
            num_heads=_env_int("NUM_HEADS", 8),
            num_layers=_env_int("NUM_LAYERS", 4),
            dim_feedforward=_env_int("DIM_FEEDFORWARD", 512),
            dropout=_env_float("DROPOUT", 0.50),
            accumulation_steps=_env_int("ACCUMULATION_STEPS", 4),
            ema_decay=_env_float("EMA_DECAY", 0.9995),
            warmup_epochs=_env_int("WARMUP_EPOCHS", 2),
            plateau_patience=_env_int("PLATEAU_PATIENCE", 2),
            plateau_factor=_env_float("PLATEAU_FACTOR", 0.5),
            cycle_amplitude=_env_float("CYCLE_AMPLITUDE", 0.2),
            cycle_length=_env_int("CYCLE_LENGTH", 5),
            topk_save_count=_env_int("TOPK_SAVE_COUNT", 3),
            grad_clip_norm=_env_float("GRAD_CLIP_NORM", 0.3),
            label_smoothing=_env_float("LABEL_SMOOTHING", 0.1),
            time_decay_rate=_env_float("TIME_DECAY_RATE", 0.001),
            early_stop_patience=_env_int("EARLY_STOP_PATIENCE", 6),
            early_stop_min_delta=_env_float("EARLY_STOP_MIN_DELTA", 0.0005),
            convergence_patience=_env_int("CONVERGENCE_PATIENCE", 5),
            convergence_min_delta=_env_float("CONVERGENCE_MIN_DELTA", 0.003),
            swa_start_ratio=_env_float("SWA_START_RATIO", 0.5),
            ret_target_scale=_env_float("RET_TARGET_SCALE", 0.05),
            mc_forward_train=_env_int("MC_FORWARD_TRAIN", 10),
            mc_forward_backtest=_env_int("MC_FORWARD_BACKTEST", 3),
            inference_batch_size=_env_int("INFERENCE_BATCH_SIZE", 64),
            focal_gamma=_env_float("FOCAL_GAMMA", 1.5),
            cls_loss_weight_initial=_env_float("CLS_LOSS_WEIGHT_INITIAL", 1.3),
            cls_loss_weight_final=_env_float("CLS_LOSS_WEIGHT_FINAL", 1.0),
            ret_loss_weight_initial=_env_float("RET_LOSS_WEIGHT_INITIAL", 0.05),
            ret_loss_weight_final=_env_float("RET_LOSS_WEIGHT_FINAL", 0.20),
            cls_time_weight_power=_env_float("CLS_TIME_WEIGHT_POWER", 0.5),
            ret_time_weight_power=_env_float("RET_TIME_WEIGHT_POWER", 1.0),
            use_balanced_sampler=_env_bool("USE_BALANCED_SAMPLER", True),
            sampler_class_power=_env_float("SAMPLER_CLASS_POWER", 1.0),
            sampler_time_power=_env_float("SAMPLER_TIME_POWER", 0.3),
            sampler_min_weight=_env_float("SAMPLER_MIN_WEIGHT", 0.05),
            sampler_max_weight=_env_float("SAMPLER_MAX_WEIGHT", 20.0),
            sampler_replacement=_env_bool("SAMPLER_REPLACEMENT", True),
            head_diag_enabled=_env_bool("HEAD_DIAG_ENABLED", True),
            head_diag_interval=max(1, _env_int("HEAD_DIAG_INTERVAL", 1)),
            head_diag_max_batches=max(1, _env_int("HEAD_DIAG_MAX_BATCHES", 32)),
            head_diag_cls_dominance_warn=_env_float("HEAD_DIAG_CLS_DOMINANCE_WARN", 0.85),
            head_diag_prob_std_warn=_env_float("HEAD_DIAG_PROB_STD_WARN", 0.02),
            head_diag_ret_std_warn=_env_float("HEAD_DIAG_RET_STD_WARN", 0.003),
        )


@dataclass
class RiskConfig:
    """风控参数"""
    max_drawdown_limit: float = -20.0
    min_profit_factor: float = 1.5
    max_position_ratio: float = 0.3
    min_sharpe_ratio: float = 0.5
    min_win_rate: float = 40.0
    min_trades: int = 15
    max_trades: int = 120
    max_daily_loss_ratio: float = 0.03
    max_correlation: float = 0.7      # 组合内持仓最大相关系�?
    max_sector_ratio: float = 0.40    # 板块集中度上�?

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            max_drawdown_limit=_env_float("MAX_DRAWDOWN_LIMIT", -20.0),
            min_profit_factor=_env_float("MIN_PROFIT_FACTOR", 1.5),
            max_position_ratio=_env_float("MAX_POSITION_RATIO", 0.3),
            min_sharpe_ratio=_env_float("MIN_SHARPE_RATIO", 0.5),
            min_win_rate=_env_float("MIN_WIN_RATE", 40.0),
            min_trades=_env_int("MIN_TRADES", 15),
            max_trades=_env_int("MAX_TRADES", 120),
            max_daily_loss_ratio=_env_float("MAX_DAILY_LOSS_RATIO", 0.03),
            max_correlation=_env_float("MAX_CORRELATION", 0.7),
            max_sector_ratio=_env_float("MAX_SECTOR_RATIO", 0.40),
        )


@dataclass
class BacktestConfig:
    """回测参数"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    n_splits: int = 5
    gap_days: int = 20
    n_optuna_trials: int = 50
    initial_capital: float = 100000.0
    expanding_window: bool = True   # 使用扩展窗口而非固定窗口
    val_early_stop_threshold: float = -5.0
    enable_val_validation: bool = True
    val_selection_metric: str = "sortino"
    precompute_transformer_cache: bool = True

    @classmethod
    def from_env(cls) -> "BacktestConfig":
        return cls(
            train_ratio=_env_float("TRAIN_RATIO", 0.7),
            val_ratio=_env_float("VAL_RATIO", 0.15),
            n_splits=_env_int("N_SPLITS", 5),
            gap_days=_env_int("GAP_DAYS", 20),
            n_optuna_trials=_env_int("N_OPTUNA_TRIALS", 50),
            initial_capital=_env_float("INITIAL_CAPITAL", 100000.0),
            expanding_window=_env_bool("EXPANDING_WINDOW", True),
            val_early_stop_threshold=_env_float("VAL_EARLY_STOP_THRESHOLD", -5.0),
            enable_val_validation=_env_bool("ENABLE_VAL_VALIDATION", True),
            val_selection_metric=_env("VAL_SELECTION_METRIC", "sortino"),
            precompute_transformer_cache=_env_bool("PRECOMPUTE_TRANSFORMER_CACHE", True),
        )


@dataclass
class PathConfig:
    """文件路径配置"""
    cache_dir: str = "./stock_cache"
    result_dir: str = "./stock_cache/optimized_strategy_results"
    market_cache_file: str = "./stock_cache/market_data.pkl"
    stock_cache_file: str = "./stock_cache/stocks_data.pkl"
    model_path: str = "model_weights.pth"
    swa_model_path: str = "swa_model_weights.pth"
    scaler_path: str = "per_stock_scalers.pkl"
    global_scaler_path: str = "global_scaler.pkl"
    model_metadata_path: str = "model_training_metadata.json"
    label_thresholds_path: str = "label_thresholds.pkl"  # ★ 修复6: 保存训练集分位数阈值
    strategy_file: str = "optimized_strategies.json"
    portfolio_file: str = "my_portfolio.json"
    topk_checkpoint_dir: str = "checkpoints"
    stock_pool_file: str = "model/stock_pool.json"
    stock_data_file: str = "stock_data_cleaned.feather"
    fundamental_cache_dir: str = "./stock_cache/fundamentals"
    ai_analysis_cache_dir: str = "./stock_cache/ai_analysis"
    wechat_webhook: str = ""
    wechat_upload_url: str = ""

    @classmethod
    def from_env(cls) -> "PathConfig":
        cache_dir = _env("CACHE_DIR", "./stock_cache")
        return cls(
            cache_dir=cache_dir,
            result_dir=_env("RESULT_DIR", f"{cache_dir}/optimized_strategy_results"),
            market_cache_file=_env("MARKET_CACHE_FILE", f"{cache_dir}/market_data.pkl"),
            stock_cache_file=_env("STOCK_CACHE_FILE", f"{cache_dir}/stocks_data.pkl"),
            model_path=_env("MODEL_PATH", "model_weights.pth"),
            swa_model_path=_env("SWA_MODEL_PATH", "swa_model_weights.pth"),
            scaler_path=_env("SCALER_PATH", "per_stock_scalers.pkl"),
            global_scaler_path=_env("GLOBAL_SCALER_PATH", "global_scaler.pkl"),
            model_metadata_path=_env("MODEL_METADATA_PATH", "model_training_metadata.json"),
            label_thresholds_path=_env("LABEL_THRESHOLDS_PATH", "label_thresholds.pkl"),
            strategy_file=_env("STRATEGY_FILE", "optimized_strategies.json"),
            portfolio_file=_env("PORTFOLIO_FILE", "my_portfolio.json"),
            topk_checkpoint_dir=_env("TOPK_CHECKPOINT_DIR", "checkpoints"),
            stock_pool_file=_env("STOCK_POOL_FILE", "model/stock_pool.json"),
            stock_data_file=_env("STOCK_DATA_FILE", "stock_data_cleaned.feather"),
            fundamental_cache_dir=_env("FUNDAMENTAL_CACHE_DIR", f"{cache_dir}/fundamentals"),
            ai_analysis_cache_dir=_env("AI_ANALYSIS_CACHE_DIR", f"{cache_dir}/ai_analysis"),
            wechat_webhook=_env("WECHAT_WEBHOOK", ""),
            wechat_upload_url=_env("WECHAT_UPLOAD_URL", ""),
        )


@dataclass
class CacheConfig:
    """缓存校验与清理配置"""
    auto_delete_invalid_cache: bool = False
    strict_freshness_check: bool = True

    @classmethod
    def from_env(cls) -> "CacheConfig":
        return cls(
            auto_delete_invalid_cache=_env_bool("CACHE_AUTO_DELETE_INVALID", False),
            strict_freshness_check=_env_bool("CACHE_STRICT_FRESHNESS_CHECK", True),
        )


@dataclass
class LLMConfig:
    """LLM / OpenAI-compatible 接口配置"""
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_seconds: int = 45
    max_prompt_chars: int = 12000
    disable_proxy: bool = False
    thinking_enabled: bool = False
    thinking_type: str = "enabled"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = _env("OPENAI_API_KEY", "")
        enabled = _env_bool("LLM_ENABLED", bool(api_key)) or bool(api_key)
        return cls(
            enabled=enabled,
            api_key=api_key,
            base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 45),
            max_prompt_chars=_env_int("LLM_MAX_PROMPT_CHARS", 12000),
            disable_proxy=_env_bool("LLM_DISABLE_PROXY", False),
            thinking_enabled=_env_bool("LLM_THINKING_ENABLED", False),
            thinking_type=_env("LLM_THINKING_TYPE", "enabled"),
        )


@dataclass
class AnalysisConfig:
    """财报 / AI 技术分析增强层配置"""
    enable_fundamental_agent: bool = True
    enable_technical_agent: bool = True
    blend_fundamental_score: bool = True
    blend_technical_score: bool = True
    fundamental_weight: float = 0.20
    technical_weight: float = 0.15
    min_fundamental_score: float = 0.35
    max_recent_days_for_technical: int = 120

    @classmethod
    def from_env(cls) -> "AnalysisConfig":
        return cls(
            enable_fundamental_agent=_env_bool("ENABLE_FUNDAMENTAL_AGENT", True),
            enable_technical_agent=_env_bool("ENABLE_TECHNICAL_AGENT", True),
            blend_fundamental_score=_env_bool("BLEND_FUNDAMENTAL_SCORE", True),
            blend_technical_score=_env_bool("BLEND_TECHNICAL_SCORE", True),
            fundamental_weight=_env_float("FUNDAMENTAL_WEIGHT", 0.20),
            technical_weight=_env_float("TECHNICAL_WEIGHT", 0.15),
            min_fundamental_score=_env_float("MIN_FUNDAMENTAL_SCORE", 0.35),
            max_recent_days_for_technical=_env_int("MAX_RECENT_DAYS_FOR_TECHNICAL", 120),
        )


@dataclass
class CommissionConfig:
    """交易佣金配置"""
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001

    @classmethod
    def from_env(cls) -> "CommissionConfig":
        min_commission = _safe_finite_float(_env_float("MIN_COMMISSION", 5.0), 5.0, "MIN_COMMISSION")
        min_commission = _clip_with_warning("MIN_COMMISSION", min_commission, 0.0, _MAX_MIN_COMMISSION)
        return cls(
            commission_rate=_normalize_rate_from_env(
                "COMMISSION_RATE", 0.00025, _MAX_COMMISSION_RATE
            ),
            min_commission=min_commission,
            stamp_duty_rate=_normalize_rate_from_env(
                "STAMP_DUTY_RATE", 0.0005, _MAX_STAMP_DUTY_RATE
            ),
            transfer_fee_rate=_normalize_rate_from_env(
                "TRANSFER_FEE_RATE", 0.00001, _MAX_TRANSFER_FEE_RATE
            ),
        )


@dataclass
class SlippageConfig:
    """滑点配置"""
    buy_slippage_rate: float = 0.001
    sell_slippage_rate: float = 0.001

    @classmethod
    def from_env(cls) -> "SlippageConfig":
        buy_slippage_rate = _normalize_slippage_rate(_env_float("BUY_SLIPPAGE_RATE", 0.001))
        sell_slippage_rate = _normalize_slippage_rate(_env_float("SELL_SLIPPAGE_RATE", 0.001))
        return cls(
            buy_slippage_rate=buy_slippage_rate,
            sell_slippage_rate=sell_slippage_rate,
        )


from enum import Enum

class RebalanceFreq(str, Enum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"


def _env_rebalance_freq(key: str, default: str = "weekly") -> RebalanceFreq:
    val = _env(key, default).lower()
    mapping = {
        "weekly": RebalanceFreq.WEEKLY,
        "biweekly": RebalanceFreq.BIWEEKLY,
        "week": RebalanceFreq.WEEKLY,
        "biweek": RebalanceFreq.BIWEEKLY,
        "双周": RebalanceFreq.BIWEEKLY,
        "两周": RebalanceFreq.BIWEEKLY,
        "周": RebalanceFreq.WEEKLY,
        "每周": RebalanceFreq.WEEKLY,
    }
    return mapping.get(val, RebalanceFreq(default))


@dataclass
class SchedulerConfig:
    """调度与频率配置"""
    rebalance_freq: RebalanceFreq = RebalanceFreq.WEEKLY
    rebalance_anchor_weekday: int = 0
    rebalance_anchor_date: str = ""

    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        return cls(
            rebalance_freq=_env_rebalance_freq("REBALANCE_FREQ", "weekly"),
            rebalance_anchor_weekday=_env_int("REBALANCE_ANCHOR_WEEKDAY", 0),
            rebalance_anchor_date=_env("REBALANCE_ANCHOR_DATE", ""),
        )


@dataclass
class RegimeConfig:
    """市场状态判断配"""
    ma_period: int = 20              # MA 周期
    volatility_period: int = 60      # 波动率周�?
    high_vol_threshold: float = 1.5  # 高波动阈值（相对中位数倍数�?
    trend_strength_period: int = 20  # 趋势强度计算周期
    weak_regime_buy_cap: float = 0.3 # 弱势市场中允许的买入仓位上限
    bear_max_position: float = 0.15  # 熊市单只最大仓�?

    @classmethod
    def from_env(cls) -> "RegimeConfig":
        return cls(
            ma_period=_env_int("REGIME_MA_PERIOD", 20),
            volatility_period=_env_int("REGIME_VOLATILITY_PERIOD", 60),
            high_vol_threshold=_env_float("REGIME_HIGH_VOL_THRESHOLD", 1.5),
            trend_strength_period=_env_int("REGIME_TREND_STRENGTH_PERIOD", 20),
            weak_regime_buy_cap=_env_float("REGIME_WEAK_BUY_CAP", 0.3),
            bear_max_position=_env_float("REGIME_BEAR_MAX_POSITION", 0.15),
        )


@dataclass
class AppConfig:
    """全局配置聚合"""
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            model=ModelConfig.from_env(),
            risk=RiskConfig.from_env(),
            backtest=BacktestConfig.from_env(),
            paths=PathConfig.from_env(),
            cache=CacheConfig.from_env(),
            commission=CommissionConfig.from_env(),
            slippage=SlippageConfig.from_env(),
            scheduler=SchedulerConfig.from_env(),
            regime=RegimeConfig.from_env(),
            llm=LLMConfig.from_env(),
            analysis=AnalysisConfig.from_env(),
        )

    def ensure_dirs(self):
        """确保所有需要的目录存在"""
        for d in [
            self.paths.cache_dir,
            self.paths.result_dir,
            self.paths.topk_checkpoint_dir,
            self.paths.fundamental_cache_dir,
            self.paths.ai_analysis_cache_dir,
        ]:
            os.makedirs(d, exist_ok=True)


# ==================== 股票代码配置 ====================
STOCK_CODES = {
    # 消费
    '贵州茅台': '600519',
    '美的集团': '000333',

    # 新能源
    '国轩高科': '002074',     # 动力电池（替代宁德时代）
    '隆基绿能': '601012',     # 光伏组件龙头
    '赛力斯':   '601127',     # 新能源车（华为合作，弹性标的）
    '比亚迪':   '002594',     # 新能源车整车龙头（与赛力斯互补）

    # AI / 科技
    '科大讯飞': '002230',
    '海康威视': '002415',

    # 金融（替代东方财富）
    '中信证券': '600030',     # 券商龙头

    # 通信/算力（替代中际旭创）
    '光迅科技': '002281',     # 光模块/光通信龙头

    # 医药（替代迈瑞医疗）
    '恒瑞医药': '600276',     # 创新药龙头

    # 黄金 / 有色 / 资源
    '山东黄金': '600547',
    '紫金矿业': '601899',     # 铜+金资源龙头

    # 电力/公用事业（防御）
    '长江电力': '600900',     # 水电龙头
    '东方电气': '600875',

    # 电子/消费电子
    '歌尔股份': '002241',
    '东山精密': '002384',

    # 军工
    '中航沈飞': '600760',

    # 农业
    '北大荒':   '600598',

    # 建材/新材料
    '北新建材': '000786',
}

# 板块分类（用于组合风控的板块集中度检查）
SECTOR_MAP = {
    '600519': '消费', '000333': '消费',
    '002074': '新能源', '601012': '新能源', '601127': '新能源', '002594': '新能源',
    '002230': '科技', '002415': '科技', '002281': '科技',
    '600030': '金融',
    '600276': '医药',
    '600547': '资源', '601899': '资源',
    '600900': '公用事业', '600875': '公用事业',
    '002241': '电子', '002384': '电子',
    '600760': '军工',
    '600598': '农业',
    '000786': '建材',
}

# 全局配置单例（延迟初始化）
_settings: Optional[AppConfig] = None


def get_settings() -> AppConfig:
    """获取全局配置单例"""
    global _settings
    if _settings is None:
        _settings = AppConfig.from_env()
        _settings.ensure_dirs()
    return _settings
