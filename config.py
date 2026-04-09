# -*- coding: utf-8 -*-
"""
集中配置管理模块
使用 dataclass + .env 实现统一配置源，避免参数散落在各文件中
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

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
    val = os.getenv(key, "").lower()
    return val in ("true", "1", "yes")


@dataclass
class ModelConfig:
    """深度学习模型超参数"""
    lookback_days: int = 120
    batch_size: int = 256
    epochs: int = 7
    learning_rate: float = 3e-5
    weight_decay: float = 0.05
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.4
    num_classes: int = 4

    # 训练高级配置
    accumulation_steps: int = 4
    ema_decay: float = 0.999
    warmup_epochs: int = 3
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    cycle_amplitude: float = 0.2
    cycle_length: int = 5
    topk_save_count: int = 3
    grad_clip_norm: float = 0.3
    label_smoothing: float = 0.1
    time_decay_rate: float = 0.001

    # 推理配置
    mc_forward_train: int = 10     # 训练/实盘 MC Dropout 采样次数
    mc_forward_backtest: int = 3   # 回测时降低采样次数加速
    inference_batch_size: int = 64 # 批量推理 batch size

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            lookback_days=_env_int("LOOKBACK_DAYS", 120),
            batch_size=_env_int("BATCH_SIZE", 256),
            epochs=_env_int("EPOCHS", 7),
            learning_rate=_env_float("LEARNING_RATE", 3e-5),
            weight_decay=_env_float("WEIGHT_DECAY", 0.05),
            num_heads=_env_int("NUM_HEADS", 8),
            num_layers=_env_int("NUM_LAYERS", 4),
            dim_feedforward=_env_int("DIM_FEEDFORWARD", 512),
            dropout=_env_float("DROPOUT", 0.4),
            accumulation_steps=_env_int("ACCUMULATION_STEPS", 4),
            ema_decay=_env_float("EMA_DECAY", 0.999),
            warmup_epochs=_env_int("WARMUP_EPOCHS", 3),
            plateau_patience=_env_int("PLATEAU_PATIENCE", 2),
            plateau_factor=_env_float("PLATEAU_FACTOR", 0.5),
            cycle_amplitude=_env_float("CYCLE_AMPLITUDE", 0.2),
            cycle_length=_env_int("CYCLE_LENGTH", 5),
            topk_save_count=_env_int("TOPK_SAVE_COUNT", 3),
            grad_clip_norm=_env_float("GRAD_CLIP_NORM", 0.3),
            label_smoothing=_env_float("LABEL_SMOOTHING", 0.1),
            time_decay_rate=_env_float("TIME_DECAY_RATE", 0.001),
            mc_forward_train=_env_int("MC_FORWARD_TRAIN", 10),
            mc_forward_backtest=_env_int("MC_FORWARD_BACKTEST", 3),
            inference_batch_size=_env_int("INFERENCE_BATCH_SIZE", 64),
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
    max_correlation: float = 0.7      # 组合内持仓最大相关系数
    max_sector_ratio: float = 0.40    # 板块集中度上限

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
    n_optuna_trials: int = 150
    initial_capital: float = 100000.0
    expanding_window: bool = True   # 使用扩展窗口而非固定窗口

    @classmethod
    def from_env(cls) -> "BacktestConfig":
        return cls(
            train_ratio=_env_float("TRAIN_RATIO", 0.7),
            val_ratio=_env_float("VAL_RATIO", 0.15),
            n_splits=_env_int("N_SPLITS", 5),
            gap_days=_env_int("GAP_DAYS", 20),
            n_optuna_trials=_env_int("N_OPTUNA_TRIALS", 150),
            initial_capital=_env_float("INITIAL_CAPITAL", 100000.0),
            expanding_window=_env_bool("EXPANDING_WINDOW", True),
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
    strategy_file: str = "optimized_strategies.json"
    portfolio_file: str = "my_portfolio.json"
    topk_checkpoint_dir: str = "checkpoints"
    stock_pool_file: str = "model/stock_pool.json"
    stock_data_file: str = "stock_data_cleaned.feather"
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
            strategy_file=_env("STRATEGY_FILE", "optimized_strategies.json"),
            portfolio_file=_env("PORTFOLIO_FILE", "my_portfolio.json"),
            topk_checkpoint_dir=_env("TOPK_CHECKPOINT_DIR", "checkpoints"),
            stock_pool_file=_env("STOCK_POOL_FILE", "model/stock_pool.json"),
            stock_data_file=_env("STOCK_DATA_FILE", "stock_data_cleaned.feather"),
            wechat_webhook=_env("WECHAT_WEBHOOK", ""),
            wechat_upload_url=_env("WECHAT_UPLOAD_URL", ""),
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
        return cls(
            commission_rate=_env_float("COMMISSION_RATE", 0.00025),
            min_commission=_env_float("MIN_COMMISSION", 5.0),
            stamp_duty_rate=_env_float("STAMP_DUTY_RATE", 0.0005),
            transfer_fee_rate=_env_float("TRANSFER_FEE_RATE", 0.00001),
        )


@dataclass
class SlippageConfig:
    """滑点配置"""
    buy_slippage_rate: float = 0.001
    sell_slippage_rate: float = 0.001

    @classmethod
    def from_env(cls) -> "SlippageConfig":
        return cls(
            buy_slippage_rate=_env_float("BUY_SLIPPAGE_RATE", 0.001),
            sell_slippage_rate=_env_float("SELL_SLIPPAGE_RATE", 0.001),
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
    """市场状态判断配置"""
    ma_period: int = 20              # MA 周期
    volatility_period: int = 60      # 波动率周期
    high_vol_threshold: float = 1.5  # 高波动阈值（相对中位数倍数）
    trend_strength_period: int = 20  # 趋势强度计算周期
    weak_regime_buy_cap: float = 0.3 # 弱势市场中允许的买入仓位上限
    bear_max_position: float = 0.15  # 熊市单只最大仓位

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
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            model=ModelConfig.from_env(),
            risk=RiskConfig.from_env(),
            backtest=BacktestConfig.from_env(),
            paths=PathConfig.from_env(),
            commission=CommissionConfig.from_env(),
            slippage=SlippageConfig.from_env(),
            scheduler=SchedulerConfig.from_env(),
            regime=RegimeConfig.from_env(),
        )

    def ensure_dirs(self):
        """确保所有需要的目录存在"""
        for d in [self.paths.cache_dir, self.paths.result_dir, self.paths.topk_checkpoint_dir]:
            os.makedirs(d, exist_ok=True)


# ==================== 股票代码配置 ====================
STOCK_CODES = {
    # 消费
    '贵州茅台': '600519',
    '美的集团': '000333',

    # 新能源
    '国轩高科': '002074',
    '隆基绿能': '601012',
    '赛力斯':   '601127',
    '比亚迪':   '002594',

    # AI / 科技
    '科大讯飞': '002230',
    '海康威视': '002415',

    # 金融
    '中信证券': '600030',

    # 通信/算力
    '光迅科技': '002281',

    # 医药
    '恒瑞医药': '600276',

    # 黄金 / 有色 / 资源
    '山东黄金': '600547',
    '紫金矿业': '601899',

    # 电力/公用事业
    '长江电力': '600900',
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
