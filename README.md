# AutoTrade for LLM Review

> **⚠️ 强烈免责声明**：本项目仅供**学习、研究以及大语言模型（LLM）代码审查的测试基底**使用。量化交易存在极高风险，本系统的回测结果不代表未来收益，**绝对不要直接将本代码连接至实盘交易账户进行自动下单**。因使用本项目造成的任何资金损失，开发者概不负责。

本项目是一个基于深度学习与经典量化策略的 A 股智能选股与仓位管理系统。整体架构按照“数据清洗 -> 特征工程 -> 模型训练 -> 滑点/成本回测 -> 实盘信号生成”的标准闭环打造，非常适合作为 LLM 进行复杂金融代码审查的基准项目。

---

## ✨ 核心特性

- **严谨的数据管线**：基于 `efinance`/`akshare` 下载 A 股日线数据，统一在 `data/indicators.py` 中计算纯量技术指标，并做严格的缺失值与异常值处理。
- **时序 Transformer 模型**：使用自定义的 Transformer 编码器进行趋势分类预测（`model/transformer.py`），训练过程集成了 EMA（指数移动平均）、SWA（随机权重平均）以及 TopK 集成策略。
- **防过拟合的回测引擎**：采用 **Walk-Forward（滚动前推）** 划分数据，并设置了 `gap=20` 天的隔离带，从根本上阻断“未来数据泄露”。
- **Optuna 多目标优化**：针对复合信号策略，使用 Optuna 同时优化夏普比率、最大回撤等多个维度，避免陷入单一收益指标的过拟合陷阱。
- **双层风控机制**：
  - **硬限制**（绝对不可违背）：单只股票最大持仓比例、行业集中度限制（`risk_manager.py`）。
  - **软目标**（用于回测过滤）：最大回撤容忍度、最小利润因子等。
- **实盘 Advisor 模式**：不直接对接券商 API，而是读取本地 `my_portfolio.json`（真实持仓），输出结合了模型预测、技术信号和风险敞口的“买卖建议+建议股数”，支持 `--dry-run` 调试。

---

## 🚀 快速开始

### 1. 环境准备



```bash```

1. 克隆项目
git clone <your-repo-url>
cd autotrade

2. 创建虚拟环境（推荐）
conda create -n autotrade python=3.10
conda activate autotrade

3. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas ta-lib efinance akshare optuna scikit-learn joblib
pip install tqdm matplotlib python-dotenv requests

ta-lib 如果 pip 安装失败，参考：https://github.com/ta-lib/ta-lib-python

## 旧模型迁移

如果你之前已经用 V2 训练过模型，**不需要重新训练**，直接复制文件到项目根目录：

你的旧项目目录/
├── model_weights.pth → 复制到 autotrade/
├── swa_model_weights.pth → 复制到 autotrade/
├── per_stock_scalers.pkl → 复制到 autotrade/
├── global_scaler.pkl → 复制到 autotarde/
├── checkpoints/ → 整个文件夹复制到 autotrade/
│ └── *.pth
└── stock_data_cleaned.feather → 复制到 autotrade/

### 2. 配置环境变量（⚠️ 极其重要）

**安全警告**：本项目已将 `.env` 从 Git 追踪中移除。请勿将包含真实 Webhook 或账户信息的 `.env` 文件提交到公共仓库。

在项目根目录创建 `.env` 文件，参考以下模板（修复了原代码中滑点配置复用佣金字段的歧义）：

```bash```

# ==================== 环境变量配置 ====================
# 代理设置
HTTP_PROXY=http://127.0.0.1:6789
HTTPS_PROXY=http://127.0.0.1:6789

# ==================== 模型配置 ====================
LOOKBACK_DAYS=120
BATCH_SIZE=256
EPOCHS=7
LEARNING_RATE=3e-5
WEIGHT_DECAY=0.05
NUM_HEADS=8
NUM_LAYERS=4
DIM_FEEDFORWARD=512
DROPOUT=0.4

# ==================== 风控配置 ====================
MAX_DRAWDOWN_LIMIT=-20.0
MIN_PROFIT_FACTOR=1.5
MAX_POSITION_RATIO=0.3
MIN_SHARPE_RATIO=0.5
MIN_WIN_RATE=40.0
MIN_TRADES=15
MAX_TRADES=120

# ==================== 回测配置 ====================
TRAIN_RATIO=0.7
TEST_RATIO=0.3
N_SPLITS=3
GAP_DAYS=20
N_OPTUNA_TRIALS=50
INITIAL_CAPITAL=100000

# ==================== 缓存配置 ====================
CACHE_DIR=./stock_cache
RESULT_DIR=./stock_cache/optimized_strategy_results
MARKET_CACHE_FILE=./stock_cache/market_data.pkl
STOCK_CACHE_FILE=./stock_cache/stocks_data.pkl

# ==================== 路径配置 ====================
MODEL_PATH=model_weights.pth
SWA_MODEL_PATH=swa_model_weights.pth
SCALER_PATH=per_stock_scalers.pkl
GLOBAL_SCALER_PATH=global_scaler.pkl
STRATEGY_FILE=optimized_strategies.json
PORTFOLIO_FILE=my_portfolio.json
TOPK_CHECKPOINT_DIR=checkpoints
STOCK_POOL_FILE=model/stock_pool.json
STOCK_DATA_FILE=stock_data_cleaned.feather

# ==================== 企业微信配置 ====================
WECHAT_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your- key
WECHAT_UPLOAD_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key=your- key

# ==================== 训练高级配置 ====================
ACCUMULATION_STEPS=4
EMA_DECAY=0.999
WARMUP_EPOCHS=3
PLATEAU_PATIENCE=2
PLATEAU_FACTOR=0.5
CYCLE_AMPLITUDE=0.2
CYCLE_LENGTH=5
TOPK_SAVE_COUNT=3
GRAD_CLIP_NORM=0.3
LABEL_SMOOTHING=0.1
TIME_DECAY_RATE=0.001

# ---- 交易成本 & 滑点 ----
COMMISSION_RATE=0.00025
MIN_COMMISSION=5.0
STAMP_DUTY_RATE=0.0005
TRANSFER_FEE_RATE=0.00001
BUY_SLIPPAGE_RATE=0.0015
SELL_SLIPPAGE_RATE=0.9985

# 调仓频率配置（weekly / biweekly）
REBALANCE_FREQ=weekly

# 调仓锚点星期几（0=周一 ... 6=周日）
REBALANCE_ANCHOR_WEEKDAY=0

### 3. 标准执行流程

按顺序执行以下脚本，完成从数据到建议的闭环：

Step 1: 下载数据 (会生成 .feather 缓存文件，已在 .gitignore 中忽略)
python run_data_download.py

Step 2: 训练模型 (生成 model_weights.pth 等)
python run_train.py

Step 3: 滚动回测与策略优化 (生成 optimized_strategies.json)
python run_backtest.py

Step 4: 基于最新模型生成全市场预测 (生成 stock_predictions.json)
python run_predict.py

Step 5: 运行实盘 Advisor (需先手动创建 my_portfolio.json)
python run_advisor.py --dry-run -v


---

## 📂 项目结构说明 (针对 LLM Review 优化)

本目录结构专门为代码审查工具做了高内聚、低耦合的设计：


```text```

├── config.py # 【核心】全局配置中心，基于 dataclass，从 .env 注入
├── run_*.py # 各模块的启动入口脚本
├── data/
│ ├── loader.py # 数据加载与合并
│ ├── indicators.py # 【唯一真理源】所有技术指标（MA, MACD, RSI等）的计算逻辑
│ ├── normalize.py # 特征归一化处理
│ └── cache.py # 数据缓存管理
├── model/
│ ├── transformer.py # PyTorch Transformer 网络结构定义
│ ├── trainer.py # 训练循环、EMA/SWA 逻辑、TopK 集成逻辑
│ └── predictor.py # 模型推理封装
├── backtest/
│ ├── engine.py # 回测引擎（处理滑点、手续费、订单撮合）
│ ├── evaluator.py # 绩效评估（夏普、回撤、胜率等指标计算）
│ └── optimizer.py # Optuna 超参/策略参数多目标寻优
├── strategies/
│ └── compound_signal.py # 复合信号策略（结合模型预测得分与技术指标阈值）
├── live/
│ ├── advisor.py # 实盘建议主逻辑（整合预测、信号、风控）
│ ├── signal_filter.py # 信号过滤（例如剔除 ST 股、停牌股、涨跌停股）
│ └── portfolio_risk.py # 持仓组合层面的实时风控计算
├── utils/
│ └── logger.py # 全局日志配置
├── my_portfolio.json # 【需手动创建】你的真实持仓 JSON 结构
├── .env.example # 环境变量模板 (已忽略真实 .env)
└── .gitignore # 忽略大文件、权重、IDE配置等


---

## 📄 开源协议

MIT License