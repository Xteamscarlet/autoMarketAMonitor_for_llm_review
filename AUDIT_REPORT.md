# 项目全面审查报告

## 🔴 严重 Bug（会导致崩溃或结果错误）

### B1. engine_no_transformer.py — CommissionConfig.from_env() 在循环内重复调用
**文件**: `backtest/engine_no_transformer.py` 第 35-40 行
```python
def calculate_transaction_cost(...):
    _commission_cfg = CommissionConfig.from_env()  # ❌ 每次调用都重新读取环境变量
```
**影响**: 回测每个交易日都会调用，严重性能问题
**修复**: 模块级缓存（engine.py 已修复，但 engine_no_transformer.py 和 engine_no_transformer_new.py 未修复）

### B2. engine_no_transformer.py — `position` 变量混用
**文件**: `backtest/engine_no_transformer.py` 第 175-195 行
```python
position = 0  # 初始化为 0
# ...
if position > 0:  # ❌ 用 position > 0 判断是否持仓，但下面赋值了 shares
    ...
    position = shares  # ❌ position 变成了股数（如 1000）
# ...
if position == 0:  # ❌ 空仓判断永远不会为 True，因为卖出时设 position = 0
```
**影响**: 卖出后 position 被重置为 0，空仓判断用 `position == 0`，但买入时设 `position = shares`，逻辑混乱但碰巧能跑（因为卖出时确实设了 0）
**严重性**: 中等，逻辑不清晰但碰巧结果正确

### B3. engine_no_transformer_new.py — `get_settings()` 递归调用自身
**文件**: `backtest/engine_no_transformer_new.py` 末尾
```python
def get_settings():
    from config import get_settings  # ❌ 函数名遮蔽了导入的函数
    return get_settings()            # ❌ 递归调用自身 → RecursionError
```
**影响**: 任何代码调用 `engine_no_transformer_new.get_settings()` 都会 RecursionError
**实际触发概率**: 低（只在 calculate_transaction_cost 内部使用，但该函数内的 get_settings() 调用是模块级的，指向这个错误函数）

### B4. engine_no_transformer_new.py — `position` 变量语义混用
**文件**: `backtest/engine_no_transformer_new.py` 第 130-170 行
```python
position = 0.0  # 用 0.0 表示空仓
# 买入时
position = 1.0  # ❌ 用 1.0 表示"满仓"，但实际上应该是持仓股数
# 卖出计算
shares = int(position * 100)  # ❌ 假设 position=1.0 表示 100 股，但这不是 ATR 动态仓位
```
**影响**: 所有交易都是固定 100 股，完全忽略了 ATR 动态仓位逻辑
**严重性**: 高 — 回测结果与现实严重不符

### B5. evaluator.py — 重复 import 和函数定义
**文件**: `backtest/evaluator.py` 第 9-11 行
```python
import numpy as np        # 第 7 行已 import
import pandas as pd       # 第 8 行已 import
from typing import Dict, Optional  # 第 6 行已 import
```
以及 `calculate_comprehensive_stats` 函数在同一个文件中被定义了两次（旧版 + 新版），旧版函数在文件前面，新版在后面覆盖了。虽然 Python 允许这样，但代码非常混乱。

### B6. compound_signal.py — 弱势市场一刀切阻断
**文件**: `strategies/compound_signal.py` 第 80 行
```python
if regime == "weak":
    return {"action": "hold", ...}  # ❌ 弱势市场完全阻断买入
```
**影响**: 与 engine.py 的改进逻辑不一致，strategy 层仍然一刀切

### B7. engine_no_transformer.py — 卖出时 T+1 限制缺失
**文件**: `backtest/engine_no_transformer.py`
买入后如果 `date <= buy_date` 不会跳过卖出逻辑，违反 A 股 T+1 规则
engine.py 有此检查，但 engine_no_transformer.py 没有

### B8. visualizer.py — regime 中文映射不完整
**文件**: `backtest/visualizer.py` 第 99 行
```python
regime_cn = {'strong': '强势', 'weak': '弱势', 'neutral': '震荡'}
```
缺少新增的 `strong_bull` 和 `bear` 映射

### B9. run_backtest_no_transformer.py — optimize_strategy_no_transformer 引用未定义
**文件**: `run_backtest_no_transformer.py` 第 128 行
```python
best_params_map, best_weights = optimize_strategy_no_transformer(
    train_df, factor_cols, settings.backtest.n_optuna_trials
)
```
但 `optimize_strategy_no_transformer` 函数定义在同一个文件中（第 174 行），而非从模块导入。不过该函数内部引用了 `full_stats` 变量（第 206 行），该变量在此作用域未定义：
```python
mdd = full_stats.get('max_drawdown', 0) if (full_stats := calculate_comprehensive_stats(trades_df)) else 0
```
这里用了海象运算符，Python 3.8+ 支持，但计算 `sharpe` 时又用了 `full_stats`（第 208 行），而此时 `full_stats` 已经在条件表达式中赋值了。**实际上这个 walrus 表达式在 `if` 分支和 `else` 分支的行为不同**——如果 `calculate_comprehensive_stats` 返回空 dict，`full_stats` 仍然是空 dict，后续 `.get()` 不会报错。所以这不是 bug，只是代码可读性差。

### B10. engine_no_transformer.py — 文件尾部混入了 BacktestEngine 类
**文件**: `backtest/engine_no_transformer.py` 第 215-370 行
文件后半段定义了一个完全不同的 `BacktestEngine` 类（使用 CashAccount），与文件前半段的函数式 `run_backtest_loop_no_transformer` 完全无关。这两套实现共存会导致：
- import 时加载不必要的依赖（CashAccount, StrategyStatistics）
- 维护困惑

---

## 🟡 中等问题（不影响运行但影响结果准确性）

### M1. engine_no_transformer_new.py — 买入不做 ATR 动态仓位
买入固定 `position = 1.0`（表示满仓），没有 ATR 波动率调整。engine.py 和 engine_no_transformer.py 都有 ATR 仓位计算，唯独这个 new 版本没有。

### M2. evaluator.py — Sharpe 计算使用交易频率而非日频
```python
sharpe = (net_returns.mean() / (net_returns.std() + 1e-12)) * np.sqrt(ann_factor)
```
这里 `ann_factor = 252`，但 `net_returns` 是逐笔交易收益而非日收益。如果平均持仓 15 天，一年只有约 17 笔交易，Sharpe 计算严重偏高。正确做法应该用 `np.sqrt(n_trades_per_year)` 或基于日收益计算。

### M3. evaluator.py — equity_final 计算错误
```python
equity_final = initial_cash * (1 + net_returns.sum())  # 简化
```
`net_returns.sum()` 是简单收益率的算术和，不是复合收益。正确应该是 `(1 + net_returns).prod() * initial_cash`。

### M4. evaluator.py — annualized return 计算逻辑有误
```python
ann_return = total_return * (ann_factor / n_trades) if n_trades > 0 else 0.0
```
这里 `n_trades` 是交易笔数而非交易日数，年化收益率应该基于实际时间跨度计算。

### M5. cache.py — 缓存有效期检查被硬屏蔽
```python
#2026/04/06今天暂时屏蔽
return True
```
缓存永远不会过期，即使数据是几个月前的也会被使用。

### M6. compound_signal.py — `take_profit` 参数名不一致
compound_signal.py 使用 `take_profit`（固定止盈比例），而 engine.py 使用 `take_profit_multiplier`（ATR 倍数止盈）。两套逻辑不一致。

### M7. indicators_no_transformer.py — get_market_regime 仍然是旧版
engine.py 和 engine_no_transformer.py 都从 `indicators_no_transformer` 导入 `get_market_regime`，但该函数仍然是旧的 3 状态版本（bull/bear/neutral）。只有 `data/regime.py` 有新的 5 状态版本，而 indicators_no_transformer.py 没有更新。

---

## 🟢 小问题（代码质量/可维护性）

### S1. engine_no_transformer.py — 无 Transformer 版本中仍引用 transformer_conf
```python
has_transformer_conf = 'transformer_conf' in df.columns and not df['transformer_conf'].isna().all()
# ...
'confidence': df['transformer_conf'].loc[buy_date] if has_transformer_conf else None,
```
虽然是防御性代码，但在"无Transformer版本"中引用 Transformer 因子不太合适。

### S2. 三个 engine 文件（engine.py, engine_no_transformer.py, engine_no_transformer_new.py）大量重复代码
`calculate_transaction_cost`、`calculate_multi_timeframe_score` 等函数在三个文件中各有一份，且实现略有不同。应该提取到公共模块。

### S3. 三个 indicators 文件（indicators.py, indicators_new.py, indicators_no_transformer.py）同样重复

### S4. config.py 新版中的 `CommissionConfig`/`SlippageConfig` 与旧版 import 路径冲突
engine_no_transformer.py 和 engine_no_transformer_new.py 仍然 `from config import CommissionConfig, SlippageConfig`，而新版 config.py 中这些是 dataclass 不是 class。需要确认旧代码的 from_env() 方法是否还能正常工作。

### S5. backtest/__init__.py — 导出但未更新
仍然导出 `from backtest.engine import run_backtest_loop`，但 engine.py 的签名已改变。

### S6. data/__init__.py — 导出了 regime 但部分模块仍从旧位置导入
engine_no_transformer.py 从 `indicators_no_transformer` 导入 `get_market_regime`，而 indicators.py 已改为从 `regime` 导入。

### S7. 多处 print + logger 混用
多个文件中同时使用 `print()` 和 `logger.info()` 输出信息，标准不统一。

### S8. run_backtest_no_transformer.py 中定义的 optimize_strategy_no_transformer 函数
应该移到 backtest/optimizer.py 或单独文件中，而不是在入口脚本中定义。
