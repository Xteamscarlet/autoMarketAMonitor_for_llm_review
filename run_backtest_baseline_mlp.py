# -*- coding: utf-8 -*-
"""
run_backtest_baseline_mlp.py

日线 MLP 基线回测完整脚本 (修正版)。
特性：
1. 完全复用现有数据管线
2. 使用 MLP 模型预测
3. 修正了可视化函数参数缺失的问题
"""

import os
import logging
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==================== 导入你现有的模块 ====================
from config import get_settings, STOCK_CODES
from data import check_and_clean_cache, save_pickle_cache, load_pickle_cache
from data.indicators_new import (
    prepare_stock_data,
    calculate_orthogonal_factors_without_transformer
)
from data.loader_new import download_market_data, download_stocks_data
from data.types import TRADITIONAL_FACTOR_COLS
from backtest.engine_no_transformer_new import run_backtest_loop_no_transformer
from backtest.evaluator import calculate_comprehensive_stats
from backtest.visualizer import visualize_backtest_with_split  # 导入可视化函数
from risk_manager import RiskManager

# ==================== 配置日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("backtest_baseline_mlp.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ==================== 结果保存目录 ====================
output_dir = "./stock_cache/baseline_mlp_results"
os.makedirs(output_dir, exist_ok=True)
viz_dir = os.path.join(output_dir, "charts")
os.makedirs(viz_dir, exist_ok=True)
model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)


# ==================== MLP 模型定义 ====================
class BaselineMLP(nn.Module):
    """
    简单 MLP 回归模型：
    输入：因子特征（维度 = n_features）
    输出：未来收益预测标量（连续值，作为 score）
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, input_dim)
        return self.net(x).squeeze(-1)  # (batch,)


class StockDataset(Dataset):
    """简单 Dataset，用于 MLP 训练 / 验证。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_mlp_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 30,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        early_stop_patience: int = 5,
) -> nn.Module:
    """
    训练一个简单的 MLP 模型，返回训练好的模型。
    """
    model = BaselineMLP(input_dim, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * len(X_batch)

        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"[MLP 训练] epoch={epoch + 1}/{epochs}, train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
            )

        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def prepare_mlp_features_from_df(
        df: pd.DataFrame,
        lookback: int = 120,
        horizon: int = 5,
        standardize: bool = True,
) -> tuple:
    """
    从已经计算好因子的 DataFrame 中构建 MLP 所需的特征矩阵 X 和标签 y。

    重要：此函数假设 df 已经包含：
    1. 技术指标因子 (由 prepare_stock_data 计算)
    2. 正交化因子 (由 calculate_orthogonal_factors_without_transformer 计算)
    3. 原始价格列
    """
    # 1. 确定特征列
    # 基于你现有的代码结构，特征列应该包括技术指标和正交因子
    # 我们排除掉非数值列和目标列
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return_1d', 'score', 'future_return']
    feature_cols = [col for col in df.columns if
                    col not in exclude_cols and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]]

    # 2. 准备特征
    X = df[feature_cols].values.copy()

    # 3. 准备标签
    future_close = df['Close'].shift(-horizon)
    ret = (future_close - df['Close']) / df['Close']
    y = ret.values

    # 4. 标准化 (全局标准化，简单但有效)
    if standardize:
        # 计算均值和标准差时，忽略 NaN
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        X = (X - mean) / (std + 1e-8)

    # 5. 清理 NaN/Inf
    valid_mask = ~np.isnan(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    dates = df.index[valid_mask]

    # 6. 对齐 lookback (虽然当前特征是单点，但保留此逻辑以备将来扩展)
    # 这里简单处理：去掉前 lookback 天，保证每个样本至少有 lookback 历史数据
    if lookback > 0 and len(X) > lookback:
        X = X[lookback:]
        y = y[lookback:]
        dates = dates[lookback:]

    return X, y, dates, feature_cols


def run_single_stock_backtest_baseline_mlp(
        stock_code: str,
        df: pd.DataFrame,
        market_data: pd.DataFrame,
        stocks_data: dict,
        params: dict,
        mlp_config: dict
) -> tuple:
    """
    对单只股票执行 MLP 基线回测，并包含可视化步骤。
    """
    # 1. 准备特征和标签
    X, y, dates, feature_cols = prepare_mlp_features_from_df(
        df,
        lookback=mlp_config['lookback'],
        horizon=mlp_config['horizon'],
        standardize=mlp_config['standardize'],
    )

    if len(X) < 200:
        logger.warning(f"[{stock_code}] 样本数 {len(X)} < 200，跳过 MLP 回测")
        return None, None, df

    # 2. 划分 train / val
    split_idx = int(len(X) * 0.8)  # 记录划分点
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 3. 训练 MLP
    input_dim = X.shape[1]
    model = train_mlp_model(
        X_train,
        y_train,
        X_val,
        y_val,
        input_dim=input_dim,
        hidden_dim=mlp_config['hidden_dim'],
        num_layers=mlp_config['num_layers'],
        dropout=mlp_config['dropout'],
        lr=mlp_config['lr'],
        batch_size=mlp_config['batch_size'],
        epochs=mlp_config['epochs'],
        device=mlp_config['device'],
        early_stop_patience=mlp_config['early_stop_patience'],
    )

    # 4. 在全部数据上预测，得到 score（对齐回测日期）
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(mlp_config['device'])
        score_array = model(X_tensor).cpu().numpy()  # (n_samples,)

    # 构造一个与 df 等长的 score 列，默认 NaN
    df["score"] = np.nan
    # 将预测值填入对应日期位置
    score_dates = dates[-len(score_array):]
    df.loc[score_dates, "score"] = score_array

    # 5. 调用原有回测引擎
    # 注意：run_backtest_loop_no_transformer 期望 weights 字典，这里传一个空字典，
    # 因为我们已经在 df['score'] 中直接存好了模型分数。
    weights = {}
    trades_df, stats, df = run_backtest_loop_no_transformer(
        df,
        stock_code,
        market_data,
        weights,
        params,
        regime=mlp_config.get('regime', None),
        stocks_data=stocks_data,
        initial_capital=mlp_config.get('initial_capital', 100000.0),
    )

    # 6. 保存模型和生成可视化
    if trades_df is not None:
        # 保存模型
        model_path = os.path.join(model_dir, f"{stock_code}_mlp.pt")
        torch.save(model.state_dict(), model_path)

        # --- 修正部分：在此处生成可视化，参数齐全 ---
        try:
            chart_path = os.path.join(viz_dir, f"{stock_code}_backtest.png")

            # 计算划分点在原图 df 中的位置（用于画分割线）
            # dates[split_idx] 是划分那天的日期
            split_date = dates[split_idx]

            # 在 df 中找到该日期对应的位置索引
            # 如果 df 是按日期索引的，get_loc 可以直接获取整数位置
            if split_date in df.index:
                split_idx_in_df = df.index.get_loc(split_date)
            else:
                # 如果日期不在（极少情况），取一个近似位置
                split_idx_in_df = int(len(df) * 0.8)

            # 调用可视化函数，填入所有必需参数：stock_name 和 split_idx
            visualize_backtest_with_split(
                df,
                trades_df,
                stock_name=stock_code,  # 填入股票代码
                split_idx=split_idx_in_df,  # 填入划分点索引
                save_path=chart_path
            )
        except Exception as e:
            logger.warning(f"[{stock_code}] 可视化生成失败: {e}")

    return trades_df, stats, df


# ==================== 多进程工作进程 ====================
_worker_market_data = None
_worker_stocks_data = None
_worker_params = None
_worker_mlp_config = None


def _init_worker(market_data, stocks_data, params, mlp_config):
    global _worker_market_data, _worker_stocks_data, _worker_params, _worker_mlp_config
    _worker_market_data = market_data
    _worker_stocks_data = stocks_data
    _worker_params = params
    _worker_mlp_config = mlp_config


def _process_single_stock(stock_code: str):
    """多进程回调"""
    global _worker_market_data, _worker_stocks_data, _worker_params, _worker_mlp_config

    df = _worker_stocks_data.get(stock_code)
    if df is None or len(df) < 60:
        return None

    # 注意：这里假设 df 已经预处理过因子
    trades_df, stats, df = run_single_stock_backtest_baseline_mlp(
        stock_code,
        df,
        _worker_market_data,
        _worker_stocks_data,
        params=_worker_params,
        mlp_config=_worker_mlp_config,
    )

    if trades_df is None:
        return None

    return {
        "code": stock_code,
        "trades_df": trades_df,
        "stats": stats,
        "df": df,
    }


def run_baseline_mlp_backtest():
    """
    主函数
    """
    settings = get_settings()
    risk_manager = RiskManager(settings)

    # 1. 数据准备
    # 下载大盘数据
    market_cache_file = "./stock_cache/no_transformer_market_data.pkl"
    try:
        if not check_and_clean_cache(market_cache_file):
            market_data = download_market_data()
            save_pickle_cache(market_cache_file, market_data)
        else:
            market_data = load_pickle_cache(market_cache_file)
    except Exception as e:
        print(f"警告: 大盘数据加载失败: {e}")
        print("尝试使用本地缓存...")
        if os.path.exists(market_cache_file):
            market_data = load_pickle_cache(market_cache_file)
        else:
            print("错误: 没有可用的大盘数据")
            return
    if market_data is None or len(market_data) == 0:
        logger.error("大盘数据下载失败")
        return

    # 下载股票数据
    print("\n[2/3] 检查个股数据...")
    # ========== 统一使用 check_and_clean_cache ==========
    stocks_cache_file = "./stock_cache/no_transformer_stocks_data.pkl"
    if not check_and_clean_cache(stocks_cache_file):
        print("下载股票数据...")
        stocks_data = download_stocks_data(STOCK_CODES)
        save_pickle_cache(stocks_cache_file, stocks_data)
    else:
        print("使用缓存的股票数据...")
        stocks_data = load_pickle_cache(stocks_cache_file)
    # 数据验证
    if not stocks_data:
        print("错误: 无法获取个股数据")
        return
    # ===== 关键修复：添加收益率计算 =====
    logger.info("步骤1.5: 计算收益率...")
    stocks_data = prepare_stock_data(stocks_data)
    print(f"stocks_data 类型: {type(stocks_data)}")
    print(f"stocks_data 长度: {len(stocks_data)}")
    print(f"stocks_data 键示例: {list(stocks_data.keys())[:3]}")

    # 3. 验证数据结构
    print(f"\n数据验证:")
    print(f" 类型: {type(stocks_data)}")
    print(f" 长度: {len(stocks_data) if stocks_data else 0}")
    if stocks_data and len(stocks_data) > 0:
        print(f" 前3个键: {list(stocks_data.keys())[:3]}")

    # 检查是否是错误的结构
    first_key = list(stocks_data.keys())[0]
    if first_key in ['stocks_data', 'last_date']:
        print("\n检测到错误的数据结构，正在修复...")
        if isinstance(stocks_data, dict) and 'stocks_data' in stocks_data:
            stocks_data = stocks_data['stocks_data']
        print(f"修复后键示例: {list(stocks_data.keys())[:3]}")

    if not stocks_data or len(stocks_data) == 0:
        print("\n错误: 无法获取有效的股票数据")
        return

    # 预处理因子
    logger.info("计算技术指标和正交因子...")
    stocks_data = prepare_stock_data(stocks_data)
    for code, df in stocks_data.items():
        stocks_data[code] = calculate_orthogonal_factors_without_transformer(df)

    # 2. 参数配置
    params = {
        "bull": {
            "buy_threshold": 0.6,
            "sell_threshold": -0.2,
            "stop_loss": -0.08,
            "hold_days": 15,
        },
        "bear": {
            "buy_threshold": 0.7,
            "sell_threshold": -0.3,
            "stop_loss": -0.10,
            "hold_days": 10,
        },
        "neutral": {
            "buy_threshold": 0.65,
            "sell_threshold": -0.25,
            "stop_loss": -0.09,
            "hold_days": 12,
        },
    }

    # 5. MLP 基线配置
    mlp_config = {
        "lookback": 120,  # 回看窗口（用于特征构建，当前特征为单点，暂未使用）
        "horizon": 5,  # 预测未来 N 天收益
        "standardize": True,  # 是否标准化特征
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 30,
        "early_stop_patience": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "regime": None,  # 自动判断市场状态
        "initial_capital": 100000.0,
    }

    # 6. 多进程并行回测
    num_workers = max(1, cpu_count() - 1)
    logger.info(f"使用 {num_workers} 个进程进行回测...")

    # 由于我们已经预处理了 stocks_data，可以直接传入
    with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(market_data, stocks_data, params, mlp_config),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(_process_single_stock, STOCK_CODES),
                total=len(STOCK_CODES),
                desc="Baseline MLP 回测",
            )
        )

    # 4. 汇总结果
    all_stats = []
    for res in results:
        if res is None:
            continue
        stats = res.get("stats")
        if stats:
            stats["code"] = res["code"]
            all_stats.append(stats)

    if not all_stats:
        logger.warning("没有有效回测结果")
        return

    stats_df = pd.DataFrame(all_stats)

    # 5. 打印统计
    comprehensive_stats = calculate_comprehensive_stats(stats_df)
    logger.info("=== MLP 基线综合统计 ===")
    for k, v in comprehensive_stats.items():
        logger.info(f"{k}: {v}")

    # 6. 保存结果
    stats_path = os.path.join(output_dir, "baseline_mlp_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"统计结果已保存: {stats_path}")


if __name__ == "__main__":
    run_baseline_mlp_backtest()
