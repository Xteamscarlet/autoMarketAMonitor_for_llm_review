# -*- coding: utf-8 -*-
"""
run_predict_baseline_mlp.py

日线 MLP 基线预测脚本：
- 训练 MLP 模型；
- 输出未来一个时间点的预测信号（买 / 卖 / 持有）；
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==================== 导入你现有的模块 ====================
from config import get_settings, STOCK_CODES
from data.indicators_new import (
    prepare_stock_data,
    calculate_orthogonal_factors_without_transformer
)
from data.loader_new import download_market_data, download_stocks_data

# ==================== 配置日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("predict_baseline_mlp.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ==================== 结果保存目录 ====================
output_dir = "./stock_cache/baseline_mlp_predict"
os.makedirs(output_dir, exist_ok=True)
model_dir = os.path.join(output_dir, "models")
os.makedirs(model_dir, exist_ok=True)
signal_dir = os.path.join(output_dir, "signals")
os.makedirs(signal_dir, exist_ok=True)


# ==================== 复用模型定义 ====================
class BaselineMLP(nn.Module):
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
        return self.net(x).squeeze(-1)


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 复用训练函数（从回测脚本复制，保持独立）
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


# 复用特征准备函数（从回测脚本复制，保持独立）
def prepare_mlp_features_from_df(
        df: pd.DataFrame,
        lookback: int = 120,
        horizon: int = 5,
        standardize: bool = True,
) -> tuple:
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'future_return_1d', 'score', 'future_return']
    feature_cols = [col for col in df.columns if
                    col not in exclude_cols and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]]

    X = df[feature_cols].values.copy()

    future_close = df['Close'].shift(-horizon)
    ret = (future_close - df['Close']) / df['Close']
    y = ret.values

    if standardize:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        X = (X - mean) / (std + 1e-8)

    valid_mask = ~np.isnan(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    dates = df.index[valid_mask]

    if lookback > 0 and len(X) > lookback:
        X = X[lookback:]
        y = y[lookback:]
        dates = dates[lookback:]

    return X, y, dates, feature_cols


def run_predict_for_stock(
        stock_code: str,
        df: pd.DataFrame,
        mlp_config: dict,
) -> dict:
    """
    对单只股票训练 MLP 并生成预测信号。
    """
    X, y, dates, feature_cols = prepare_mlp_features_from_df(
        df,
        lookback=mlp_config['lookback'],
        horizon=mlp_config['horizon'],
        standardize=mlp_config['standardize'],
    )

    if len(X) < 200:
        logger.warning(f"[{stock_code}] 样本数 {len(X)} < 200，跳过预测")
        return None

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

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

    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float().to(mlp_config['device'])
        scores = model(X_tensor).cpu().numpy()

    latest_score = scores[-1]
    latest_date = dates[-1]

    if latest_score > mlp_config['buy_threshold']:
        signal = "buy"
    elif latest_score < mlp_config['sell_threshold']:
        signal = "sell"
    else:
        signal = "hold"

    model_path = os.path.join(model_dir, f"{stock_code}_mlp_predict.pt")
    torch.save(model.state_dict(), model_path)

    return {
        "code": stock_code,
        "latest_date": latest_date,
        "latest_score": float(latest_score),
        "signal": signal,
        "buy_threshold": mlp_config['buy_threshold'],
        "sell_threshold": mlp_config['sell_threshold'],
    }


def run_baseline_mlp_predict():
    """
    主函数：对所有股票执行 MLP 预测。
    """
    settings = get_settings()

    # 1. 市场数据
    logger.info("下载市场数据...")
    market_data = download_market_data()
    if market_data is None:
        logger.error("下载市场数据失败")
        return

    # 2. 股票数据
    logger.info("下载/加载股票数据...")
    stocks_data = download_stocks_data(STOCK_CODES)
    if not stocks_data:
        logger.error("没有有效股票数据")
        return

    # 3. 预处理数据（复用你现有的管线）
    logger.info("计算技术指标和正交因子...")
    stocks_data = prepare_stock_data(stocks_data)
    for code, df in stocks_data.items():
        stocks_data[code] = calculate_orthogonal_factors_without_transformer(df)

    # 4. 预测配置
    mlp_config = {
        "lookback": 120,
        "horizon": 5,
        "standardize": True,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 30,
        "early_stop_patience": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "buy_threshold": 0.6,
        "sell_threshold": -0.2,
    }

    # 5. 遍历股票
    all_signals = []
    for stock_code in tqdm(STOCK_CODES, desc="MLP 预测"):
        df = stocks_data.get(stock_code)
        if df is None or len(df) < 60:
            continue

        result = run_predict_for_stock(stock_code, df, mlp_config)

        if result is not None:
            all_signals.append(result)

    # 6. 汇总信号
    if not all_signals:
        logger.warning("没有生成任何有效信号")
        return

    signals_df = pd.DataFrame(all_signals)
    signals_path = os.path.join(signal_dir, "latest_signals.csv")
    signals_df.to_csv(signals_path, index=False)
    logger.info(f"信号已保存: {signals_path}")

    # 7. 打印简要统计
    signal_counts = signals_df["signal"].value_counts()
    logger.info("=== 信号统计 ===")
    for sig, cnt in signal_counts.items():
        logger.info(f"{sig}: {cnt}")


if __name__ == "__main__":
    run_baseline_mlp_predict()
