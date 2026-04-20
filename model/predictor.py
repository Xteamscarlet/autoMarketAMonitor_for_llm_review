# -*- coding: utf-8 -*-
"""
模型推理模块 V2 — 批量推理优化
主要改进：
1. calculate_transformer_factor_series 使用批量推理替代逐条推理
2. 降低回测时 MC Dropout 采样次数（3次替代10次）
3. 优化内存管理
"""
import os
import time
import logging
from typing import Optional, List, Dict

from data.indicators_no_transformer import safe_sma
from utils.stock_filter import should_intercept_stock

import numpy as np
import pandas as pd
import torch
import joblib
import talib as ta
from tqdm import tqdm

from config import get_settings
from data.types import FEATURES
from model.transformer import StockTransformer
from exceptions import ModelLoadError

logger = logging.getLogger(__name__)

try:
    import efinance as ef
    _EF_AVAILABLE = True
except ImportError:
    _EF_AVAILABLE = False
    ef = None


# ★ 次要修复: 模型加载缓存，避免回测时每只股票都重复加载几百 MB 权重
_MODEL_CACHE: Dict[str, List[tuple]] = {}


def _build_model(device: torch.device) -> StockTransformer:
    settings = get_settings()
    return StockTransformer(
        input_dim=len(FEATURES),
        lookback_days=settings.model.lookback_days,
        num_heads=settings.model.num_heads,
        dim_feedforward=settings.model.dim_feedforward,
        num_layers=settings.model.num_layers,
        num_classes=settings.model.num_classes,
        dropout=settings.model.dropout,
    ).to(device)


def _load_ensemble_models(device: torch.device) -> List[tuple]:
    """动态加载所有可用模型（带进程内缓存）"""
    cache_key = str(device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    settings = get_settings()
    models = []

    if os.path.exists(settings.paths.model_path):
        try:
            m = _build_model(device)
            m.load_state_dict(torch.load(settings.paths.model_path, map_location=device))
            m.eval()
            models.append(("EMA", m))
        except Exception as e:
            raise ModelLoadError(f"EMA模型加载失败: {e}", model_type="EMA", path=settings.paths.model_path)

    if os.path.exists(settings.paths.swa_model_path):
        try:
            m = _build_model(device)
            swa_state = torch.load(settings.paths.swa_model_path, map_location=device)
            clean_state = {k.replace('module.', ''): v for k, v in swa_state.items() if k != 'n_averaged'}
            m.load_state_dict(clean_state)
            m.eval()
            models.append(("SWA", m))
        except Exception as e:
            logger.warning(f"SWA 模型加载失败: {e}")

    topk_dir = settings.paths.topk_checkpoint_dir
    if os.path.exists(topk_dir):
        for fname in sorted(os.listdir(topk_dir)):
            if fname.startswith("topk_rawclose_") and fname.endswith(".pth"):
                try:
                    m = _build_model(device)
                    ckpt = torch.load(os.path.join(topk_dir, fname), map_location=device)
                    clean_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items() if k != 'n_averaged'}
                    m.load_state_dict(clean_ckpt)
                    m.eval()
                    models.append((f"TopK_{fname}", m))
                except Exception as e:
                    logger.warning(f"TopK 模型 {fname} 加载失败: {e}")

    _MODEL_CACHE[cache_key] = models
    return models


def _load_scalers() -> tuple:
    settings = get_settings()
    scalers = {}
    global_scaler = None

    if os.path.exists(settings.paths.scaler_path):
        scalers = joblib.load(settings.paths.scaler_path)
    if os.path.exists(settings.paths.global_scaler_path):
        global_scaler = joblib.load(settings.paths.global_scaler_path)

    return scalers, global_scaler


# ★ 修复6: 训练集分位数阈值缓存（用于预测时把"上涨/下跌"映射到真实涨跌幅区间）
_THRESHOLD_CACHE: Optional[dict] = None


def _load_label_thresholds() -> Optional[dict]:
    global _THRESHOLD_CACHE
    if _THRESHOLD_CACHE is not None:
        return _THRESHOLD_CACHE
    settings = get_settings()
    path = settings.paths.label_thresholds_path
    if os.path.exists(path):
        try:
            _THRESHOLD_CACHE = joblib.load(path)
            return _THRESHOLD_CACHE
        except Exception as e:
            logger.warning(f"加载分位数阈值失败: {e}")
    return None


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp['MA5'] = safe_sma(temp['Close'], period=5)
    temp['MA10'] = safe_sma(temp['Close'], period=10)
    temp['MA20'] = safe_sma(temp['Close'], period=20)
    temp['MACD'], temp['MACD_Signal'], temp['MACD_Hist'] = ta.MACD(temp['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    temp['K'], temp['D'] = ta.STOCH(temp['High'], temp['Low'], temp['Close'], fastk_period=9, slowk_period=3, slowd_period=3)
    temp['J'] = 3 * temp['K'] - 2 * temp['D']
    temp['RSI'] = ta.RSI(temp['Close'], timeperiod=14)
    temp['ADX'] = ta.ADX(temp['High'], temp['Low'], temp['Close'], timeperiod=14)
    temp['BB_Upper'], temp['BB_Middle'], temp['BB_Lower'] = ta.BBANDS(temp['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    temp['OBV'] = ta.OBV(temp['Close'], temp['Volume'])
    temp['CCI'] = ta.CCI(temp['High'], temp['Low'], temp['Close'], timeperiod=20)
    # ★ 新增收益率特征
    for lag in [1, 3, 5, 10]:
        temp[f'ret_{lag}'] = temp['Close'].pct_change(lag)
    temp.bfill(inplace=True)
    temp.dropna(inplace=True)
    return temp


def _select_scaler(code: str, scalers: dict, global_scaler, features_data: np.ndarray):
    if code in scalers:
        scaled = scalers[code].transform(features_data)
        return np.clip(scaled, -5, 5), "专用"
    elif global_scaler is not None:
        scaled = global_scaler.transform(features_data)
        return np.clip(scaled, -5, 5), "全局"
    else:
        median = np.median(features_data, axis=0)
        q75 = np.percentile(features_data, 75, axis=0)
        q25 = np.percentile(features_data, 25, axis=0)
        iqr = np.clip(q75 - q25, 1e-6, None)
        scaled = (features_data - median) / iqr
        return np.clip(scaled, -5, 5), "在线标准化"


def predict_stocks(target_codes: List[str], models: Optional[List] = None) -> pd.DataFrame:
    """集成预测多只股票"""
    settings = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if models is None:
        models = _load_ensemble_models(device)

    if not models:
        logger.error("没有找到任何模型文件！")
        return pd.DataFrame()

    scalers, global_scaler = _load_scalers()
    thresholds = _load_label_thresholds()
    predictions = []

    for i, target_code in enumerate(target_codes):
        try:
            # ★ 次要修复: 先检查依赖再 sleep，避免没装 efinance 还浪费时间
            if not _EF_AVAILABLE:
                continue

            time.sleep(10)
            if i > 0 and i % 100 == 0:
                logger.warning('每100只暂停100秒')
                time.sleep(100)

            df = ef.stock.get_quote_history(target_code, beg='20210101', end='20270101')
            if df is None or df.empty:
                continue
            if '日期' not in df.columns:
                continue

            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期').sort_index()

            temp = pd.DataFrame(index=df.index)
            temp['Open'] = df['开盘']
            temp['High'] = df['最高']
            temp['Low'] = df['最低']
            temp['Close'] = df['收盘']
            temp['Volume'] = df['成交量']
            temp['Turnover Rate'] = df['换手率']

            temp = _prepare_features(temp)
            skip, reason = should_intercept_stock(target_code, "", temp)
            if skip:
                logger.warning(f"[拦截-预测] 跳过 {target_code}: {reason}")
                continue

            if len(temp) < settings.model.lookback_days:
                continue

            scaled, scaler_type = _select_scaler(
                target_code, scalers, global_scaler, temp[FEATURES].values
            )

            last_seq = torch.FloatTensor(scaled[-settings.model.lookback_days:]).unsqueeze(0).to(device)

            all_probs = []
            all_rets = []
            with torch.no_grad():
                for name, m in models:
                    mean_probs, _, mean_ret = m.mc_predict(last_seq, n_forward=settings.model.mc_forward_train)
                    all_probs.append(mean_probs)
                    all_rets.append(mean_ret)

            ensemble_probs = torch.stack(all_probs).mean(dim=0)
            ensemble_ret = torch.stack(all_rets).mean(dim=0)
            inter_model_uncertainty = torch.stack(all_probs).std(dim=0).mean().item()

            probs_1d = ensemble_probs.squeeze(0)
            up_prob = (probs_1d[2] + probs_1d[3]).item()
            # ★ 修复5: 训练时 ret target 除以 ret_target_scale=0.05，推理时需还原
            pred_ret = ensemble_ret.item() * settings.model.ret_target_scale
            trend = "上涨" if up_prob > 0.5 else "下跌"
            confidence = up_prob if trend == "上涨" else (1 - up_prob)
            risk_flag = "⚠️ 高模型分歧" if inter_model_uncertainty > 0.05 else "正常"
            expected_score = up_prob * pred_ret if pred_ret > 0 else pred_ret * (1 - up_prob)

            predictions.append({
                'code': target_code,
                'trend': trend,
                'probability': round(float(confidence), 4),
                'predicted_ret': round(float(pred_ret), 4),
                'uncertainty': round(float(inter_model_uncertainty), 4),
                'risk_flag': risk_flag,
                'expected_score': round(float(expected_score), 4),
                'ensemble_size': len(models),
                'scaler_type': scaler_type,
                # ★ 修复6: 附带训练集分位数阈值，让"上涨"的语义清晰
                'label_threshold_q75': round(float(thresholds['q75']), 4) if thresholds else None,
                'label_threshold_q90': round(float(thresholds['q90']), 4) if thresholds else None,
            })

        except Exception as e:
            logger.warning(f"{target_code} 预测失败: {e}")

    if predictions:
        df_result = pd.DataFrame(predictions)
        df_result = df_result.sort_values(by='expected_score', ascending=False).reset_index(drop=True)
        df_result.insert(0, 'rank', df_result.index + 1)
        return df_result
    return pd.DataFrame()


def calculate_transformer_factor_series(
    df: pd.DataFrame,
    code: str,
    device=None,
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    ★ 批量推理版 Transformer 因子计算
    将所有时间步的序列一次性打包成 batch，大幅加速推理
    """
    settings = get_settings()
    if lookback_days is None:
        lookback_days = settings.model.lookback_days
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        models = _load_ensemble_models(device)
        if not models:
            logger.warning(f"找不到任何模型，跳过 {code}")
            return pd.DataFrame(
                index=df.index,
                columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
            ).fillna(0.5)

        temp = df.copy()
        col_mapping = {'开盘': 'Open', '最高': 'High', '最低': 'Low', '收盘': 'Close', '成交量': 'Volume', '换手率': 'Turnover Rate'}
        for cn, en in col_mapping.items():
            if cn in temp.columns and en not in temp.columns:
                temp[en] = temp[cn]

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if any(c not in temp.columns for c in required):
            return pd.DataFrame(
                index=df.index,
                columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
            ).fillna(0.5)

        temp['Turnover Rate'] = temp.get('Turnover Rate', pd.Series(3.0, index=temp.index)).fillna(3.0)
        temp = _prepare_features(temp)

        if len(temp) < lookback_days:
            return pd.DataFrame(
                index=df.index,
                columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
            ).fillna(0.5)

        scalers, global_scaler = _load_scalers()
        scaled_data, _ = _select_scaler(code, scalers, global_scaler, temp[FEATURES].values)

        # ★ 批量构建序列
        sequences = np.array([scaled_data[i - lookback_days: i] for i in range(lookback_days, len(scaled_data) + 1)])
        valid_indices = [temp.index[i - 1] for i in range(lookback_days, len(scaled_data) + 1)]

        if len(sequences) == 0:
            return pd.DataFrame(
                index=df.index,
                columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
            ).fillna(0.5)

        # ★ 分批推理
        batch_size = settings.model.inference_batch_size
        mc_forward = settings.model.mc_forward_backtest  # 回测用更少的 MC 采样

        all_probs_list = []
        all_rets_list = []

        for name, m in models:
            m.eval()
            probs_per_model = []
            rets_per_model = []

            for batch_start in range(0, len(sequences), batch_size):
                batch_end = min(batch_start + batch_size, len(sequences))
                batch_seqs = sequences[batch_start:batch_end]
                batch_tensor = torch.FloatTensor(batch_seqs).to(device)

                with torch.no_grad():
                    # ★ 使用批量 MC predict
                    mean_probs, _, mean_ret = m.mc_predict(batch_tensor, n_forward=mc_forward)
                    probs_per_model.append(mean_probs.cpu())
                    rets_per_model.append(mean_ret.cpu())

                # 释放显存
                del batch_tensor
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            all_probs_list.append(torch.cat(probs_per_model, dim=0))
            all_rets_list.append(torch.cat(rets_per_model, dim=0))

        stack_probs = torch.stack(all_probs_list)
        ensemble_probs = stack_probs.mean(dim=0)
        inter_model_std = stack_probs.std(dim=0).mean(dim=-1).numpy()
        up_probs = (ensemble_probs[:, 2] + ensemble_probs[:, 3]).numpy()
        # ★ 修复5: 还原 ret 尺度（训练时 target 除以 ret_target_scale）
        mean_rets = torch.stack(all_rets_list).mean(dim=0).squeeze(-1).numpy() * settings.model.ret_target_scale

        result_df = pd.DataFrame({
            'transformer_prob': up_probs,
            'transformer_pred_ret': mean_rets,
            'transformer_uncertainty': inter_model_std,
        }, index=valid_indices).reindex(df.index)

        result_df['transformer_prob'] = result_df['transformer_prob'].fillna(0.5)
        result_df['transformer_pred_ret'] = result_df['transformer_pred_ret'].fillna(0.0)
        result_df['transformer_uncertainty'] = result_df['transformer_uncertainty'].fillna(0.15)

        return result_df

    except Exception as e:
        logger.error(f"Transformer 因子计算错误 ({code}): {e}")
        return pd.DataFrame(
            index=df.index,
            columns=['transformer_prob', 'transformer_pred_ret', 'transformer_uncertainty'],
        ).fillna(0.5)
