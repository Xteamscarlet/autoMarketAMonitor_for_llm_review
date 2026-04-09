# -*- coding: utf-8 -*-
"""
可视化模块
生成包含价格走势、买卖点、因子面板的完整回测图表
从 backtest_market_v2.py 的 visualize_backtest_with_split() 提取
"""
import logging
from typing import Optional, Dict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data.indicators import get_market_regime

logger = logging.getLogger(__name__)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def visualize_backtest_with_split(
    df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame],
    stock_name: str,
    split_idx: int,
    market_data: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    strat: Optional[Dict] = None,
) -> Optional[object]:
    """可视化回测结果（测试集部分）"""
    if split_idx >= len(df):
        split_idx = int(len(df) * 0.7)

    df = df.iloc[split_idx:].copy()
    if df.empty:
        return None

    if trades_df is not None and len(trades_df) > 0:
        test_start_date = df.index[0]
        trades_df = trades_df[trades_df['buy_date'] >= test_start_date].copy()

    if 'Combined_Score' not in df.columns:
        # 不再调用有Transformer版本的 calculate_multi_timeframe_score
        # 直接用权重加权求和，逻辑与无Transformer版本的单日评分完全一致
        if strat and isinstance(strat, dict) and 'weights' in strat and strat['weights']:
            weights_to_use = strat['weights']
        else:
            factor_cols = [
                col for col in df.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'MA20']
                                             and pd.api.types.is_numeric_dtype(df[col])
            ]
            weights_to_use = {col: 1.0 / len(factor_cols) for col in factor_cols} if factor_cols else {}

        # ===== 核心修复 =====
        if weights_to_use:
            valid_factors = [
                col for col in weights_to_use if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]
            if valid_factors:
                w = pd.Series(weights_to_use).reindex(valid_factors)
                w_sum = w.sum()
                if w_sum > 0:
                    df['Combined_Score'] = df[valid_factors].mul(w).sum(axis=1) / w_sum
                else:
                    # 修复B1b: 权重全为0时(如优化器失败返回-999)，改用等权均值
                    print(f"{stock_name} 因子权重全为0(w_sum=0)，改用等权均值计算综合得分")
                    df['Combined_Score'] = df[valid_factors].mean(axis=1)
            else:
                # 修复B2: 权重key与df列名不匹配时，自动检测因子列
                print(f"{stock_name} 权重字典中无匹配的因子列，自动检测因子列")
                exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA20',
                                'MA5', 'MA60', 'buy_signal', 'sell_signal',
                                'position', 'Combined_Score', 'transformer_prob']
                auto_factors = [
                    col for col in df.columns
                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
                ]
                if auto_factors:
                    df['Combined_Score'] = df[auto_factors].mean(axis=1)
                else:
                    print(f"{stock_name} 未找到任何因子列，综合得分设为默认值0.5")
                    df['Combined_Score'] = 0.5
        else:
            # 修复C: 权重为空字典时，自动检测因子列
            print(f"{stock_name} 权重为空，自动检测因子列计算综合得分")
            exclude_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA20',
                            'MA5', 'MA60', 'buy_signal', 'sell_signal',
                            'position', 'Combined_Score', 'transformer_prob']
            auto_factors = [
                col for col in df.columns
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
            ]
            if auto_factors:
                df['Combined_Score'] = df[auto_factors].mean(axis=1)
            else:
                print(f"{stock_name} 未找到任何因子列，综合得分设为默认值0.5")
                df['Combined_Score'] = 0.5
    else:
        # 修复D: Combined_Score已存在时保留原值，不再覆盖为0.5
        print(f"{stock_name} 使用已有的Combined_Score列")

    fig = plt.figure(figsize=(20, 27))
    fig.suptitle(f'{stock_name} 测试集回测可视化 (全因子展示)', fontsize=16, fontweight='bold')

    factor_subplots = [
        ('mom_10', 'mom_20', '动量因子', 'blue', 'cyan'),
        ('atr_pct', 'bias_20', '波动/乖离因子', 'orange', 'brown'),
        ('vol_price_res', None, '量价因子', 'green', None),
        ('rsi_norm', None, 'RSI因子', 'purple', None),
        ('macd_hist_norm', None, 'MACD因子', 'red', None),
        ('bb_width', None, '布林带宽度', 'navy', None),
    ]

    n_subplots = 3 + len(factor_subplots) + (1 if 'transformer_prob' in df.columns else 0)
    gs = fig.add_gridspec(n_subplots, 1, height_ratios=[3] + [1] * (n_subplots - 1), hspace=0.35)

    # 子图1: 价格走势
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['Close'], label='收盘价', color='black', linewidth=1.5, alpha=0.8)
    if 'MA20' in df.columns:
        ax1.plot(df.index, df['MA20'], label='MA20', color='orange', linewidth=1, linestyle='--', alpha=0.7)

    ma20 = df['Close'].rolling(20, min_periods=1).mean()
    std20 = df['Close'].rolling(20, min_periods=1).std()
    ax1.fill_between(df.index, ma20 - 2 * std20, ma20 + 2 * std20, alpha=0.1, color='gray', label='布林带')

    if trades_df is not None and len(trades_df) > 0:
        for idx_t, trade in trades_df.iterrows():
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            buy_price = df.loc[buy_date, 'Close'] if buy_date in df.index else None
            sell_price = df.loc[sell_date, 'Close'] if sell_date in df.index else None

            if buy_price:
                ax1.scatter(buy_date, buy_price, marker='^', color='red', s=200, zorder=5,
                           edgecolors='darkred', linewidths=2,
                           label='买入' if idx_t == trades_df.index[0] else "")
                score = df.loc[buy_date, 'Combined_Score'] if 'Combined_Score' in df.columns else 0
                regime = get_market_regime(market_data, buy_date) if market_data is not None else 'neutral'
                regime_cn = {'strong_bull': '强多头', 'bull': '多头', 'weak': '弱势', 'bear': '空头', 'neutral': '震荡', 'strong': '强势'}
                ax1.annotate(f'买入\n得分:{score:.2f}\n大盘:{regime_cn.get(regime, regime)}',
                           xy=(buy_date, buy_price), xytext=(15, 25), textcoords='offset points',
                           fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.4),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))

            if sell_price:
                ax1.scatter(sell_date, sell_price, marker='v', color='green', s=200, zorder=5,
                           edgecolors='darkgreen', linewidths=2,
                           label='卖出' if idx_t == trades_df.index[0] else "")
                reason = trade.get('reason', 'signal/time')
                reason_cn = {'stop_loss': '止损', 'trailing': '移动止损', 'signal_decay': '信号衰减',
                            'take_profit': '止盈', 'ai_bearish': 'AI看空', 'time_stop': '时间止损'}
                net_return = trade.get('net_return', 0) * 100
                ax1.annotate(f'卖出\n原因:{reason_cn.get(reason, reason)}\n收益:{net_return:.2f}%',
                           xy=(sell_date, sell_price), xytext=(15, -35), textcoords='offset points',
                           fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='green', alpha=0.4),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green'))

    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 子图2: 成交量
    ax2 = fig.add_subplot(gs[1])
    if 'Volume' in df.columns:
        ax2.bar(df.index, df['Volume'], alpha=0.5, color='gray', width=0.8)
    ax2.set_ylabel('成交量', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 子图3: 综合得分
    ax3 = fig.add_subplot(gs[2])
    if 'Combined_Score' in df.columns:
        score_series = df['Combined_Score']
        score_min, score_max = score_series.min(), score_series.max()
        buy_thresh = 0.5 + (score_max - 0.5) * 0.2
        sell_thresh = 0.5 - (0.5 - score_min) * 0.2
        ax3.plot(df.index, score_series, label='综合得分', color='purple', linewidth=1.5)
        ax3.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5)
        ax3.fill_between(df.index, buy_thresh, score_max, alpha=0.2, color='red', label='买入区')
        ax3.fill_between(df.index, score_min, sell_thresh, alpha=0.2, color='green', label='卖出区')
        ax3.set_ylim(score_min - 0.1, score_max + 0.1)
    ax3.set_ylabel('综合得分', fontsize=10)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 因子子图
    plot_idx = 3
    for col1, col2, title, color1, color2 in factor_subplots:
        ax = fig.add_subplot(gs[plot_idx])
        if col1 in df.columns:
            ax.plot(df.index, df[col1], label=col1, color=color1, linewidth=1)
            if col2 and col2 in df.columns:
                ax.plot(df.index, df[col2], label=col2, color=color2, linewidth=1, linestyle='--')

            if col1 == 'vol_price_res':
                ax.fill_between(df.index, 0.5, df[col1], where=(df[col1] > 0.5), alpha=0.3, color='red')
                ax.fill_between(df.index, 0.5, df[col1], where=(df[col1] < 0.5), alpha=0.3, color='green')
            elif col1 == 'bb_width':
                ax.fill_between(df.index, 0.5, df[col1], where=(df[col1] > 0.5), alpha=0.3, color='red')
                ax.fill_between(df.index, 0.5, df[col1], where=(df[col1] < 0.5), alpha=0.3, color='green')

        ax.set_ylabel(title, fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plot_idx += 1

    # Transformer 子图
    if 'transformer_prob' in df.columns:
        ax_t = fig.add_subplot(gs[plot_idx])
        ax_t.plot(df.index, df['transformer_prob'], label='AI预测概率', color='magenta', linewidth=1.5)
        ax_t.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax_t.fill_between(df.index, 0.5, df['transformer_prob'], where=(df['transformer_prob'] > 0.5), alpha=0.3, color='red')
        ax_t.fill_between(df.index, 0.5, df['transformer_prob'], where=(df['transformer_prob'] < 0.5), alpha=0.3, color='green')
        ax_t.set_ylabel('AI因子', fontsize=10)
        ax_t.set_xlabel('日期', fontsize=10)
        ax_t.legend(loc='upper left', fontsize=8)
        ax_t.grid(True, alpha=0.3)
        ax_t.set_ylim(0, 1)
        plot_idx += 1

    # 日期格式
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")
    plt.close()
    return fig
