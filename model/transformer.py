# -*- coding: utf-8 -*-
"""
StockTransformer 模型定义
包含 ResidualConv1d、EfficientAttention、OptimizedTransformerLayer、StockTransformer
从 TransformerStock.py 中提取，保持完全一致的架构
"""
import torch
import torch.nn as nn


class ResidualConv1d(nn.Module):
    """残差卷积块：深度可分离卷积 + 1x1卷积 + 残差连接"""

    def __init__(self, in_channels, out_channels, dropout=0.4):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            self.depthwise_conv,
            nn.SiLU(),
            nn.BatchNorm1d(in_channels),
            self.pointwise_conv,
            nn.SiLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class EfficientAttention(nn.Module):
    """高效注意力层：使用 PyTorch fused SDPA + 外部 Dropout 控制"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Keep attention deterministic in eval/inference, while preserving MC dropout.
        dropout_p = 0.1 if self.attn_dropout.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=False
        )
        attn_output = self.attn_dropout(attn_output)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn_output)


class OptimizedTransformerLayer(nn.Module):
    """优化的 Transformer 层：Pre-Norm + 多头注意力 + MLP"""

    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StockTransformer(nn.Module):
    """股票预测 Transformer 模型

    多任务输出：
    - logits: 4分类预测（大涨/涨/跌/大跌）
    - ret_pred: 收益率回归预测

    支持 MC Dropout 不确定性估计
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        num_layers: int = 4,
        lookback_days: int = 120,
        num_classes: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lookback_days = lookback_days

        self.conv_block = ResidualConv1d(input_dim, dim_feedforward, dropout=dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, lookback_days, dim_feedforward))

        self.transformer_layers = nn.ModuleList([
            OptimizedTransformerLayer(dim_feedforward, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.query = nn.Parameter(torch.randn(1, 1, dim_feedforward))
        self.pool_attn = nn.MultiheadAttention(dim_feedforward, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_feedforward)
        self.dropout = nn.Dropout(0.2)

        self.shared = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 2),
            nn.BatchNorm1d(dim_feedforward // 2),
        )

        self.head_cls = nn.Linear(dim_feedforward // 2, num_classes)
        self.head_ret = nn.Linear(dim_feedforward // 2, 1)

    def forward(self, x):
        # x: [B, lookback_days, input_dim]
        x = x.permute(0, 2, 1)  # [B, input_dim, lookback_days]
        x = self.conv_block(x)  # [B, dim_feedforward, lookback_days]
        x = x.permute(0, 2, 1)  # [B, lookback_days, dim_feedforward]

        x = x + self.pos_encoder
        x = self.norm(x)
        x = self.dropout(x)

        for layer in self.transformer_layers:
            x = layer(x)

        # Attention Pooling
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, x, x)
        feat = pooled.squeeze(1)

        feat = self.shared(feat)
        logits = self.head_cls(feat)
        ret_pred = self.head_ret(feat)

        return logits, ret_pred

    @torch.no_grad()
    def mc_predict(self, x, n_forward=10):
        """MC Dropout 预测

        Returns:
            mean_probs: 平均类别概率 [B, num_classes]
            probs_uncertainty: 概率标准差 [B, num_classes]
            mean_ret: 平均预测收益 [B, 1]
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        all_probs = []
        all_rets = []
        for _ in range(n_forward):
            logits, ret_pred = self(x)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs)
            all_rets.append(ret_pred)

        mean_probs = torch.stack(all_probs).mean(dim=0)
        probs_uncertainty = torch.stack(all_probs).std(dim=0)
        mean_ret = torch.stack(all_rets).mean(dim=0)

        self.eval()
        return mean_probs, probs_uncertainty, mean_ret
