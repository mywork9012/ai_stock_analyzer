# models/lstm_model.py
"""
LSTM模型定义 (V1.0 Path B 轻量版)
- 单层LSTM，64个隐藏单元
- Dropout=0.3
- Sigmoid输出：次日上涨概率
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    单层LSTM分类器

    Args:
        input_size:  输入特征数（因子数量）
        hidden_size: 隐藏单元数，默认64
        num_layers:  LSTM层数，默认1
        dropout:     Dropout概率，默认0.3
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                   # 输入格式: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout（应用在LSTM输出上）
        self.dropout = nn.Dropout(p=dropout)

        # BatchNorm（提升训练稳定性）
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，shape = (batch_size, seq_len, input_size)

        Returns:
            输出张量，shape = (batch_size, 1)，值域 [0,1]（sigmoid激活后）
        """
        # LSTM前向传播
        # out shape: (batch_size, seq_len, hidden_size)
        out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        last_out = out[:, -1, :]           # shape: (batch_size, hidden_size)

        # Dropout
        last_out = self.dropout(last_out)

        # BatchNorm
        last_out = self.batch_norm(last_out)

        # 全连接 + Sigmoid
        logits = self.fc(last_out)         # shape: (batch_size, 1)
        return torch.sigmoid(logits)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """推理模式（无梯度计算）"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_model_info(self) -> dict:
        """返回模型基本信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "type": "LSTM (Path B - Lightweight)",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
