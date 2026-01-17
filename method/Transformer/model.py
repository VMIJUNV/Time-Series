import sys
sys.path.append(".")

import torch
import torch.nn as nn
from dataclasses import dataclass

from utils.more_utils import BaseDataclass


@dataclass
class Hparams(BaseDataclass):
    input_size: int = 9
    hidden_size: int = 1024
    output_size: int = 1
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1


import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, hparams: Hparams):
        super().__init__()

        self.hidden_size = hparams.hidden_size

        # 输入映射到 d_model
        self.input_proj = nn.Linear(
            hparams.input_size,
            hparams.hidden_size
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hparams.hidden_size,
            nhead=hparams.num_heads,
            dropout=hparams.dropout,
            batch_first=True  # 关键：保持 (B, T, C)
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=hparams.num_layers
        )

        # 输出层
        self.fc = nn.Linear(
            hparams.hidden_size,
            hparams.output_size
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """

        # (B, T, input_size) -> (B, T, hidden_size)
        x = self.input_proj(x)

        # Transformer Encoder
        out = self.encoder(x)

        # 取最后一个时间步（和 LSTM 一样）
        out = out[:, -1:, :]  # (B, hidden_size)

        out = self.fc(out)   # (B, output_size)
        return out
