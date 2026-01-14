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

class LSTM(nn.Module):
    def __init__(self, hparams: Hparams):
        super(LSTM, self).__init__()
        self.hidden_size = hparams.hidden_size
        self.lstm = nn.LSTM(hparams.input_size, hparams.hidden_size, batch_first=True)
        self.fc = nn.Linear(hparams.hidden_size, hparams.output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
