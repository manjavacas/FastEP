import torch
import torch.nn as nn


class LSTMDynamics(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)  # out: [batch, seq_len, hidden_size]
        last_out = out[:, -1, :]  # [batch, hidden_size]
        return self.output_linear(last_out)  # [batch, output_size]
