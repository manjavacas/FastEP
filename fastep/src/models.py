import torch.nn as nn


class LSTMDynamics(nn.Module):
    """
    LSTM surrogate model.
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        """
        Initializes the model with the given architecture parameters.

        Args:
            input_size (int): Dimensionality of the input features.
            output_size (int): Dimensionality of the output (number of classes or target features).
            hidden_size (int): Number of hidden units in each LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability applied between LSTM layers (0.0 means no dropout).
        """

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)  # out: [batch, seq_len, hidden_size]
        last_out = out[:, -1, :]  # [batch, hidden_size]
        return self.output_linear(last_out)  # [batch, output_size]
