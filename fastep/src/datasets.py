import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


class SequenceDataset(Dataset):
    """
    Dataset of sequences for sequence-to-one prediction tasks.

    Each sample is a sequence of features (states + actions) of length `seq_len`,
    and the target is the next partial state.

    Example for `seq_len=4`:
        X = [x_0, x_1, x_2, x_3]
        y = [y_4]

    Attributes:
        X (torch.Tensor): Input sequences of shape [N, seq_len, features].
        y (torch.Tensor): Targets of shape [N, target_features].
        seq_len (int): Length of input sequences.
        state_cols (list[str]): Column names corresponding to state features.
        action_cols (list[str]): Column names corresponding to action features.
        target_cols (list[str]): Column names corresponding to target features.
    """

    def __init__(self, df, seq_len: int):
        """
        Initialize the SequenceDataset.

        Args:
            df (polars.DataFrame): DataFrame containing the raw data.
            seq_len (int): Length of the input sequences.

        Raises:
            ValueError: If the DataFrame does not contain the expected columns
                        with prefixes 'STATE_', 'ACTION_', or 'NEXT_STATE_'.
        """
        self.seq_len = seq_len

        # Detect columns by prefix
        cols = {
            prefix: [c for c in df.columns if c.startswith(prefix)]
            for prefix in ["STATE_", "ACTION_", "NEXT_STATE_"]
        }

        if not all(cols.values()):
            raise ValueError(
                "DataFrame must contain columns with prefixes 'STATE_', 'ACTION_', and 'NEXT_STATE_'."
            )

        self.state_cols = cols["STATE_"]
        self.action_cols = cols["ACTION_"]
        self.target_cols = cols["NEXT_STATE_"]

        # Prepare input and target data
        data_X = df.select(self.state_cols + self.action_cols).to_numpy()
        data_y = df.select(self.target_cols).to_numpy()

        # Create sequences using sliding window
        seq_X = (
            sliding_window_view(data_X, (seq_len, data_X.shape[1]))
            .squeeze(axis=1)
            .copy()
        )
        seq_y = data_y[seq_len:].copy()

        # Fix length mismatch due to sliding window
        seq_X = seq_X[: len(seq_y)]

        # Convert to torch tensors
        self.X = torch.from_numpy(seq_X).float()  # [N, seq_len, features]
        self.y = torch.from_numpy(seq_y).float()  # [N, target_features]

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx: int):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Input sequence of shape [seq_len, features]
                - Target of shape [target_features]
        """
        return self.X[idx], self.y[idx]
