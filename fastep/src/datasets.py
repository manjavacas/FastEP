import torch

from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


class SequenceDataset(Dataset):
    '''
    Dataset of sequences. Each sample is a sequence of features (states + actions) of length `seq_len`, and the target is the next partial state.
    E.g. for seq_len=4:
        X = [x_0, x_1, x_2, x3],
        y = [y_4]
    Args:
        df: polars dataframe containing the data
        seq_len: length of the input sequences
    Outputs:
        self.X: tensor of shape [N, seq_len, features]
        self.y: tensor of shape [N, targets]
    '''

    def __init__(self, df, seq_len=4):
        '''
        Initialize the dataset.
        Args:
            df: polars dataframe containing the data
            seq_len: length of the input sequences
        '''
        self.seq_len = seq_len

        cols = {prefix: [c for c in df.columns if c.startswith(prefix)]
                for prefix in ['STATE_', 'ACTION_', 'NEXT_STATE_']}

        self.state_cols = cols['STATE_']
        self.action_cols = cols['ACTION_']
        self.target_cols = cols['NEXT_STATE_']

        data_X = df.select(self.state_cols + self.action_cols).to_numpy()
        data_y = df.select(self.target_cols).to_numpy()

        seq_X = sliding_window_view(
            data_X, (seq_len, data_X.shape[1])).squeeze(axis=1).copy()
        seq_y = data_y[seq_len:].copy()

        seq_X = seq_X[:len(seq_y)]  # fix length mismatch due to sliding window

        self.X = torch.from_numpy(seq_X).float()  # [N, seq_len, features]
        self.y = torch.from_numpy(seq_y).float()  # [N, targets]

    def __len__(self):
        '''
        Return the number of samples in the dataset.
        Returns:
            Length of the dataset.
        '''
        return len(self.X)

    def __getitem__(self, idx):
        '''
        Get a sample from the dataset.
        Args:
            idx: Index of the sample to retrieve.
        Returns:
            Tuple of (input sequence, target).
        '''
        return self.X[idx], self.y[idx]
