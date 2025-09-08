import numpy as np
import polars as pl

import torch
import torch.nn as nn

from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(__file__).parent.parent / 'data'
TRAIN_DATA_DIR = DATA_DIR / 'train'


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

        self.X = torch.from_numpy(seq_X).float()  # [N, seq_len, features]
        self.y = torch.from_numpy(seq_y).float()  # [N, targets]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():

    # Load datasets...
    df_names = [f.name for f in TRAIN_DATA_DIR.iterdir() if f.is_dir()]
    datasets = []
    for df_name in df_names:
        df = SequenceDataset(pl.read_csv(
            TRAIN_DATA_DIR / df_name / (df_name + '.csv')))
        datasets.append(df)

if __name__ == '__main__':
    main()
