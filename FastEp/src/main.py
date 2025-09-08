import numpy as np
import polars as pl
import torch

from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(__file__).parent.parent / 'data'
TRAIN_DATA_DIR = DATA_DIR / 'train'


class SequenceLoader(Dataset):
    '''
    Creates a dataset of sequences. Each sample is a sequence of features (states + actions) of length `seq_len`, and the target is the next partial state. 
    E.g. for seq_len=4:
        X = [x_0, x_1, x_2, x3],
        y = [y_4]
    Args:
        df: polars dataframe containing the data
        state_cols: list of column names for the state variables
        target_cols: list of column names for the target variables
        seq_len: length of the input sequences
    Outputs:
        self.X: tensor of shape [N, seq_len, features]
        self.y: tensor of shape [N, targets]
    '''

    def __init__(self, df, state_cols, action_cols, target_cols, seq_len=4):
        self.seq_len = seq_len

        data_X = df.select(state_cols+action_cols).to_numpy()
        data_y = df.select(target_cols).to_numpy()

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

    df = pl.read_csv(TRAIN_DATA_DIR / 'df_1' / 'df_1.csv')

    cols = {prefix: [c for c in df.columns if c.startswith(prefix)]
            for prefix in ["STATE_", "ACTION_", "NEXT_STATE_"]}

    dataloader = SequenceLoader(
        df, cols["STATE_"], cols["ACTION_"], cols["NEXT_STATE_"])

    print(dataloader[:2])

if __name__ == "__main__":
    main()
