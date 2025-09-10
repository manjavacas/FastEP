import polars as pl
import numpy as np

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import SequenceDataset
from models import LSTMDynamics

from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / 'data'
TRAIN_DATA_DIR = DATA_DIR / 'train'

BATCH_SIZE = 32
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_datasets(data_dir):
    '''
    Load datasets from the specified directory. Each subdirectory is expected to contain a CSV file with the same name as the 
    directory. The first dataset is used to fit a StandardScaler for normalization, which is then applied to all datasets.
    Args:
        data_dir: Path to the directory containing dataset subdirectories.
    Returns:
        List of SequenceDataset objects.
        StandardScaler fitted on the first dataset.
    '''
    datasets, scaler = [], None
    for folder in data_dir.iterdir():
        if folder.is_dir():
            csv_file = folder / f'{folder.name}.csv'
            df = pl.read_csv(csv_file)
            df, scaler = preprocess_dataframe(df, scaler)
            datasets.append(SequenceDataset(df))
    return datasets, scaler


def preprocess_dataframe(df, scaler=None):
    '''
    Preprocess the dataframe by normalizing all numeric columns and applying
    cyclic encoding for 'month', 'hour', 'day_of_month' if they exist.

    Args:
        df: Input Polars DataFrame.
        scaler: Optional StandardScaler object. If None, a new scaler is fitted.

    Returns:
        Tuple of (normalized DataFrame, fitted scaler). If scaler was provided, it is returned unchanged.
    '''
    numeric_cols = [c for c in df.columns if df[c].dtype in (
        pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
    data = df.select(numeric_cols).to_numpy()

    # Normalization
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    df = df.with_columns([pl.Series(col, data_scaled[:, i])
                         for i, col in enumerate(numeric_cols)])

    # Cyclic encoding for time features
    for col, period in {'month': 12, 'hour': 24, 'day_of_month': 31}.items():
        if col in df.columns:
            vals = df[col].to_numpy()
            df = df.with_columns([
                pl.Series(f'{col}_sin', np.sin(2*np.pi*vals/period)),
                pl.Series(f'{col}_cos', np.cos(2*np.pi*vals/period))
            ]).drop(col)
    return df, scaler


def train_one_epoch(model, loaders, criterion, optimizer, device):
    '''
    Train the model for one epoch over multiple data loaders.
    Each loader corresponds to a different dataset. 
    Args:
        model: The neural network model to train.
        loaders: List of DataLoader objects for different datasets.
        criterion: Loss function.
        optimizer: Optimizer for model parameters.
        device: Device to run the training on (CPU or GPU).
    Returns:
        Average loss over all datasets.
    '''
    model.train()
    total_loss = 0
    for loader in loaders:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loaders)


def main():
    '''
    Main training loop to load data, initialize model, and train.
    '''
    # Load datasets and create loaders
    datasets, _ = load_datasets(TRAIN_DATA_DIR)
    loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
               for ds in datasets]

    # Define model, optimizer, criterion
    input_size = datasets[0].X.shape[2]
    output_size = datasets[0].y.shape[1]
    model = LSTMDynamics(input_size, output_size,
                         hidden_size=128, num_layers=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = train_one_epoch(
            model, loaders, criterion, optimizer, DEVICE)
        print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}')

    torch.save(model.state_dict(), 'trained_model.pth')


if __name__ == '__main__':
    main()
