import polars as pl
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import SequenceDataset
from models import LSTMDynamics

from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_datasets(data_dir: Path, seq_len: int):
    """
    Load all datasets from the specified directory and normalize them using StandardScaler.

    The first dataset is used to fit the scaler, which is then applied to all subsequent datasets.
    Each subdirectory should contain a Parquet file with the same name as the directory.

    Args:
        data_dir (Path): Path to the directory containing dataset subdirectories.
        seq_len (int): Length of input sequences for the SequenceDataset.

    Returns:
        tuple:
            - List[SequenceDataset]: List of datasets ready for training.
            - StandardScaler: Fitted scaler used for normalization.
    """
    datasets, scaler = [], None
    for folder in data_dir.iterdir():
        if folder.is_dir():
            parquet_file = folder / f"{folder.name}.parquet"
            # Fallback to CSV if Parquet doesn't exist
            if parquet_file.exists():
                df = pl.read_parquet(parquet_file)
            else:
                csv_file = folder / f"{folder.name}.csv"
                df = pl.read_csv(csv_file)
            df, scaler = preprocess_dataframe(df, scaler)
            datasets.append(SequenceDataset(df, seq_len))
    return datasets, scaler


def preprocess_dataframe(df: pl.DataFrame, scaler: StandardScaler = None):
    """
    Preprocess a DataFrame by normalizing numeric columns and applying cyclic encoding
    to time-related features ('month', 'hour', 'day_of_month') if present.

    Args:
        df (pl.DataFrame): Input dataframe to preprocess.
        scaler (StandardScaler, optional): Pre-fitted scaler. If None, a new one is fitted.

    Returns:
        tuple:
            - pl.DataFrame: Normalized and processed dataframe.
            - StandardScaler: Fitted scaler.
    """
    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    data = df.select(numeric_cols).to_numpy()

    # Normalization
    if scaler is None:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = scaler.transform(data)

    df = df.with_columns(
        [pl.Series(col, data_scaled[:, i]) for i, col in enumerate(numeric_cols)]
    )

    # Cyclic encoding for time features
    for col, period in {"month": 12, "hour": 24, "day_of_month": 31}.items():
        if col in df.columns:
            vals = df[col].to_numpy()
            df = df.with_columns(
                [
                    pl.Series(f"{col}_sin", np.sin(2 * np.pi * vals / period)),
                    pl.Series(f"{col}_cos", np.cos(2 * np.pi * vals / period)),
                ]
            ).drop(col)
    return df, scaler


def train_one_epoch(
    model: nn.Module, loaders: list[DataLoader], criterion, optimizer, device
):
    """
    Train the model for a single epoch over multiple datasets.

    Args:
        model (nn.Module): Neural network model to train.
        loaders (list[DataLoader]): List of DataLoader objects for different datasets.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
        float: Average loss over all datasets for this epoch.
    """
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


def train(
    batch_size: int,
    seq_len: int,
    num_epochs: int,
    learning_rate: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    model_save_path: str = "trained_model.pth",
):
    """
    Main training loop: load datasets, initialize model, and train for the specified number of epochs.

    Args:
        batch_size (int): Number of samples per batch during training.
        seq_len (int): Length of input sequences for the model.
        num_epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
        hidden_size (int): Number of hidden units in each LSTM layer.
        num_layers (int): Number of stacked LSTM layers in the model.
        dropout (float): Dropout probability applied between LSTM layers (0.0 means no dropout).
        model_save_path (str, optional): Path to save the trained model. Defaults to 'trained_model.pth'.

    Returns:
        None
    """
    # Load datasets and create loaders
    datasets, _ = load_datasets(TRAIN_DATA_DIR, seq_len)
    loaders = [DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in datasets]

    # Define model, optimizer, criterion
    input_size = datasets[0].X.shape[2]
    output_size = datasets[0].y.shape[1]
    model = LSTMDynamics(input_size, output_size, hidden_size, num_layers, dropout).to(
        DEVICE
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, loaders, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
