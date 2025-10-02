from pathlib import Path
import polars as pl

from variables import STATE_VARIABLES, ACTION_VARIABLES, TARGET_VARIABLES

DATASET_FILENAME = "observations.csv"
DATASET_PARQUET_FILENAME = "observations.parquet"


def load_data(df_dir: Path) -> pl.DataFrame:
    """
    Load and merge (state_t, action_t, state_{t+1}) transitions into a single DataFrame.

    This function reads the dataset from CSV or Parquet, then constructs:
        - STATE_: features at time t
        - ACTION_: actions at time t
        - NEXT_STATE_: features at time t+1

    Args:
        df_dir (Path): Directory containing the dataset file.

    Returns:
        pl.DataFrame: Merged dataframe with columns prefixed by 'STATE_', 'ACTION_',
                      and 'NEXT_STATE_' ready for sequence dataset creation.

    Raises:
        FileNotFoundError: If neither the dataset CSV nor Parquet file exists in `df_dir`.
    """
    parquet_path = df_dir / DATASET_PARQUET_FILENAME
    csv_path = df_dir / DATASET_FILENAME

    if parquet_path.exists():
        df = pl.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pl.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Dataset file not found: {csv_path} or {parquet_path}")

    # States at time t
    df_states = df.select(STATE_VARIABLES)[:-1].rename(
        lambda colname: "STATE_" + colname
    )

    # Actions at time t (shifted by one timestep)
    df_actions = df.select(ACTION_VARIABLES)[1:].rename(
        lambda colname: "ACTION_" + colname
    )

    # States at time t+1 (shifted by one timestep)
    df_next_states = df.select(TARGET_VARIABLES)[1:].rename(
        lambda colname: "NEXT_STATE_" + colname
    )

    # Merge all into a single dataframe
    df_merged = df_states.hstack(df_actions).hstack(df_next_states)

    return df_merged


def main():
    """
    Process all observation datasets in the training directory and save them as preprocessed Parquet files.

    For each subdirectory in the training data directory whose name starts with 'df_',
    this function loads the raw observation data, merges state and action transitions,
    and writes the resulting dataframe as `<directory_name>.parquet`.
    """
    data_dir_train = Path(__file__).parent.parent / "data" / "train"

    for df_dir in data_dir_train.iterdir():
        if df_dir.is_dir() and df_dir.name.startswith("df_"):
            df = load_data(df_dir)
            parquet_path = df_dir / (df_dir.name + ".parquet")
            df.write_parquet(parquet_path)
            print(f"Saved {parquet_path}")


if __name__ == "__main__":
    main()
