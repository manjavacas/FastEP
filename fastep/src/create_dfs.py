from pathlib import Path
import polars as pl

from variables import STATE_VARIABLES, ACTION_VARIABLES, TARGET_VARIABLES

DATASET_FILENAME = "observations.csv"


def load_data(df_dir: Path) -> pl.DataFrame:
    """
    Load and merge (state_t, action_t, state_{t+1}) transitions into a single DataFrame.

    This function reads the dataset CSV, then constructs:
        - STATE_: features at time t
        - ACTION_: actions at time t
        - NEXT_STATE_: features at time t+1

    Args:
        df_dir (Path): Directory containing the dataset CSV file.

    Returns:
        pl.DataFrame: Merged dataframe with columns prefixed by 'STATE_', 'ACTION_', 
                      and 'NEXT_STATE_' ready for sequence dataset creation.

    Raises:
        FileNotFoundError: If the dataset CSV file does not exist in `df_dir`.
    """
    csv_path = df_dir / DATASET_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pl.read_csv(csv_path)

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
    Process all `observation.csv` datasets in the training directory and save them as preprocessed CSV files.

    For each subdirectory in the training data directory whose name starts with 'df_', 
    this function loads the raw observation CSV, merges state and action transitions, 
    and writes the resulting dataframe back as `<directory_name>.csv`.
    """
    data_dir_train = Path(__file__).parent.parent / "data" / "train"

    for df_dir in data_dir_train.iterdir():
        if df_dir.is_dir() and df_dir.name.startswith("df_"):
            df = load_data(df_dir)
            df.write_csv(df_dir / (df_dir.name + ".csv"))


if __name__ == "__main__":
    main()
