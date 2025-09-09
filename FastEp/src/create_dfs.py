from pathlib import Path

import numpy as np

import polars as pl
import polars.selectors as ps

from constants import STATE_VARIABLES, ACTION_VARIABLES, TARGET_VARIABLES

DATASET_FILENAME = 'observations.csv'


def load_data(df_dir):
    '''
    Loads and merges (state_t, action_t, state_{t+1}) transitions into a single dataframe.
    Args:
        df_dir (Path): Directory containing the dataset file.
    Returns:
        pl.DataFrame: Merged dataframe with states, actions, and next states.
    '''
    df = pl.read_csv(
        df_dir / DATASET_FILENAME)[:-1]  # removes duplicated last row

    # States (t)
    df_states = df.select(STATE_VARIABLES)[
        :-1].rename(lambda colname: 'STATE_' + colname)

    # Actions (t) (shifted by one timestep)
    df_actions = df.select(ACTION_VARIABLES)[1:].rename(
        lambda colname: 'ACTION_' + colname)

    # States (t+1) (shifted by one timestep)
    df_next_states = df.select(TARGET_VARIABLES)[1:].rename(
        lambda colname: 'NEXT_STATE_' + colname)

    df = df_states.hstack(df_actions).hstack(df_next_states)

    return df


def main():
    '''
    Main function to process all `observation.csv` datasets and save them as CSV files.
    '''
    data_dir_train = Path(__file__).parent.parent / 'data' / 'train'

    for df_dir in data_dir_train.iterdir():
        if df_dir.is_dir() and df_dir.name.startswith('df_'):
            df = load_data(df_dir)
            df.write_csv(df_dir / (df_dir.name + '.csv'))


if __name__ == '__main__':
    main()
