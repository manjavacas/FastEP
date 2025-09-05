from pathlib import Path
import argparse

import polars as pl
import polars.selectors as ps

parser = argparse.ArgumentParser(
    description='Create a dataframe from data files in a specified folder.')
parser.add_argument('input_folder', type=str,
                    help='Path to the input folder containing data files.')
parser.add_argument('--output_file', '-o', type=str,
                    help='Output file name', default='df.csv')
args = parser.parse_args()

df_dir = Path(__file__).parent / args.input_folder

df_act = pl.read_csv(df_dir / 'simulated_actions.csv')[:-1]
df_obs = pl.read_csv(df_dir / 'observations.csv')[:-2]
df_next_obs = pl.read_csv(df_dir / 'observations.csv').select(
    ps.starts_with(['air_temperature_', 'heat_source_electricity_rate']))[1:-1]

print(df_obs.shape) # (61055, 45)
print(df_act.shape) # (61055, 6)
print(df_next_obs.shape) # (61055, 6)

df_x = df_obs.hstack(df_act)
df_y = df_next_obs.rename(lambda colname: 'next_' + colname)

df = df_x.hstack(df_y)
print(df)