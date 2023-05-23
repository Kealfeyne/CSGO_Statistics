import pandas as pd
import numpy as np


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("-", np.NaN)

    # pl#kills
    for index in range(1, 11):
        kills_column = f"pl{index}kills"
        headshots_column = f"pl{index}hs"
        df[kills_column] = df[kills_column].str.replace("(", "", regex=True)
        df[kills_column] = df[kills_column].str.replace(")", "", regex=True)
        df[[kills_column, headshots_column]] = df[kills_column].str.split(expand=True)

    # pl#asts
    for index in range(1, 11):
        assists_column = f"pl{index}asts"
        flash_assists_column = f"pl{index}flasts"
        df[assists_column] = df[assists_column].str.replace("(", "", regex=True)
        df[assists_column] = df[assists_column].str.replace(")", "", regex=True)
        df[[assists_column, flash_assists_column]] = df[assists_column].str.split(expand=True)

    # pl#kast
    for index in range(1, 11):
        kast_column = f"pl{index}kast"
        df[kast_column] = df[kast_column].str.replace("%", "", regex=True)

    # typing
    date_column = ['date']
    string_columns = ['tm1', 'tm2', 'map', 'pl1nm', 'pl2nm', 'pl3nm', 'pl4nm', 'pl5nm',
                      'pl6nm', 'pl7nm', 'pl8nm', 'pl9nm', 'pl10nm']
    float_columns = list(set(df.columns) - set(string_columns) - set(date_column))

    df[float_columns] = df[float_columns].astype(float)
    df[string_columns] = df[string_columns].astype(str)

    return df


dataset = pd.read_csv("data/dataset.csv", header=0, index_col=0, parse_dates=['date'])

preprocess_dataset(dataset).to_csv("data/preprocessed_dataset.csv", index=True)
