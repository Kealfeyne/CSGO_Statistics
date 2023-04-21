import pandas as pd
import numpy as np
from tqdm import tqdm

general_columns = ['map', 'tm1', 'tm2', 'target', 'class_target']

first_team_columns = ['tm1rt', 'tm1fk', 'tm1clwon',
                      'pl1nm', 'pl1kills', 'pl1asts', 'pl1deaths', 'pl1kast', 'pl1diffk-d', 'pl1adr', 'pl1fkdiff',
                      'pl1hs', 'pl1flasts', 'pl1rt',
                      'pl2nm', 'pl2kills', 'pl2asts', 'pl2deaths', 'pl2kast', 'pl2diffk-d', 'pl2adr', 'pl2fkdiff',
                      'pl2hs', 'pl2flasts', 'pl2rt',
                      'pl3nm', 'pl3kills', 'pl3asts', 'pl3deaths', 'pl3kast', 'pl3diffk-d', 'pl3adr', 'pl3fkdiff',
                      'pl3hs', 'pl3flasts', 'pl3rt',
                      'pl4nm', 'pl4kills', 'pl4asts', 'pl4deaths', 'pl4kast', 'pl4diffk-d', 'pl4adr', 'pl4fkdiff',
                      'pl4hs', 'pl4flasts', 'pl4rt',
                      'pl5nm', 'pl5kills', 'pl5asts', 'pl5deaths', 'pl5kast', 'pl5diffk-d', 'pl5adr', 'pl5fkdiff',
                      'pl5hs', 'pl5flasts', 'pl5rt']

second_team_columns = ['tm2rt', 'tm2fk', 'tm2clwon',
                       'pl6nm', 'pl6kills', 'pl6asts', 'pl6deaths', 'pl6kast', 'pl6diffk-d', 'pl6adr', 'pl6fkdiff',
                       'pl6hs', 'pl6flasts', 'pl6rt',
                       'pl7nm', 'pl7kills', 'pl7asts', 'pl7deaths', 'pl7kast', 'pl7diffk-d', 'pl7adr', 'pl7fkdiff',
                       'pl7hs', 'pl7flasts', 'pl7rt',
                       'pl8nm', 'pl8kills', 'pl8asts', 'pl8deaths', 'pl8kast', 'pl8diffk-d', 'pl8adr', 'pl8fkdiff',
                       'pl8hs', 'pl8flasts', 'pl8rt',
                       'pl9nm', 'pl9kills', 'pl9asts', 'pl9deaths', 'pl9kast', 'pl9diffk-d', 'pl9adr', 'pl9fkdiff',
                       'pl9hs', 'pl9flasts', 'pl9rt',
                       'pl10nm', 'pl10kills', 'pl10asts', 'pl10deaths', 'pl10kast', 'pl10diffk-d', 'pl10adr',
                       'pl10fkdiff', 'pl10hs', 'pl10flasts', 'pl10rt']

new_columns = ['map', 'tm1', 'tm2', 'target', 'class_target', 'tmsrt', 'tmsfk', 'tmsclwon',
               'leaders1kills', 'leaders1asts', 'leaders1deaths', 'leaders1kast', 'leaders1diffk-d', 'leaders1adr',
               'leaders1fkdiff', 'leaders1hs', 'leaders1flasts', 'leaders1rt',
               'leaders2kills', 'leaders2asts', 'leaders2deaths', 'leaders2kast', 'leaders2diffk-d', 'leaders2adr',
               'leaders2fkdiff', 'leaders2hs', 'leaders2flasts', 'leaders2rt',
               'leaders3kills', 'leaders3asts', 'leaders3deaths', 'leaders3kast', 'leaders3diffk-d', 'leaders3adr',
               'leaders3fkdiff', 'leaders3hs', 'leaders3flasts', 'leaders3rt',
               'leaders4kills', 'leaders4asts', 'leaders4deaths', 'leaders4kast', 'leaders4diffk-d', 'leaders4adr',
               'leaders4fkdiff', 'leaders4hs', 'leaders4flasts', 'leaders4rt',
               'leaders5kills', 'leaders5asts', 'leaders5deaths', 'leaders5kast', 'leaders5diffk-d', 'leaders5adr',
               'leaders5fkdiff', 'leaders5hs', 'leaders5flasts', 'leaders5rt']

datasets_params_grid = [(1, 1, 1), (1, 5, 5), (1, 10, 10),
                        (2, 1, 1), (2, 5, 5), (2, 10, 10),
                        (5, 1, 1), (5, 5, 5), (5, 10, 10)]
tag = ""

for common, tm1, tm2 in datasets_params_grid:
    print(f"Выделяется {common}, {tm1}, {tm2}...")
    processed_dataset = pd.read_csv(f"data/datasets_to_model/{common}_{tm1}_{tm2}_{tag}_dataset.csv",
                                    header=0, index_col=0)

    observations = []

    for index, row in processed_dataset.iterrows():

        observation = np.array([])

        for i in range(common + tm1 + tm2):

            other_columns = [f"{x}_{i}" for x in general_columns]
            tm1_columns = [f"{x}_{i}" for x in first_team_columns]
            tm2_columns = [f"{x}_{i}" for x in second_team_columns]

            extracted_features = np.concatenate([row[other_columns], row[tm1_columns]/row[tm2_columns]])
            observation = np.concatenate([observation, extracted_features])

        observations.append(observation)

    observation_features = []

    for i in range(common + tm1 + tm2):
        extracted_columns = [f"{x}_{i}" for x in new_columns]
        observation_features.extend(extracted_columns)

    extracted_df = pd.DataFrame(observations, columns=observation_features)

    extracted_df.to_csv(f"data/datasets_to_model/{common}_{tm1}_{tm2}_{tag}_extracted.csv")
