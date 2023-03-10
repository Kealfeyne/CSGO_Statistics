import pandas as pd
import numpy as np
from tqdm import tqdm


class Dataset:
    def __init__(self, number_of_matches_to_filter: int = 100,
                 number_of_common_past_matches: int = 1,
                 number_of_past_tm1_matches: int = 2,
                 number_of_past_tm2_matches: int = 2):
        """
        Ининициирует экземпляр датасета
        :param number_of_matches_to_filter:
        :param number_of_common_past_matches:
        :param number_of_past_tm1_matches:
        :param number_of_past_tm2_matches:
        """
        # TODO: Добавить поля данных и параметров
        self.number_of_matches_to_filter = number_of_matches_to_filter
        self.number_of_common_past_matches = number_of_common_past_matches
        self.number_of_past_tm1_matches = number_of_past_tm1_matches
        self.number_of_past_tm2_matches = number_of_past_tm2_matches

    def structure_team_to_first_place(self, matches: pd.DataFrame, team_name: str) -> pd.DataFrame:
        """
        Меняет местами команды для определенности данных.
        :param matches:
        :param team_name:
        :return:
        """
        for i, r in matches.iterrows():  # Обязательно перепроверить как меняются местами команды
            if team_name == r["tm2"]:
                buffer = r[first_team_columns]
                r[first_team_columns] = r[second_team_columns].values
                r[second_team_columns] = buffer.values

            r["target"] = r["tm1pts"] / max(r["tm2pts"], 1)
            r["class_target"] = int(r["tm1pts"] >= r["tm2pts"])

        return matches


    def process_dataset(self, df: pd.DataFrame, drop_nans: bool = False):
        if drop_nans:
            df = df.dropna()
        # df = df.sort_values(by="date") # При сортировке может нарушаться дата

        tm1_value_counts = df["tm1"].value_counts()
        tm1_value_counts_filter = tm1_value_counts[tm1_value_counts >= self.number_of_matches_to_filter]
        df = df[df["tm1"].isin(tm1_value_counts_filter.index.tolist())].reset_index(drop=True)

        return df

    def build_dataset(self, df: pd.DataFrame, df_columns: np.array) -> pd.DataFrame:
        """
        Строит датасет по заданным количествам исторических матчей
        :param df:
        :param df_columns:
        :return:
        """
        features_names = np.array([])
        for i in range(self.number_of_common_past_matches + self.number_of_past_tm1_matches
                       + self.number_of_past_tm2_matches):
            features_names = np.concatenate((features_names, [f"{item_value}_{i}" for item_value in features_columns]))

        dataset_list = np.array([])

        for index, row in tqdm(df.iterrows()):
            tm1_name = row["tm1"]
            tm2_name = row["tm2"]
            # Все матчи, следующие после итерируемого
            past_matches = df.iloc[index + 1:]

            # Условие, что команды играли в порядке первая, вторая и вторая, первая
            is_first_second_played = (past_matches['tm1'] == tm1_name) & (past_matches['tm2'] == tm2_name)
            is_second_first_played = (past_matches['tm1'] == tm2_name) & (past_matches['tm2'] == tm1_name)
            # Общие последние между командами матчи
            common_past_matches = past_matches[is_first_second_played |
                                               is_second_first_played].iloc[:self.number_of_common_past_matches]

            past_matches = past_matches.drop(common_past_matches.index)

            # Рассматриваем прошлые матчи каждой из команд отдельно без учета совместных
            tm1_past_matches = past_matches[(past_matches['tm1'] == tm1_name) | (past_matches['tm2'] == tm1_name)].iloc[
                               :self.number_of_past_tm1_matches]
            tm2_past_matches = past_matches[(past_matches['tm1'] == tm2_name) | (past_matches['tm2'] == tm2_name)].iloc[
                               :self.number_of_past_tm2_matches]

            # Проверка на достаточность истории матчей
            if ((common_past_matches.shape[0] >= self.number_of_common_past_matches) &
                    (tm1_past_matches.shape[0] >= self.number_of_past_tm1_matches) &
                    (tm2_past_matches.shape[0] >= self.number_of_past_tm2_matches)):
                continue

            # Смена команд местами для упорядочивания
            structured_common_matches = self.structure_team_to_first_place(common_past_matches, tm1_name)
            structured_tm1_matches = self.structure_team_to_first_place(tm1_past_matches, tm1_name)
            structured_tm2_matches = self.structure_team_to_first_place(tm2_past_matches, tm2_name)

            # Строим наблюдение
            observation = np.array([])
            observation_features = np.array([])

            for i, r in pd.concat([structured_common_matches, structured_tm1_matches, structured_tm2_matches]) \
                    .reset_index(drop=True):
                observation = np.concatenate((observation, r[features_columns].values))
                observation_features = np.concatenate((observation_features, features_names))

            target_value = row["tm1pts"] / max(row["tm2pts"], 1)
            observation = np.concatenate((observation, np.array([target_value])))

            if len(features_names) == len(features_columns) + 1:
                dataset_list = np.concatenate((dataset_list, observation))

        return pd.DataFrame(dataset_list, columns=df_columns)


match_columns = ['date', 'map']

first_team_columns = ['tm1', 'tm1pts', 'tm1side1pts', 'tm1side2pts', 'tm1rt', 'tm1fk', 'tm1clwon',
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

second_team_columns = ['tm2', 'tm2pts', 'tm2side1pts', 'tm2side2pts', 'tm2rt', 'tm2fk', 'tm2clwon',
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

features_columns = np.concatenate((np.array(['map']), np.concatenate(
    (np.array(first_team_columns), np.array(second_team_columns), np.array(['target'])))))

# TODO: Обернуть в экземпляр класса

preprocessed_dataset = pd.read_csv("data/preprocessed_dataset.csv", header=0, index_col=0, parse_dates=['date'])
processed_dataset = process_dataset(preprocessed_dataset, drop_nans=True)
dataset = build_dataset(processed_dataset,
                        number_of_common_past_matches=,
                        number_of_past_tm1_matches=,
                        number_of_past_tm2_matches=)


dataset.to_csv("data/dataset_")
