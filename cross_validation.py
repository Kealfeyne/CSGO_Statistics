import numpy as np
import pandas as pd
import json

from knn_models.knn_model import KnnModel
from random_forest_models.random_forest_model import RandomForestModel
from catboost_models.catboost_model import CatBoostModel
from decision_tree_models.decision_tree_model import TreeModel


class CrossValidation:
    def __init__(self, data: pd.DataFrame, part_counts: int):
        self.part_counts = part_counts
        self.data = data
        self.test_size = data.shape[0] // part_counts

    def validate_model(self, model, experiment_id: str, normalize: bool, path_to_logs: str):
        logs = {}
        accuracies, f1_scores, recalls, precisions = [], [], [], []

        for part in range(self.part_counts):
            test_data = self.data.iloc[part * self.test_size: (part + 1) * self.test_size, :]
            train_data = self.data.drop(test_data.index)

            model_logs = model.analyse(train_data, test_data, experiment_id, normalize)
            logs[f"cross_val_{part}"] = model_logs

            accuracies.append(model_logs['validation_accuracy'])
            f1_scores.append(model_logs['validation_f1'])
            recalls.append(model_logs["validation_recall"])
            precisions.append(model_logs["validation_precision"])

        experiment_logs = {
            'cross_validation': logs,
            'accuracy': {
                'min': min(accuracies),
                'max': max(accuracies),
                'mean': np.mean(accuracies)
            },
            'f1_score': {
                'min': min(f1_scores),
                'max': max(f1_scores),
                'mean': np.mean(f1_scores)
            },
            'recall': {
                'min': min(recalls),
                'max': max(recalls),
                'mean': np.mean(recalls)
            },
            'precision': {
                'min': min(precisions),
                'max': max(precisions),
                'mean': np.mean(precisions)
            }
        }

        with open(
                f'{path_to_logs}/{experiment_id}_metrics.json',
                'w') as fp:
            json.dump(experiment_logs, fp)

        return experiment_logs


dataset_params_grid = [(1, 1, 1), (1, 5, 5), (1, 10, 10),
                       (2, 1, 1), (2, 5, 5), (2, 10, 10),
                       (5, 1, 1), (5, 5, 5), (5, 10, 10)]

# model_params_grid = [0.01, 0.05, 0.1, 0.2]
# model_params_grid = [2, 3, 5, 7, 10, 12, 15]
# model_params_grid = [10, 25, 100, 250]
model_params_grid = [2, 3, 5, 10]


for dataset_params in dataset_params_grid:
    print(f"Dataset params: {dataset_params}...")
    df = pd.read_csv(
        f"data/datasets_to_model/{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_wonans_dataset.csv",
        index_col=0)

    for model_params in model_params_grid:
        print(f"Model params: {model_params}...")

        cv = CrossValidation(df, 5)

        # cv.validate_model(model=KnnModel(n_neighbors=int(df.shape[0] * 0.8 * model_params)),
        #                   experiment_id=f'{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_wonans_knn_normalized_{model_params}',
        #                   normalize=True,
        #                   path_to_logs="knn_models")

        # cv.validate_model(model=TreeModel(max_depth=model_params),
        #                   experiment_id=f'{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_wonans_tree_{model_params}',
        #                   normalize=False,
        #                   path_to_logs="decision_tree_models")

        # cv.validate_model(model=RandomForestModel(n_estimators=model_params),
        #                   experiment_id=f'{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_wonans_rf_normalized_{model_params}',
        #                   normalize=True,
        #                   path_to_logs="random_forest_models")

        cv.validate_model(model=CatBoostModel(iterations=500, depth=model_params),
                          experiment_id=f'{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_wonans_catb_{model_params}',
                          normalize=False,
                          path_to_logs="catboost_models")
