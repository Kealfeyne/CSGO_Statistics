import numpy as np
import pandas as pd
import json

from catboost_models.catboost_model import CatBoostModel
from knn_models.knn_model import KnnModel


class CrossValidation:
    def __init__(self, data: pd.DataFrame, part_counts: int):
        self.part_counts = part_counts
        self.data = data
        self.test_size = data.shape[0] // part_counts

    def validate_model(self, model, experiment_id: str):
        logs = {}
        accuracies, f1_scores, recalls, precisions = [], [], [], []

        for part in range(self.part_counts):
            test_data = self.data.iloc[part * self.test_size: (part + 1) * self.test_size, :]
            train_data = self.data.drop(test_data.index)

            model_logs = model.analyse(train_data, test_data, experiment_id)
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
                f'cross_validation_evaluating/{experiment_id}_metrics.json',
                'w') as fp:
            json.dump(experiment_logs, fp)

        return experiment_logs


df = pd.read_csv(f"data/datasets_to_model/{1}_{1}_{1}_wonans_dataset.csv",
                 index_col=0)

experiment_id = '1_1_1_wonans_catb_2'

cv = CrossValidation(df, 5)

cv.validate_model(model=CatBoostModel(iterations=500, depth=2),
                  experiment_id=experiment_id)

# cv.validate_model(model=KnnModel(n_neighbors=int(df.shape[0] * 0.8 * neighbours_part)),
#                   experiment_id=f'5_10_10_wonans_knn_{neighbours_part}')
