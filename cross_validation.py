import numpy as np
import pandas as pd
import json


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
            train_data = self.data.drop(test_data)

            model_logs = model.analyse(train_data, test_data, f"{part}_{experiment_id}")
            logs[f"cross_val_{part}"] = model_logs

            accuracies.append(model_logs['validation_accuracy'])
            f1_scores.append(model_logs['validation_f1score'])
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
