import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier


class KnnModel:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def analyse(self, train_data, test_data, experiment_id):
        numeric_features = train_data.dtypes[
            (train_data.dtypes == np.float64) | (train_data.dtypes == np.int64)].index.tolist()
        print("Обучение...")
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        knn.fit(X=train_data[numeric_features].drop(['target', 'class_target'], axis=1),
                y=train_data['class_target'])

        preds = knn.predict(X=test_data[numeric_features].drop(['target', 'class_target'], axis=1))

        crossval_logs = {
            'validation_accuracy': accuracy_score(test_data['class_target'], preds),
            'validation_f1': f1_score(test_data['class_target'], preds),
            'validation_recall': recall_score(test_data['class_target'], preds),
            'validation_precision': precision_score(test_data['class_target'], preds)

        }

        print(crossval_logs)

        return crossval_logs
