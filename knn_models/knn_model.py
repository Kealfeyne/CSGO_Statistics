import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


class KnnModel:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def analyse(self, train_data, test_data, experiment_id, normalize):
        numeric_features = train_data.dtypes[
            (train_data.dtypes == np.float64) | (train_data.dtypes == np.int64)].index.tolist()
        print(f"kNN, {self.n_neighbors} neighbours, Обучение...")
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        if normalize:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(train_data[numeric_features].drop(["target", "class_target"], axis=1))
            X_test = scaler.transform(test_data[numeric_features].drop(["target", "class_target"], axis=1))
        else:
            X_train = train_data[numeric_features].drop(["target", "class_target"], axis=1)
            X_test = test_data[numeric_features].drop(["target", "class_target"], axis=1)

        y_train = train_data["class_target"]
        y_test = test_data["class_target"]

        knn.fit(X=X_train,
                y=y_train)

        preds = knn.predict(X=X_test)

        print(np.unique(np.array(preds), return_counts=True))

        crossval_logs = {
            'validation_accuracy': accuracy_score(y_test, preds),
            'validation_f1': f1_score(y_test, preds, zero_division=0),
            'validation_recall': recall_score(y_test, preds, zero_division=0),
            'validation_precision': precision_score(y_test, preds, zero_division=0)

        }

        print(crossval_logs)

        return crossval_logs, preds
