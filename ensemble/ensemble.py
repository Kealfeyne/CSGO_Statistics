import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from knn_models.knn_model import KnnModel
from random_forest_models.random_forest_model import RandomForestModel
from catboost_models.catboost_model import CatBoostModel


class Ensemble:

    def analyse(self, train_data, test_data, experiment_id, normalize):
        knn = KnnModel(n_neighbors=int(train_data.shape[0] * 0.01))
        knn_logs, knn_preds = knn.analyse(train_data, test_data, experiment_id, normalize=True)

        rf = RandomForestModel(n_estimators=500)
        rf_logs, rf_preds = rf.analyse(train_data, test_data, experiment_id, normalize=False)

        cb = CatBoostModel(iterations=300, depth=5)
        cb_logs, cb_preds = cb.analyse(train_data, test_data, experiment_id, normalize=False)

        ensemble_preds = np.apply_along_axis(lambda x: int(np.sum(x) > 1), axis=0,
                                             arr=np.array([knn_preds, rf_preds, cb_preds]))

        print(np.unique(np.array(ensemble_preds), return_counts=True))

        crossval_logs = {
            'validation_accuracy': accuracy_score(test_data['class_target'], ensemble_preds),
            'validation_f1': f1_score(test_data['class_target'], ensemble_preds),
            'validation_recall': recall_score(test_data['class_target'], ensemble_preds),
            'validation_precision': precision_score(test_data['class_target'], ensemble_preds)

        }

        print(crossval_logs)

        return crossval_logs
