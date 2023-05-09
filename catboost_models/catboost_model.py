import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from catboost import CatBoostRegressor, CatBoostClassifier, Pool


class CatBoostModel:
    def __init__(self, iterations, depth):
        self.iterations = iterations
        self.depth = depth

    def analyse(self, train_data, test_data, experiment_id, normalize):
        categorical_features = train_data.dtypes[
            (train_data.dtypes != np.float64) & (train_data.dtypes != np.int64)].index.tolist()

        pooled_train = Pool(data=train_data.drop(['target', 'class_target'], axis=1),
                            label=train_data['class_target'],
                            cat_features=categorical_features)

        pooled_eval = Pool(data=test_data.drop(['target', 'class_target'], axis=1),
                           label=test_data['class_target'],
                           cat_features=categorical_features)

        cbc = CatBoostClassifier(iterations=self.iterations,
                                 depth=self.depth,
                                 random_seed=42,
                                 # task_type="GPU",
                                 devices="0:1",
                                 loss_function='CrossEntropy',
                                 eval_metric="F1")

        print("Обучение...")
        cbc.fit(pooled_train,
                eval_set=pooled_eval,
                use_best_model=True,
                verbose=False)

        # print("Сохранение...")
        # cbc.save_model(
        #     f"catboost_models/models/{experiment_id}.cbm",
        #     format="cbm")

        print("Логирование модели...")
        train_results = cbc.get_evals_result()
        eval_metrics = cbc.eval_metrics(data=pooled_eval, metrics=["Accuracy", "F1", "Recall", "Precision"])

        features_importance = cbc.feature_importances_
        important_ids = np.array(cbc.feature_importances_).argsort()[-30:]

        model_logs = {
            'train_loss': train_results["learn"]["CrossEntropy"],
            'train_f1': train_results["learn"]["F1"],
            'val_loss': train_results["validation"]["CrossEntropy"],
            'val_f1': train_results["validation"]["F1"],
            'val_metrics': {
                'val_accuracy': eval_metrics["Accuracy"],
                'val_f1': eval_metrics["F1"],
                'val_recall': eval_metrics["Recall"],
                'val_precision': eval_metrics["Precision"]},
            'importance': {
                'important_features': train_data.columns[important_ids].to_list(),
                'importance_values': list(features_importance[important_ids])}
        }

        with open(
                f'catboost_models/logs/{experiment_id}.json',
                'w') as fp:
            json.dump(model_logs, fp)

        preds = cbc.predict(pooled_eval)

        print(np.unique(np.array(preds), return_counts=True))

        crossval_logs = {
            'validation_accuracy': accuracy_score(test_data['class_target'], preds),
            'validation_f1': f1_score(test_data['class_target'], preds),
            'validation_recall': recall_score(test_data['class_target'], preds),
            'validation_precision': precision_score(test_data['class_target'], preds)

        }

        print(crossval_logs)

        return crossval_logs
