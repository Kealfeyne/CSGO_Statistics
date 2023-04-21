import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv


def pool_dataset(dataset: pd.DataFrame, drop_nans: bool = True, test_size: float = 0.2,
                 random_state: int = 42):
    if drop_nans:
        dataset = dataset.dropna()
    print(f"Dataset shape:", dataset.shape)

    categorial_features = dataset.dtypes[(dataset.dtypes != np.float64) & (dataset.dtypes != np.int64)].index.tolist()

    X_train, X_test, y_train, y_test = train_test_split(dataset.drop(["target", "class_target"], axis=1),
                                                        dataset["class_target"], test_size=test_size,
                                                        random_state=random_state, stratify=dataset["class_target"])

    pooled_train = Pool(data=X_train,
                        label=y_train,
                        cat_features=categorial_features)

    pooled_eval = Pool(data=X_test,
                       label=y_test,
                       cat_features=categorial_features)
    return pooled_train, pooled_eval


def fit_catboost(dataset_params: tuple, pooled_train: Pool, pooled_eval: Pool, iterations: int, depth: int,
                 loss_function: str, metric: str, dataset_tag: str):
    cbc = CatBoostClassifier(iterations=iterations,
                             depth=depth,
                             random_seed=42,
                             task_type="GPU",
                             devices="0:1",
                             loss_function=loss_function,
                             eval_metric=metric)

    print("Обучение...")
    cbc.fit(pooled_train,
            eval_set=pooled_eval,
            use_best_model=True,
            verbose=False)
    print("Сохранение...")
    cbc.save_model(
        f"models/{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_{iterations}iterations_{depth}depth.cbm",
        format="cbm")

    print("Логирование...")
    train_results = cbc.get_evals_result()
    eval_metrics = cbc.eval_metrics(data=pooled_eval, metrics=["Accuracy", "F1", "Recall", "Precision"])

    features_importance = cbc.feature_importances_
    important_ids = np.array(cbc.feature_importances_).argsort()[-30:]

    logs = {
        'train_loss': train_results["learn"][loss_function],
        'train_f1': train_results["learn"][metric],
        'val_loss': train_results["validation"][loss_function],
        'val_f1': train_results["validation"][metric],
        'val_metrics': {
            'val_accuracy': eval_metrics["Accuracy"],
            'val_f1': eval_metrics["F1"],
            'val_recall': eval_metrics["Recall"],
            'val_precision': eval_metrics["Precision"]},
        'importance': {
            'important_features': dataset.columns[important_ids].to_list(),
            'importance_values': list(features_importance[important_ids])}
    }

    with open(
            f'logs/{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_{iterations}iterations_{depth}depth_{dataset_tag}_logs.json',
            'w') as fp:
        json.dump(logs, fp)

    return cbc, logs


dataset_params_grid = [(1, 5, 5), (2, 5, 5), (2, 10, 10),
                       (3, 5, 5), (3, 10, 10), (4, 1, 1), (4, 5, 5), (5, 1, 1), (5, 5, 5)]

tag = ""

for dataset_params in dataset_params_grid:
    dataset = pd.read_csv(
        f"../data/datasets_to_model/{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_{tag}_dataset.csv",
        index_col=0)
    min_counts = min(dataset[dataset["class_target"] == 0].shape[0], dataset[dataset["class_target"] == 1].shape[0])
    dataset = pd.concat(
        [dataset[dataset["class_target"] == 0][:min_counts], dataset[dataset["class_target"] == 1][:min_counts]])

    pooled_train, pooled_eval = pool_dataset(dataset=dataset, random_state=90)

    model, metrics = fit_catboost(dataset_params=dataset_params,
                                  pooled_train=pooled_train,
                                  pooled_eval=pooled_eval,
                                  iterations=300,
                                  depth=5,
                                  loss_function="CrossEntropy",
                                  metric="F1",
                                  dataset_tag=tag)

# model = CatBoostClassifier()  # parameters not required.
# model.load_model('model_name')
