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
                                                        random_state=random_state)

    pooled_train = Pool(data=X_train,
                        label=y_train,
                        cat_features=categorial_features)

    pooled_eval = Pool(data=X_test,
                       label=y_test,
                       cat_features=categorial_features)
    return pooled_train, pooled_eval


def fit_catboost(dataset_params: tuple, pooled_train: Pool, pooled_eval: Pool, iterations: int, depth: int,
                 loss_function: str, metric: str):
    cbc = CatBoostClassifier(iterations=iterations,
                             depth=depth,
                             random_seed=42,
                             task_type="GPU",
                             devices="0:1",
                             loss_function=loss_function,
                             eval_metric=metric)

    cbc.fit(pooled_train,
            eval_set=pooled_eval,
            use_best_model=True,
            verbose=False)

    cbc.save_model(
        f"models/{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_{iterations}iterations_{depth}depth.cbm",
        format="cbm")

    train_results = cbc.get_evals_result()
    eval_metrics = cbc.eval_metrics(data=pooled_eval, metrics=["Accuracy", "F1", "Recall", "Precision"])

    features_importance = cbc.feature_importances_
    important_ids = np.array(cbc.feature_importances_).argsort()[-15:]
    # print("important", features_importance[important_ids])
    # print("important_columns", dataset.columns[important_ids].to_list())


    print(dataset.columns[important_ids].to_list())
    print(list(features_importance[important_ids]))

    graphs = {
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

    with open(f'evaluating/{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_{iterations}iterations_{depth}depth_logs.json', 'w') as fp:
        json.dump(graphs, fp)

    print(graphs['val_metrics'])

    return cbc, graphs


dataset_params = (3, 5, 5)
dataset = pd.read_csv(f"../preprocessed_data/{dataset_params[0]}_{dataset_params[1]}_{dataset_params[2]}_dataset.csv",
                      index_col=0)

pooled_train, pooled_eval = pool_dataset(dataset=dataset, random_state=90)

model, graphs = fit_catboost(dataset_params=dataset_params,
                             pooled_train=pooled_train,
                             pooled_eval=pooled_eval,
                             iterations=300,
                             depth=5,
                             loss_function="CrossEntropy",
                             metric="F1")

# model = CatBoostClassifier()  # parameters not required.
# model.load_model('model_name')
