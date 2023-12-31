from functools import partial
from typing import Union

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from prefect import flow, task
from prefect_aws import S3Bucket
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


# Utility functions used inside the tasks
def objective(params, dtrain, dval, y_val):
    """Objective function trying to minimize area under ROC"""
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50
    )
    y_pred = booster.predict(dval)

    print(f'y_val = {y_val}')
    print(f'y_pred = {y_pred}')

    return {
        'loss': -1*roc_auc_score(y_val, y_pred, multi_class='ovr'),
        'status': STATUS_OK
    }


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read dataset into dataframe"""

    s3_bucket_block = S3Bucket.load("s3-bucket-block")
    s3_bucket_block.download_folder_to_path(from_folder='data', to_folder="data")

    # Make sure to use only Cohort1 for training
    data_both_cohorts = pd.read_csv(filename)
    data = data_both_cohorts[data_both_cohorts['patient_cohort'] == 'Cohort1']

    # Transform data
    data['diagnosis'] = data['diagnosis'].apply(lambda d: d-1)
    dummies = pd.get_dummies(data['sex'], drop_first=True)
    data = pd.concat([data.drop('sex', axis=1), dummies], axis=1)

    return data


@task
def add_features(df: pd.DataFrame) -> tuple(
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray
    ]
):
    """ Add features to the model """

    # Feature selection
    categorical = ['M']
    numerical = ['age','creatinine', 'LYVE1', 'REG1B', 'TFF1']
    target = ['diagnosis']

    X = df[categorical + numerical].values
    y = np.squeeze(df[target].values) # Squeeze cause of dimension errors in XGBoost

    # Perform the train - test - validation split here already
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
    return X_train, X_val, X_test, y_train, y_val, y_test


@task
def tune_model(
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray
) -> dict[str, Union[float, int]]:
    """Hyperparameter optimization"""

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
        'subsample': hp.uniform('subsample', .5, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -7, 0),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 7),
        'reg_alpha': hp.loguniform('reg_alpha', -10, 10),
        'reg_lambda': hp.loguniform('reg_lambda', -10, 10),
        'gamma': hp.loguniform('gamma', -10, 10),
        'objective': 'multi:softprob',
        'eval_metric': 'auc',
        'num_class': 3
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    fmin_obj = partial(objective, dtrain=dtrain, dval=dval, y_val=y_val)

    best_params = fmin(
        fn=fmin_obj,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )

    return best_params


@task(log_prints=True)
def train_model(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        params: dict[str, Union[float, int]]
) -> xgb.core.Booster:
    """Train the model and record in MLFlow"""
    with mlflow.start_run():
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params['max_depth'] = int(params['max_depth'])

        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000
        )

        y_pred_float = booster.predict(dtest)
        y_pred = [round(value) for value in y_pred_float]
        test_score = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("f1_score", test_score)
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return booster


@flow
def training_flow(
        train_path: str = "./data/Debernardi_2020_data.csv"
) -> None:
    """The main training pipeline

    FOR PEER REVIEWER WITHOUT AWS ACCESS
    set TRACKING_SERVER_HOST as localhost and on a separate window, run "mlflow start"
    """

    #MLFlow Settings
    TRACKING_SERVER_HOST = "ec2-3-79-25-97.eu-central-1.compute.amazonaws.com"
    #TRACKING_SERVER_HOST = 'localhost'
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("pancreatic-cancer-experiment")

    # Load
    df = read_data(train_path)

    # Transform
    X_train, X_val, X_test, y_train, y_val, y_test = add_features(df)

    # Tune
    best_params = tune_model(X_train, X_val, y_train, y_val)

    # Train
    train_model(X_train, X_test, y_train, y_test, best_params)

if __name__ == "__main__":
    training_flow()
