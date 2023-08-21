# pylint: disable=duplicate-code

import os

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, request


def get_model_location(run_id):
    model_location = os.getenv('MODEL_LOCATION')
    if model_location is not None:
        return model_location
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '1')
    model_bucket = os.getenv('MODEL_BUCKET', 'mlflow-artifacts-remote-tilemachos')

    model_location = (
        f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models_mlflow/'
    )
    return model_location


def load_model(run_id):
    model_path = get_model_location(run_id)
    model = mlflow.xgboost.load_model(model_path)
    return model


# A simplified version of read_dataframe
def prepare_features(patient):
    patient['diagnosis'] = patient['diagnosis'].apply(lambda d: d - 1)
    dummies = pd.get_dummies(patient['sex'])
    patient = pd.concat([patient.drop('sex', axis=1), dummies], axis=1)
    return patient


def select_features(df: pd.DataFrame) -> np.ndarray:
    """Add features to the model"""

    # Feature selection
    categorical = ['M']
    numerical = ['age', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']

    X = df[categorical + numerical].values
    return X


def predict(patient, model):
    df_patient = prepare_features(patient)
    X = select_features(df_patient)
    preds = np.round(model.predict(xgb.DMatrix(X)))
    return preds


app = Flask('cancer-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    patient_dict = request.get_json()
    patient_df = pd.DataFrame.from_dict(patient_dict)
    model = load_model(os.getenv("RUN_ID"))
    pred = predict(patient_df, model)

    # predict() returns numpy array of floats, so get the first int
    result = {
        'sample_id': patient_df['sample_id'].iloc[0],
        'diagnosis': int(pred[0]),
        'model_run_id': os.getenv("RUN_ID"),
    }

    return jsonify(result)


if __name__ == "__main__":
    os.environ["RUN_ID"] = "79b4b49914ad48598aac9946c1a61c3d"
    app.run(debug=True, host='0.0.0.0', port=9696)
