# pylint: disable=duplicate-code

import numpy as np
import pandas as pd
import xgboost as xgb

import batch_predict


class ModelMock:
    def predict(self, X):
        n = X.num_row()
        return [1] * n


def test_predict():
    model = ModelMock()

    X = xgb.DMatrix(
        np.expand_dims(np.array([1, 51, 0.78039, 0.1455889, 102.366, 461.141]), axis=0)
    )
    actual_prediction = model.predict(X)
    expected_prediction = [1]

    assert actual_prediction == expected_prediction


def test_select_features():
    patient100 = pd.DataFrame.from_dict(
        {
            "sample_id": ["S100"],
            "patient_cohort": ["Cohort2"],
            "sample_origin": ["BPTB"],
            "age": [51],
            "M": [1],
            "diagnosis": [0],
            "stage": [None],
            "benign_sample_diagnosis": [None],
            "plasma_CA19_9": [7],
            "creatinine": [0.78039],
            "LYVE1": [0.1455889],
            "REG1B": [102.366],
            "TFF1": [461.141],
            "REG1A": [None],
        }
    )

    actual_features = batch_predict.select_features(patient100).flatten()
    expected_features = np.array([1, 51, 0.78039, 0.1455889, 102.366, 461.141])

    assert (actual_features == expected_features).all()
