import mlflow
import xgboost as xgb
import numpy as np

logged_model = f's3://mlflow-artifacts-remote-tilemachos/1/79b4b49914ad48598aac9946c1a61c3d/artifacts/models_mlflow/'
model = mlflow.xgboost.load_model(logged_model)

def prepare_features(patient):
    features = {}
    features['male'] = int(patient['sex'] == 'M')
    numerical = ['age','creatinine', 'LYVE1', 'REG1B', 'TFF1']
    features.update(dict(filter(lambda i:i[0] in numerical, patient.items())))

    return features.values()

def predict(patient):
    features = prepare_features(patient)
    X = xgb.DMatrix(np.array(list(features)).reshape(-1, len(features)))
    preds = model.predict(X)
    return preds

