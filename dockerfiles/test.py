import requests
import predict
import pandas as pd

patient100 = {
    "sample_id": ["S100"],
    "patient_cohort": ["Cohort2"],
    "sample_origin": ["BPTB"],
    "age": [51],
    "sex": ["M"],
    "diagnosis": [1],
    "stage": [None],
    "benign_sample_diagnosis": [None],
    "plasma_CA19_9": [7],
    "creatinine": [0.78039],
    "LYVE1": [0.1455889],
    "REG1B": [102.366],
    "TFF1": [461.141],
    "REG1A": [None]
}


url = 'http://localhost:9696/predict'
response = requests.post(url, json=patient100, timeout=20)
print(response.json())

