# pylint: disable=duplicate-code

import requests
from deepdiff import DeepDiff

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
    "REG1A": [None],
}


url = 'http://localhost:9696/predict'
actual_response = requests.post(url, json=patient100, timeout=20).json()

expected_response = {
    'diagnosis': 1,
    'model_run_id': '79b4b49914ad48598aac9946c1a61c3d',
    'sample_id': 'S100',
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)

print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff
