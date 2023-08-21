# MLOps-Zoomcamp 2023 Final Project
## Tilemachos S. Doganis

## Problem Statement
Using urinary biomarkers to predict pancreatic cancer

Data source: [Kaggle](https://www.kaggle.com/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer)

Diagnosing pancreatic cancer early offers much greater chances of survival, and prediction using biomarkers would be enormously helpful. An XGBoost model is trained for predicting the tumors via the given biomarkers.

The study included two cohorts of patients, therefore one is used as the initial training set and the next as the validation and monitoring one.

## Prepare Environment
1. Install Anaconda (Anaconda3-2023.07-1-Linux-x86_64.sh)
2. conda env create -f mlops-zoomcamp-env.yaml
3. conda activate mlops-zoomcamp

Since AWS Access will be problematic, the data and the MLFlow best model are included in the folder PEER_REVIEWER_DATA

In the train_model.py a few comment changes at lines 165-166 can allow you to run it locally.

For the integration test, the model dir can also be copied to integration-tests/model, to be bind-mounted in the container.

For the batch_prediction the lines 26-27 can be commented out, and the local filenames of input and output data can be given instead of the S3 paths, as well as the model using the local mlflow training output.

## Infrastructure
The data exploration and development takes place locally (Ubuntu 20.04).

The data have been manually donwloaded from Kaggle and then uploaded to an AWS S3 Bucket.

The MLFlow server used for experiment tracking is hosted on AWS, and the models stored in an S3 Bucket.

The orchestration is carried out via a Prefect server deployed locally, which connects via prefect_aws to the S3 Buckets for downloading the data, downloading the model and uploading the results.

Monitoring is performed via a Grafana server connected with a local postgresql database (which itself is accessible via adminer). All 3 run locally as docker containers.

The model is deployed locally in a Docker container, which downloads the model from its S3 bucket and can serve requests via localhost.

## Model Development
As this is a classification problem, XGBoost was used to predict the diagnosis for each patient. Since the dataset contains 2 cohorts, it was decided to use Cohort1 for the training. It was split in training, test, and validation sets. The features present in all patients are used, since missing biomedical data of this nature would be tricky to statistically fill in. The model is tuned using Hyperopt's hyperparameter optimization, with each training performed with the multiclass Softprob objective and then cross-validated using the area under ROC score. The best parameters are then used to train the final model, rate it using the F1-Score for convenience, and be logged in MLFlow.

## Orchestration
### Start Training
1. prefect server start
2. prefect worker start -p local-worker
3. Go to http://localhost:4200 and run the 'pancreatic-cancer-training' deployment of 'training-flow'

#### Without AWS Access
3. Uncomment line 166 at `train_model.py` to use localhost for mlflow
    in another terminal run `mlflow start`

### Batch Prediction
1. prefect server start
2. prefect worker start -p local-worker
3. Go to http://localhost:4200 and run the 'pancreatic_batch_prediction' Deployment of 'pancreatic-cancer-prediction'

## Monitoring
### Monitor dataset and generate Dashboard
1. prefect server start
2. prefect worker start -p local-worker
3. Navigate to mlops-project/monitoring
4. `docker-compose up` in `monitoring/`
5. Go to http://localhost:4200 and run the 'prediction_monitoring' Deployment of 'monitoring-flow'

### Read Grafana Dashboard
1. `docker-compose up` in `monitoring/`
2. Go to http://localhost:3000 and login using admin as username and password
3. Go to Dashboards (http://localhost:3000/dashboards)
4. Click on "Drift as Bar Dashboard" to see the monitoring metrics compared to the reference (training) dataset (Cohort1)

## Model Deployment
The deployment code to be used in the docker container is contained under `dockefiles/`. `predict.py` serves predictions, `Pipfile` and `Pipefile.lock` contain the deployment environment setup and `Dockerfile` builds the deployment container. An example of a single prediction can be done via the Integration Test.

## Best Practices

- Unit Testing is performed via pytest using `make test`
- Integration Testing of the deployment container is done via `make integration_test`. It creates a container, fetches the model from S3, serves a sample request, and compares the response.
- Linting and formatting are performed via `make quality_checks` (`pylint`, `black`, `isort`). Their parameters can be found under `pyproject.toml`
- `.pre-commit-config.yaml` contains the pre-commit hooks that test the code before commits
- `.github/workflows/ci-tests.yaml` uses Git Actions to perform CI tests on merges with develop

# CI/CD
.github:
    Contains the yaml file for the CI tests for Github Actions. (Performed upon Pull Requests with develop)
    Sets up the environment and performs python, lint tests and an integration test for the deployment
