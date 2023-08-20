from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context
from prefect_aws import S3Bucket

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

import psycopg

def read_dataframe(filename: str, src_dir: str, dst_dir: str) -> pd.DataFrame:
    """ Read dataset into dataframe """

    s3_bucket_block = S3Bucket.load("s3-bucket-block")
    s3_bucket_block.download_folder_to_path(from_folder=src_dir, to_folder=dst_dir)

    data = pd.read_csv(filename)
    data['diagnosis'] = data['diagnosis'].apply(lambda d: d-1)
    dummies = pd.get_dummies(data['sex'], drop_first=True)
    data = pd.concat([data.drop('sex', axis=1), dummies], axis=1)

    return data

def select_features(df: pd.DataFrame) -> np.ndarray:
    """ Add features to the model """

    # Feature selection
    categorical = ['M']
    numerical = ['age','creatinine', 'LYVE1', 'REG1B', 'TFF1']

    X = df[categorical + numerical].values
    return X

def load_model(run_id: str) -> xgb.Booster:
    logged_model = f's3://mlflow-artifacts-remote-tilemachos/1/{run_id}/artifacts/models_mlflow/'
    model = mlflow.xgboost.load_model(logged_model)
    return model

def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    feat_list = ['M', 'age','creatinine', 'LYVE1', 'REG1B', 'TFF1']
    for feat in feat_list:
        df_result[feat] = df[feat]
    
    df_result['actual_diagnosis'] = df['diagnosis']
    df_result['predicted_diagnosis'] = y_pred
    df_result['model_version'] = run_id

    df_result.to_csv(output_file, index=False)

@task
def prep_db(create_table_statement: str):
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)

@task(log_prints=True)
def generate_report(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    conn: psycopg.Connection
):
    # Selected features
    categorical = ['M']
    numerical = ['age','creatinine', 'LYVE1', 'REG1B', 'TFF1']
    target = ['diagnosis']

    # Tell evidently which columns correspond to what kind of feature
    column_mapping = ColumnMapping(
        target=None, # we're not analyzing the ground truth
        prediction=target[0],
        numerical_features=numerical,
        categorical_features=categorical
    )

    # Select metrics to calculate
    report = Report(metrics=[
        ColumnDriftMetric(column_name=target[0]), # Prediction Drift
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ]
    )

    # Generate evidently report and store as dict for database insertion
    report.run(reference_data=training_data, current_data=validation_data, column_mapping=column_mapping)
    report_dict = report.as_dict()

    # Select the metrics we want to save
    prediction_drift = report_dict['metrics'][0]['result']['drift_score']
    num_drifted_columns = report_dict['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = report_dict['metrics'][2]['result']['current']['share_of_missing_values']

    # Insert selected metrics into database
    conn.execute(
        "insert into monitoring_metrics(cohort, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
        ("cohort2", prediction_drift, num_drifted_columns, share_missing_values)
    )

@flow
def monitoring_flow(
    train_path: str = "./monitoring/data/Debernardi_2020_data.csv",
    validation_path: str="./monitoring/data/Debernardi_2020_data_cohort2.csv"
) -> None:
    """ Evidently Monitoring Report Pipeline """
    df_train = read_dataframe(train_path, "data", "monitoring/data")
    df_val = read_dataframe(validation_path, "cohort2", "monitoring/data")

    create_table_statement = """
    drop table if exists monitoring_metrics;
    create table monitoring_metrics(
        cohort text,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_values float
    )
    """

    prep_db(create_table_statement)

    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn: 
        generate_report(df_train, df_val, conn)


@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()

    logger.info('reading the data from %s...', input_file)
    df = read_dataframe(input_file, "data", "data")

    logger.info('selecting features from data...')
    X = select_features(df)

    logger.info('loading the model with RUN_ID=%s...', run_id)
    model = load_model(run_id)

    logger.info('applying the model...')
    y_pred = np.round(model.predict(xgb.DMatrix(X)))

    logger.info('saving the result to %s...', output_file)

    save_results(df, y_pred, run_id, output_file)
    return output_file

@flow
def pancreatic_cancer_prediction(
    input_file: str,
    output_dir: str,
    run_id: str,
    run_date: datetime = None):
        if run_date is None:
            ctx = get_run_context()
            run_date = ctx.flow_run.expected_start_time
        
        output_file = output_dir + f'cohort2_predictions/{run_id}.csv'
        
        apply_model(
             input_file=input_file,
             run_id=run_id,
             output_file=output_file
        )

def run_pred():
    input_file = 's3://mlops-zoomcamp-2013/cohort2/Debernardi_2020_data_cohort2.csv'
    output_dir = 's3://mlops-zoomcamp-2013/results/'
    run_id = '79b4b49914ad48598aac9946c1a61c3d'

    pancreatic_cancer_prediction(
        input_file=input_file,
        output_dir=output_dir,
        run_id=run_id,
        run_date=datetime.today().strftime('%Y-%m-%d')
    )

def run_monitor():

    train_path = "./monitoring/data/Debernardi_2020_data.csv"
    validation_path ="./monitoring/data/Debernardi_2020_data_cohort2.csv"
    
    monitoring_flow(
        train_path=train_path,
        validation_path=validation_path
    )


if __name__ == "__main__":
#    run_pred()
    run_monitor()
