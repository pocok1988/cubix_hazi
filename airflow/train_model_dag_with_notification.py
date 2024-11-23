from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import requests
import mlflow
from mlflow import MlflowClient

MLFLOW_URL = 'http://192.168.11.10:12650'
MODEL_REST_URL = 'http://192.168.11.71:8080/model/train'

# Define the MLflow client
mlflow.set_tracking_uri(MLFLOW_URL)
client = MlflowClient()

# Set the default model name
model_name = "clf_model"

# Path to the CSV file
csv_file_path = "/opt/bitnami/airflow/dags/spam.csv"

def train_model(**kwargs):
    # Call the train endpoint with the CSV file
    with open(csv_file_path, 'rb') as f:
        files = {'file': ('spam.csv', f)}
        response = requests.post(MODEL_REST_URL, files=files)

    print(response.status_code)
    
    if response.status_code != 200:
        data = response.json()
        raise Exception(f"Training failed: {data.get('error', 'Unknown error')}")

    data = response.json()
    new_accuracy = data["test_accuracy"]

    latest_version_info = client.get_latest_versions(model_name, stages=["Staging"])

    if latest_version_info:
        latest_version = latest_version_info[0]
        staging_accuracy = client.get_metric_history(latest_version.run_id, "test_accuracy")[-1].value

        print(new_accuracy)
        print(staging_accuracy)

        if new_accuracy > staging_accuracy:
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Archived"
            )
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Staging"
            )
            return "skip_notification"
    return "send_notification"

# Define the branching function
def branch_decision(**kwargs):
    return kwargs['ti'].xcom_pull(task_ids='train_and_compare_model') # ti: TaskInstance, xcom_pull: cross-communication - data sharing between tasks

# Define the Airflow DAG
with DAG(
    dag_id="_dani_daily_model_training_with_notification",
    start_date=datetime(2024, 11, 20),
    schedule_interval="0 2 * * *", # every day at 2am
    catchup=False,
) as dag:

    train_and_compare_task = PythonOperator(
        task_id="train_and_compare_model",
        python_callable=train_model,
        retries=1,
        retry_delay=timedelta(minutes=1),
        provide_context=True,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=branch_decision,
        provide_context=True,
    )

    notification_task = EmailOperator(
        task_id="send_notification",
        to="teszt@teszt.com",
        subject="pocok1988 - Model Accuracy Notification",
        html_content="The new model's accuracy is equal to or lower than the old model's accuracy.",
    )

    # Dummy task to mark the end if no notification is needed
    skip_notification = PythonOperator(
        task_id="skip_notification",
        python_callable=lambda: print("No notification sent."),
    )

    train_and_compare_task >> branch_task
    branch_task >> [notification_task, skip_notification]
