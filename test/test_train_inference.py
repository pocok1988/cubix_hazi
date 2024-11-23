import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path so we can import MLModel
current_file_path = Path(__file__).resolve()  # Get the path of the current file
parent_directory = current_file_path.parent.parent  # Get the parent directory (two levels up)
sys.path.append(str(parent_directory))  # Add the parent directory to sys.path

from constants import EXPERIMENT_NAME, MLFLOW_URL, SPAM
from MLModel import MLModel

import mlflow
from mlflow import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_URL)

# Set default experiment
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

def test_prediction_accuracy():
    client = MlflowClient()

    obj_mlmodel = MLModel(client=client)  
    data_path = 'data/spam.csv'
    df = pd.read_csv(data_path)  

    # Training pipeline
    df_preprocessed = obj_mlmodel.preprocessing_pipeline(df)
    y_expected = df_preprocessed[SPAM]
    accuracy_train_pipeline_full = obj_mlmodel.get_accuracy_full(
                                            df_preprocessed.drop(columns=SPAM), 
                                            y_expected)
    accuracy_train_pipeline_full = np.round(accuracy_train_pipeline_full, 2)

    # Inference pipeline
    obj_mlmodel = MLModel(client=client)  
    df = pd.read_csv(data_path)

    accuracy_inference_pipeline_full = obj_mlmodel.get_accuracy_full(df, y_expected)
    accuracy_inference_pipeline_full = np.round(accuracy_inference_pipeline_full, 2)
    print(accuracy_train_pipeline_full, accuracy_inference_pipeline_full)

    assert accuracy_train_pipeline_full == accuracy_inference_pipeline_full, 'Inference prediction accuracy is not as expected'