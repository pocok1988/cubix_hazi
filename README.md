# ML Spam Detection Project

This project is a Flask-based API using a machine learning model to detect spam messages. The application can train new models based on uploaded CSV files and make predictions on text data.
The main goal of the project is to transform [this notebook](https://www.kaggle.com/code/ahmedraafatmohamed/spam-emails-detection-using-naive-bayes-99) content into a production-ready code, applying clean code principles.

## Installation

### Prerequisites
Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Required Python Packages
```
flask
flask-restx
pandas
scikit-learn
numpy
matplotlib
seaborn
mlflow
streamlit
evidently
```

## Project Structure
```
├── app.py                                  # Flask application and API endpoints
├── MLModel.py                              # Machine learning model implementation
├── constants.py                            # Constants definitions
├── airflow/                                # Airflow stack, requirements.txt and train model pipeline for airflow
│   ├── airflow_stack.yaml                  # Airflow stack configuration
│   ├── requirements_airflow.txt            # Airflow requirements
│   └── train_model_dag_with_notification.py# Airflow DAG for training model with notification
├── monitor_with_streamlit_train_data.py    # Streamlit monitoring dashboard
├── pages/
│   └── test_data.py                        # Streamlit monitoring dashboard component
├── test/
│   └── test_train_inference.py             # Unit test
├── artifacts/
│   └── models/                             # Saved models directory
│       └── clf_model.pkl                   # Trained model
├── data/
│   ├── spam.csv                            # Dataset
│   ├── spam_10.csv                         # Dataset sample
│   ├── spam_20.csv                         # Dataset sample
│   └── spam_80.csv                         # Dataset sample
├── orig_notebook/                          # Original notebook
│   └── spam-emails-detection-using-naive-bayes-99.ipynb # Original notebook file
├── test/
│   └── spam-emails-detection-save_model.ipynb # Test notebook
├── inference_pipeline.ipynb                # Inference pipeline notebook
├── train_pipeline.ipynb                    # Training pipeline notebook
├── Dockerfile                              # Dockerfile for containerizing the application
├── environment.yml                         # Conda environment configuration
├── requirements.txt                        # Python requirements
├── build.sh                                # Build script
├── stop.sh                                 # Stop script
└── README.md                               # Project documentation
```

## API Endpoints

### 1. Train Model
- **Endpoint:** `/model/train`
- **Method:** POST
- **Description:** Upload CSV file and train model
- **Input:** CSV file (multipart/form-data)
- **Response:** Training and testing accuracy

Example curl command:
```bash
curl -X POST -F "file=@training_data.csv" http://localhost:8080/model/train
```

Example response:
```json
{
    "message": "Model Trained Successfully",
    "train_accuracy": 0.98,
    "test_accuracy": 0.97
}
```

### 2. Prediction
- **Endpoint:** `/model/predict`
- **Method:** POST
- **Description:** Detect spam for multiple messages
- **Input:** JSON with array of messages
- **Response:** Prediction result for each message (0: not spam, 1: spam)

Example request:
```json
{
    "inference_row": [
        {
            "Message": "Hello, how are you today?"
        },
        {
            "Message": "Win a free iPhone! Click here now!"
        },
        {
            "Message": "Meeting tomorrow at 10 AM"
        }
    ]
}
```

Example response:
```json
{
    "message": "Inference Successful",
    "predictions": [
        {
            "message": "Hello, how are you today?",
            "is_spam": 0,
            "prediction_label": "NOT SPAM"
        },
        {
            "message": "Win a free iPhone! Click here now!",
            "is_spam": 1,
            "prediction_label": "SPAM"
        },
        {
            "message": "Meeting tomorrow at 10 AM",
            "is_spam": 0,
            "prediction_label": "NOT SPAM"
        }
    ]
}
```

## Docker

1. After starting the container:
- Port 8081: Streamlit monitoring dashboard for visualizing ML model data.
- Port 8080: REST API endpoints for `train` and `predict` operations.

2. Build the Docker Image
- Run the following command to build the Docker image:

```bash
docker build -t ml_spam_detection .
```

3. Start the Container
- Start the container with the following command:

```bash
docker run -p 8081:8081 -p 8080:8080 ml_spam_detection
```

4. Access the Services
- Streamlit dashboard: http://localhost:8081
- REST API:
    - Train: http://localhost:8080/model/train
    - Predict: http://localhost:8080/model/predict

## Usage

1. Start the server:
```bash
python app.py
```

2. Access the API documentation at:
```
http://localhost:8080/
```

## Model Details
- The system uses a Multinomial Naive Bayes classifier
- The model is automatically saved to the `artifacts/models/` directory
- During training, the system splits data 80-20 (training-test)

## API Documentation
Documentation and API models can be accessed at / when the application is running.

## Files and Functions
- `app.py`: Creates and manages the Flask API, which includes the `/train` and `/predict` endpoints.
- `constants.py`: Defines key constants, such as the category names.
- `MLModel.py`: The machine learning model class, which handles training, preprocessing, model saving, and loading.
- `monitor_with_streamlit_train_data.py`: Streamlit dashboard for monitoring training data.
- `pages/test_data.py`: Streamlit component for browsing test datasets.
- `test/test_train_inference.py`: Unit tests for training and inference pipelines.