import os

SPAM = 'spam'
CATEGORY = 'Category'
MLFLOW_URL = os.getenv('MLFLOW_TRACKING_URI', 'http://192.168.11.10:12650')
EXPERIMENT_NAME = 'dani_experiment'
FLASK_PORT = int(os.getenv('FLASK_PORT', '8080'))