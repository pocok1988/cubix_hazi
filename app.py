from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import os
import pandas as pd
from MLModel import MLModel
import mlflow
from mlflow import MlflowClient
from datetime import datetime
from mlflow.exceptions import MlflowException
from constants import MLFLOW_URL, EXPERIMENT_NAME, CATEGORY

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_URL)

# Set default experiment
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

# Initialize Flask app and API
app = Flask(__name__)
api = Api(app, version='1.0', title='API Documentation')

# Initialize MLFlow client
client = MlflowClient()

# Attempt to load the latest "Staging" model version if it exists
try:
    obj_mlmodel = MLModel(client=client)
    if obj_mlmodel.model is None:
        print("""⚠️  Warning: No 'Staging' model found. 
              Training is still possible.""")
except Exception as e:
    print(f"""⚠️  Warning: Could not load 'Staging' model. 
          Training is still possible. Error: {e}""")
    obj_mlmodel = MLModel(client=client)  # Create an empty MLModel instance

predict_model = api.model('PredictModel', {
    'inference_row': fields.List(fields.Raw, required=True, 
                                 description='A row of data for inference')
})

file_upload = api.parser()
file_upload.add_argument('file', location='files',
                         type=FileStorage, required=True,
                         help='CSV file for training')

ns = api.namespace('model', description='Model operations')

@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        """
            Uploads a CSV file for training the model.
                - Validates file extension to ensure it's CSV.
                - Saves the file temporarily and loads it for preprocessing and training.
                - If training is successful, model and accuracies are returned.
        """
        args = file_upload.parse_args()
        uploaded_file = args['file']

        # Ensure uploaded file is CSV format
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'error': 'Invalid file type'}, 400
        
        data_path = 'temp_data.csv'
        uploaded_file.save(data_path)
        
        try:
            # Set a unique name for the training run based on timestamp
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_name = "clf_model"  # Name of your registered model

            with mlflow.start_run(run_name=run_name) as run:
                # Load and preprocess data
                df = pd.read_csv(data_path)
                input_example = df.drop(CATEGORY, axis =1).iloc[:1]
                signature = mlflow.models.infer_signature(df.drop(CATEGORY, axis =1), pd.Series([1]))
                df = obj_mlmodel.preprocessing_pipeline(df)

                # Log dataset to MLflow
                mlflow.log_artifact(data_path, artifact_path="datasets")

                # Train the model and get accuracy metrics      
                train_accuracy, test_accuracy, xgb = obj_mlmodel.train_and_save_model(df)

                # Log metrics to MLflow
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)

                # Log the trained model with input example and signature
                mlflow.sklearn.log_model(
                    sk_model=xgb, 
                    artifact_path="model", 
                    input_example=input_example, 
                    signature=signature
                )
                
                # Register the model in the MLflow Model Registry
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
                
                # Transition model version to "Staging"
                mlflow_client = mlflow.tracking.MlflowClient()
                mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=registered_model_version.version,
                    stage="Staging" # or Production
                )

                # Clean up the temporary data file
                os.remove(data_path)

                return {'message': 'Model Trained Successfully', 
                        'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}, 200
        except MlflowException as mfe:
            return {'message': 'MLflow Error', 'error': str(mfe)}, 500
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500
        
@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        """
        Predicts output using the trained model based on the provided data rows.
            - Accepts JSON with 'inference_row' containing multiple messages
            - Returns prediction for each message (0: not spam, 1: spam)
        """
        try:
            data = request.get_json()

            if 'inference_row' not in data:
                return {'error': 'No inference_row found'}, 400
            if obj_mlmodel.model is None:
                return {'error': """No staging model is loaded. 
                        Train a model first."""}, 400
            
            # Set a unique name for the inference run based on timestamp
            run_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Start the MLflow run
            with mlflow.start_run(run_name=run_name) as run:            
                # Get predictions for all messages
                predictions = obj_mlmodel.predict(request)

                # Log batch input and output to MLflow
                mlflow.log_param("batch_inference_input", data['inference_row'])
                mlflow.log_param("batch_inference_output", predictions)
                
                # Create response with predictions for each message
                response = {
                    'message': 'Inference Successful',
                    'predictions': [
                        {
                            'message': msg['Message'],
                            'is_spam': pred,
                            'prediction_label': 'SPAM' if pred == 1 else 'NOT SPAM'
                        }
                        for msg, pred in zip(data['inference_row'], predictions)
                    ]
                }

                # Log each message prediction as a separate row
                for i, (msg, pred) in enumerate(zip(data['inference_row'], predictions)):
                    mlflow.log_param(f"inference_input_{i}", msg)
                    mlflow.log_param(f"inference_output_{i}", pred)
                
                return response, 200
            
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500

if __name__ == '__main__':
    print("Starting Flask API on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=False)
