from pathlib import Path
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
from flask import jsonify
import json

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from constants import CATEGORY,SPAM
import mlflow
from mlflow.artifacts import download_artifacts


class MLModel:
    def __init__(self,client):
        """
        Initialize the MLModel with the given MLflow client and 
        load the staging model if available.

        Parameters:
            client (MlflowClient): The MLflow client used to 
            interact with the MLflow registry.

        Attributes:
            model (object): The loaded model, or None if no model 
                is loaded.
        """
        self.client = client
        self.model = None
        self.load_staging_model()

    def load_staging_model(self):
        """
        Load the latest model tagged with 'Staging' stage from MLflow 
        if available.
        
        If a model with the 'Staging' tag exists, it loads the model 
        and associated artifacts. Otherwise, prints a warning.

        Returns:
            None
        """
        try:
            latest_staging_model = None
            for model in self.client.search_registered_models():
                for latest_version in model.latest_versions:
                    if latest_version.current_stage == "Staging":
                        latest_staging_model = latest_version
                        break
                if latest_staging_model:
                    break
            
            if latest_staging_model:
                model_uri = latest_staging_model.source
                self.model = mlflow.sklearn.load_model(model_uri)
                print("Staging model loaded successfully.")
                
            else:
                print("No staging model found.")
                
        except Exception as e:
            print(f"Error loading model or artifacts: {e}")

    def predict(self, inference_rows):
        """
        Predicts the outcome for multiple rows of data.
        Args:
            inference_rows: List of messages to predict
        Returns:
            List of predictions (0: not spam, 1: spam) or error message in JSON
        """
        try:
            # Process multiple messages
            df = self.preprocessing_pipeline_inference(inference_rows)
            
            # Get predictions for all messages
            predictions = self.model.predict(df)
            
            # Convert numpy array to list of integers
            return predictions.tolist()

        except Exception as e:
            return jsonify({'message': 'Internal Server Error. ',
                        'error': str(e)}), 500

    def preprocessing_pipeline(self, df):
        """
        Preprocesses the data for training by handling missing values,
        and normalizing data.
        
        Keyword arguments:
            df (DataFrame) -- DataFrame with the data

        Returns:
            pandas.DataFrame -- DataFrame with the preprocessed data
        """

        df[SPAM] = np.where(df[CATEGORY] == SPAM, 1, 0)
        df.drop(CATEGORY,inplace =True, axis =1)

        return df

    def preprocessing_pipeline_inference(self, request_data):
        """
        Preprocesses multiple rows of inference data.
        
        Args:
            request_data: Request object containing JSON data
        Returns:
            List: List of processed messages ready for prediction
        """
        try:
            parsed_data = json.loads(request_data.data)
            messages = [item['Message'] for item in parsed_data['inference_row']]
            return messages
        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")

    def get_accuracy(self, X_train, X_test, y_train, y_test):
        """
        Computes model accuracy on both training and test data.
        Returns tuple with training and test accuracy scores.
        
        Args:
            X_train: Features for the training set.
            X_test: Features for the test set.
            y_train: Actual labels for the training set.
            y_test: Actual labels for the test set.

        Returns:
            A tuple containing the training accuracy and the test accuracy.
        """

        train_accuracy = self.model.score(X_train.astype(str), y_train.to_list())
        test_accuracy = self.model.score(X_test.astype(str), y_test.to_list())

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy
    
    def get_accuracy_full(self, X, y):
        """
        Calculate and print the overall accuracy of the model using a data set.

        Args:
            X: Features for the data set.
            y: Actual labels for the data set.

        Returns:
            The accuracy of the model on the provided data set.
        """
        y_pred = self.model.predict(X['Message'].tolist())

        accuracy = self.model.score(y.astype(str), y_pred)

        print("Accuracy: ", accuracy)

        return accuracy

    def train_and_save_model(self, df):
        """Trains the model on the given data and saves it.
        Returns training and testing accuracy, and the trained model.
        
        Keyword arguments:
        df -- DataFrame with the preprocessed data

        Returns:
        train_accuracy -- Accuracy of the model on the training set
        test_accuracy -- Accuracy of the model on the test set
        """

        messages = df.Message
        spam_labels= df['spam']
        x_train,x_test,y_train,y_test = train_test_split(messages,spam_labels,test_size=0.2)

        clf = Pipeline([('vectorizer', CountVectorizer()),('nb', MultinomialNB())])
        clf.fit(x_train,y_train)

        self.model = clf

        train_accuracy, test_accuracy = self.get_accuracy(x_train, x_test, y_train, y_test)

        return train_accuracy, test_accuracy, clf

    @staticmethod
    def create_new_folder(folder):
        """Create a new folder if it doesn't exist.
        
        Keyword arguments:
        folder -- Path to the folder

        Returns:
        None
        """
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_model(model, file_path):
        """Saves trained model to specified path."""
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path):
        """Loads model from specified path."""
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    