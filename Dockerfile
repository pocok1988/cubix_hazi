# Base Docker image: Anaconda
FROM continuumio/anaconda3:latest

# Set working directory within the container
WORKDIR /app

# Copy environment.yml, app.py, and additional files
COPY environment.yml /app
COPY app.py /app
COPY constants.py /app
COPY MLModel.py /app
COPY monitor_with_streamlit_train_data.py /app
COPY pages /app/pages
COPY data /app/data

# Install environment based on environment.yml
RUN conda env create -f environment.yml

# Activate the environment and install additional pip dependencies
RUN /bin/bash -c "source activate cubix_mlops_pipelines"
RUN /bin/bash -c "pip install flask-restx==1.3.0"
RUN /bin/bash -c "pip install streamlit==1.37.1"
RUN /bin/bash -c "pip install evidently==0.4.39"
RUN /bin/bash -c "pip install mlflow==2.17.2"

# Set environment variables
ENV MLFLOW_TRACKING_URI="http://192.168.11.10:12650"
ENV FLASK_PORT="8080"

# Start MLflow server and app.py in the cubix_mlops_pipelines environment
CMD ["/bin/bash", "-c", "source activate cubix_mlops_pipelines && streamlit run monitor_with_streamlit_train_data.py --server.port 8081 & python app.py"]
