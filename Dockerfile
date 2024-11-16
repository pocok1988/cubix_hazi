# Base Docker image: Anaconda
FROM continuumio/anaconda3

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

# Activate the environment and install additional pip dependencies (flask-restx and streamlit specifically)
RUN /bin/bash -c "source activate cubix_mlops_pipelines"
RUN /bin/bash -c "pip install flask-restx"
RUN /bin/bash -c "pip install streamlit"
RUN /bin/bash -c "pip install evidently"
RUN /bin/bash -c "pip install mlflow"

# Set environment variables
ENV MLFLOW_TRACKING_URI="http://192.168.11.10:12650"

# Start MLflow server and app.py in the cubix_mlops_pipelines environment
CMD ["/bin/bash", "-c", "source activate cubix_mlops_pipelines && streamlit run monitor_with_streamlit_train_data.py --server.port 8081 & python app.py"]
