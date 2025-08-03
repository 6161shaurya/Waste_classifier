# Dockerfile

# Use an official Python runtime as a parent image
# Using 3.9-slim-buster for broad compatibility, matching your Anaconda environment's likely version.
FROM python:3.9-slim-buster 

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install Python dependencies
# This is done separately to leverage Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container's /app folder.
# This copies all your Python files, including the updated app.py, classifier_model.py, utils/, etc.
COPY . .

# Create necessary directories within the container where your scripts write files.
# 'models' for the trained model, 'data' for the dataset download, 'output' (though less used here).
RUN mkdir -p data models output

# Command to execute when the container starts.
# THIS IS THE CRITICAL LINE for ensuring the correct app runs.
# It runs 'streamlit run app.py' directly.
# --server.port 8501: Tells Streamlit to run on this port.
# --server.enableCORS false --server.enableXsrfProtection false: Needed for public cloud deployment.
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]

# Expose the port Streamlit runs on.
EXPOSE 8501