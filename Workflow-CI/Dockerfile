FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY MLProject/conda.yaml .

# Install dependencies
RUN pip install --no-cache-dir mlflow==2.19.0 \
    scikit-learn>=1.2.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    matplotlib>=3.5.0 \
    seaborn>=0.12.0 \
    joblib>=1.1.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0

# Copy model and server code
COPY MLProject/artifacts/model /app/model
COPY MLProject/inference.py /app/inference.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
