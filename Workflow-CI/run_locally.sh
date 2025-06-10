#!/bin/bash
# Local execution script for MLflow workflow
# This script replicates the GitHub Actions workflow steps locally

set -e  # Exit on error

# Check for Python3
PYTHON_CMD=""
for cmd in "python3" "python" ; do
    if command -v $cmd &> /dev/null; then
        echo "Found Python command: $cmd"
        PYTHON_CMD=$cmd
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python not found! Please install Python 3.x before continuing."
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON_CMD --version | cut -d" " -f2)
echo "Using Python version: $PY_VERSION"

echo "=== Setting up environment ==="
# Create a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install mlflow==2.19.0 scikit-learn pandas numpy matplotlib seaborn joblib pytest docker dagshub
pip install "mlflow[extras]"
# Check that MLflow was installed correctly
mlflow --version || { echo "Error: MLflow installation failed"; exit 1; }

echo "=== Configuring MLflow tracking ==="
# Configure MLflow to use local directory for tracking
mkdir -p mlruns

# Use a relative path for MLflow tracking to avoid permission issues
export MLFLOW_TRACKING_URI="file:./mlruns"

# Create an explicit artifacts directory with full permissions
mkdir -p "$(pwd)/artifacts"
chmod -R 777 "$(pwd)/artifacts"

# Set MLflow to use the artifact directory with proper permissions
export MLFLOW_ARTIFACT_ROOT="file://$(pwd)/artifacts"
export MLFLOW_EXPERIMENT_BASE_PATH="$(pwd)/mlruns"
echo "Using MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo "Using MLflow artifact root: $MLFLOW_ARTIFACT_ROOT"

# Initialize local experiment
python -c "\
import mlflow
from mlflow.exceptions import MlflowException
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')

try:
    experiment = mlflow.get_experiment_by_name('Iris-Classification-CI')
    if experiment:
        print('Using existing experiment: Iris-Classification-CI')
    else:
        experiment_id = mlflow.create_experiment('Iris-Classification-CI')
        print(f'Created new experiment: {experiment_id}')
except MlflowException as e:
    experiment_id = mlflow.create_experiment('Iris-Classification-CI')
    print(f'Created new experiment: {experiment_id}')

print('Local MLflow tracking initialized')
"

echo "=== Running MLflow Project ==="
# Run the model training script
cd MLProject
python modelling.py --data_path="../namadataset_preprocessing/iris_preprocessed.csv" --tune_hyperparameters=true
cd ..

echo "=== Getting Run ID and Model Path ==="
# Find MLflow experiment ID
EXPERIMENT_ID=$(python -c "\
import mlflow
import os
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
print('Searching for experiment: Iris-Classification-CI')
print(f'Current MLflow tracking URI: {mlflow.get_tracking_uri()}')
exp = mlflow.get_experiment_by_name('Iris-Classification-CI')
if exp:
    print(f'Found experiment with ID: {exp.experiment_id}')
    print(exp.experiment_id)
else:
    print('Experiment not found!')
    print('')
" || echo "")

if [ -n "$EXPERIMENT_ID" ]; then
    echo "Found experiment ID: $EXPERIMENT_ID"
    
    # Get the latest run ID from the experiment
    LATEST_RUN_ID=$(python -c "\
import mlflow
import os
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
print('Searching for runs in experiment ID: $EXPERIMENT_ID')
try:
    runs = mlflow.search_runs(experiment_ids=['$EXPERIMENT_ID'], max_results=1, order_by=['start_time DESC'])
    if not runs.empty:
        run_id = runs['run_id'].iloc[0]
        print('Found latest run with ID: ' + run_id)
        print(run_id)
    else:
        print('No runs found!')
        print('')
except Exception as e:
    print('Error searching for runs: ' + str(e))
    print('')
" || echo "")
    
    if [ -n "$LATEST_RUN_ID" ]; then
        echo "Found Run ID: ${LATEST_RUN_ID}"
        
        # Find the model path from the run ID
        MODEL_DIR=$(python -c "\
import mlflow
import os
mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI')
print('Getting artifact path for run: $LATEST_RUN_ID')
try:
    run = mlflow.get_run('$LATEST_RUN_ID')
    artifact_uri = run.info.artifact_uri
    print('Raw artifact URI: ' + artifact_uri)
    
    # Handle relative paths properly
    if artifact_uri.startswith('file:'):
        if artifact_uri.startswith('file:./') or artifact_uri.startswith('file:.//'):  
            # For relative paths, convert to absolute for Docker
            project_dir = os.path.abspath(os.getcwd())
            rel_path = artifact_uri[5:].lstrip('/')
            artifact_uri = os.path.join(project_dir, rel_path)
        else:
            # For absolute paths, just strip the file: prefix
            artifact_uri = artifact_uri[5:]
    
    model_path = os.path.join(artifact_uri, 'model')
    print('Final model path: ' + model_path)
    print(model_path)
except Exception as e:
    print('Error getting model path: ' + str(e))
    print('')
" || echo "")
        
        if [ -n "$MODEL_DIR" ]; then
            echo "Found model at: ${MODEL_DIR}"
        fi
    else
        echo "No run ID found!"
    fi
else
    echo "No experiment ID found!"
fi

echo "=== Building Docker Image ==="
# Set Docker image name
IMAGE_NAME="iris-classifier:latest"
echo "Using Docker image name: ${IMAGE_NAME}"

# Create conda.yaml for the model if needed
if [ -n "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
    # Using echo statements instead of here-document to avoid syntax issues
    echo "channels:" > "$MODEL_DIR/conda.yaml"
    echo "- conda-forge" >> "$MODEL_DIR/conda.yaml"
    echo "dependencies:" >> "$MODEL_DIR/conda.yaml"
    echo "- python=3.9" >> "$MODEL_DIR/conda.yaml"
    echo "- pip" >> "$MODEL_DIR/conda.yaml"
    echo "- pip:" >> "$MODEL_DIR/conda.yaml"
    echo "  - mlflow>=2.19.0" >> "$MODEL_DIR/conda.yaml"
    echo "  - scikit-learn" >> "$MODEL_DIR/conda.yaml"
    echo "  - pandas" >> "$MODEL_DIR/conda.yaml"
    echo "  - numpy" >> "$MODEL_DIR/conda.yaml"
    echo "name: mlflow-env" >> "$MODEL_DIR/conda.yaml"
    echo "Created conda.yaml for model environment"
fi

# Build Docker image
if [ -n "$LATEST_RUN_ID" ]; then
    echo "Building Docker image using Run ID: $LATEST_RUN_ID"
    python build_docker_mlflow.py --run-id $LATEST_RUN_ID --image-name "${IMAGE_NAME}"
elif [ -n "$MODEL_DIR" ]; then
    echo "Building Docker image using model path: $MODEL_DIR"
    
    echo "Checking contents of model directory:"
    ls -la "$MODEL_DIR"
    
    # Build Docker image with absolute path
    ABSOLUTE_MODEL_PATH="$(pwd)/$MODEL_DIR"
    echo "Using absolute path: ${ABSOLUTE_MODEL_PATH}"
    python build_docker_mlflow.py --model-uri "file://${ABSOLUTE_MODEL_PATH}" --image-name "${IMAGE_NAME}"
else
    echo "Error: No run ID or model path found!"
    exit 1
fi

echo "=== Testing the Model Deployment ==="
echo "Starting model server in the background..."
docker run -d -p 5001:8080 ${IMAGE_NAME}

echo "Waiting for server to start..."
sleep 15

echo "Testing with a sample request..."
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"], "data": [[5.1, 3.5, 1.4, 0.2]]}'

echo -e "\n\n=== Workflow completed! ==="
echo "To stop the Docker container, use: docker ps and docker stop <container_id>"

# Deactivate virtual environment
deactivate
