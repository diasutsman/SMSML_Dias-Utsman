#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Script to build Docker image using MLflow's CLI functionality
"""

import os
import argparse
import subprocess
import shutil
import mlflow

def build_docker_image(run_id=None, model_uri=None, image_name="diasutsman/iris-classifier", enable_mlserver=True):
    """
    Build a Docker image for serving the MLflow model using MLflow CLI
    
    Args:
        run_id: The MLflow run ID to use if model_uri is not provided
        model_uri: The model URI in MLflow format (e.g., "runs:/your-run-id/model")
        image_name: The name for the Docker image
        enable_mlserver: Whether to enable MLServer
    """
    # Clean up the model URI if needed
    if model_uri is None and run_id is not None:
        model_uri = f"runs:/{run_id}/model"
    elif model_uri is not None:
        # Handle file paths, especially on CI environments
        if model_uri.startswith('file://') and '/artifacts' in model_uri and not model_uri.endswith('/model'):
            model_uri = f"{model_uri}/model"
    
    # Validate the model URI
    if model_uri is None:
        raise ValueError("Either run_id or model_uri must be provided")
    
    print(f"Building Docker image for model: {model_uri}")
    
    # Validate Docker image name
    if not image_name or image_name.startswith('/'):
        # If image name starts with /, it's not valid
        image_name = image_name.lstrip('/')
        if ':' not in image_name:
            image_name = f"{image_name}:latest"
    
    # First attempt: Try MLflow CLI
    print(f"\n=== ATTEMPT 1: Building Docker image with MLflow CLI ===")
    cmd = f"mlflow models build-docker -n {image_name} "
    if model_uri:
        cmd += f"-m {model_uri} "
    elif run_id:
        cmd += f"-m runs:/{run_id}/model "
    if enable_mlserver:
        cmd += "--enable-mlserver"
    
    print(f"Building Docker image for model: {model_uri or f'runs:/{run_id}/model'}")
    print(f"Running command: {cmd}")
    
    try:
        result = subprocess.run(cmd, check=True, shell=True, 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        print("Docker image built successfully")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building Docker image with MLflow CLI: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        
        # Second attempt: Try creating a minimal Dockerfile directly
        print(f"\n=== ATTEMPT 2: Building with direct Dockerfile approach ===")
        return build_docker_image_fallback(run_id, model_uri, image_name)


def build_docker_image_fallback(run_id: str = None, model_uri: str = None, image_name: str = "diasutsman/iris-classifier") -> bool:
    """Fallback method to build Docker image using direct Dockerfile creation"""
    # Determine the model path
    model_path = None
    if model_uri and model_uri.startswith("file://"):
        model_path = model_uri[7:]  # Remove file:// prefix
    elif run_id:
        # Try to find model path from local mlruns directory
        try:
            run_path = os.path.join("mlruns", "0", run_id)
            if os.path.exists(os.path.join(run_path, "artifacts", "model")):
                model_path = os.path.join(run_path, "artifacts", "model")
        except Exception as e:
            print(f"Error finding model path from run_id: {e}")
    
    # Use artifacts/model directory as fallback
    if not model_path or not os.path.exists(model_path):
        model_path = "artifacts/model"
    
    print(f"Using model path for Docker build: {model_path}")
    
    # Check if required model files exist
    required_files = ["MLmodel", "conda.yaml", "model.pkl"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_files:
        print(f"Warning: Missing required model files: {missing_files}")
        return False
    
    # Create a temporary Dockerfile
    docker_dir = "docker_build_temp"
    os.makedirs(docker_dir, exist_ok=True)
    
    with open(os.path.join(docker_dir, "Dockerfile"), "w") as f:
        f.write("""\
FROM python:3.9-slim

WORKDIR /app

RUN pip install mlflow>=2.19.0 scikit-learn pandas numpy flask gunicorn

COPY model /app/model

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "/app/model", "-p", "5000", "--host", "0.0.0.0"]
""")
    
    # Copy model files to docker build directory
    model_docker_dir = os.path.join(docker_dir, "model")
    os.makedirs(model_docker_dir, exist_ok=True)
    
    try:
        for file in required_files:
            src = os.path.join(model_path, file)
            dst = os.path.join(model_docker_dir, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Build the Docker image
        build_cmd = f"docker build -t {image_name} {docker_dir}"
        print(f"Building Docker image with command: {build_cmd}")
        
        result = subprocess.run(build_cmd, check=True, shell=True, 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        print("Docker image built successfully with fallback method")
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Error in fallback Docker build: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Command output: {e.stdout}")
            print(f"Command error: {e.stderr}")
        return False
    finally:
        # Clean up
        try:
            shutil.rmtree(docker_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary build directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Docker image for MLflow model")
    parser.add_argument("--run-id", type=str, help="MLflow run ID to use")
    parser.add_argument("--model-uri", type=str, help="MLflow model URI (overrides run-id)")
    parser.add_argument("--image-name", type=str, default="diasutsman/iris-classifier", 
                        help="Name for the Docker image")
    parser.add_argument("--enable-mlserver", action="store_true", default=True,
                        help="Enable MLServer for model serving")
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI (use DagsHub if configured)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    else:
        mlflow_uri = "http://localhost:5000"
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    
    print(f"Using MLflow tracking URI: {mlflow_uri}")
    
    build_docker_image(
        run_id=args.run_id,
        model_uri=args.model_uri,
        image_name=args.image_name,
        enable_mlserver=args.enable_mlserver
    )
