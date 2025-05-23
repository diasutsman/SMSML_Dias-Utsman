#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Script to build Docker image using MLflow's built-in functionality
"""

import mlflow.models.docker
import os
import argparse

def build_docker_image(run_id, model_uri=None, image_name="diasutsman/iris-classifier", enable_mlserver=True):
    """
    Build a Docker image for serving the MLflow model
    
    Args:
        run_id: The MLflow run ID to use if model_uri is not provided
        model_uri: The model URI in MLflow format (e.g., "runs:/your-run-id/model")
        image_name: The name for the Docker image
        enable_mlserver: Whether to enable MLServer
    """
    if model_uri is None:
        model_uri = f"runs:/{run_id}/model"
    
    print(f"Building Docker image for model: {model_uri}")
    
    # Use MLflow's built-in function to build the Docker image
    mlflow.models.docker.build_docker_image(
        model_uri=model_uri,
        image_name=image_name,
        enable_mlserver=enable_mlserver
    )
    
    print(f"Docker image built successfully: {image_name}")
    print("\nTo run the Docker image locally:")
    print(f"docker run -p 5001:8080 {image_name}")
    
    print("\nTo push the image to Docker Hub:")
    print("docker login")
    print(f"docker push {image_name}")

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
