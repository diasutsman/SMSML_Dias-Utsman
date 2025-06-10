#!/usr/bin/env python3
"""
Direct Docker build script for MLflow models
This script builds a Docker image directly without using mlflow models build-docker
"""

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path


def build_docker_image(model_dir, image_name, dockerfile_template):
    """
    Build a Docker image for an MLflow model using a direct Dockerfile approach
    
    Args:
        model_dir (str): Path to the model directory containing MLmodel, conda.yaml, and model.pkl
        image_name (str): Name for the Docker image
        dockerfile_template (str): Path to the Dockerfile template
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Building Docker image using direct approach")
    print(f"Model directory: {model_dir}")
    print(f"Image name: {image_name}")
    
    # Create a temporary build directory
    build_dir = Path("docker_build_temp")
    model_artifacts_dir = build_dir / "model_artifacts"
    
    try:
        # Create build directory
        build_dir.mkdir(exist_ok=True)
        model_artifacts_dir.mkdir(exist_ok=True)
        
        # Copy Dockerfile template
        with open(dockerfile_template, 'r') as src_file:
            with open(build_dir / "Dockerfile", 'w') as dest_file:
                dest_file.write(src_file.read())
        
        # Check for required model files
        required_files = ["MLmodel", "conda.yaml", "model.pkl"]
        model_dir_path = Path(model_dir)
        
        missing_files = []
        for file in required_files:
            if not (model_dir_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"Error: Missing required model files: {missing_files}")
            print(f"Please ensure all required files exist in {model_dir}")
            return False
        
        # Copy model files to build directory
        for file in os.listdir(model_dir):
            src = os.path.join(model_dir, file)
            dst = os.path.join(model_artifacts_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        
        # Display contents of build directory for debugging
        print("Files copied to build directory:")
        subprocess.run(f"ls -la {model_artifacts_dir}", shell=True, check=False)
        
        # Build the Docker image
        build_cmd = f"docker build -t {image_name} {build_dir}"
        print(f"Running Docker build command: {build_cmd}")
        
        process = subprocess.run(
            build_cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Docker image built successfully")
        print(f"Image name: {image_name}")
        return True
        
    except Exception as e:
        print(f"Error building Docker image: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Command output: {e.stdout}")
            print(f"Command error: {e.stderr}")
        return False
        
    finally:
        # Clean up build directory
        if build_dir.exists():
            shutil.rmtree(build_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Docker image for MLflow model")
    parser.add_argument("--model-dir", required=True, type=str, 
                        help="Path to the model directory containing MLmodel, conda.yaml, and model.pkl")
    parser.add_argument("--image-name", required=True, type=str,
                        help="Name for the Docker image")
    parser.add_argument("--dockerfile", type=str, default=".github/scripts/Dockerfile.template",
                        help="Path to the Dockerfile template")
    
    args = parser.parse_args()
    
    success = build_docker_image(
        model_dir=args.model_dir,
        image_name=args.image_name,
        dockerfile_template=args.dockerfile
    )
    
    sys.exit(0 if success else 1)
