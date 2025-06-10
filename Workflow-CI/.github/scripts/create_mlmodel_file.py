#!/usr/bin/env python
"""
Script to create a proper MLmodel file for MLflow model artifacts.
"""
import sys
import uuid
import os

def create_mlmodel_file(output_path):
    """Create an MLmodel file at the specified path."""
    mlmodel_content = f"""artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.12.10
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.3.0
model_uuid: {str(uuid.uuid4())}
mlflow_version: 2.19.0
model_size_bytes: 4096
"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the MLmodel file
    with open(output_path, 'w') as f:
        f.write(mlmodel_content)
    print(f"Successfully created MLmodel file at {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_mlmodel_file.py <output_path>")
        sys.exit(1)
    
    output_path = sys.argv[1]
    success = create_mlmodel_file(output_path)
    if not success:
        sys.exit(1)
