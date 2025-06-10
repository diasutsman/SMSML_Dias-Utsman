#!/usr/bin/env python
"""
Script to create a placeholder sklearn model for MLflow model artifacts.
"""
import sys
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def create_model_pkl(output_path):
    """Create a placeholder model.pkl file at the specified path."""
    # Create a minimal sklearn model
    model = RandomForestClassifier(n_estimators=10)
    model.fit(np.array([[5.1, 3.5, 1.4, 0.2]]), [0])
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Successfully created placeholder model.pkl file at {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_model_pkl.py <output_path>")
        sys.exit(1)
    
    output_path = sys.argv[1]
    success = create_model_pkl(output_path)
    if not success:
        sys.exit(1)
