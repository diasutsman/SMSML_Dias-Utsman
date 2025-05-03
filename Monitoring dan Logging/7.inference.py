#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Inference API with Prometheus monitoring
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import mlflow
import mlflow.sklearn
import uvicorn
import random
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST

# Create FastAPI app
app = FastAPI(title="Iris Classification API - Dias Utsman", 
              description="Model serving API with Prometheus monitoring")

# Define Prometheus metrics
REQUESTS = Counter('iris_api_requests_total', 'Total number of requests to the Iris API', ['method', 'endpoint', 'status'])
PREDICTIONS = Counter('iris_api_predictions_total', 'Total number of predictions made', ['class'])
PREDICTION_TIME = Histogram('iris_api_prediction_seconds', 'Time spent processing prediction request')
MODEL_CONFIDENCE = Histogram('iris_api_model_confidence', 'Confidence scores for predictions', ['class'])
REQUEST_LATENCY = Summary('iris_api_request_latency_seconds', 'Request latency in seconds', ['endpoint'])
FEATURE_GAUGE = Gauge('iris_api_feature_value', 'Feature values from requests', ['feature'])
PREDICTION_ERRORS = Counter('iris_api_prediction_errors', 'Prediction errors', ['error_type'])
SYSTEM_MEMORY = Gauge('iris_api_system_memory_bytes', 'System memory usage')
SYSTEM_CPU = Gauge('iris_api_system_cpu_percent', 'System CPU usage percent')
MODEL_LOAD_TIME = Gauge('iris_api_model_load_time_seconds', 'Time to load model')

# Track active requests
ACTIVE_REQUESTS = Gauge('iris_api_active_requests', 'Number of currently active requests')

# Input data model
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output prediction model
class IrisPrediction(BaseModel):
    prediction: int
    class_name: str
    probability: Dict[str, float]
    processing_time: float

# Load the model
def load_model():
    start_time = time.time()
    try:
        # Try loading from MLflow
        try:
            model_uri = "models:/iris-model/latest"
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            # Fallback to local model file
            model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
            model = joblib.load(model_path)
        
        MODEL_LOAD_TIME.set(time.time() - start_time)
        return model
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="model_load_error").inc()
        print(f"Error loading model: {e}")
        # For demo purposes, return a dummy model
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()

# Load the model
model = load_model()

# Class names mapping
class_names = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Middleware to track request latency
@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time() - start_time)
        REQUESTS.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
        return response
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="request_processing_error").inc()
        raise e
    finally:
        ACTIVE_REQUESTS.dec()

# Endpoint to expose metrics to Prometheus
@app.get("/metrics")
async def metrics():
    # Simulate system metrics for demo
    SYSTEM_MEMORY.set(random.randint(1000000, 2000000))
    SYSTEM_CPU.set(random.uniform(10, 90))
    
    return generate_latest()

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Iris Classification API - Dias Utsman</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                }
                a {
                    color: #2c7be5;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
                .endpoint {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-left: 4px solid #2c7be5;
                }
                code {
                    background-color: #eee;
                    padding: 2px 4px;
                    border-radius: 3px;
                }
                .footer {
                    margin-top: 30px;
                    text-align: center;
                    color: #666;
                    font-size: 0.85em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Iris Classification API</h1>
                <p>API for predicting Iris species based on flower measurements.</p>
                
                <div class="endpoint">
                    <h3>Predict Endpoint</h3>
                    <p>Make predictions by sending POST requests to <code>/predict</code> with JSON data.</p>
                    <p>Example:</p>
                    <pre><code>{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}</code></pre>
                </div>
                
                <div class="endpoint">
                    <h3>Metrics Endpoint</h3>
                    <p>Access Prometheus metrics at <a href="/metrics">/metrics</a></p>
                </div>
                
                <div class="endpoint">
                    <h3>Health Check</h3>
                    <p>Check API health at <a href="/health">/health</a></p>
                </div>
                
                <div class="footer">
                    <p>Created by Dias Utsman | MLOps System</p>
                </div>
            </div>
        </body>
    </html>
    """

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Prediction endpoint
@app.post("/predict", response_model=IrisPrediction)
def predict(features: IrisFeatures):
    # Start timing
    with PREDICTION_TIME.time():
        start_time = time.time()
        
        try:
            # Log feature values
            feature_array = np.array([[
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ]])
            
            # Update feature gauges
            FEATURE_GAUGE.labels(feature="sepal_length").set(features.sepal_length)
            FEATURE_GAUGE.labels(feature="sepal_width").set(features.sepal_width)
            FEATURE_GAUGE.labels(feature="petal_length").set(features.petal_length)
            FEATURE_GAUGE.labels(feature="petal_width").set(features.petal_width)
            
            # Make prediction
            prediction = int(model.predict(feature_array)[0])
            PREDICTIONS.labels(class=class_names[prediction]).inc()
            
            # Get probabilities if available
            probabilities = {}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feature_array)[0]
                for i, p in enumerate(proba):
                    class_name = class_names[i]
                    probabilities[class_name] = float(p)
                    MODEL_CONFIDENCE.labels(class=class_name).observe(p)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return prediction
            return {
                "prediction": prediction,
                "class_name": class_names[prediction],
                "probability": probabilities,
                "processing_time": processing_time
            }
            
        except Exception as e:
            PREDICTION_ERRORS.labels(error_type="prediction_error").inc()
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
