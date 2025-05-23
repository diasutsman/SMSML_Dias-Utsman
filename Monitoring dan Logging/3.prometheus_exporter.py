#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Prometheus exporter for monitoring ML model
"""

from prometheus_client import Counter, Gauge, Summary, Histogram
import time
import random
import http.server
from prometheus_client import start_http_server, generate_latest, CONTENT_TYPE_LATEST
import threading

# Create metrics
REQUEST_COUNT = Counter('iris_api_requests_total', 'Total number of requests received')
REQUEST_LATENCY = Summary('iris_api_request_latency_seconds', 'Request latency in seconds')
PREDICTION_COUNT = Counter('iris_model_predictions_total', 'Total number of predictions made', ['class_name'])
MODEL_CONFIDENCE = Histogram('iris_model_confidence', 'Confidence scores of model predictions', ['class_name'])
FEATURE_GAUGE = Gauge('iris_feature_values', 'Feature values used for prediction', ['feature'])
SYSTEM_MEMORY = Gauge('system_memory_usage_percent', 'System memory usage percentage')
SYSTEM_CPU = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')

# Class names
class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Simulate metrics for demo purposes
def simulate_metrics():
    while True:
        # Simulate requests
        REQUEST_COUNT.inc()
        
        # Simulate latency
        with REQUEST_LATENCY.time():
            # Simulate some processing time
            time.sleep(random.uniform(0.01, 0.1))
        
        # Simulate predictions
        prediction = random.randint(0, 2)
        PREDICTION_COUNT.labels(class_name=class_names[prediction]).inc()
        
        # Simulate confidence scores
        for i in range(3):
            conf = random.random() if i == prediction else random.random() * 0.5
            MODEL_CONFIDENCE.labels(class_name=class_names[i]).observe(conf)
        
        # Simulate feature values
        FEATURE_GAUGE.labels(feature='sepal_length').set(random.uniform(4.5, 7.5))
        FEATURE_GAUGE.labels(feature='sepal_width').set(random.uniform(2.0, 4.0))
        FEATURE_GAUGE.labels(feature='petal_length').set(random.uniform(1.0, 6.5))
        FEATURE_GAUGE.labels(feature='petal_width').set(random.uniform(0.1, 2.5))
        
        # Simulate system metrics
        SYSTEM_MEMORY.set(random.uniform(40, 90))
        SYSTEM_CPU.set(random.uniform(10, 80))
        
        time.sleep(1)

# Metrics HTTP handler
class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', CONTENT_TYPE_LATEST)
        self.end_headers()
        self.wfile.write(generate_latest())
    
    def log_message(self, format, *args):
        # Suppress log messages
        return

# Start the exporter
if __name__ == '__main__':
    # Start metrics server
    start_http_server(8000)
    print("Prometheus metrics server started on port 8000")
    
    # Start metrics simulation in a background thread
    sim_thread = threading.Thread(target=simulate_metrics, daemon=True)
    sim_thread.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
