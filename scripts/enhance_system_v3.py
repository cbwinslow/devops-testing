#!/usr/bin/env python3

import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Dict, List
import joblib
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from quantum_web import QuantumWeb
from quantum_metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhance_system_v3')

class ModelEnhancer:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("quantum_system_models")
        
    def create_model_signature(self, model_type: str) -> ModelSignature:
        """Create MLflow model signature based on model type"""
        if model_type == "resource":
            input_schema = Schema([
                ColSpec("double", "current_usage"),
                ColSpec("integer", "total_processes"),
                ColSpec("integer", "hour"),
                ColSpec("integer", "day_of_week")
            ])
            output_schema = Schema([ColSpec("double", "predicted_usage")])
        elif model_type == "anomaly":
            input_schema = Schema([
                ColSpec("double", "cpu_usage"),
                ColSpec("double", "memory_usage"),
                ColSpec("double", "disk_usage"),
                ColSpec("double", "network_usage")
            ])
            output_schema = Schema([ColSpec("integer", "is_anomaly")])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return ModelSignature(inputs=input_schema, outputs=output_schema)
        
    def create_example_input(self, model_type: str) -> pd.DataFrame:
        """Create example input data for model"""
        if model_type == "resource":
            return pd.DataFrame({
                "current_usage": [45.5],
                "total_processes": [100],
                "hour": [14],
                "day_of_week": [2]
            })
        elif model_type == "anomaly":
            return pd.DataFrame({
                "cpu_usage": [75.0],
                "memory_usage": [65.0],
                "disk_usage": [80.0],
                "network_usage": [45.0]
            })
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def train_and_log_models(self):
        """Train and log models with signatures and examples"""
        try:
            # Get training data
            history = self.metrics_collector.metrics_history[-1000:]  # Use last 1000 data points
            if len(history) < 24:  # Need at least 24 hours of data
                logger.warning("Insufficient historical data for training")
                return
                
            # Prepare features and targets for resource models
            resource_features = []
            targets = {resource: [] for resource in ['cpu', 'memory', 'disk', 'network']}
            
            for metrics in history:
                feature = [
                    metrics['cpu']['usage'],
                    metrics['system']['processes']['total'],
                    float(metrics['time_features']['hour']),
                    float(metrics['time_features']['day_of_week'])
                ]
                resource_features.append(feature)
                
                for resource in targets.keys():
                    targets[resource].append(metrics[f'{resource}.usage'])
                    
            X_resource = StandardScaler().fit_transform(np.array(resource_features))
            
            # Train and log resource prediction models
            resource_signature = self.create_model_signature("resource")
            resource_example = self.create_example_input("resource")
            
            for resource in targets.keys():
                with mlflow.start_run(run_name=f"{resource}_model"):
                    model = RandomForestRegressor(n_estimators=100)
                    model.fit(X_resource, targets[resource])
                    
                    mlflow.sklearn.log_model(
                        model,
                        f"{resource}_model",
                        signature=resource_signature,
                        input_example=resource_example
                    )
                    
                    # Save model locally
                    joblib.dump(model, f'static/models/{resource}_model.joblib')
                    
            # Prepare features for anomaly detection
            anomaly_features = []
            for metrics in history:
                anomaly_features.append([
                    metrics['cpu']['usage'],
                    metrics['memory']['usage'],
                    metrics['disk']['usage'],
                    metrics['network']['total_sent'] + metrics['network']['total_recv']
                ])
                
            X_anomaly = StandardScaler().fit_transform(np.array(anomaly_features))
            
            # Train and log anomaly detection model
            with mlflow.start_run(run_name="anomaly_model"):
                anomaly_model = IsolationForest(contamination=0.1)
                anomaly_model.fit(X_anomaly)
                
                mlflow.sklearn.log_model(
                    anomaly_model,
                    "anomaly_model",
                    signature=self.create_model_signature("anomaly"),
                    input_example=self.create_example_input("anomaly")
                )
                
                # Save model locally
                joblib.dump(anomaly_model, 'static/models/anomaly_model.joblib')
                
            logger.info("Successfully trained and logged all models")
            
        except Exception as e:
            logger.error(f"Error in model training and logging: {e}")
            raise

def main():
    try:
        # Create necessary directories
        Path('static/models').mkdir(parents=True, exist_ok=True)
        Path('static/data').mkdir(parents=True, exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        
        # Initialize and run model enhancement
        enhancer = ModelEnhancer()
        enhancer.train_and_log_models()
        
        # Initialize and run web interface
        quantum_web = QuantumWeb()
        
        # Run web server
        config = uvicorn.Config(
            quantum_web.app,
            host="0.0.0.0",
            port=45678,  # Use the port that worked previously
            log_level="info"
        )
        server = uvicorn.Server(config)
        server.run()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

