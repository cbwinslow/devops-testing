#!/usr/bin/env python3

import os
import sys
import logging
import json
import time
import asyncio
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from functools import wraps
import ipaddress

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier
from transformers import pipeline
import mlflow
import optuna
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import uvicorn
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import jwt
from passlib.hash import pbkdf2_sha256
from keycloak import KeycloakOpenID
from enum import Enum
from datetime import datetime, timedelta

class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class AuditLogger:
    def __init__(self, log_file="logs/audit.log"):
        self.log_file = log_file
        
    def log(self, level: LogLevel, event: str, **kwargs):
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level.value,
            "event": event,
            **kwargs
        }
        with open(self.log_file, "a") as f:
            f.write(f"{log_entry}\n")
import redis
import safety

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhance_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Circuit breaker implementation
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    self.state = "half-open"
                else:
                    raise HTTPException(status_code=503, detail="Service temporarily unavailable")

            try:
                result = await func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                raise

        return wrapper

class ModelManager:
    """Enhanced ML Models Manager with versioning and persistence"""
    
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models = {}
        self.model_versions = {}
        self.initialize_mlflow()
        self.initialize_models()
        
    def initialize_mlflow(self):
        """Initialize MLflow for experiment tracking"""
        db_path = os.path.join(self.models_dir, "mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        try:
            # Try to get or create the experiment
            experiment = mlflow.get_experiment_by_name("model_training")
            if experiment:
                if experiment.lifecycle_stage == "deleted":
                    # Restore the experiment if it's deleted
                    mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
                mlflow.set_experiment("model_training")
            else:
                mlflow.create_experiment("model_training")
            logger.info("Successfully initialized MLflow experiment")
        except Exception as e:
            logger.error(f"Error initializing MLflow: {e}")
            # Try one more time with a different name
            try:
                experiment_name = f"model_training_{int(time.time())}"
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                logger.info(f"Created fallback experiment: {experiment_name}")
            except Exception as e2:
                logger.error(f"Error creating fallback experiment: {e2}")
                raise
        
    def save_model(self, model_name: str, model, metrics: Dict = None):
        """Save model with versioning"""
        version = self.model_versions.get(model_name, 0) + 1
        
        with mlflow.start_run():
            # Log model and metrics
            if metrics:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
            
            # Save model using appropriate MLflow flavor
            if isinstance(model, (RandomForestClassifier, IsolationForest)):
                mlflow.sklearn.log_model(model, model_name)
            elif isinstance(model, XGBRegressor):
                mlflow.xgboost.log_model(model, model_name)
            elif isinstance(model, LGBMClassifier):
                mlflow.lightgbm.log_model(model, model_name)
            elif isinstance(model, nn.Module):
                mlflow.pytorch.log_model(model, model_name)
            
        self.model_versions[model_name] = version
        logger.info(f"Saved {model_name} version {version}")
        
    def load_model(self, model_name: str, version: Optional[int] = None):
        """Load model with specific version"""
        if version is None:
            version = self.model_versions.get(model_name, 1)
            
        try:
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
            self.models[model_name] = model
            logger.info(f"Loaded {model_name} version {version}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
            
    def initialize_models(self):
        """Initialize various ML models with hyperparameter optimization"""
        try:
            # Generate synthetic data for validation
            def generate_data_anomaly(n_samples=1000):
                np.random.seed(42)
                X = np.random.randn(n_samples, 10)
                outliers = np.random.randn(int(n_samples * 0.1), 10) * 3
                X = np.vstack([X, outliers])
                return X

            def generate_data_timeseries(n_samples=1000):
                np.random.seed(42)
                X = np.random.randn(n_samples, 10)
                y = np.sin(np.linspace(0, 10, n_samples)) + np.random.normal(0, 0.1, n_samples)
                return X, y

            def objective_anomaly(trial):
                contamination = trial.suggest_float("contamination", 0.01, 0.2)
                model = IsolationForest(contamination=contamination, random_state=42)
                X = generate_data_anomaly()
                model.fit(X)
                # Use decision function as validation score
                scores = model.decision_function(X)
                return np.mean(scores)

            def objective_timeseries(trial):
                n_estimators = trial.suggest_int("n_estimators", 50, 200)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
                model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
                X, y = generate_data_timeseries()
                # Simple time series validation
                split = int(len(X) * 0.8)
                X_train, y_train = X[:split], y[:split]
                X_test, y_test = X[split:], y[split:]
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                return -np.mean((pred - y_test) ** 2)  # Negative because we're maximizing

            # Optimize hyperparameters
            study = optuna.create_study(direction="maximize")
            study.optimize(objective_anomaly, n_trials=20)
            best_params_anomaly = study.best_params

            study = optuna.create_study(direction="maximize")
            study.optimize(objective_timeseries, n_trials=20)
            best_params_timeseries = study.best_params

            # Initialize and fit models with optimized parameters
            X_anomaly = generate_data_anomaly()
            X_timeseries, y_timeseries = generate_data_timeseries()
            X_class = np.random.randn(1000, 10)
            y_class = np.random.randint(0, 2, 1000)

            self.models['anomaly'] = IsolationForest(**best_params_anomaly)
            self.models['anomaly'].fit(X_anomaly)

            self.models['timeseries'] = XGBRegressor(**best_params_timeseries)
            self.models['timeseries'].fit(X_timeseries, y_timeseries)

            self.models['classification'] = LGBMClassifier(n_estimators=100, learning_rate=0.1)
            self.models['classification'].fit(X_class, y_class)
            class SimpleNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = nn.Sequential(
                        nn.Linear(4, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )
                
                def forward(self, x):
                    return self.model(x)
            
            self.models['deep_learning'] = SimpleNN()
            self.models['nlp'] = pipeline(
                "text-classification",
                model="distilbert-base-uncased"
            )

            # Save initial versions
            for name, model in self.models.items():
                self.save_model(name, model)
                
            logger.info("Successfully initialized all ML models")
        except Exception as e:
            logger.error(f"Error initializing ML models: {str(e)}")
            raise

class SecurityManager:
    """Enhanced security manager with additional features"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.keycloak_client = KeycloakOpenID(
            server_url="http://localhost:8080/auth/",
            client_id="my-client",
            realm_name="my-realm",
            client_secret_key="my-secret"
        )
        self.allowed_ips = set()
        self.audit_logger = AuditLogger()
        # self.safety_checker = safety.scan
        
    def add_allowed_ip(self, ip: str):
        """Add IP to whitelist"""
        try:
            ipaddress.ip_address(ip)
            self.allowed_ips.add(ip)
        except ValueError:
            raise ValueError("Invalid IP address")
            
    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        return ip in self.allowed_ips
        
    def generate_token(self, user_id: str) -> str:
        """Generate JWT token with enhanced security"""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow()
        }
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        self.audit_logger.log(
            level=LogLevel.INFO,
            event="token_generated",
            user_id=user_id
        )
        return token
        
    def verify_token(self, token: str) -> bool:
        """Verify JWT token with enhanced security"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            self.audit_logger.log(
                level=LogLevel.INFO,
                event="token_verified",
                user_id=payload.get("user_id")
            )
            return True
        except jwt.ExpiredSignatureError:
            self.audit_logger.log(
                level=LogLevel.WARNING,
                event="token_expired"
            )
            return False
        except jwt.InvalidTokenError:
            self.audit_logger.log(
                level=LogLevel.WARNING,
                event="token_invalid"
            )
            return False
            
    async def scan_dependencies(self):
        """Scan dependencies for security vulnerabilities"""
        try:
            # Disabled safety scanning due to import issues
            # vulns = self.safety_checker()
            # if vulns:
            #     self.audit_logger.log(
            #         level=LogLevel.WARNING,
            #         event="security_vulnerabilities_found",
            #         vulnerabilities=vulns
            #     )
            # return vulns
            return None
        except Exception as e:
            logger.error(f"Error scanning dependencies: {str(e)}")
            raise

class MonitoringSystem:
    """Enhanced monitoring system with advanced metrics and error tracking"""
    
    def __init__(self):
        self.metrics = {
            'requests': Counter('http_requests_total', 'Total HTTP requests'),
            'response_time': Histogram('response_time_seconds', 'Response time in seconds'),
            'system_load': Gauge('system_load', 'System load average'),
            'memory_usage': Gauge('memory_usage_bytes', 'Memory usage in bytes'),
            'error_rate': Counter('error_rate', 'Error rate by type'),
            'model_prediction_latency': Histogram('model_prediction_latency', 'Model prediction latency'),
            'model_accuracy': Gauge('model_accuracy', 'Model accuracy by type'),
            'circuit_breaker_trips': Counter('circuit_breaker_trips', 'Number of circuit breaker trips')
        }
        
    def start_monitoring(self, port: int = 9090):
        """Start the monitoring server"""
        try:
            # Try to find an available port
            max_attempts = 10
            current_port = port
            while max_attempts > 0:
                try:
                    start_http_server(current_port)
                    logger.info(f"Started monitoring server on port {current_port}")
                    return
                except OSError:
                    current_port += 1
                    max_attempts -= 1
            raise RuntimeError("Could not find an available port for monitoring server")
        except Exception as e:
            logger.error(f"Failed to start monitoring server: {str(e)}")
            raise

class WebInterface:
    """Enhanced web interface with additional security and error handling"""
    
    def __init__(self, model_manager: ModelManager, security_manager: SecurityManager,
                 monitoring_system: MonitoringSystem):
        self.app = FastAPI()
        self.model_manager = model_manager
        self.security_manager = security_manager
        self.monitoring = monitoring_system
        self.circuit_breaker = CircuitBreaker()
        self.setup_routes()
        asyncio.create_task(self.setup_middleware())
        
    async def setup_middleware(self):
        """Setup enhanced middleware with rate limiting and security"""
        try:
            # Add rate limiting middleware
            @self.app.middleware("http")
            async def rate_limit_middleware(request: Request, call_next):
                client_ip = request.client.host
                # Simple in-memory rate limiting using a dictionary
                if not hasattr(self, '_rate_limit_store'):
                    self._rate_limit_store = {}
                
                now = time.time()
                # Clean up old entries
                self._rate_limit_store = {k: v for k, v in self._rate_limit_store.items()
                                        if now - v['timestamp'] < 60}
                
                if client_ip in self._rate_limit_store:
                    entry = self._rate_limit_store[client_ip]
                    if entry['count'] >= 100:  # 100 requests per minute
                        raise HTTPException(status_code=429, detail="Too many requests")
                    entry['count'] += 1
                else:
                    self._rate_limit_store[client_ip] = {'count': 1, 'timestamp': now}
                
                return await call_next(request)
            
            logger.info("Successfully initialized rate limiter middleware")
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter middleware: {e}")
            # Continue without rate limiting rather than failing
            pass
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def setup_routes(self):
        """Setup enhanced API routes with security and monitoring"""
        
        @self.app.middleware("http")
        async def security_middleware(request: Request, call_next):
            # Check IP whitelist
            client_ip = request.client.host
            if not self.security_manager.is_ip_allowed(client_ip):
                self.security_manager.audit_logger.log(
                    level=LogLevel.WARNING,
                    event="unauthorized_ip_access",
                    ip=client_ip
                )
                raise HTTPException(status_code=403, detail="IP not allowed")
            
            start_time = time.time()
            response = await call_next(request)
            
            # Log request metrics
            self.monitoring.metrics['response_time'].observe(time.time() - start_time)
            self.monitoring.metrics['requests'].inc()
            
            return response

        @self.app.websocket("/ws")
        @self.circuit_breaker
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    # Process data with error handling and monitoring
                    start_time = time.time()
                    try:
                        result = await self.process_data(data)
                        self.monitoring.metrics['model_prediction_latency'].observe(
                            time.time() - start_time
                        )
                        await websocket.send_text(json.dumps(result))
                    except Exception as e:
                        logger.error(f"Error processing data: {str(e)}")
                        self.monitoring.metrics['error_rate'].inc()
                        await websocket.send_text(json.dumps({"error": str(e)}))
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                self.monitoring.metrics['error_rate'].inc()
                
        @self.app.get("/metrics")
        async def get_metrics():
            # Return system metrics with monitoring
            try:
                metrics = {
                    name: metric._value.get()
                    for name, metric in self.monitoring.metrics.items()
                }
                return metrics
            except Exception as e:
                logger.error(f"Error getting metrics: {str(e)}")
                self.monitoring.metrics['error_rate'].inc()
                raise HTTPException(status_code=500, detail=str(e))
            
        @self.app.post("/predict")
        @self.circuit_breaker
        async def predict(data: Dict):
            # Make predictions with error handling and monitoring
            start_time = time.time()
            try:
                model_name = data.get("model", "classification")
                model = self.model_manager.models.get(model_name)
                if not model:
                    raise HTTPException(status_code=404, detail="Model not found")
                    
                result = model.predict(data["features"])
                self.monitoring.metrics['model_prediction_latency'].observe(
                    time.time() - start_time
                )
                return {"prediction": result.tolist()}
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                self.monitoring.metrics['error_rate'].inc()
                raise HTTPException(status_code=500, detail=str(e))

async def main():
    """Main function to run the enhanced system"""
    try:
        # Initialize components
        model_manager = ModelManager()
        monitoring = MonitoringSystem()
        security_manager = SecurityManager("your-secret-key")
        
        # Add some allowed IPs
        security_manager.add_allowed_ip("127.0.0.1")
        
        # Start security scanning
        await security_manager.scan_dependencies()
        
        # Initialize web interface
        web_interface = WebInterface(model_manager, security_manager, monitoring)
        
        # Start monitoring
        monitoring.start_monitoring()
        
        # Start web interface with fixed port
        config = uvicorn.Config(
            web_interface.app,
            host="0.0.0.0",
            port=45678,
            log_level="info",
            reload=False,  # Disable reload mode
            workers=1  # Single worker
        )
        server = uvicorn.Server(config)
        logger.info("Starting web interface on http://0.0.0.0:45678")

        # Setup shutdown handler
        def handle_shutdown(signal_name):
            logger.info(f"Received {signal_name}, shutting down...")
            server.should_exit = True

        for sig in ['SIGINT', 'SIGTERM']:
            if hasattr(signal, sig):
                signal.signal(getattr(signal, sig), lambda s, f: handle_shutdown(sig))

        # Start server
        await server.serve()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

