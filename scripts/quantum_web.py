#!/usr/bin/env python3

from fastapi import FastAPI, WebSocket, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import aiohttp
from quantum_metrics import MetricsCollector
from quantum_optimization_advanced import AdvancedOptimizationOrchestrator
from quantum_visualizer import QuantumVisualizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_web')

class QuantumWeb:
    """Web interface for quantum optimization system"""
    def __init__(self):
        self.app = FastAPI(title="Quantum Optimization System")
        self.setup_components()
        self.setup_routes()
        self.setup_ml_models()
        self.active_websockets: List[WebSocket] = []

    def setup_components(self):
        """Initialize system components"""
        self.metrics_collector = MetricsCollector()
        self.optimizer = AdvancedOptimizationOrchestrator()
        self.visualizer = QuantumVisualizer()
        
        # Create static directories
        Path('static').mkdir(parents=True, exist_ok=True)
        Path('static/models').mkdir(parents=True, exist_ok=True)
        Path('static/data').mkdir(parents=True, exist_ok=True)

    def setup_routes(self):
        """Setup FastAPI routes"""
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

        # API routes
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            return self.get_dashboard_html()

        @self.app.get("/api/metrics")
        async def get_metrics():
            return await self.metrics_collector.collect_system_metrics()

        @self.app.get("/api/health")
        async def get_health():
            metrics = await self.metrics_collector.collect_system_metrics()
            return {
                'status': 'healthy' if metrics['analysis']['system_health'] > 80 else 'warning',
                'score': metrics['analysis']['system_health'],
                'timestamp': datetime.now().isoformat()
            }

        @self.app.post("/api/optimize")
        async def optimize_system(background_tasks: BackgroundTasks):
            metrics = await self.metrics_collector.collect_system_metrics()
            background_tasks.add_task(self.run_optimization, metrics)
            return {"status": "optimization_started"}

        @self.app.get("/api/predictions")
        async def get_predictions(resource: str = 'cpu', hours: int = 24):
            return await self.predict_resource_usage(resource, hours)

        @self.app.get("/api/anomalies")
        async def get_anomalies():
            return await self.detect_anomalies()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_websockets.append(websocket)
            try:
                while True:
                    # Keep connection alive and send updates
                    metrics = await self.metrics_collector.collect_system_metrics()
                    await websocket.send_json(metrics)
                    await asyncio.sleep(1)
            except:
                self.active_websockets.remove(websocket)

    def setup_ml_models(self):
        """Initialize machine learning models"""
        self.models = {}
        model_path = Path('static/models')
        
        # Resource prediction models
        for resource in ['cpu', 'memory', 'disk', 'network']:
            model_file = model_path / f"{resource}_model.joblib"
            if model_file.exists():
                self.models[resource] = joblib.load(model_file)
            else:
                self.models[resource] = RandomForestRegressor(n_estimators=100)

        # Anomaly detection model
        anomaly_model_path = model_path / "anomaly_model.joblib"
        if anomaly_model_path.exists():
            self.anomaly_model = joblib.load(anomaly_model_path)
        else:
            self.anomaly_model = IsolationForest(contamination=0.1)

        # Feature scaler
        scaler_path = model_path / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = StandardScaler()

    async def run_optimization(self, metrics: Dict):
        """Run system optimization"""
        try:
            optimized, applied = self.optimizer.optimize(metrics)
            
            # Notify connected clients
            for websocket in self.active_websockets:
                await websocket.send_json({
                    'type': 'optimization_complete',
                    'result': applied
                })

            # Update ML models
            await self.update_models(metrics)

        except Exception as e:
            logger.error(f"Error in optimization: {e}")

    async def predict_resource_usage(self, resource: str, hours: int) -> Dict:
        """Predict future resource usage"""
        try:
            # Get historical data
            history = self.metrics_collector.metrics_history[-24:]  # Last 24 hours
            if not history:
                raise HTTPException(status_code=400, detail="Insufficient historical data")

            # Prepare features
            features = []
            for metrics in history:
                features.append([
                    metrics[f'{resource}.usage'],
                    metrics['system']['processes']['total'],
                    float(metrics['time_features']['hour']),
                    float(metrics['time_features']['day_of_week'])
                ])

            X = np.array(features)
            X = self.scaler.fit_transform(X)

            # Make prediction
            model = self.models[resource]
            future_features = X[-1].reshape(1, -1)
            predictions = []

            for hour in range(hours):
                pred = model.predict(future_features)[0]
                predictions.append({
                    'timestamp': (datetime.now() + pd.Timedelta(hours=hour)).isoformat(),
                    'value': float(pred)
                })
                # Update features for next prediction
                future_features[0][0] = pred

            return {
                'resource': resource,
                'predictions': predictions,
                'confidence': float(model.score(X, [m[f'{resource}.usage'] for m in history]))
            }

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def detect_anomalies(self) -> Dict:
        """Detect system anomalies"""
        try:
            # Get recent metrics
            recent_metrics = self.metrics_collector.metrics_history[-100:]  # Last 100 data points
            if not recent_metrics:
                raise HTTPException(status_code=400, detail="Insufficient data")

            # Prepare features
            features = []
            for metrics in recent_metrics:
                features.append([
                    metrics['cpu']['usage'],
                    metrics['memory']['usage'],
                    metrics['disk']['usage'],
                    metrics['network']['total_sent'] + metrics['network']['total_recv']
                ])

            X = self.scaler.fit_transform(np.array(features))
            
            # Detect anomalies
            predictions = self.anomaly_model.predict(X)
            anomaly_scores = self.anomaly_model.score_samples(X)

            # Find anomalies
            anomalies = []
            for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                if pred == -1:  # Anomaly detected
                    anomalies.append({
                        'timestamp': recent_metrics[i]['timestamp'],
                        'metrics': {
                            'cpu': recent_metrics[i]['cpu']['usage'],
                            'memory': recent_metrics[i]['memory']['usage'],
                            'disk': recent_metrics[i]['disk']['usage'],
                            'network': recent_metrics[i]['network']['total_sent'] + recent_metrics[i]['network']['total_recv']
                        },
                        'score': float(score)
                    })

            return {
                'total_anomalies': len(anomalies),
                'anomalies': anomalies,
                'analysis_timeframe': {
                    'start': recent_metrics[0]['timestamp'],
                    'end': recent_metrics[-1]['timestamp']
                }
            }

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def update_models(self, new_metrics: Dict):
        """Update ML models with new data"""
        try:
            # Update resource prediction models
            history = self.metrics_collector.metrics_history[-1000:]  # Use last 1000 data points
            if len(history) < 24:  # Need at least 24 hours of data
                return

            # Prepare features and targets
            features = []
            targets = {resource: [] for resource in ['cpu', 'memory', 'disk', 'network']}

            for metrics in history:
                feature = [
                    metrics['cpu']['usage'],
                    metrics['system']['processes']['total'],
                    float(metrics['time_features']['hour']),
                    float(metrics['time_features']['day_of_week'])
                ]
                features.append(feature)

                for resource in targets.keys():
                    targets[resource].append(metrics[f'{resource}.usage'])

            X = self.scaler.fit_transform(np.array(features))

            # Update each resource model
            for resource, y in targets.items():
                self.models[resource].fit(X, y)

            # Update anomaly detection model
            self.anomaly_model.fit(X)

            # Save updated models
            for resource, model in self.models.items():
                joblib.dump(model, f'static/models/{resource}_model.joblib')
            joblib.dump(self.anomaly_model, 'static/models/anomaly_model.joblib')
            joblib.dump(self.scaler, 'static/models/scaler.joblib')

        except Exception as e:
            logger.error(f"Error updating models: {e}")

    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Optimization System</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { 
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #1a1a1a;
                    color: #ffffff;
                }
                .dashboard {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                }
                .card {
                    background: #2d2d2d;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .full-width {
                    grid-column: 1 / -1;
                }
                .metric {
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }
                .status {
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                .status.healthy { background: #00ff00; color: #000000; }
                .status.warning { background: #ffff00; color: #000000; }
                .status.critical { background: #ff0000; color: #ffffff; }
                #optimizeBtn {
                    background: #00ff00;
                    color: #000000;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                #optimizeBtn:hover {
                    background: #00cc00;
                }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="card full-width">
                    <h2>System Health</h2>
                    <div id="healthScore" class="metric">Loading...</div>
                    <button id="optimizeBtn" onclick="optimizeSystem()">Optimize System</button>
                </div>
                <div class="card">
                    <h2>Resource Usage</h2>
                    <div id="resourceChart"></div>
                </div>
                <div class="card">
                    <h2>Predictions</h2>
                    <div id="predictionChart"></div>
                </div>
                <div class="card">
                    <h2>Anomalies</h2>
                    <div id="anomalyChart"></div>
                </div>
                <div class="card full-width">
                    <h2>System Metrics</h2>
                    <div id="metricsTable"></div>
                </div>
            </div>

            <script>
                let ws = new WebSocket(`ws://${window.location.host}/ws`);
                let resourceData = {
                    cpu: [],
                    memory: [],
                    disk: [],
                    network: []
                };

                ws.onmessage = function(event) {
                    const metrics = JSON.parse(event.data);
                    updateDashboard(metrics);
                };

                function updateDashboard(metrics) {
                    // Update health score
                    const healthScore = metrics.analysis.system_health;
                    const healthStatus = healthScore > 80 ? 'healthy' : 'warning';
                    document.getElementById('healthScore').innerHTML = `
                        System Health: ${healthScore}%
                        <span class="status ${healthStatus}">
                            ${healthStatus.toUpperCase()}
                        </span>
                    `;

                    // Update resource chart
                    resourceData.cpu.push(metrics.cpu.usage);
                    resourceData.memory.push(metrics.memory.usage);
                    resourceData.disk.push(metrics.disk.usage);
                    resourceData.network.push(metrics.network.load);

                    // Keep last 100 points
                    Object.keys(resourceData).forEach(key => {
                        if (resourceData[key].length > 100) {
                            resourceData[key].shift();
                        }
                    });

                    updateResourceChart();
                    updateMetricsTable(metrics);
                }

                function updateResourceChart() {
                    const data = [{
                        y: resourceData.cpu,
                        name: 'CPU',
                        line: {color: '#00ff00'}
                    }, {
                        y: resourceData.memory,
                        name: 'Memory',
                        line: {color: '#ff00ff'}
                    }, {
                        y: resourceData.disk,
                        name: 'Disk',
                        line: {color: '#00ffff'}
                    }, {
                        y: resourceData.network,
                        name: 'Network',
                        line: {color: '#ffffff'}
                    }];

                    const layout = {
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: {color: '#ffffff'},
                        yaxis: {range: [0, 100]},
                        showlegend: true
                    };

                    Plotly.newPlot('resourceChart', data, layout);
                }

                function updateMetricsTable(metrics) {
                    const table = document.getElementById('metricsTable');
                    table.innerHTML = `
                        <table style="width:100%; border-collapse: collapse;">
                            <tr>
                                <th style="text-align: left; padding: 8px;">Metric</th>
                                <th style="text-align: right; padding: 8px;">Value</th>
                            </tr>
                            <tr>
                                <td>CPU Usage</td>
                                <td style="text-align: right;">${metrics.cpu.usage}%</td>
                            </tr>
                            <tr>
                                <td>Memory Usage</td>
                                <td style="text-align: right;">${metrics.memory.usage}%</td>
                            </tr>
                            <tr>
                                <td>Disk Usage</td>
                                <td style="text-align: right;">${metrics.disk.usage}%</td>
                            </tr>
                            <tr>
                                <td>Network Load</td>
                                <td style="text-align: right;">${metrics.network.load}%</td>
                            </tr>
                        </table>
                    `;
                }

                async function optimizeSystem() {
                    const button = document.getElementById('optimizeBtn');
                    button.disabled = true;
                    button.textContent = 'Optimizing...';

                    try {
                        const response = await fetch('/api/optimize', {
                            method: 'POST'
                        });
                        const result = await response.json();
                        
                        if (result.status === 'optimization_started') {
                            button.textContent = 'Optimization Started';
                            setTimeout(() => {
                                button.textContent = 'Optimize System';
                                button.disabled = false;
                            }, 5000);
                        }
                    } catch (error) {
                        console.error('Optimization failed:', error);
                        button.textContent = 'Optimization Failed';
                        setTimeout(() => {
                            button.textContent = 'Optimize System';
                            button.disabled = false;
                        }, 3000);
                    }
                }

                // Initial load
                updateResourceChart();
                
                // Load predictions
                fetch('/api/predictions?resource=cpu&hours=24')
                    .then(response => response.json())
                    .then(data => {
                        const predictionData = [{
                            x: data.predictions.map(p => p.timestamp),
                            y: data.predictions.map(p => p.value),
                            name: 'CPU Prediction',
                            line: {color: '#00ff00', dash: 'dot'}
                        }];

                        const layout = {
                            paper_bgcolor: '#2d2d2d',
                            plot_bgcolor: '#2d2d2d',
                            font: {color: '#ffffff'},
                            yaxis: {range: [0, 100]},
                            title: {
                                text: `Prediction Confidence: ${(data.confidence * 100).toFixed(1)}%`,
                                font: {color: '#ffffff'}
                            }
                        };

                        Plotly.newPlot('predictionChart', predictionData, layout);
                    });

                // Load anomalies
                fetch('/api/anomalies')
                    .then(response => response.json())
                    .then(data => {
                        const anomalyData = [{
                            type: 'scatter',
                            mode: 'markers',
                            x: data.anomalies.map(a => a.timestamp),
                            y: data.anomalies.map(a => a.score),
                            marker: {
                                size: 10,
                                color: '#ff0000'
                            },
                            name: 'Anomalies'
                        }];

                        const layout = {
                            paper_bgcolor: '#2d2d2d',
                            plot_bgcolor: '#2d2d2d',
                            font: {color: '#ffffff'},
                            title: {
                                text: `Found ${data.total_anomalies} anomalies`,
                                font: {color: '#ffffff'}
                            }
                        };

                        Plotly.newPlot('anomalyChart', anomalyData, layout);
                    });
            </script>
        </body>
        </html>
        """

async def main():
    quantum_web = QuantumWeb()
    config = uvicorn.Config(
        quantum_web.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())

