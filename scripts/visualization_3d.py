#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import logging
from typing import List, Dict, Optional
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualization_3d')

class SystemVisualizer:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.data = []
        self.anomalies = []

    def setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            html.H1('System State Visualization'),
            
            # Time range selector
            dcc.RangeSlider(
                id='time-slider',
                min=0,
                max=24,
                step=1,
                value=[0, 24],
                marks={i: f'{i}h' for i in range(0, 25, 4)}
            ),
            
            # 3D visualization
            dcc.Graph(id='3d-system-state'),
            
            # Metrics selection
            dcc.Dropdown(
                id='metrics-selector',
                options=[
                    {'label': 'CPU Usage', 'value': 'cpu_usage'},
                    {'label': 'Memory Usage', 'value': 'memory_usage'},
                    {'label': 'Disk Usage', 'value': 'disk_usage'},
                    {'label': 'Network Traffic', 'value': 'network'},
                    {'label': 'Error Rate', 'value': 'error_rate'}
                ],
                value=['cpu_usage', 'memory_usage', 'disk_usage'],
                multi=True
            ),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            ),
            
            # Anomaly alerts
            html.Div(id='anomaly-alerts', className='alerts'),
            
            # System state summary
            html.Div(id='system-summary', className='summary')
        ])
        
        self.setup_callbacks()

    def setup_callbacks(self):
        """Set up dashboard callbacks"""
        @self.app.callback(
            Output('3d-system-state', 'figure'),
            [Input('metrics-selector', 'value'),
             Input('time-slider', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_3d_graph(selected_metrics, time_range, n_intervals):
            return self.create_3d_visualization(selected_metrics, time_range)
        
        @self.app.callback(
            Output('anomaly-alerts', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_anomaly_alerts(n_intervals):
            return self.create_anomaly_alerts()
        
        @self.app.callback(
            Output('system-summary', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_system_summary(n_intervals):
            return self.create_system_summary()

    def create_3d_visualization(self, selected_metrics: List[str], time_range: List[int]) -> go.Figure:
        """Create 3D visualization of system state"""
        # Create figure
        fig = go.Figure()
        
        # Get data for selected time range
        df = pd.DataFrame(self.data)
        if not df.empty:
            df = df[(df['timestamp'] >= time_range[0]) & 
                   (df['timestamp'] <= time_range[1])]
            
            # Add traces for each metric
            for metric in selected_metrics:
                fig.add_trace(go.Scatter3d(
                    x=df['timestamp'],
                    y=df[metric],
                    z=df['load'],
                    mode='lines+markers',
                    name=metric,
                    marker=dict(
                        size=4,
                        opacity=0.8
                    )
                ))
            
            # Add anomaly points if any
            anomalies_df = pd.DataFrame(self.anomalies)
            if not anomalies_df.empty:
                fig.add_trace(go.Scatter3d(
                    x=anomalies_df['timestamp'],
                    y=anomalies_df['value'],
                    z=anomalies_df['load'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='x'
                    )
                ))
        
        # Update layout
        fig.update_layout(
            title='System State 3D Visualization',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Metric Value',
                zaxis_title='System Load',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig

    def create_anomaly_alerts(self) -> html.Div:
        """Create anomaly alert components"""
        alerts = []
        for anomaly in self.anomalies[-5:]:  # Show last 5 anomalies
            alerts.append(html.Div([
                html.H4(f"Anomaly Detected: {anomaly['metric']}"),
                html.P(f"Value: {anomaly['value']:.2f}"),
                html.P(f"Timestamp: {anomaly['timestamp']}")
            ], className='alert-item'))
        
        return html.Div(alerts)

    def create_system_summary(self) -> html.Div:
        """Create system state summary"""
        if not self.data:
            return html.Div("No data available")
        
        latest = self.data[-1]
        return html.Div([
            html.H3("System Summary"),
            html.Table([
                html.Tr([html.Td("CPU Usage:"), html.Td(f"{latest['cpu_usage']:.1f}%")]),
                html.Tr([html.Td("Memory Usage:"), html.Td(f"{latest['memory_usage']:.1f}%")]),
                html.Tr([html.Td("Disk Usage:"), html.Td(f"{latest['disk_usage']:.1f}%")]),
                html.Tr([html.Td("Network Traffic:"), html.Td(f"{latest['network']:.1f} MB/s")]),
                html.Tr([html.Td("Error Rate:"), html.Td(f"{latest['error_rate']:.2f}%")])
            ])
        ])

    async def fetch_metrics(self):
        """Fetch metrics from monitoring system"""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get('http://localhost:8000/metrics') as response:
                        if response.status == 200:
                            metrics = await response.json()
                            self.update_data(metrics)
                except Exception as e:
                    logger.error(f"Error fetching metrics: {e}")
                
                await asyncio.sleep(5)  # Update every 5 seconds

    def update_data(self, metrics: Dict):
        """Update visualization data"""
        # Add new data point
        self.data.append({
            'timestamp': datetime.now(),
            **metrics
        })
        
        # Keep last 24 hours of data
        cutoff = datetime.now() - timedelta(hours=24)
        self.data = [d for d in self.data if d['timestamp'] >= cutoff]
        
        # Check for anomalies
        self.detect_anomalies(metrics)

    def detect_anomalies(self, metrics: Dict):
        """Simple anomaly detection"""
        thresholds = {
            'cpu_usage': 90,
            'memory_usage': 90,
            'disk_usage': 90,
            'error_rate': 5
        }
        
        for metric, value in metrics.items():
            if metric in thresholds and value > thresholds[metric]:
                self.anomalies.append({
                    'timestamp': datetime.now(),
                    'metric': metric,
                    'value': value,
                    'load': metrics.get('load', 0)
                })
        
        # Keep last 100 anomalies
        self.anomalies = self.anomalies[-100:]

    def run(self, host: str = 'localhost', port: int = 8050):
        """Run the visualization server"""
        # Start metrics collection in background
        loop = asyncio.get_event_loop()
        loop.create_task(self.fetch_metrics())
        
        # Run Dash app
        self.app.run_server(host=host, port=port, debug=True)

def main():
    visualizer = SystemVisualizer()
    visualizer.run()

if __name__ == '__main__':
    main()

