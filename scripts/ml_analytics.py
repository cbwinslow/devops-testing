#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, models

@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    features: Dict[str, float]
    timestamp: str

class MLAnalytics:
    def __init__(self, model_dir: str = '~/devops-testing/models'):
        self.model_dir = os.path.expanduser(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.anomaly_detector = self._load_or_create_model('anomaly_detector', IsolationForest)
        self.pattern_classifier = self._load_or_create_model('pattern_classifier', RandomForestClassifier)
        self.sequence_predictor = self._load_or_create_sequence_model()
        
        self.feature_columns = [
            'cpu_usage', 'memory_usage', 'disk_usage',
            'network_in', 'network_out', 'error_count',
            'latency', 'request_count'
        ]

    def _load_or_create_model(self, name: str, model_class):
        """Load existing model or create new one"""
        model_path = os.path.join(self.model_dir, f'{name}.joblib')
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return model_class()

    def _load_or_create_sequence_model(self):
        """Create or load sequence prediction model"""
        model_path = os.path.join(self.model_dir, 'sequence_model')
        if os.path.exists(model_path):
            return models.load_model(model_path)
        
        model = models.Sequential([
            layers.LSTM(64, input_shape=(24, len(self.feature_columns))),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.feature_columns))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _save_models(self):
        """Save all models"""
        joblib.dump(self.anomaly_detector, os.path.join(self.model_dir, 'anomaly_detector.joblib'))
        joblib.dump(self.pattern_classifier, os.path.join(self.model_dir, 'pattern_classifier.joblib'))
        self.sequence_predictor.save(os.path.join(self.model_dir, 'sequence_model'))

    def prepare_features(self, metrics: Dict) -> np.ndarray:
        """Extract and normalize features from metrics"""
        features = []
        for column in self.feature_columns:
            features.append(metrics.get(column, 0))
        return np.array(features).reshape(1, -1)

    def detect_anomalies(self, metrics: List[Dict]) -> List[Dict]:
        """Detect anomalies in metrics data"""
        features = np.array([self.prepare_features(m)[0] for m in metrics])
        scaled_features = self.scaler.fit_transform(features)
        
        predictions = self.anomaly_detector.predict(scaled_features)
        scores = self.anomaly_detector.score_samples(scaled_features)
        
        results = []
        for i, (prediction, score) in enumerate(zip(predictions, scores)):
            results.append({
                'timestamp': metrics[i].get('timestamp'),
                'is_anomaly': prediction == -1,
                'anomaly_score': score,
                'metrics': metrics[i]
            })
        
        return results

    def train_anomaly_detector(self, metrics: List[Dict]):
        """Train anomaly detection model"""
        features = np.array([self.prepare_features(m)[0] for m in metrics])
        scaled_features = self.scaler.fit_transform(features)
        self.anomaly_detector.fit(scaled_features)
        self._save_models()

    def classify_patterns(self, metrics: List[Dict], labels: List[str]) -> Dict[str, float]:
        """Classify patterns in metrics data"""
        features = np.array([self.prepare_features(m)[0] for m in metrics])
        scaled_features = self.scaler.transform(features)
        
        probabilities = self.pattern_classifier.predict_proba(scaled_features)
        class_names = self.pattern_classifier.classes_
        
        return {name: float(prob) for name, prob in zip(class_names, probabilities[0])}

    def train_pattern_classifier(self, metrics: List[Dict], labels: List[str]):
        """Train pattern classification model"""
        features = np.array([self.prepare_features(m)[0] for m in metrics])
        scaled_features = self.scaler.fit_transform(features)
        
        self.pattern_classifier.fit(scaled_features, labels)
        self._save_models()

    def predict_sequence(self, metrics: List[Dict], horizon: int = 24) -> List[Dict]:
        """Predict future metrics"""
        # Prepare sequence data
        features = np.array([self.prepare_features(m)[0] for m in metrics])
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X = scaled_features[-24:].reshape(1, 24, -1)
        
        # Make predictions
        predictions = []
        current_sequence = X.copy()
        
        for _ in range(horizon):
            next_pred = self.sequence_predictor.predict(current_sequence)
            predictions.append(next_pred[0])
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = next_pred[0]
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(predictions)
        
        # Format results
        results = []
        current_time = datetime.now()
        for i, pred in enumerate(predictions):
            results.append({
                'timestamp': (current_time + timedelta(hours=i)).isoformat(),
                'predictions': {
                    column: float(value)
                    for column, value in zip(self.feature_columns, pred)
                }
            })
        
        return results

    def train_sequence_model(self, metrics: List[Dict], sequence_length: int = 24):
        """Train sequence prediction model"""
        features = np.array([self.prepare_features(m)[0] for m in metrics])
        scaled_features = self.scaler.fit_transform(features)
        
        # Prepare sequences
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:i+sequence_length])
            y.append(scaled_features[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.sequence_predictor.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
        self._save_models()

    def analyze_metrics(self, metrics: List[Dict]) -> Dict:
        """Comprehensive metric analysis"""
        # Detect anomalies
        anomalies = self.detect_anomalies(metrics)
        
        # Classify patterns
        recent_metrics = metrics[-10:]  # Use last 10 data points
        patterns = self.classify_patterns(recent_metrics, ['normal', 'degraded', 'critical'])
        
        # Predict future values
        predictions = self.predict_sequence(metrics)
        
        # Calculate trends
        trends = self._calculate_trends(metrics)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'anomalies': anomalies,
            'patterns': patterns,
            'predictions': predictions,
            'trends': trends
        }

    def _calculate_trends(self, metrics: List[Dict]) -> Dict:
        """Calculate trends in metrics"""
        df = pd.DataFrame(metrics)
        
        trends = {}
        for column in self.feature_columns:
            if column in df.columns:
                values = df[column].values
                trends[column] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'decreasing'
                }
        
        return trends

    def save_analysis(self, analysis: Dict):
        """Save analysis results"""
        output_dir = os.path.expanduser('~/devops-testing/reports/analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'analysis_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(analysis, f, indent=2)

def main():
    # Example usage
    analytics = MLAnalytics()
    
    # Generate some example metrics
    metrics = []
    start_time = datetime.now() - timedelta(days=1)
    for i in range(24):
        metrics.append({
            'timestamp': (start_time + timedelta(hours=i)).isoformat(),
            'cpu_usage': np.random.normal(50, 10),
            'memory_usage': np.random.normal(60, 15),
            'disk_usage': np.random.normal(70, 5),
            'network_in': np.random.normal(1000, 200),
            'network_out': np.random.normal(800, 150),
            'error_count': np.random.poisson(2),
            'latency': np.random.normal(0.1, 0.02),
            'request_count': np.random.poisson(100)
        })
    
    # Train models
    analytics.train_anomaly_detector(metrics)
    analytics.train_pattern_classifier(metrics, ['normal'] * len(metrics))
    analytics.train_sequence_model(metrics)
    
    # Perform analysis
    analysis = analytics.analyze_metrics(metrics)
    analytics.save_analysis(analysis)
    
    print("Analysis complete. Results saved to reports/analysis/")

if __name__ == '__main__':
    main()

