#!/usr/bin/env python3

import unittest
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import tempfile
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ai_resolver import AIResolver, Alert

class MLEnhancedResolver(AIResolver):
    def __init__(self, config_path: str = '~/devops-testing/config/ai_resolver_config.yaml'):
        super().__init__(config_path)
        self.model_path = os.path.expanduser('~/devops-testing/models')
        os.makedirs(self.model_path, exist_ok=True)
        self.anomaly_detector = self._load_or_create_model()
        self.scaler = StandardScaler()

    def _load_or_create_model(self):
        model_file = os.path.join(self.model_path, 'anomaly_detector.joblib')
        if os.path.exists(model_file):
            return joblib.load(model_file)
        return IsolationForest(contamination=0.1, random_state=42)

    def _save_model(self):
        model_file = os.path.join(self.model_path, 'anomaly_detector.joblib')
        joblib.dump(self.anomaly_detector, model_file)

    def _extract_features(self, alert: Alert) -> np.ndarray:
        """Extract numerical features from alert for anomaly detection"""
        features = [
            alert.priority,
            len(str(alert.details)),
            hash(alert.type) % 1000,  # Simple hash of alert type
            int(datetime.fromisoformat(alert.timestamp).timestamp())
        ]
        return np.array(features).reshape(1, -1)

    def detect_anomaly(self, alert: Alert) -> bool:
        """Detect if an alert represents an anomaly"""
        features = self._extract_features(alert)
        scaled_features = self.scaler.fit_transform(features)
        return self.anomaly_detector.predict(scaled_features)[0] == -1

    def train_model(self, alerts: list[Alert]):
        """Train the anomaly detection model"""
        features = np.array([self._extract_features(alert)[0] for alert in alerts])
        scaled_features = self.scaler.fit_transform(features)
        self.anomaly_detector.fit(scaled_features)
        self._save_model()

class TestMLEnhancedResolver(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'test_config.yaml')
        
        # Create test configuration
        with open(self.config_path, 'w') as f:
            json.dump({
                'notification': {'slack_webhook': 'test_webhook'},
                'priorities': {
                    'security': 5,
                    'permission': 4,
                    'config_change': 3,
                    'package': 2
                }
            }, f)
        
        self.resolver = MLEnhancedResolver(config_path=self.config_path)

    def generate_test_alerts(self, num_normal: int = 100, num_anomalies: int = 10) -> list[Alert]:
        """Generate synthetic alerts for testing"""
        alerts = []
        
        # Generate normal alerts
        for _ in range(num_normal):
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24)
            )
            alert_type = np.random.choice(['security', 'permission', 'config_change', 'package'])
            alerts.append(Alert(
                type=alert_type,
                timestamp=timestamp.isoformat(),
                details={'event': 'test_event'},
                priority=np.random.randint(1, 4)
            ))
        
        # Generate anomalous alerts
        for _ in range(num_anomalies):
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, 30),
                hours=np.random.randint(0, 24)
            )
            alerts.append(Alert(
                type='security',
                timestamp=timestamp.isoformat(),
                details={'event': 'suspicious_activity' * 100},  # Unusual detail length
                priority=5  # Highest priority
            ))
        
        return alerts

    def test_anomaly_detection(self):
        """Test anomaly detection capabilities"""
        # Generate and train on normal alerts
        normal_alerts = self.generate_test_alerts(num_normal=100, num_anomalies=0)
        self.resolver.train_model(normal_alerts)
        
        # Test normal alert
        normal_alert = Alert(
            type='permission',
            timestamp=datetime.now().isoformat(),
            details={'event': 'test_event'},
            priority=2
        )
        self.assertFalse(self.resolver.detect_anomaly(normal_alert))
        
        # Test anomalous alert
        anomalous_alert = Alert(
            type='security',
            timestamp=datetime.now().isoformat(),
            details={'event': 'suspicious_activity' * 100},
            priority=5
        )
        self.assertTrue(self.resolver.detect_anomaly(anomalous_alert))

    def test_model_persistence(self):
        """Test model saving and loading"""
        # Train model
        alerts = self.generate_test_alerts()
        self.resolver.train_model(alerts)
        
        # Create new resolver instance
        new_resolver = MLEnhancedResolver(config_path=self.config_path)
        
        # Test both models produce same predictions
        test_alert = Alert(
            type='security',
            timestamp=datetime.now().isoformat(),
            details={'event': 'test_event'},
            priority=3
        )
        self.assertEqual(
            self.resolver.detect_anomaly(test_alert),
            new_resolver.detect_anomaly(test_alert)
        )

    def test_feature_extraction(self):
        """Test feature extraction process"""
        alert = Alert(
            type='security',
            timestamp=datetime.now().isoformat(),
            details={'event': 'test_event'},
            priority=5
        )
        features = self.resolver._extract_features(alert)
        self.assertEqual(features.shape, (1, 4))  # Expecting 4 features

    def test_adaptive_learning(self):
        """Test model adaptation to new patterns"""
        # Initial training
        initial_alerts = self.generate_test_alerts(num_normal=50, num_anomalies=5)
        self.resolver.train_model(initial_alerts)
        
        # New pattern of alerts
        new_alerts = [Alert(
            type='new_alert_type',
            timestamp=datetime.now().isoformat(),
            details={'event': 'new_pattern'},
            priority=3
        ) for _ in range(20)]
        
        # Retrain with new pattern
        self.resolver.train_model(new_alerts + initial_alerts)
        
        # Test detection of new pattern
        test_alert = Alert(
            type='new_alert_type',
            timestamp=datetime.now().isoformat(),
            details={'event': 'new_pattern'},
            priority=3
        )
        self.assertFalse(self.resolver.detect_anomaly(test_alert))

if __name__ == '__main__':
    unittest.main()

