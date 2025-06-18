#!/usr/bin/env python3

import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch
import sys

# Add parent directory to path to import ai_resolver
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.ai_resolver import AIResolver, Alert, Resolution

class TestAIResolver(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'test_config.yaml')
        self.alerts_file = os.path.join(self.test_dir, 'test_alerts.json')
        
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
        
        # Initialize resolver with test config
        self.resolver = AIResolver(config_path=self.config_path)

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def test_alert_parsing(self):
        """Test alert parsing functionality"""
        test_alert = {
            'type': 'security',
            'timestamp': '2025-06-18T22:00:00Z',
            'details': {'event': 'failed_login'}
        }
        
        alert = self.resolver.parse_alert(test_alert)
        self.assertEqual(alert.type, 'security')
        self.assertEqual(alert.priority, 5)

    def test_permission_resolution(self):
        """Test permission issue resolution"""
        test_file = os.path.join(self.test_dir, 'test_file')
        with open(test_file, 'w') as f:
            f.write('test content')
        
        alert = Alert(
            type='permission',
            timestamp=datetime.now().isoformat(),
            details={'path': test_file, 'expected': '644'},
            priority=4
        )
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            resolution = self.resolver.resolve_permission_issue(alert)
            self.assertEqual(resolution.status, 'success')

    def test_package_resolution(self):
        """Test package issue resolution"""
        alert = Alert(
            type='package',
            timestamp=datetime.now().isoformat(),
            details={'event': 'missing_package', 'package': 'test-pkg'},
            priority=2
        )
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            resolution = self.resolver.resolve_package_issue(alert)
            self.assertEqual(resolution.status, 'success')

    def test_security_resolution(self):
        """Test security issue resolution"""
        alert = Alert(
            type='security',
            timestamp=datetime.now().isoformat(),
            details={'event': 'failed_login', 'details': 'Failed login attempt'},
            priority=5
        )
        
        resolution = self.resolver.resolve_security_issue(alert)
        self.assertEqual(resolution.status, 'notified')

    def test_config_change_resolution(self):
        """Test configuration change resolution"""
        alert = Alert(
            type='config_change',
            timestamp=datetime.now().isoformat(),
            details={'file': 'config.yaml'},
            priority=3
        )
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            resolution = self.resolver.resolve_config_change(alert)
            self.assertEqual(resolution.status, 'success')

    def test_report_generation(self):
        """Test report generation"""
        alert = Alert(
            type='security',
            timestamp=datetime.now().isoformat(),
            details={'event': 'test_event'},
            priority=5
        )
        
        resolution = Resolution(
            alert=alert,
            action='test_action',
            status='success',
            details='test details'
        )
        
        self.resolver.resolutions.append(resolution)
        report = self.resolver.generate_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('resolutions', report)
        self.assertEqual(len(report['resolutions']), 1)
        self.assertEqual(report['resolutions'][0]['status'], 'success')

    def test_alert_priority_sorting(self):
        """Test that alerts are processed in priority order"""
        test_alerts = [
            {'type': 'package', 'details': {'event': 'test'}},
            {'type': 'security', 'details': {'event': 'test'}},
            {'type': 'permission', 'details': {'event': 'test'}}
        ]
        
        with open(self.alerts_file, 'w') as f:
            json.dump(test_alerts, f)
        
        with patch.object(self.resolver, 'resolve_security_issue') as mock_security, \
             patch.object(self.resolver, 'resolve_permission_issue') as mock_permission, \
             patch.object(self.resolver, 'resolve_package_issue') as mock_package:
            
            self.resolver.process_alerts(self.alerts_file)
            
            # Check that security was called first, then permission, then package
            call_order = []
            for call in mock_security.call_args_list + \
                       mock_permission.call_args_list + \
                       mock_package.call_args_list:
                call_order.append(call[0][0].type)
            
            self.assertEqual(call_order, ['security', 'permission', 'package'])

    def test_error_handling(self):
        """Test error handling in resolution process"""
        # Test with invalid alerts file
        with self.assertLogs() as captured:
            self.resolver.process_alerts('nonexistent_file')
            self.assertIn('ERROR', captured.output[0])

        # Test with invalid configuration
        resolver = AIResolver(config_path='nonexistent_config.yaml')
        self.assertEqual(resolver.config, {})

if __name__ == '__main__':
    unittest.main()

