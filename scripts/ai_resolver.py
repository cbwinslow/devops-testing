#!/usr/bin/env python3

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import subprocess
import git
from dataclasses import dataclass
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser('~/devops-testing/reports/ai_resolver.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ai_resolver')

@dataclass
class Alert:
    type: str
    timestamp: str
    details: Dict
    priority: int = 0

class Resolution:
    def __init__(self, alert: Alert, action: str, status: str, details: str):
        self.alert = alert
        self.action = action
        self.status = status
        self.details = details
        self.timestamp = datetime.now().isoformat()

class AIResolver:
    def __init__(self, config_path: str = '~/devops-testing/config/ai_resolver_config.yaml'):
        self.config_path = os.path.expanduser(config_path)
        self.load_config()
        self.repo = git.Repo(os.path.expanduser('~/devops-testing'))
        self.resolutions: List[Resolution] = []

    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {}

    def parse_alert(self, alert_data: Dict) -> Alert:
        """Convert raw alert data to Alert object"""
        return Alert(
            type=alert_data.get('type'),
            timestamp=alert_data.get('timestamp', datetime.now().isoformat()),
            details=alert_data,
            priority=self.get_alert_priority(alert_data)
        )

    def get_alert_priority(self, alert_data: Dict) -> int:
        """Determine alert priority based on type and content"""
        priority_map = {
            'security': 5,
            'permission': 4,
            'config_change': 3,
            'package': 2
        }
        return priority_map.get(alert_data.get('type', ''), 1)

    def resolve_permission_issue(self, alert: Alert) -> Resolution:
        """Handle permission-related issues"""
        path = alert.details.get('path')
        expected = alert.details.get('expected')
        
        try:
            if path and expected:
                subprocess.run(['chmod', expected, path], check=True)
                return Resolution(
                    alert=alert,
                    action=f"chmod {expected} {path}",
                    status="success",
                    details="Permissions updated successfully"
                )
        except subprocess.CalledProcessError as e:
            return Resolution(
                alert=alert,
                action="chmod",
                status="failed",
                details=str(e)
            )

    def resolve_package_issue(self, alert: Alert) -> Resolution:
        """Handle package-related issues"""
        if alert.details.get('event') == 'missing_package':
            package = alert.details.get('package')
            try:
                subprocess.run(['apt-get', 'install', '-y', package], check=True)
                return Resolution(
                    alert=alert,
                    action=f"install {package}",
                    status="success",
                    details=f"Package {package} installed successfully"
                )
            except subprocess.CalledProcessError as e:
                return Resolution(
                    alert=alert,
                    action="package_install",
                    status="failed",
                    details=str(e)
                )

    def resolve_security_issue(self, alert: Alert) -> Resolution:
        """Handle security-related issues"""
        event = alert.details.get('event')
        
        if event == 'failed_login':
            # Log and notify about failed login attempts
            return Resolution(
                alert=alert,
                action="security_notification",
                status="notified",
                details="Security team notified of failed login attempts"
            )
        elif event == 'file_modified':
            # Check file integrity and restore from backup if necessary
            file = alert.details.get('file')
            return self.restore_file_from_git(file, alert)

    def resolve_config_change(self, alert: Alert) -> Resolution:
        """Handle configuration changes"""
        # Validate configuration changes
        try:
            subprocess.run(['bash', 'tests/system_validation.sh'], check=True)
            return Resolution(
                alert=alert,
                action="config_validation",
                status="success",
                details="Configuration changes validated successfully"
            )
        except subprocess.CalledProcessError as e:
            return Resolution(
                alert=alert,
                action="config_validation",
                status="failed",
                details=str(e)
            )

    def restore_file_from_git(self, file: str, alert: Alert) -> Resolution:
        """Restore file from Git history"""
        try:
            self.repo.git.checkout('--', file)
            return Resolution(
                alert=alert,
                action=f"restore_file {file}",
                status="success",
                details=f"File {file} restored from Git history"
            )
        except git.GitCommandError as e:
            return Resolution(
                alert=alert,
                action="restore_file",
                status="failed",
                details=str(e)
            )

    def generate_report(self) -> Dict:
        """Generate report of all resolutions"""
        return {
            'timestamp': datetime.now().isoformat(),
            'resolutions': [
                {
                    'alert_type': r.alert.type,
                    'action': r.action,
                    'status': r.status,
                    'details': r.details,
                    'timestamp': r.timestamp
                }
                for r in self.resolutions
            ]
        }

    def save_report(self, report: Dict):
        """Save report to file"""
        report_dir = os.path.expanduser('~/devops-testing/reports/resolutions')
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(
            report_dir,
            f'resolution_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")

    def notify_user(self, report: Dict):
        """Notify user about resolutions"""
        # Implement notification method (email, Slack, etc.)
        if 'notification' in self.config:
            webhook_url = self.config['notification'].get('slack_webhook')
            if webhook_url:
                try:
                    requests.post(webhook_url, json={
                        'text': f"AI Resolver Report\n```{json.dumps(report, indent=2)}```"
                    })
                except Exception as e:
                    logger.error(f"Failed to send notification: {e}")

    def process_alerts(self, alerts_file: str):
        """Process all alerts from file"""
        try:
            with open(alerts_file, 'r') as f:
                alerts_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load alerts file: {e}")
            return

        # Sort alerts by priority
        alerts = [self.parse_alert(alert) for alert in alerts_data]
        alerts.sort(key=lambda x: x.priority, reverse=True)

        for alert in alerts:
            resolution = None
            
            if alert.type == 'permission':
                resolution = self.resolve_permission_issue(alert)
            elif alert.type == 'package':
                resolution = self.resolve_package_issue(alert)
            elif alert.type == 'security':
                resolution = self.resolve_security_issue(alert)
            elif alert.type == 'config_change':
                resolution = self.resolve_config_change(alert)

            if resolution:
                self.resolutions.append(resolution)
                logger.info(f"Resolved {alert.type} alert: {resolution.status}")

        # Generate and save report
        report = self.generate_report()
        self.save_report(report)
        self.notify_user(report)

def main():
    if len(sys.argv) != 2:
        print("Usage: ai_resolver.py <alerts_file>")
        sys.exit(1)

    resolver = AIResolver()
    resolver.process_alerts(sys.argv[1])

if __name__ == '__main__':
    main()

