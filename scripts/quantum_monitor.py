#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import yaml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from quantum_metrics import MetricsCollector
from quantum_optimization_strategies import OptimizationOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_monitor')

class AlertConfig:
    """Alert configuration and thresholds"""
    def __init__(self, config_path: str = 'config/alert_config.yaml'):
        self.config_path = Path(config_path)
        self.load_config()

    def load_config(self):
        """Load alert configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {
                    'thresholds': {
                        'critical': {
                            'cpu_usage': 90,
                            'memory_usage': 90,
                            'disk_usage': 90,
                            'network_errors': 100
                        },
                        'warning': {
                            'cpu_usage': 80,
                            'memory_usage': 80,
                            'disk_usage': 80,
                            'network_errors': 50
                        }
                    },
                    'notification': {
                        'email': {
                            'enabled': False,
                            'smtp_server': 'smtp.gmail.com',
                            'smtp_port': 587,
                            'username': '',
                            'password': '',
                            'from_address': '',
                            'to_addresses': []
                        },
                        'slack': {
                            'enabled': False,
                            'webhook_url': ''
                        }
                    },
                    'alert_frequency': {
                        'critical': 300,  # 5 minutes
                        'warning': 900    # 15 minutes
                    }
                }
                # Save default config
                self.save_config()

        except Exception as e:
            logger.error(f"Error loading alert configuration: {e}")
            raise

    def save_config(self):
        """Save current configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving alert configuration: {e}")

class AlertManager:
    """Manage system alerts and notifications"""
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alert_history: List[Dict] = []
        self.last_notification_time: Dict[str, datetime] = {}

    async def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check metrics against thresholds and generate alerts"""
        try:
            current_alerts = []
            timestamp = datetime.now()

            # Check CPU usage
            cpu_usage = metrics['cpu']['usage']
            if cpu_usage >= self.config.config['thresholds']['critical']['cpu_usage']:
                current_alerts.append(self._create_alert('critical', 'CPU', cpu_usage))
            elif cpu_usage >= self.config.config['thresholds']['warning']['cpu_usage']:
                current_alerts.append(self._create_alert('warning', 'CPU', cpu_usage))

            # Check memory usage
            memory_usage = metrics['memory']['usage']
            if memory_usage >= self.config.config['thresholds']['critical']['memory_usage']:
                current_alerts.append(self._create_alert('critical', 'Memory', memory_usage))
            elif memory_usage >= self.config.config['thresholds']['warning']['memory_usage']:
                current_alerts.append(self._create_alert('warning', 'Memory', memory_usage))

            # Check disk usage
            disk_usage = metrics['disk']['usage']
            if disk_usage >= self.config.config['thresholds']['critical']['disk_usage']:
                current_alerts.append(self._create_alert('critical', 'Disk', disk_usage))
            elif disk_usage >= self.config.config['thresholds']['warning']['disk_usage']:
                current_alerts.append(self._create_alert('warning', 'Disk', disk_usage))

            # Check network errors
            network = metrics['network']
            total_errors = network['errin'] + network['errout']
            if total_errors >= self.config.config['thresholds']['critical']['network_errors']:
                current_alerts.append(self._create_alert('critical', 'Network', total_errors))
            elif total_errors >= self.config.config['thresholds']['warning']['network_errors']:
                current_alerts.append(self._create_alert('warning', 'Network', total_errors))

            # Process current alerts
            for alert in current_alerts:
                await self._process_alert(alert)

            return current_alerts

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []

    def _create_alert(self, severity: str, resource: str, value: float) -> Dict:
        """Create an alert dictionary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'resource': resource,
            'value': value,
            'message': f"{severity.upper()} alert: {resource} usage at {value}%"
        }

    async def _process_alert(self, alert: Dict):
        """Process and send alert notifications"""
        try:
            # Add to history
            self.alert_history.append(alert)

            # Check if we should send notification
            alert_key = f"{alert['severity']}_{alert['resource']}"
            last_time = self.last_notification_time.get(alert_key)
            current_time = datetime.now()

            if last_time is None or (current_time - last_time).total_seconds() >= self.config.config['alert_frequency'][alert['severity']]:
                # Send notifications
                await self._send_notifications(alert)
                self.last_notification_time[alert_key] = current_time

        except Exception as e:
            logger.error(f"Error processing alert: {e}")

    async def _send_notifications(self, alert: Dict):
        """Send alert notifications through configured channels"""
        try:
            # Send email notification
            if self.config.config['notification']['email']['enabled']:
                await self._send_email_notification(alert)

            # Send Slack notification
            if self.config.config['notification']['slack']['enabled']:
                await self._send_slack_notification(alert)

        except Exception as e:
            logger.error(f"Error sending notifications: {e}")

    async def _send_email_notification(self, alert: Dict):
        """Send email notification"""
        try:
            email_config = self.config.config['notification']['email']
            msg = MIMEMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"System Alert: {alert['severity'].upper()} - {alert['resource']}"

            body = f"""
            Alert Details:
            Severity: {alert['severity']}
            Resource: {alert['resource']}
            Value: {alert['value']}
            Time: {alert['timestamp']}
            Message: {alert['message']}
            """

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)

        except Exception as e:
            logger.error(f"Error sending email notification: {e}")

    async def _send_slack_notification(self, alert: Dict):
        """Send Slack notification"""
        try:
            webhook_url = self.config.config['notification']['slack']['webhook_url']
            
            color = '#ff0000' if alert['severity'] == 'critical' else '#ffff00'
            message = {
                'attachments': [{
                    'color': color,
                    'title': f"System Alert: {alert['severity'].upper()} - {alert['resource']}",
                    'text': alert['message'],
                    'fields': [
                        {'title': 'Severity', 'value': alert['severity'], 'short': True},
                        {'title': 'Resource', 'value': alert['resource'], 'short': True},
                        {'title': 'Value', 'value': str(alert['value']), 'short': True},
                        {'title': 'Time', 'value': alert['timestamp'], 'short': True}
                    ]
                }]
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message) as response:
                    if response.status != 200:
                        logger.error(f"Error sending Slack notification: {await response.text()}")

        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

class SystemMonitor:
    """Monitor system metrics and manage alerts"""
    def __init__(self, interval: int = 60):
        self.interval = interval
        self.metrics_collector = MetricsCollector()
        self.alert_config = AlertConfig()
        self.alert_manager = AlertManager(self.alert_config)
        self.optimization_orchestrator = OptimizationOrchestrator()
        self.setup_storage()

    def setup_storage(self):
        """Setup storage directories"""
        self.alert_log_path = Path('logs/alerts')
        self.alert_log_path.mkdir(parents=True, exist_ok=True)

    async def run(self):
        """Run the monitoring system"""
        try:
            while True:
                # Collect metrics
                metrics = await self.metrics_collector.collect_system_metrics()
                
                if metrics:
                    # Check for alerts
                    alerts = await self.alert_manager.check_alerts(metrics)
                    
                    # If critical alerts exist, trigger optimization
                    if any(alert['severity'] == 'critical' for alert in alerts):
                        optimized_metrics, applied_optimizations = self.optimization_orchestrator.optimize(metrics)
                        
                        # Log optimization results
                        if applied_optimizations:
                            logger.info("Applied optimizations:")
                            logger.info(json.dumps(applied_optimizations, indent=2))
                    
                    # Save alerts to log file
                    if alerts:
                        await self._save_alerts(alerts)
                
                # Wait for next interval
                await asyncio.sleep(self.interval)

        except Exception as e:
            logger.error(f"Error in monitoring system: {e}")

    async def _save_alerts(self, alerts: List[Dict]):
        """Save alerts to log file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = self.alert_log_path / f"alerts_{timestamp}.json"
            
            # Load existing alerts
            existing_alerts = []
            if log_file.exists():
                with open(log_file, 'r') as f:
                    existing_alerts = json.load(f)
            
            # Append new alerts
            existing_alerts.extend(alerts)
            
            # Save updated alerts
            with open(log_file, 'w') as f:
                json.dump(existing_alerts, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving alerts: {e}")

async def main():
    # Create and run monitor
    monitor = SystemMonitor(interval=60)
    
    try:
        await monitor.run()
    except KeyboardInterrupt:
        logger.info("Monitoring system stopped")

if __name__ == '__main__':
    asyncio.run(main())

