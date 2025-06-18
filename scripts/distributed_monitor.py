#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
import asyncio
import aiohttp
import prometheus_client
from datetime import datetime
from typing import Dict, List, Optional
import yaml
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import socket

# Metrics
ALERT_COUNTER = Counter('alerts_total', 'Total number of alerts by type', ['alert_type'])
RESOLUTION_TIME = Histogram('resolution_time_seconds', 'Time taken to resolve alerts')
ACTIVE_ALERTS = Gauge('active_alerts', 'Number of active alerts by type', ['alert_type'])
NODE_STATUS = Gauge('node_status', 'Status of monitoring nodes', ['node'])

class DistributedMonitor:
    def __init__(self, config_path: str = '~/devops-testing/config/distributed_monitor_config.yaml'):
        self.config_path = os.path.expanduser(config_path)
        self.load_config()
        self.hostname = socket.gethostname()
        self.node_id = f"{self.hostname}_{os.getpid()}"
        self.setup_logging()
        self.setup_metrics()

    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            self.config = {
                'monitoring': {
                    'interval': 60,
                    'metrics_port': 8000,
                    'nodes': ['localhost'],
                    'alert_threshold': 5
                }
            }

    def setup_logging(self):
        """Configure logging"""
        log_dir = os.path.expanduser('~/devops-testing/reports/distributed')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/distributed_monitor_{self.node_id}.log"),
                logging.StreamHandler()
            ]
        )

    def setup_metrics(self):
        """Initialize Prometheus metrics server"""
        metrics_port = self.config['monitoring'].get('metrics_port', 8000)
        start_http_server(metrics_port)
        logging.info(f"Metrics server started on port {metrics_port}")

    async def monitor_node(self, node: str):
        """Monitor a single node"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{node}:8000/metrics"
                async with session.get(url) as response:
                    if response.status == 200:
                        NODE_STATUS.labels(node=node).set(1)
                        data = await response.text()
                        self.process_node_metrics(node, data)
                    else:
                        NODE_STATUS.labels(node=node).set(0)
                        logging.error(f"Failed to fetch metrics from {node}: {response.status}")
        except Exception as e:
            NODE_STATUS.labels(node=node).set(0)
            logging.error(f"Error monitoring node {node}: {e}")

    def process_node_metrics(self, node: str, metrics_data: str):
        """Process metrics from a node"""
        # Parse and analyze metrics
        alerts = self.parse_metrics(metrics_data)
        for alert_type, count in alerts.items():
            ACTIVE_ALERTS.labels(alert_type=alert_type).set(count)
            if count > self.config['monitoring']['alert_threshold']:
                self.trigger_alert(node, alert_type, count)

    def parse_metrics(self, metrics_data: str) -> Dict[str, int]:
        """Parse metrics data and return alert counts by type"""
        alerts = {}
        for line in metrics_data.split('\n'):
            if line.startswith('active_alerts'):
                try:
                    parts = line.split()
                    alert_type = parts[1].split('=')[1].strip('"')
                    count = float(parts[2])
                    alerts[alert_type] = count
                except Exception:
                    continue
        return alerts

    def trigger_alert(self, node: str, alert_type: str, count: int):
        """Trigger an alert for a node"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'node': node,
            'alert_type': alert_type,
            'count': count,
            'severity': 'high' if count > self.config['monitoring']['alert_threshold'] * 2 else 'medium'
        }
        
        # Save alert to file
        alert_dir = os.path.expanduser('~/devops-testing/reports/alerts')
        os.makedirs(alert_dir, exist_ok=True)
        alert_file = f"{alert_dir}/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)
        
        ALERT_COUNTER.labels(alert_type=alert_type).inc()
        logging.warning(f"Alert triggered for {node}: {alert_type} ({count} instances)")

    async def monitor_cluster(self):
        """Monitor all nodes in the cluster"""
        while True:
            tasks = []
            for node in self.config['monitoring']['nodes']:
                tasks.append(self.monitor_node(node))
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.config['monitoring']['interval'])

    def run(self):
        """Start the distributed monitoring system"""
        logging.info(f"Starting distributed monitoring on {self.node_id}")
        try:
            asyncio.run(self.monitor_cluster())
        except KeyboardInterrupt:
            logging.info("Shutting down distributed monitoring")
        except Exception as e:
            logging.error(f"Error in monitoring system: {e}")

class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage percentage'),
            'memory_usage': Gauge('memory_usage_percent', 'Memory usage percentage'),
            'disk_usage': Gauge('disk_usage_percent', 'Disk usage percentage'),
            'network_in': Counter('network_in_bytes', 'Network inbound traffic'),
            'network_out': Counter('network_out_bytes', 'Network outbound traffic')
        }

    def collect_system_metrics(self):
        """Collect system metrics"""
        # CPU usage
        try:
            cpu_percent = psutil.cpu_percent()
            self.metrics['cpu_usage'].set(cpu_percent)
        except Exception as e:
            logging.error(f"Error collecting CPU metrics: {e}")

        # Memory usage
        try:
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].set(memory.percent)
        except Exception as e:
            logging.error(f"Error collecting memory metrics: {e}")

        # Disk usage
        try:
            disk = psutil.disk_usage('/')
            self.metrics['disk_usage'].set(disk.percent)
        except Exception as e:
            logging.error(f"Error collecting disk metrics: {e}")

        # Network usage
        try:
            net_io = psutil.net_io_counters()
            self.metrics['network_in'].inc(net_io.bytes_recv)
            self.metrics['network_out'].inc(net_io.bytes_sent)
        except Exception as e:
            logging.error(f"Error collecting network metrics: {e}")

def main():
    monitor = DistributedMonitor()
    collector = MetricsCollector()
    
    # Start metrics collection in a separate thread
    import threading
    metrics_thread = threading.Thread(
        target=lambda: asyncio.run(collect_metrics_periodically(collector)),
        daemon=True
    )
    metrics_thread.start()
    
    # Start monitoring
    monitor.run()

async def collect_metrics_periodically(collector: MetricsCollector):
    while True:
        collector.collect_system_metrics()
        await asyncio.sleep(15)  # Collect metrics every 15 seconds

if __name__ == '__main__':
    main()

