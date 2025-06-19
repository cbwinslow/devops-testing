#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import psutil
import platform
import socket
import uuid
import ssl
import base64
import os
import signal
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_remote_agent')

class RemoteAgent:
    """Remote monitoring agent for quantum optimization system"""
    def __init__(self, config_path: str = 'config/remote_agent_config.yaml'):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.setup_encryption()
        self.agent_id = self.generate_agent_id()
        self.setup_storage()

    def load_config(self) -> Dict:
        """Load agent configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Create default config
                config = {
                    'server': {
                        'host': 'localhost',
                        'port': 8000,
                        'ssl': False,
                        'verify_ssl': True
                    },
                    'monitoring': {
                        'interval': 60,
                        'metrics': ['cpu', 'memory', 'disk', 'network'],
                        'processes': True,
                        'services': True
                    },
                    'security': {
                        'encryption': True,
                        'token': '',
                        'key_rotation': 86400  # 24 hours
                    },
                    'logging': {
                        'level': 'INFO',
                        'file': 'logs/remote_agent.log',
                        'max_size': 10485760,  # 10MB
                        'backup_count': 5
                    }
                }
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f)
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def setup_encryption(self):
        """Setup data encryption"""
        try:
            key_file = Path('.agent_key')
            if not key_file.exists():
                key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(key)
            else:
                with open(key_file, 'rb') as f:
                    key = f.read()
            self.cipher = Fernet(key)
        except Exception as e:
            logger.error(f"Error setting up encryption: {e}")
            self.cipher = None

    def generate_agent_id(self) -> str:
        """Generate unique agent identifier"""
        try:
            # Combine system information for unique ID
            system_info = {
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'mac': uuid.getnode(),
                'cpu_id': self.get_cpu_id(),
                'boot_time': psutil.boot_time()
            }
            # Create deterministic UUID
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, json.dumps(system_info)))
        except Exception as e:
            logger.error(f"Error generating agent ID: {e}")
            return str(uuid.uuid4())

    def get_cpu_id(self) -> str:
        """Get CPU identifier"""
        try:
            if platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('Serial'):
                            return line.split(':')[1].strip()
            return platform.processor()
        except Exception:
            return platform.processor()

    def setup_storage(self):
        """Setup storage directories"""
        try:
            # Create directories
            Path('logs').mkdir(parents=True, exist_ok=True)
            Path('data/metrics').mkdir(parents=True, exist_ok=True)
            Path('data/reports').mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error setting up storage: {e}")

    async def collect_metrics(self) -> Dict:
        """Collect system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'agent_id': self.agent_id,
                'system': {
                    'hostname': socket.gethostname(),
                    'platform': platform.platform(),
                    'uptime': int(datetime.now().timestamp() - psutil.boot_time())
                },
                'cpu': self.collect_cpu_metrics(),
                'memory': self.collect_memory_metrics(),
                'disk': self.collect_disk_metrics(),
                'network': self.collect_network_metrics()
            }

            if self.config['monitoring'].get('processes', True):
                metrics['processes'] = self.collect_process_metrics()

            if self.config['monitoring'].get('services', True):
                metrics['services'] = await self.collect_service_metrics()

            return metrics

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {}

    def collect_cpu_metrics(self) -> Dict:
        """Collect CPU metrics"""
        try:
            cpu_times = psutil.cpu_times()
            cpu_freq = psutil.cpu_freq()
            return {
                'usage': psutil.cpu_percent(interval=1),
                'per_cpu': psutil.cpu_percent(interval=1, percpu=True),
                'times': {
                    'user': cpu_times.user,
                    'system': cpu_times.system,
                    'idle': cpu_times.idle
                },
                'frequency': {
                    'current': cpu_freq.current if cpu_freq else None,
                    'min': cpu_freq.min if cpu_freq else None,
                    'max': cpu_freq.max if cpu_freq else None
                },
                'cores': {
                    'physical': psutil.cpu_count(logical=False),
                    'logical': psutil.cpu_count()
                },
                'load_average': psutil.getloadavg()
            }
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
            return {}

    def collect_memory_metrics(self) -> Dict:
        """Collect memory metrics"""
        try:
            virtual = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                'virtual': {
                    'total': virtual.total,
                    'available': virtual.available,
                    'used': virtual.used,
                    'free': virtual.free,
                    'percent': virtual.percent,
                    'cached': virtual.cached if hasattr(virtual, 'cached') else None
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                }
            }
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
            return {}

    def collect_disk_metrics(self) -> Dict:
        """Collect disk metrics"""
        try:
            disk_metrics = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_metrics[partition.device] = {
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': usage.percent
                    }
                except Exception:
                    continue

            # Add disk I/O statistics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_metrics['io'] = {
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_time': disk_io.read_time,
                    'write_time': disk_io.write_time
                }

            return disk_metrics

        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
            return {}

    def collect_network_metrics(self) -> Dict:
        """Collect network metrics"""
        try:
            network_metrics = {}
            
            # Network interfaces
            for interface, stats in psutil.net_if_stats().items():
                if stats.isup:
                    network_metrics[interface] = {
                        'stats': {
                            'speed': stats.speed,
                            'mtu': stats.mtu,
                            'duplex': str(stats.duplex)
                        }
                    }
                    
                    # Add interface counters
                    counters = psutil.net_io_counters(pernic=True).get(interface)
                    if counters:
                        network_metrics[interface]['counters'] = {
                            'bytes_sent': counters.bytes_sent,
                            'bytes_recv': counters.bytes_recv,
                            'packets_sent': counters.packets_sent,
                            'packets_recv': counters.packets_recv,
                            'errin': counters.errin,
                            'errout': counters.errout,
                            'dropin': counters.dropin,
                            'dropout': counters.dropout
                        }

            # Network connections
            connections = psutil.net_connections(kind='inet')
            network_metrics['connections'] = {
                'established': len([c for c in connections if c.status == 'ESTABLISHED']),
                'listen': len([c for c in connections if c.status == 'LISTEN']),
                'total': len(connections)
            }

            return network_metrics

        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            return {}

    def collect_process_metrics(self) -> Dict:
        """Collect process metrics"""
        try:
            process_metrics = {
                'total': len(psutil.pids()),
                'running': 0,
                'sleeping': 0,
                'top_cpu': [],
                'top_memory': []
            }

            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    if pinfo['status'] == 'running':
                        process_metrics['running'] += 1
                    elif pinfo['status'] == 'sleeping':
                        process_metrics['sleeping'] += 1
                    
                    processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort processes by CPU and memory usage
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            process_metrics['top_cpu'] = processes[:10]

            processes.sort(key=lambda x: x.get('memory_percent', 0), reverse=True)
            process_metrics['top_memory'] = processes[:10]

            return process_metrics

        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return {}

    async def collect_service_metrics(self) -> Dict:
        """Collect service metrics"""
        try:
            service_metrics = {
                'system': {},
                'custom': {}
            }

            # System services (systemd on Linux)
            if platform.system() == "Linux":
                try:
                    import systemd.journal
                    service_metrics['system']['systemd'] = {
                        'active': len([u for u in systemd.journal.Reader().query_unique('_SYSTEMD_UNIT')
                                    if u.endswith('.service')]),
                        'failed': len([u for u in systemd.journal.Reader().query_unique('_SYSTEMD_UNIT')
                                    if u.endswith('.service') and 'failed' in u.lower()])
                    }
                except ImportError:
                    pass

            # Custom service checks
            for service in self.config.get('monitoring', {}).get('services', []):
                try:
                    if 'port' in service:
                        # Check if port is open
                        result = await self.check_port(service['host'], service['port'])
                        service_metrics['custom'][service['name']] = {
                            'status': 'up' if result else 'down',
                            'port': service['port']
                        }
                    elif 'url' in service:
                        # Check if URL is accessible
                        result = await self.check_url(service['url'])
                        service_metrics['custom'][service['name']] = {
                            'status': 'up' if result else 'down',
                            'url': service['url']
                        }
                except Exception as e:
                    service_metrics['custom'][service['name']] = {
                        'status': 'error',
                        'error': str(e)
                    }

            return service_metrics

        except Exception as e:
            logger.error(f"Error collecting service metrics: {e}")
            return {}

    async def check_port(self, host: str, port: int) -> bool:
        """Check if port is open"""
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    async def check_url(self, url: str) -> bool:
        """Check if URL is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False

    async def send_metrics(self, metrics: Dict) -> bool:
        """Send metrics to central server"""
        try:
            server_config = self.config['server']
            url = f"{'https' if server_config['ssl'] else 'http'}://{server_config['host']}:{server_config['port']}/api/metrics"

            # Encrypt metrics if enabled
            if self.config['security']['encryption'] and self.cipher:
                encrypted_data = self.cipher.encrypt(json.dumps(metrics).encode())
                payload = {
                    'agent_id': self.agent_id,
                    'data': base64.b64encode(encrypted_data).decode()
                }
            else:
                payload = metrics

            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f"Bearer {self.config['security']['token']}"} if self.config['security']['token'] else {}
                async with session.post(url, json=payload, headers=headers,
                                     ssl=False if not server_config['verify_ssl'] else None) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Error sending metrics: {e}")
            return False

    async def run(self):
        """Run the remote agent"""
        try:
            logger.info(f"Starting remote agent {self.agent_id}")
            
            while True:
                try:
                    # Collect metrics
                    metrics = await self.collect_metrics()
                    
                    # Send metrics
                    success = await self.send_metrics(metrics)
                    if success:
                        logger.debug("Metrics sent successfully")
                    else:
                        logger.warning("Failed to send metrics")
                    
                    # Wait for next interval
                    await asyncio.sleep(self.config['monitoring']['interval'])
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(5)  # Wait before retry

        except asyncio.CancelledError:
            logger.info("Remote agent stopped")

class AgentManager:
    """Manage the remote agent process"""
    def __init__(self):
        self.agent = None
        self.running = False
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}")
        self.stop()

    def start(self):
        """Start the remote agent"""
        if not self.running:
            self.running = True
            self.agent = RemoteAgent()
            asyncio.run(self.agent.run())

    def stop(self):
        """Stop the remote agent"""
        if self.running:
            self.running = False
            if self.agent:
                # Cleanup
                logger.info("Stopping remote agent...")

    def restart(self):
        """Restart the remote agent"""
        self.stop()
        self.start()

def main():
    manager = AgentManager()
    manager.start()

if __name__ == "__main__":
    main()

