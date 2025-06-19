#!/usr/bin/env python3

import psutil
import time
import json
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_metrics')

class MetricsCollector:
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_history: List[Dict] = []
        self.setup_storage()

    def setup_storage(self):
        """Setup storage for metrics"""
        self.storage_path = Path('data/metrics')
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def collect_system_metrics(self) -> Dict:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()

            # Get network interfaces
            net_interfaces = psutil.net_if_stats()
            net_io = {iface: psutil.net_io_counters(pernic=True)[iface]
                     for iface in net_interfaces.keys() if net_interfaces[iface].isup}

            # Get disk IO stats
            disk_io = psutil.disk_io_counters()
            
            # Get system temperature if available
            try:
                temperatures = psutil.sensors_temperatures()
                cpu_temp = max([temp.current for temps in temperatures.values() 
                              for temp in temps]) if temperatures else None
            except Exception:
                cpu_temp = None

            # Get fan speeds if available
            try:
                fans = psutil.sensors_fans()
                fan_speeds = [fan.current for fans in fans.values() 
                            for fan in fans] if fans else []
            except Exception:
                fan_speeds = []

            # Get battery info if available
            try:
                battery = psutil.sensors_battery()
                battery_info = {
                    'percent': battery.percent,
                    'power_plugged': battery.power_plugged,
                    'time_left': battery.secsleft
                } if battery else None
            except Exception:
                battery_info = None

            # Collect process information
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    if pinfo['cpu_percent'] > 1.0 or pinfo['memory_percent'] > 1.0:
                        processes.append(pinfo)
                except Exception:
                    continue

            # Get system uptime
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = (datetime.now() - boot_time).total_seconds()

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage': cpu_percent,
                    'per_core': psutil.cpu_percent(percpu=True),
                    'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                    'temperature': cpu_temp,
                    'load_average': psutil.getloadavg()
                },
                'memory': {
                    'usage': memory.percent,
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'free': memory.free,
                    'cached': memory.cached if hasattr(memory, 'cached') else None,
                    'swap': psutil.swap_memory()._asdict()
                },
                'disk': {
                    'usage': disk.percent,
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'io': {
                        'read_bytes': disk_io.read_bytes if disk_io else None,
                        'write_bytes': disk_io.write_bytes if disk_io else None,
                        'read_count': disk_io.read_count if disk_io else None,
                        'write_count': disk_io.write_count if disk_io else None
                    }
                },
                'network': {
                    'interfaces': {
                        iface: {
                            'stats': vars(stats),
                            'io': vars(net_io[iface])
                        } for iface, stats in net_interfaces.items() if stats.isup
                    },
                    'total_sent': network.bytes_sent,
                    'total_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv,
                    'errin': network.errin,
                    'errout': network.errout,
                    'dropin': network.dropin,
                    'dropout': network.dropout
                },
                'system': {
                    'boot_time': psutil.boot_time(),
                    'uptime': uptime,
                    'processes': {
                        'total': len(psutil.pids()),
                        'running': len([p for p in processes if p.get('status') == 'running']),
                        'top_cpu': sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:5],
                        'top_memory': sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:5]
                    }
                },
                'hardware': {
                    'fans': fan_speeds,
                    'battery': battery_info,
                    'temperature': cpu_temp
                }
            }

            # Collect per-CPU metrics
            metrics['cpu_per_core'] = psutil.cpu_percent(percpu=True)

            # Collect detailed memory metrics
            metrics['memory_details'] = {
                'total': memory.total / 1024 / 1024,  # MB
                'available': memory.available / 1024 / 1024,  # MB
                'used': memory.used / 1024 / 1024,  # MB
                'cached': memory.cached / 1024 / 1024 if hasattr(memory, 'cached') else 0  # MB
            }

            # Collect network interfaces metrics
            metrics['network_interfaces'] = {}
            for interface, stats in psutil.net_if_stats().items():
                if stats.isup:
                    metrics['network_interfaces'][interface] = {
                        'speed': stats.speed,
                        'mtu': stats.mtu
                    }

            return metrics

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {}

    def analyze_metrics(self, metrics: Dict) -> Dict:
        """Analyze collected metrics and add derived insights"""
        try:
            analysis = {}

            # System health score (0-100)
            health_scores = {
                'cpu': max(0, 100 - metrics['cpu_usage']),
                'memory': max(0, 100 - metrics['memory_usage']),
                'disk': max(0, 100 - metrics['disk_usage']),
                'network': max(0, 100 - min(metrics['network_load'], 100))
            }
            analysis['system_health'] = sum(health_scores.values()) / len(health_scores)

            # Resource pressure indicators
            analysis['resource_pressure'] = {
                'cpu_pressure': metrics['cpu_usage'] > 80,
                'memory_pressure': metrics['memory_usage'] > 85,
                'disk_pressure': metrics['disk_usage'] > 90
            }

            # Performance trends
            if len(self.metrics_history) > 0:
                last_metrics = self.metrics_history[-1]
                analysis['trends'] = {
                    'cpu_trend': metrics['cpu_usage'] - last_metrics['cpu_usage'],
                    'memory_trend': metrics['memory_usage'] - last_metrics['memory_usage'],
                    'disk_trend': metrics['disk_usage'] - last_metrics['disk_usage']
                }

            # Load prediction (simple linear extrapolation)
            if len(self.metrics_history) >= 5:
                recent_cpu = [m['cpu_usage'] for m in self.metrics_history[-5:]]
                analysis['load_prediction'] = {
                    'cpu_next_hour': min(100, max(0, np.polyval(np.polyfit(range(5), recent_cpu, 1), 5)))
                }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing metrics: {e}")
            return {}

    async def enhance_metrics(self, metrics: Dict) -> Dict:
        """Enhance metrics with additional computed values"""
        try:
            enhanced = metrics.copy()

            # Add timestamp-based features
            current_time = datetime.fromisoformat(metrics['timestamp'])
            enhanced['time_features'] = {
                'hour': current_time.hour,
                'day_of_week': current_time.weekday(),
                'is_weekend': current_time.weekday() >= 5,
                'is_business_hours': 9 <= current_time.hour <= 17
            }

            # Calculate resource efficiency
            if 'cpu_usage' in metrics and 'memory_usage' in metrics:
                enhanced['resource_efficiency'] = {
                    'cpu_memory_ratio': metrics['cpu_usage'] / max(1, metrics['memory_usage']),
                    'resource_utilization': (metrics['cpu_usage'] + metrics['memory_usage']) / 2
                }

            # Add system stability indicators
            enhanced['stability_indicators'] = {
                'high_cpu': metrics['cpu_usage'] > 80,
                'high_memory': metrics['memory_usage'] > 85,
                'high_disk': metrics['disk_usage'] > 90,
                'resource_pressure': any([
                    metrics['cpu_usage'] > 80,
                    metrics['memory_usage'] > 85,
                    metrics['disk_usage'] > 90
                ])
            }

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing metrics: {e}")
            return metrics

    def save_metrics(self, metrics: Dict):
        """Save metrics to storage"""
        try:
            timestamp = datetime.fromisoformat(metrics['timestamp'])
            filename = self.storage_path / f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Keep metrics history limited to last 1000 entries
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:
                self.metrics_history.pop(0)

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    async def run_collection(self):
        """Run continuous metrics collection"""
        try:
            while True:
                # Collect base metrics
                metrics = await self.collect_system_metrics()
                if metrics:
                    # Analyze metrics
                    analysis = self.analyze_metrics(metrics)
                    metrics['analysis'] = analysis

                    # Enhance metrics
                    enhanced_metrics = await self.enhance_metrics(metrics)

                    # Save metrics
                    self.save_metrics(enhanced_metrics)

                    logger.info(f"Collected and processed metrics: {json.dumps(enhanced_metrics, indent=2)}")

                await asyncio.sleep(self.collection_interval)

        except Exception as e:
            logger.error(f"Error in metrics collection loop: {e}")

class MetricsOptimizer:
    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    async def get_optimization_recommendations(self) -> Dict:
        """Generate optimization recommendations based on metrics"""
        try:
            if not self.collector.metrics_history:
                return {}

            current_metrics = self.collector.metrics_history[-1]
            recommendations = {
                'immediate_actions': [],
                'long_term_actions': [],
                'resource_optimization': {}
            }

            # CPU optimization
            if current_metrics['cpu_usage'] > 80:
                recommendations['immediate_actions'].append({
                    'type': 'cpu',
                    'action': 'Consider scaling up CPU resources or optimizing high-CPU processes',
                    'priority': 'high'
                })

            # Memory optimization
            if current_metrics['memory_usage'] > 85:
                recommendations['immediate_actions'].append({
                    'type': 'memory',
                    'action': 'Investigate memory leaks or consider increasing memory allocation',
                    'priority': 'high'
                })

            # Disk optimization
            if current_metrics['disk_usage'] > 90:
                recommendations['immediate_actions'].append({
                    'type': 'disk',
                    'action': 'Clean up disk space or expand storage',
                    'priority': 'high'
                })

            # Long-term recommendations
            if len(self.collector.metrics_history) >= 10:
                cpu_trend = [m['cpu_usage'] for m in self.collector.metrics_history[-10:]]
                if np.mean(cpu_trend) > 70:
                    recommendations['long_term_actions'].append({
                        'type': 'capacity',
                        'action': 'Plan for capacity increase based on consistent high CPU usage',
                        'priority': 'medium'
                    })

            # Resource optimization suggestions
            recommendations['resource_optimization'] = {
                'cpu_optimization': self._get_cpu_optimization(current_metrics),
                'memory_optimization': self._get_memory_optimization(current_metrics),
                'disk_optimization': self._get_disk_optimization(current_metrics)
            }

            return recommendations

        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            return {}

    def _get_cpu_optimization(self, metrics: Dict) -> Dict:
        """Get CPU-specific optimization recommendations"""
        return {
            'suggested_limit': min(85, metrics['cpu_usage'] * 1.2),
            'optimization_target': max(50, metrics['cpu_usage'] * 0.8),
            'action_items': [
                'Identify CPU-intensive processes',
                'Consider process nice values adjustment',
                'Evaluate containerization options'
            ] if metrics['cpu_usage'] > 70 else []
        }

    def _get_memory_optimization(self, metrics: Dict) -> Dict:
        """Get memory-specific optimization recommendations"""
        return {
            'suggested_limit': min(90, metrics['memory_usage'] * 1.15),
            'optimization_target': max(60, metrics['memory_usage'] * 0.85),
            'action_items': [
                'Review memory leak possibilities',
                'Consider memory limits for applications',
                'Evaluate swap usage optimization'
            ] if metrics['memory_usage'] > 80 else []
        }

    def _get_disk_optimization(self, metrics: Dict) -> Dict:
        """Get disk-specific optimization recommendations"""
        return {
            'suggested_limit': min(95, metrics['disk_usage'] * 1.1),
            'optimization_target': max(70, metrics['disk_usage'] * 0.9),
            'action_items': [
                'Implement log rotation',
                'Clean temporary files',
                'Consider volume expansion'
            ] if metrics['disk_usage'] > 85 else []
        }

async def main():
    # Create metrics collector
    collector = MetricsCollector(collection_interval=60)
    
    # Create metrics optimizer
    optimizer = MetricsOptimizer(collector)
    
    # Start metrics collection
    collection_task = asyncio.create_task(collector.run_collection())
    
    try:
        # Run for a while to collect some data
        await asyncio.sleep(300)  # 5 minutes
        
        # Get optimization recommendations
        recommendations = await optimizer.get_optimization_recommendations()
        print("\nOptimization Recommendations:")
        print(json.dumps(recommendations, indent=2))
        
    except KeyboardInterrupt:
        collection_task.cancel()
        try:
            await collection_task
        except asyncio.CancelledError:
            pass

if __name__ == '__main__':
    asyncio.run(main())

