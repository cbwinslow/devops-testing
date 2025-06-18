#!/usr/bin/env python3

import numpy as np
import asyncio
import aiohttp
import logging
import os
import json
import sys
import docker
from kubernetes import client, config
from typing import List, Dict, Optional
import subprocess
import psutil
from datetime import datetime, timedelta
import yaml
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('auto_actions')

@dataclass
class ResourceLimits:
    cpu_limit: float
    memory_limit: float
    disk_limit: float
    network_limit: float

class AutomatedActions:
    def __init__(self, config_path: str = '~/devops-testing/config/auto_actions.yaml'):
        self.config_path = os.path.expanduser(config_path)
        self.load_config()
        self.setup_clients()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = {
                'resource_limits': {
                    'cpu_limit': 90,
                    'memory_limit': 90,
                    'disk_limit': 90,
                    'network_limit': 1000
                },
                'scaling': {
                    'enabled': True,
                    'min_replicas': 1,
                    'max_replicas': 10,
                    'cpu_threshold': 80,
                    'memory_threshold': 80
                },
                'recovery': {
                    'max_attempts': 3,
                    'backoff_time': 30
                }
            }

    def setup_clients(self):
        """Set up client connections"""
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None

        try:
            config.load_kube_config()
            self.k8s_client = client.CoreV1Api()
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            self.k8s_client = None

    async def monitor_and_act(self):
        """Main monitoring and action loop"""
        while True:
            try:
                # Get current metrics
                metrics = await self.get_system_metrics()
                
                # Check and act on metrics
                actions = []
                
                # Resource scaling
                if await self.needs_scaling(metrics):
                    actions.append(self.scale_resources(metrics))
                
                # Service health
                if await self.check_service_health():
                    actions.append(self.heal_services())
                
                # System maintenance
                if await self.needs_maintenance(metrics):
                    actions.append(self.perform_maintenance())
                
                # Execute actions concurrently
                if actions:
                    await asyncio.gather(*actions)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            await asyncio.sleep(60)  # Check every minute

    async def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_usage'] = psutil.cpu_percent()
        metrics['cpu_load'] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
        metrics['memory_available'] = memory.available / 1024 / 1024  # MB
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_usage'] = disk.percent
        metrics['disk_free'] = disk.free / 1024 / 1024 / 1024  # GB
        
        # Network metrics
        net_io = psutil.net_io_counters()
        metrics['network_in'] = net_io.bytes_recv / 1024 / 1024  # MB
        metrics['network_out'] = net_io.bytes_sent / 1024 / 1024  # MB
        
        return metrics

    async def needs_scaling(self, metrics: Dict) -> bool:
        """Determine if system needs scaling"""
        if not self.config['scaling']['enabled']:
            return False
        
        return (metrics['cpu_usage'] > self.config['scaling']['cpu_threshold'] or
                metrics['memory_usage'] > self.config['scaling']['memory_threshold'])

    async def scale_resources(self, metrics: Dict):
        """Scale system resources"""
        logger.info("Scaling resources...")
        
        try:
            # Docker container scaling
            if self.docker_client:
                await self.scale_docker_containers(metrics)
            
            # Kubernetes scaling
            if self.k8s_client:
                await self.scale_kubernetes_resources(metrics)
            
            # System resource scaling
            await self.scale_system_resources(metrics)
            
        except Exception as e:
            logger.error(f"Error scaling resources: {e}")

    async def scale_docker_containers(self, metrics: Dict):
        """Scale Docker containers"""
        def _scale_container(container, cpu_limit, memory_limit):
            try:
                container.update(
                    cpu_quota=int(cpu_limit * 100000),
                    memory=f"{memory_limit}m"
                )
                logger.info(f"Scaled container {container.name}")
            except Exception as e:
                logger.error(f"Error scaling container {container.name}: {e}")
        
        containers = self.docker_client.containers.list()
        for container in containers:
            self.executor.submit(
                _scale_container,
                container,
                self.config['resource_limits']['cpu_limit'],
                self.config['resource_limits']['memory_limit']
            )

    async def scale_kubernetes_resources(self, metrics: Dict):
        """Scale Kubernetes resources"""
        try:
            # Update deployments
            apps_v1 = client.AppsV1Api()
            deployments = apps_v1.list_deployment_for_all_namespaces()
            
            for deployment in deployments.items:
                if metrics['cpu_usage'] > self.config['scaling']['cpu_threshold']:
                    replicas = min(
                        deployment.spec.replicas * 2,
                        self.config['scaling']['max_replicas']
                    )
                    
                    apps_v1.patch_namespaced_deployment(
                        name=deployment.metadata.name,
                        namespace=deployment.metadata.namespace,
                        body={'spec': {'replicas': replicas}}
                    )
                    
                    logger.info(f"Scaled deployment {deployment.metadata.name} to {replicas} replicas")
        
        except Exception as e:
            logger.error(f"Error scaling Kubernetes resources: {e}")

    async def scale_system_resources(self, metrics: Dict):
        """Scale system-level resources"""
        try:
            # Adjust process priorities
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                if proc.info['cpu_percent'] > 50:
                    p = psutil.Process(proc.info['pid'])
                    p.nice(10)  # Lower priority for CPU-intensive processes
            
            # Adjust I/O scheduling
            for disk in psutil.disk_partitions():
                if disk.fstype == 'ext4':
                    subprocess.run(['ionice', '-c', '2', '-n', '7', '-p', str(os.getpid())])
        
        except Exception as e:
            logger.error(f"Error scaling system resources: {e}")

    async def check_service_health(self) -> bool:
        """Check if services need healing"""
        try:
            unhealthy_containers = []
            
            # Check Docker containers
            if self.docker_client:
                containers = self.docker_client.containers.list()
                for container in containers:
                    if container.status != 'running':
                        unhealthy_containers.append(container.name)
            
            # Check Kubernetes pods
            if self.k8s_client:
                pods = self.k8s_client.list_pod_for_all_namespaces()
                for pod in pods.items:
                    if pod.status.phase not in ['Running', 'Succeeded']:
                        unhealthy_containers.append(f"{pod.metadata.namespace}/{pod.metadata.name}")
            
            return len(unhealthy_containers) > 0
        
        except Exception as e:
            logger.error(f"Error checking service health: {e}")
            return False

    async def heal_services(self):
        """Heal unhealthy services"""
        logger.info("Healing services...")
        
        try:
            # Restart unhealthy Docker containers
            if self.docker_client:
                containers = self.docker_client.containers.list(all=True)
                for container in containers:
                    if container.status != 'running':
                        logger.info(f"Restarting container {container.name}")
                        container.restart()
            
            # Restart unhealthy Kubernetes pods
            if self.k8s_client:
                pods = self.k8s_client.list_pod_for_all_namespaces()
                for pod in pods.items:
                    if pod.status.phase not in ['Running', 'Succeeded']:
                        logger.info(f"Deleting unhealthy pod {pod.metadata.name}")
                        self.k8s_client.delete_namespaced_pod(
                            name=pod.metadata.name,
                            namespace=pod.metadata.namespace
                        )
        
        except Exception as e:
            logger.error(f"Error healing services: {e}")

    async def needs_maintenance(self, metrics: Dict) -> bool:
        """Check if system needs maintenance"""
        return (metrics['disk_usage'] > self.config['resource_limits']['disk_limit'] or
                len(psutil.Process().open_files()) > 1000)

    async def perform_maintenance(self):
        """Perform system maintenance"""
        logger.info("Performing maintenance...")
        
        try:
            # Clean up disk space
            await self.clean_disk_space()
            
            # Clean up Docker resources
            await self.clean_docker_resources()
            
            # Clean up system resources
            await self.clean_system_resources()
            
        except Exception as e:
            logger.error(f"Error performing maintenance: {e}")

    async def clean_disk_space(self):
        """Clean up disk space"""
        try:
            # Clean old logs
            subprocess.run(['find', '/var/log', '-type', 'f', '-mtime', '+30', '-delete'])
            
            # Clean package cache
            subprocess.run(['apt-get', 'clean'])
            subprocess.run(['apt-get', 'autoremove', '-y'])
            
            # Clean temporary files
            subprocess.run(['find', '/tmp', '-type', 'f', '-atime', '+7', '-delete'])
            
        except Exception as e:
            logger.error(f"Error cleaning disk space: {e}")

    async def clean_docker_resources(self):
        """Clean up Docker resources"""
        if self.docker_client:
            try:
                # Remove stopped containers
                self.docker_client.containers.prune()
                
                # Remove unused images
                self.docker_client.images.prune()
                
                # Remove unused volumes
                self.docker_client.volumes.prune()
                
                # Remove unused networks
                self.docker_client.networks.prune()
                
            except Exception as e:
                logger.error(f"Error cleaning Docker resources: {e}")

    async def clean_system_resources(self):
        """Clean up system resources"""
        try:
            # Clear page cache
            subprocess.run(['sync'])
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
            
            # Clean up user cache
            cache_dir = os.path.expanduser('~/.cache')
            if os.path.exists(cache_dir):
                subprocess.run(['rm', '-rf', cache_dir])
            
        except Exception as e:
            logger.error(f"Error cleaning system resources: {e}")

def main():
    actions = AutomatedActions()
    asyncio.run(actions.monitor_and_act())

if __name__ == '__main__':
    main()

