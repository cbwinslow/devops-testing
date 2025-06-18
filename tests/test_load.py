#!/usr/bin/env python3

import unittest
import asyncio
import aiohttp
import time
import multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pytest
from typing import List, Dict
import psutil
import os
import json
from datetime import datetime, timedelta

from scripts.distributed_monitor import DistributedMonitor, MetricsCollector

class LoadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.monitor = DistributedMonitor()
        cls.collector = MetricsCollector()
        cls.start_time = time.time()
        cls.metrics = []

    def setUp(self):
        """Set up individual test"""
        self.test_start = time.time()

    def tearDown(self):
        """Clean up after test"""
        duration = time.time() - self.test_start
        self.metrics.append({
            'test': self._testMethodName,
            'duration': duration,
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent
        })

    @classmethod
    def tearDownClass(cls):
        """Generate test report"""
        total_duration = time.time() - cls.start_time
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'tests': len(cls.metrics),
            'metrics': cls.metrics,
            'summary': {
                'avg_duration': np.mean([m['duration'] for m in cls.metrics]),
                'max_cpu': max(m['cpu_usage'] for m in cls.metrics),
                'max_memory': max(m['memory_usage'] for m in cls.metrics)
            }
        }

        os.makedirs('reports/load_tests', exist_ok=True)
        with open(f'reports/load_tests/report_{datetime.now():%Y%m%d_%H%M%S}.json', 'w') as f:
            json.dump(report, f, indent=2)

    async def generate_load(self, duration: int, rate: int) -> List[Dict]:
        """Generate synthetic load"""
        results = []
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration:
                tasks = []
                for _ in range(rate):
                    tasks.append(self.make_request(session))
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)
                await asyncio.sleep(1)
        
        return results

    async def make_request(self, session: aiohttp.ClientSession) -> Dict:
        """Make a single request"""
        start_time = time.time()
        try:
            async with session.get('http://localhost:8000/metrics') as response:
                data = await response.text()
                return {
                    'status': response.status,
                    'latency': time.time() - start_time,
                    'size': len(data)
                }
        except Exception as e:
            return {
                'status': 'error',
                'latency': time.time() - start_time,
                'error': str(e)
            }

    @pytest.mark.benchmark
    def test_constant_load(self):
        """Test system under constant load"""
        results = asyncio.run(self.generate_load(duration=60, rate=10))
        
        latencies = [r['latency'] for r in results if r.get('status') == 200]
        self.assertTrue(len(latencies) > 0)
        self.assertLess(np.mean(latencies), 0.1)  # Average latency under 100ms
        self.assertLess(np.percentile(latencies, 95), 0.2)  # 95th percentile under 200ms

    @pytest.mark.benchmark
    def test_increasing_load(self):
        """Test system under increasing load"""
        async def run_test():
            results = []
            for rate in range(10, 110, 10):
                batch = await self.generate_load(duration=30, rate=rate)
                results.extend(batch)
                await asyncio.sleep(5)  # Cool-down period
            return results

        results = asyncio.run(run_test())
        success_rate = len([r for r in results if r.get('status') == 200]) / len(results)
        self.assertGreater(success_rate, 0.95)  # 95% success rate

    @pytest.mark.benchmark
    def test_burst_load(self):
        """Test system under burst load"""
        async def burst():
            return await self.generate_load(duration=10, rate=100)

        results = asyncio.run(burst())
        errors = [r for r in results if r.get('status') != 200]
        self.assertLess(len(errors) / len(results), 0.05)  # Less than 5% errors

    @pytest.mark.benchmark
    def test_concurrent_clients(self):
        """Test system with multiple concurrent clients"""
        async def client(client_id: int):
            results = await self.generate_load(duration=30, rate=5)
            return {'client_id': client_id, 'results': results}

        async def run_clients():
            tasks = [client(i) for i in range(20)]
            return await asyncio.gather(*tasks)

        all_results = asyncio.run(run_clients())
        for client_results in all_results:
            success_rate = len([r for r in client_results['results'] if r.get('status') == 200]) / len(client_results['results'])
            self.assertGreater(success_rate, 0.95)

    def test_memory_leak(self):
        """Test for memory leaks"""
        initial_memory = psutil.Process().memory_info().rss
        
        # Run intensive operations
        asyncio.run(self.generate_load(duration=60, rate=50))
        
        # Allow for garbage collection
        import gc
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_growth = (final_memory - initial_memory) / initial_memory
        
        self.assertLess(memory_growth, 0.1)  # Less than 10% memory growth

    def test_cpu_utilization(self):
        """Test CPU utilization under load"""
        cpu_percentages = []
        
        async def monitor_cpu():
            start_time = time.time()
            while time.time() - start_time < 60:
                cpu_percentages.append(psutil.cpu_percent(interval=1))
                await asyncio.sleep(1)

        async def run_test():
            cpu_task = asyncio.create_task(monitor_cpu())
            load_task = asyncio.create_task(self.generate_load(duration=60, rate=20))
            await asyncio.gather(cpu_task, load_task)

        asyncio.run(run_test())
        
        avg_cpu = np.mean(cpu_percentages)
        self.assertLess(avg_cpu, 80)  # Average CPU usage under 80%
        self.assertLess(max(cpu_percentages), 95)  # Peak CPU under 95%

    @pytest.mark.benchmark
    def test_recovery_time(self):
        """Test system recovery after heavy load"""
        async def measure_latency():
            async with aiohttp.ClientSession() as session:
                start = time.time()
                async with session.get('http://localhost:8000/metrics') as response:
                    return time.time() - start

        async def run_test():
            # Generate heavy load
            await self.generate_load(duration=30, rate=100)
            
            # Measure recovery
            latencies = []
            start_time = time.time()
            while time.time() - start_time < 30:
                latency = await measure_latency()
                latencies.append(latency)
                if len(latencies) >= 5 and np.mean(latencies[-5:]) < 0.1:
                    break
                await asyncio.sleep(1)
            
            return time.time() - start_time, latencies

        recovery_time, latencies = asyncio.run(run_test())
        self.assertLess(recovery_time, 30)  # Should recover within 30 seconds

if __name__ == '__main__':
    unittest.main()

