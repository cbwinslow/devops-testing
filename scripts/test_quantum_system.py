#!/usr/bin/env python3

import asyncio
import json
import logging
import yaml
import logging.config
from pathlib import Path
from datetime import datetime
from quantum_automation import QuantumAutomation
from quantum_metrics import MetricsCollector, MetricsOptimizer

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_config_path = Path('config/logging_config.yaml')
    if log_config_path.exists():
        with open(log_config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

# Test scenarios
TEST_SCENARIOS = {
    'normal_load': {
        'cpu_usage': 45.5,
        'memory_usage': 55.2,
        'disk_usage': 35.8,
        'network_load': 25.3,
        'process_count': 150,
        'swap_usage': 15.0,
        'load_average': [1.5, 1.2, 1.0]
    },
    'high_cpu': {
        'cpu_usage': 85.5,
        'memory_usage': 55.2,
        'disk_usage': 35.8,
        'network_load': 25.3,
        'process_count': 200,
        'swap_usage': 20.0,
        'load_average': [4.5, 4.0, 3.5]
    },
    'high_memory': {
        'cpu_usage': 45.5,
        'memory_usage': 95.2,
        'disk_usage': 35.8,
        'network_load': 25.3,
        'process_count': 180,
        'swap_usage': 75.0,
        'load_average': [2.5, 2.2, 2.0]
    },
    'high_disk': {
        'cpu_usage': 45.5,
        'memory_usage': 55.2,
        'disk_usage': 95.8,
        'network_load': 25.3,
        'process_count': 160,
        'swap_usage': 25.0,
        'load_average': [1.8, 1.5, 1.2]
    },
    'system_stress': {
        'cpu_usage': 95.5,
        'memory_usage': 95.2,
        'disk_usage': 95.8,
        'network_load': 85.3,
        'process_count': 300,
        'swap_usage': 90.0,
        'load_average': [8.5, 8.0, 7.5]
    },
    'network_congestion': {
        'cpu_usage': 65.5,
        'memory_usage': 75.2,
        'disk_usage': 45.8,
        'network_load': 95.3,
        'process_count': 200,
        'swap_usage': 30.0,
        'load_average': [3.5, 3.0, 2.5]
    },
    'memory_leak': {
        'cpu_usage': 55.5,
        'memory_usage': 98.2,
        'disk_usage': 55.8,
        'network_load': 35.3,
        'process_count': 250,
        'swap_usage': 95.0,
        'load_average': [2.8, 2.5, 2.2]
    },
    'io_bottleneck': {
        'cpu_usage': 35.5,
        'memory_usage': 65.2,
        'disk_usage': 92.8,
        'network_load': 15.3,
        'process_count': 180,
        'swap_usage': 40.0,
        'load_average': [4.2, 4.0, 3.8]
    },
    'resource_exhaustion': {
        'cpu_usage': 98.5,
        'memory_usage': 99.2,
        'disk_usage': 99.8,
        'network_load': 95.3,
        'process_count': 350,
        'swap_usage': 98.0,
        'load_average': [12.5, 12.0, 11.5]
    },
    'cascade_failure': {
        'cpu_usage': 100.0,
        'memory_usage': 100.0,
        'disk_usage': 100.0,
        'network_load': 100.0,
        'process_count': 400,
        'swap_usage': 100.0,
        'load_average': [15.0, 14.5, 14.0]
    }
}

class QuantumSystemTester:
    def __init__(self):
        self.automation = QuantumAutomation()
        self.metrics_collector = MetricsCollector()
        self.metrics_optimizer = MetricsOptimizer(self.metrics_collector)
        self.logger = logging.getLogger('quantum_system_tester')

    async def run_scenario(self, scenario_name: str, metrics: dict):
        """Run a test scenario"""
        self.logger.info(f"Running scenario: {scenario_name}")
        
        try:
            # Add timestamp to metrics
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Collect and analyze metrics
            enhanced_metrics = await self.metrics_collector.enhance_metrics(metrics)
            analysis = self.metrics_collector.analyze_metrics(enhanced_metrics)
            enhanced_metrics['analysis'] = analysis
            
            # Get optimization recommendations
            self.metrics_collector.metrics_history.append(enhanced_metrics)
            recommendations = await self.metrics_optimizer.get_optimization_recommendations()
            
            # Run quantum optimization
            optimizations = await self.automation.optimize_system(metrics)
            
            # Combine results
            results = {
                'scenario': scenario_name,
                'metrics': enhanced_metrics,
                'recommendations': recommendations,
                'optimizations': optimizations
            }
            
            # Log results
            self.logger.info(f"Scenario {scenario_name} completed")
            self.logger.debug(f"Results: {json.dumps(results, indent=2)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running scenario {scenario_name}: {e}")
            return None

    async def run_all_scenarios(self):
        """Run all test scenarios"""
        results = {}
        
        for scenario_name, metrics in TEST_SCENARIOS.items():
            self.logger.info(f"Starting scenario: {scenario_name}")
            result = await self.run_scenario(scenario_name, metrics)
            if result:
                results[scenario_name] = result
            await asyncio.sleep(1)  # Brief pause between scenarios
        
        return results

async def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger('quantum_system_tester')
    
    try:
        # Create output directory
        output_dir = Path('test_results')
        output_dir.mkdir(exist_ok=True)
        
        # Create and run tester
        tester = QuantumSystemTester()
        logger.info("Starting system tests")
        
        results = await tester.run_all_scenarios()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")
        
        # Print summary
        print("\nTest Summary:")
        for scenario, result in results.items():
            print(f"\nScenario: {scenario}")
            print(f"System Health: {result['metrics']['analysis']['system_health']:.2f}")
            print("Immediate Actions Required:", len(result['recommendations']['immediate_actions']))
            print("Optimization Recommendations:", len(result['recommendations']['resource_optimization']))
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == '__main__':
    asyncio.run(main())

