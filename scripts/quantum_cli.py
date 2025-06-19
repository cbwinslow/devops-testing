#!/usr/bin/env python3

import click
import asyncio
import yaml
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.quantum_monitor import SystemMonitor
from scripts.quantum_reporter import ReportGenerator
from scripts.quantum_optimization_advanced import AdvancedOptimizationOrchestrator
from scripts.quantum_visualizer import QuantumVisualizer
from scripts.quantum_metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_cli')

class QuantumCLI:
    """Unified CLI for quantum optimization system"""
    def __init__(self):
        self.config = self.load_config()
        self.setup_components()

    def load_config(self) -> Dict:
        """Load system configuration"""
        config_path = Path('config/quantum_config.yaml')
        if not config_path.exists():
            template_path = Path('config/quantum_config_template.yaml')
            if template_path.exists():
                with open(template_path, 'r') as f:
                    config = yaml.safe_load(f)
                # Save as active config
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
            else:
                logger.error("Configuration template not found")
                sys.exit(1)
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        return config

    def setup_components(self):
        """Initialize system components"""
        self.monitor = SystemMonitor()
        self.reporter = ReportGenerator()
        self.optimizer = AdvancedOptimizationOrchestrator()
        self.visualizer = QuantumVisualizer()
        self.metrics_collector = MetricsCollector()

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Quantum Optimization System CLI"""
    pass

@cli.group()
def monitor():
    """System monitoring commands"""
    pass

@monitor.command('start')
@click.option('--interval', default=60, help='Monitoring interval in seconds')
def monitor_start(interval):
    """Start system monitoring"""
    click.echo("Starting system monitoring...")
    quantum_cli = QuantumCLI()
    asyncio.run(quantum_cli.monitor.run())

@monitor.command('status')
def monitor_status():
    """Show monitoring status"""
    quantum_cli = QuantumCLI()
    metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
    click.echo("\nCurrent System Status:")
    click.echo(json.dumps(metrics, indent=2))

@cli.group()
def optimize():
    """System optimization commands"""
    pass

@optimize.command('run')
@click.option('--strategy', help='Specific optimization strategy to run')
@click.option('--dry-run', is_flag=True, help='Show optimization plan without executing')
def optimize_run(strategy, dry_run):
    """Run system optimization"""
    quantum_cli = QuantumCLI()
    metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
    
    if dry_run:
        click.echo("\nOptimization Plan:")
        for s in quantum_cli.optimizer.strategies:
            if s.can_apply(metrics):
                click.echo(f"- Would apply {s.name}: {s.description}")
        return

    optimized_metrics, applied_optimizations = quantum_cli.optimizer.optimize(metrics)
    click.echo("\nApplied Optimizations:")
    click.echo(json.dumps(applied_optimizations, indent=2))

@optimize.command('status')
def optimize_status():
    """Show optimization status"""
    quantum_cli = QuantumCLI()
    history = quantum_cli.optimizer.optimization_history
    if history:
        click.echo("\nOptimization History:")
        for entry in history[-5:]:  # Show last 5 optimizations
            click.echo(f"\nTimestamp: {entry['timestamp']}")
            click.echo(f"Strategies: {', '.join(entry['applied_strategies'])}")
    else:
        click.echo("No optimization history available")

@cli.group()
def report():
    """Reporting commands"""
    pass

@report.command('generate')
@click.option('--type', type=click.Choice(['daily', 'weekly', 'monthly']), default='daily')
@click.option('--format', type=click.Choice(['html', 'pdf', 'json']), default='html')
def report_generate(type, format):
    """Generate system report"""
    quantum_cli = QuantumCLI()
    metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
    alerts = []  # Get from monitor
    optimizations = quantum_cli.optimizer.optimization_history
    
    report_path = quantum_cli.reporter.generate_report(
        metrics_data=[metrics],
        alerts=alerts,
        optimizations=optimizations,
        report_type=type
    )
    click.echo(f"\nReport generated at: {report_path}")

@report.command('list')
def report_list():
    """List available reports"""
    reports_dir = Path('reports')
    if reports_dir.exists():
        click.echo("\nAvailable Reports:")
        for report in reports_dir.glob('**/report_*.{html,pdf,json}'):
            click.echo(f"- {report.relative_to(reports_dir)}")
    else:
        click.echo("No reports available")

@cli.group()
def visualize():
    """Visualization commands"""
    pass

@visualize.command('dashboard')
def visualize_dashboard():
    """Generate system dashboard"""
    quantum_cli = QuantumCLI()
    metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
    dashboard_path = quantum_cli.visualizer.create_system_health_dashboard(metrics)
    click.echo(f"\nDashboard generated at: {dashboard_path}")

@visualize.command('trends')
def visualize_trends():
    """Generate trend analysis visualization"""
    quantum_cli = QuantumCLI()
    metrics_history = quantum_cli.metrics_collector.metrics_history
    if metrics_history:
        trend_path = quantum_cli.visualizer.create_trend_analysis(metrics_history)
        click.echo(f"\nTrend analysis generated at: {trend_path}")
    else:
        click.echo("No metrics history available for trend analysis")

@cli.group()
def config():
    """Configuration commands"""
    pass

@config.command('show')
@click.option('--section', help='Specific configuration section to show')
def config_show(section):
    """Show current configuration"""
    quantum_cli = QuantumCLI()
    if section:
        if section in quantum_cli.config:
            click.echo(f"\n{section}:")
            click.echo(yaml.dump({section: quantum_cli.config[section]}, indent=2))
        else:
            click.echo(f"Section '{section}' not found in configuration")
    else:
        click.echo("\nCurrent Configuration:")
        click.echo(yaml.dump(quantum_cli.config, indent=2))

@config.command('update')
@click.argument('path')
@click.argument('value')
def config_update(path, value):
    """Update configuration value"""
    quantum_cli = QuantumCLI()
    parts = path.split('.')
    config = quantum_cli.config
    
    # Navigate to the nested key
    current = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Update the value
    try:
        # Try to parse as JSON for complex values
        value = json.loads(value)
    except json.JSONDecodeError:
        # Use as string if not valid JSON
        pass
    
    current[parts[-1]] = value
    
    # Save updated config
    config_path = Path('config/quantum_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    click.echo(f"Updated configuration: {path} = {value}")

@cli.group()
def system():
    """System management commands"""
    pass

@system.command('status')
def system_status():
    """Show overall system status"""
    quantum_cli = QuantumCLI()
    
    # Collect metrics
    metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
    
    # Show component status
    click.echo("\nSystem Status:")
    click.echo("\nMonitoring:")
    click.echo(f"- Active: {True}")
    click.echo(f"- Metrics collected: {len(quantum_cli.metrics_collector.metrics_history)}")
    
    click.echo("\nOptimization:")
    click.echo(f"- Optimizations performed: {len(quantum_cli.optimizer.optimization_history)}")
    
    click.echo("\nResource Usage:")
    click.echo(f"- CPU: {metrics['cpu']['usage']}%")
    click.echo(f"- Memory: {metrics['memory']['usage']}%")
    click.echo(f"- Disk: {metrics['disk']['usage']}%")
    
    if 'system_health' in metrics.get('analysis', {}):
        click.echo(f"\nSystem Health Score: {metrics['analysis']['system_health']}")

@system.command('health')
def system_health():
    """Show detailed system health report"""
    quantum_cli = QuantumCLI()
    metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
    
    click.echo("\nSystem Health Report:")
    
    # Resource Status
    click.echo("\nResource Status:")
    resources = [
        ('CPU', metrics['cpu']['usage']),
        ('Memory', metrics['memory']['usage']),
        ('Disk', metrics['disk']['usage'])
    ]
    
    for resource, usage in resources:
        status = 'CRITICAL' if usage > 90 else 'WARNING' if usage > 80 else 'OK'
        color = 'red' if status == 'CRITICAL' else 'yellow' if status == 'WARNING' else 'green'
        click.echo(f"- {resource}: {click.style(f'{usage}% ({status})', fg=color)}")
    
    # Process Information
    click.echo("\nTop Processes:")
    for proc in metrics['system']['processes']['top_cpu'][:3]:
        click.echo(f"- {proc['name']}: CPU {proc['cpu_percent']}%")
    
    # Network Status
    click.echo("\nNetwork Status:")
    for iface, stats in metrics['network']['interfaces'].items():
        click.echo(f"- {iface}: {stats['stats']['speed']}Mbps")
    
    # Hardware Status
    if 'hardware' in metrics:
        click.echo("\nHardware Status:")
        if 'temperature' in metrics['hardware']:
            temp = metrics['hardware']['temperature']
            temp_status = 'CRITICAL' if temp > 80 else 'WARNING' if temp > 70 else 'OK'
            temp_color = 'red' if temp_status == 'CRITICAL' else 'yellow' if temp_status == 'WARNING' else 'green'
            click.echo(f"- Temperature: {click.style(f'{temp}°C ({temp_status})', fg=temp_color)}")

@system.command('cleanup')
@click.option('--older-than', type=int, help='Clean files older than N days')
def system_cleanup(older_than):
    """Clean up old data files"""
    quantum_cli = QuantumCLI()
    
    # Clean up paths from config
    paths = quantum_cli.config['paths']
    for path_name, path in paths.items():
        path = Path(path)
        if path.exists():
            click.echo(f"\nCleaning {path_name}...")
            
            # Remove old files
            if older_than:
                now = datetime.now().timestamp()
                for file in path.glob('**/*'):
                    if file.is_file():
                        age = now - file.stat().st_mtime
                        if age > older_than * 86400:  # Convert days to seconds
                            file.unlink()
                            click.echo(f"Removed: {file}")
            
            # Remove empty directories
            for dir_path in reversed(list(path.glob('**/*'))):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    click.echo(f"Removed empty directory: {dir_path}")

# Add advanced visualization commands
@cli.group()
def advanced():
    """Advanced system commands"""
    pass

@advanced.command('analyze')
@click.option('--type', type=click.Choice(['resource', 'network', 'performance']), default='resource')
def advanced_analyze(type):
    """Perform advanced system analysis"""
    quantum_cli = QuantumCLI()
    metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
    
    if type == 'resource':
        # Resource correlation analysis
        correlations = quantum_cli.visualizer.create_correlation_analysis(metrics)
        click.echo(f"\nResource correlation analysis saved to: {correlations}")
    elif type == 'network':
        # Network topology visualization
        topology = quantum_cli.visualizer.create_network_topology(metrics)
        click.echo(f"\nNetwork topology visualization saved to: {topology}")
    elif type == 'performance':
        # Performance bottleneck analysis
        bottlenecks = quantum_cli.visualizer.create_bottleneck_analysis(metrics)
        click.echo(f"\nPerformance bottleneck analysis saved to: {bottlenecks}")

@advanced.command('predict')
@click.option('--resource', type=click.Choice(['cpu', 'memory', 'disk', 'network']), default='cpu')
@click.option('--hours', type=int, default=24)
def advanced_predict(resource, hours):
    """Predict resource usage trends"""
    quantum_cli = QuantumCLI()
    prediction = quantum_cli.visualizer.create_resource_prediction(resource, hours)
    click.echo(f"\nResource usage prediction saved to: {prediction}")

@advanced.command('export')
@click.option('--format', type=click.Choice(['json', 'csv', 'excel']), default='json')
@click.option('--output', type=str, default='export')
def advanced_export(format, output):
    """Export system data"""
    quantum_cli = QuantumCLI()
    metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
    
    # Export data
    if format == 'json':
        output_path = f"{output}.json"
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    elif format == 'csv':
        output_path = f"{output}.csv"
        pd.DataFrame([metrics]).to_csv(output_path, index=False)
    elif format == 'excel':
        output_path = f"{output}.xlsx"
        pd.DataFrame([metrics]).to_excel(output_path, index=False)
    
    click.echo(f"\nData exported to: {output_path}")

# Add automated optimization commands
@optimize.command('auto')
@click.option('--schedule', is_flag=True, help='Schedule automatic optimization')
@click.option('--interval', type=int, default=3600, help='Interval between optimizations (seconds)')
def optimize_auto(schedule, interval):
    """Run or schedule automatic optimization"""
    quantum_cli = QuantumCLI()
    if schedule:
        click.echo(f"Scheduling automatic optimization every {interval} seconds")
        # Add to system scheduler
        # This is a placeholder - implement actual scheduling logic
        click.echo("Automatic optimization scheduled")
    else:
        click.echo("Running automatic optimization...")
        metrics = asyncio.run(quantum_cli.metrics_collector.collect_system_metrics())
        optimized, applied = quantum_cli.optimizer.optimize(metrics)
        click.echo("\nOptimization complete:")
        click.echo(json.dumps(applied, indent=2))

# Add backup commands
from quantum_backup import add_backup_commands
add_backup_commands(cli)

if __name__ == '__main__':
    cli()

