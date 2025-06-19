#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_visualizer')

class QuantumVisualizer:
    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_style()

    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('dark_background')
        sns.set_theme(style="darkgrid")
        
    def create_system_health_dashboard(self, metrics_data: Dict) -> str:
        """Create an interactive system health dashboard"""
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(rows=2, cols=2,
                              subplot_titles=('Resource Usage', 'System Health Score',
                                            'Resource Efficiency', 'Network Performance'))

            # Resource Usage
            fig.add_trace(go.Scatter(y=[metrics_data['cpu_usage']], name="CPU",
                                   line=dict(color="#00ff00")), row=1, col=1)
            fig.add_trace(go.Scatter(y=[metrics_data['memory_usage']], name="Memory",
                                   line=dict(color="#ff00ff")), row=1, col=1)
            fig.add_trace(go.Scatter(y=[metrics_data['disk_usage']], name="Disk",
                                   line=dict(color="#00ffff")), row=1, col=1)

            # System Health Score
            health_score = metrics_data['analysis']['system_health']
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=health_score,
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': self._get_health_color(health_score)}},
                title={'text': "System Health"}
            ), row=1, col=2)

            # Resource Efficiency
            efficiency = metrics_data['resource_efficiency']['resource_utilization']
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=efficiency,
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': self._get_health_color(efficiency)}},
                title={'text': "Resource Efficiency"}
            ), row=2, col=1)

            # Network Performance
            fig.add_trace(go.Scatter(y=[metrics_data['network_load']], name="Network Load",
                                   line=dict(color="#ffffff")), row=2, col=2)

            # Update layout
            fig.update_layout(
                title_text="System Health Dashboard",
                title_x=0.5,
                height=800,
                showlegend=True,
                paper_bgcolor='black',
                plot_bgcolor='black',
                font=dict(color='white')
            )

            # Save dashboard
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"dashboard_{timestamp}.html"
            fig.write_html(str(output_file))
            
            return str(output_file)

        except Exception as e:
            logger.error(f"Error creating system health dashboard: {e}")
            return ""

    def create_optimization_summary(self, results: Dict) -> str:
        """Create optimization summary visualization"""
        try:
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 3)
            
            # Resource Usage Plot
            ax1 = fig.add_subplot(gs[0, :2])
            resources = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_load']
            values = [results['metrics'][r] for r in resources]
            ax1.bar(resources, values, color=['#00ff00', '#ff00ff', '#00ffff', '#ffffff'])
            ax1.set_title('Resource Usage')
            ax1.set_ylim(0, 100)
            
            # System Health Gauge
            ax2 = fig.add_subplot(gs[0, 2])
            health_score = results['metrics']['analysis']['system_health']
            self._create_gauge(ax2, health_score, 'System Health')
            
            # Optimization Recommendations
            ax3 = fig.add_subplot(gs[1, :])
            self._create_recommendations_plot(ax3, results['recommendations'])
            
            # Style and save
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"optimization_summary_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            
            return str(output_file)

        except Exception as e:
            logger.error(f"Error creating optimization summary: {e}")
            return ""

    def create_trend_analysis(self, metrics_history: List[Dict]) -> str:
        """Create trend analysis visualization"""
        try:
            # Convert metrics history to DataFrame
            df = pd.DataFrame(metrics_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create figure
            fig = make_subplots(rows=2, cols=2,
                              subplot_titles=('Resource Usage Trends',
                                            'System Health Trend',
                                            'Resource Efficiency Trend',
                                            'Network Performance Trend'))
            
            # Resource Usage Trends
            for resource in ['cpu_usage', 'memory_usage', 'disk_usage']:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df[resource],
                             name=resource.replace('_', ' ').title()),
                    row=1, col=1
                )
            
            # System Health Trend
            health_scores = [m['analysis']['system_health'] for m in metrics_history]
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=health_scores,
                         name='System Health'),
                row=1, col=2
            )
            
            # Resource Efficiency Trend
            efficiency_scores = [m['resource_efficiency']['resource_utilization']
                               for m in metrics_history]
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=efficiency_scores,
                         name='Resource Efficiency'),
                row=2, col=1
            )
            
            # Network Performance Trend
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['network_load'],
                         name='Network Load'),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                template='plotly_dark',
                title_text='System Performance Trends'
            )
            
            # Save visualization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"trend_analysis_{timestamp}.html"
            fig.write_html(str(output_file))
            
            return str(output_file)

        except Exception as e:
            logger.error(f"Error creating trend analysis: {e}")
            return ""

    def create_optimization_impact(self, before: Dict, after: Dict) -> str:
        """Create visualization of optimization impact"""
        try:
            # Create figure
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)
            
            # Resource Usage Comparison
            ax1 = fig.add_subplot(gs[0, 0])
            resources = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_load']
            x = np.arange(len(resources))
            width = 0.35
            
            before_values = [before['metrics'][r] for r in resources]
            after_values = [after['metrics'][r] for r in resources]
            
            ax1.bar(x - width/2, before_values, width, label='Before', color='#ff0000')
            ax1.bar(x + width/2, after_values, width, label='After', color='#00ff00')
            ax1.set_xticks(x)
            ax1.set_xticklabels(resources)
            ax1.set_title('Resource Usage Impact')
            ax1.legend()
            
            # System Health Impact
            ax2 = fig.add_subplot(gs[0, 1])
            before_health = before['metrics']['analysis']['system_health']
            after_health = after['metrics']['analysis']['system_health']
            self._create_health_comparison(ax2, before_health, after_health)
            
            # Optimization Metrics
            ax3 = fig.add_subplot(gs[1, :])
            self._create_optimization_metrics(ax3, before, after)
            
            # Style and save
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"optimization_impact_{timestamp}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            
            return str(output_file)

        except Exception as e:
            logger.error(f"Error creating optimization impact visualization: {e}")
            return ""

    def _get_health_color(self, score: float) -> str:
        """Get color based on health score"""
        if score >= 80:
            return "#00ff00"  # Green
        elif score >= 60:
            return "#ffff00"  # Yellow
        else:
            return "#ff0000"  # Red

    def _create_gauge(self, ax, value: float, title: str):
        """Create a gauge chart"""
        colors = [(0, '#ff0000'), (0.5, '#ffff00'), (1, '#00ff00')]
        norm = plt.Normalize(0, 100)
        cmap = plt.cm.RdYlGn
        
        theta = np.linspace(0, 180, 100)
        r = 1.0
        
        for t in theta:
            color = cmap(norm(t))
            ax.plot([0, r * np.cos(np.radians(t))],
                   [0, r * np.sin(np.radians(t))],
                   color=color, alpha=0.1)
        
        value_theta = np.radians(value * 180 / 100)
        ax.plot([0, r * np.cos(value_theta)],
                [0, r * np.sin(value_theta)],
                color=self._get_health_color(value), linewidth=3)
        
        ax.set_title(f"{title}\n{value:.1f}%")
        ax.set_axis_off()

    def _create_recommendations_plot(self, ax, recommendations: Dict):
        """Create recommendations summary plot"""
        categories = ['Immediate Actions', 'Long Term Actions']
        values = [len(recommendations['immediate_actions']),
                 len(recommendations['long_term_actions'])]
        
        ax.bar(categories, values, color=['#ff0000', '#00ff00'])
        ax.set_title('Optimization Recommendations')
        ax.set_ylim(0, max(values) + 1)

    def _create_health_comparison(self, ax, before: float, after: float):
        """Create health score comparison"""
        ax.arrow(0, 0, np.cos(np.radians(before * 180 / 100)),
                np.sin(np.radians(before * 180 / 100)),
                head_width=0.05, head_length=0.1, fc='#ff0000', ec='#ff0000')
        ax.arrow(0, 0, np.cos(np.radians(after * 180 / 100)),
                np.sin(np.radians(after * 180 / 100)),
                head_width=0.05, head_length=0.1, fc='#00ff00', ec='#00ff00')
        
        ax.set_title('System Health Impact')
        ax.legend(['Before', 'After'])
        ax.set_axis_off()

    def _create_optimization_metrics(self, ax, before: Dict, after: Dict):
        """Create optimization metrics comparison"""
        metrics = ['resource_efficiency', 'system_health', 'performance_score']
        before_values = [
            before['metrics']['resource_efficiency']['resource_utilization'],
            before['metrics']['analysis']['system_health'],
            (before['metrics']['cpu_usage'] + before['metrics']['memory_usage']) / 2
        ]
        after_values = [
            after['metrics']['resource_efficiency']['resource_utilization'],
            after['metrics']['analysis']['system_health'],
            (after['metrics']['cpu_usage'] + after['metrics']['memory_usage']) / 2
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, before_values, width, label='Before', color='#ff0000')
        ax.bar(x + width/2, after_values, width, label='After', color='#00ff00')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title('Optimization Metrics Impact')
        ax.legend()

async def main():
    # Example usage
    visualizer = QuantumVisualizer()
    
    # Load some test results
    results_dir = Path('test_results')
    if results_dir.exists():
        result_files = list(results_dir.glob('*.json'))
        if result_files:
            latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
            with open(latest_result, 'r') as f:
                results = json.load(f)
            
            # Create visualizations
            for scenario, result in results.items():
                dashboard_file = visualizer.create_system_health_dashboard(result['metrics'])
                summary_file = visualizer.create_optimization_summary(result)
                print(f"\nScenario: {scenario}")
                print(f"Dashboard: {dashboard_file}")
                print(f"Summary: {summary_file}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

