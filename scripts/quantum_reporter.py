#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import jinja2
import pdfkit
import yaml
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_reporter')

class ReportGenerator:
    """Generate system performance reports"""
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_templates()

    def setup_templates(self):
        """Setup Jinja2 templates"""
        self.template_dir = Path('templates')
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default template if it doesn't exist
        self.template_path = self.template_dir / 'report_template.html'
        if not self.template_path.exists():
            self._create_default_template()

    def _create_default_template(self):
        """Create default HTML template"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333366; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ccc; }
                .metric { margin: 10px 0; }
                .critical { color: #ff0000; }
                .warning { color: #ff9900; }
                .normal { color: #009900; }
                .chart { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{{ summary }}</p>
            </div>
            
            <div class="section">
                <h2>System Health Overview</h2>
                {{ health_overview }}
            </div>
            
            <div class="section">
                <h2>Resource Utilization</h2>
                {{ resource_charts }}
            </div>
            
            <div class="section">
                <h2>Alert Summary</h2>
                {{ alert_summary }}
            </div>
            
            <div class="section">
                <h2>Optimization Results</h2>
                {{ optimization_summary }}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {{ recommendations }}
            </div>
        </body>
        </html>
        """
        with open(self.template_path, 'w') as f:
            f.write(template)

    def generate_report(self, 
                       metrics_data: List[Dict],
                       alerts: List[Dict],
                       optimizations: List[Dict],
                       report_type: str = 'daily') -> str:
        """Generate a comprehensive system performance report"""
        try:
            # Create report directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = self.output_dir / f"{report_type}_report_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)

            # Generate visualizations
            viz_dir = report_dir / 'visualizations'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            charts = self._generate_visualizations(metrics_data, viz_dir)
            
            # Prepare report data
            report_data = {
                'title': f"System Performance Report - {report_type.capitalize()}",
                'summary': self._generate_executive_summary(metrics_data, alerts, optimizations),
                'health_overview': self._generate_health_overview(metrics_data),
                'resource_charts': self._embed_charts(charts),
                'alert_summary': self._generate_alert_summary(alerts),
                'optimization_summary': self._generate_optimization_summary(optimizations),
                'recommendations': self._generate_recommendations(metrics_data, alerts, optimizations)
            }
            
            # Generate HTML report
            html_report = self._generate_html_report(report_data)
            html_path = report_dir / f"report_{timestamp}.html"
            with open(html_path, 'w') as f:
                f.write(html_report)
            
            # Generate PDF report
            pdf_path = report_dir / f"report_{timestamp}.pdf"
            pdfkit.from_string(html_report, str(pdf_path))
            
            # Generate JSON data export
            json_path = report_dir / f"report_data_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'metrics': metrics_data,
                    'alerts': alerts,
                    'optimizations': optimizations,
                    'report_metadata': {
                        'type': report_type,
                        'generated_at': timestamp,
                        'summary_stats': self._calculate_summary_stats(metrics_data)
                    }
                }, f, indent=2)
            
            return str(report_dir)

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""

    def _generate_visualizations(self, metrics_data: List[Dict], viz_dir: Path) -> Dict[str, str]:
        """Generate visualization charts"""
        charts = {}
        
        try:
            # Convert metrics to DataFrame
            df = pd.DataFrame(metrics_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Resource utilization over time
            fig = self._create_resource_utilization_plot(df)
            chart_path = viz_dir / 'resource_utilization.html'
            fig.write_html(str(chart_path))
            charts['resource_utilization'] = str(chart_path)
            
            # System health score
            fig = self._create_health_score_plot(df)
            chart_path = viz_dir / 'health_score.html'
            fig.write_html(str(chart_path))
            charts['health_score'] = str(chart_path)
            
            # Resource correlation heatmap
            fig = self._create_correlation_heatmap(df)
            chart_path = viz_dir / 'correlation_heatmap.png'
            plt.savefig(chart_path)
            plt.close()
            charts['correlation_heatmap'] = str(chart_path)
            
            return charts

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return {}

    def _create_resource_utilization_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create resource utilization plot"""
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('CPU Usage', 'Memory Usage',
                                        'Disk Usage', 'Network Load'))
        
        # CPU Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu.usage'],
                      name='CPU', line=dict(color='#00ff00')),
            row=1, col=1
        )
        
        # Memory Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory.usage'],
                      name='Memory', line=dict(color='#ff00ff')),
            row=1, col=2
        )
        
        # Disk Usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['disk.usage'],
                      name='Disk', line=dict(color='#00ffff')),
            row=2, col=1
        )
        
        # Network Load
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['network.load'],
                      name='Network', line=dict(color='#ffffff')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Resource Utilization Over Time")
        return fig

    def _create_health_score_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create health score plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=df['analysis.system_health'].iloc[-1],
            delta={'reference': df['analysis.system_health'].mean()},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ]
            },
            title={'text': "System Health Score"}
        ))
        
        fig.update_layout(height=400)
        return fig

    def _create_correlation_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """Create correlation heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[[
            'cpu.usage', 'memory.usage', 'disk.usage', 'network.load'
        ]].corr(), annot=True, cmap='RdYlGn_r')
        plt.title('Resource Correlation Heatmap')
        return plt.gcf()

    def _generate_executive_summary(self, 
                                  metrics_data: List[Dict],
                                  alerts: List[Dict],
                                  optimizations: List[Dict]) -> str:
        """Generate executive summary"""
        try:
            df = pd.DataFrame(metrics_data)
            latest_metrics = df.iloc[-1]
            
            critical_alerts = sum(1 for a in alerts if a['severity'] == 'critical')
            warning_alerts = sum(1 for a in alerts if a['severity'] == 'warning')
            
            return f"""
            System Performance Summary for {datetime.now().strftime('%Y-%m-%d')}
            
            Overall System Health: {latest_metrics['analysis.system_health']:.1f}%
            
            Alert Status:
            - Critical Alerts: {critical_alerts}
            - Warning Alerts: {warning_alerts}
            
            Resource Utilization:
            - CPU: {latest_metrics['cpu.usage']:.1f}%
            - Memory: {latest_metrics['memory.usage']:.1f}%
            - Disk: {latest_metrics['disk.usage']:.1f}%
            - Network Load: {latest_metrics['network.load']:.1f}%
            
            Optimizations Applied: {len(optimizations)}
            """

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Error generating executive summary"

    def _generate_health_overview(self, metrics_data: List[Dict]) -> str:
        """Generate health overview section"""
        try:
            df = pd.DataFrame(metrics_data)
            latest = df.iloc[-1]
            
            status_html = """
            <table>
                <tr>
                    <th>Resource</th>
                    <th>Current Usage</th>
                    <th>Status</th>
                    <th>Trend</th>
                </tr>
            """
            
            resources = [
                ('CPU', 'cpu.usage'),
                ('Memory', 'memory.usage'),
                ('Disk', 'disk.usage'),
                ('Network', 'network.load')
            ]
            
            for resource_name, metric_key in resources:
                current = latest[metric_key]
                trend = self._calculate_trend(df[metric_key])
                status = self._get_status_class(current)
                
                status_html += f"""
                <tr>
                    <td>{resource_name}</td>
                    <td>{current:.1f}%</td>
                    <td class="{status}">{status.upper()}</td>
                    <td>{trend}</td>
                </tr>
                """
            
            status_html += "</table>"
            return status_html

        except Exception as e:
            logger.error(f"Error generating health overview: {e}")
            return "Error generating health overview"

    def _generate_alert_summary(self, alerts: List[Dict]) -> str:
        """Generate alert summary section"""
        try:
            if not alerts:
                return "<p>No alerts during this period.</p>"
            
            alert_html = """
            <table>
                <tr>
                    <th>Time</th>
                    <th>Severity</th>
                    <th>Resource</th>
                    <th>Message</th>
                </tr>
            """
            
            for alert in sorted(alerts, key=lambda x: x['timestamp'], reverse=True):
                severity_class = 'critical' if alert['severity'] == 'critical' else 'warning'
                alert_html += f"""
                <tr class="{severity_class}">
                    <td>{alert['timestamp']}</td>
                    <td>{alert['severity'].upper()}</td>
                    <td>{alert['resource']}</td>
                    <td>{alert['message']}</td>
                </tr>
                """
            
            alert_html += "</table>"
            return alert_html

        except Exception as e:
            logger.error(f"Error generating alert summary: {e}")
            return "Error generating alert summary"

    def _generate_optimization_summary(self, optimizations: List[Dict]) -> str:
        """Generate optimization summary section"""
        try:
            if not optimizations:
                return "<p>No optimizations applied during this period.</p>"
            
            opt_html = """
            <table>
                <tr>
                    <th>Time</th>
                    <th>Strategy</th>
                    <th>Impact</th>
                </tr>
            """
            
            for opt in sorted(optimizations, key=lambda x: x['timestamp'], reverse=True):
                impact = self._calculate_optimization_impact(opt)
                opt_html += f"""
                <tr>
                    <td>{opt['timestamp']}</td>
                    <td>{opt['strategy']}</td>
                    <td>{impact}</td>
                </tr>
                """
            
            opt_html += "</table>"
            return opt_html

        except Exception as e:
            logger.error(f"Error generating optimization summary: {e}")
            return "Error generating optimization summary"

    def _generate_recommendations(self,
                                metrics_data: List[Dict],
                                alerts: List[Dict],
                                optimizations: List[Dict]) -> str:
        """Generate recommendations section"""
        try:
            df = pd.DataFrame(metrics_data)
            latest = df.iloc[-1]
            
            recommendations = []
            
            # Resource-based recommendations
            if latest['cpu.usage'] > 80:
                recommendations.append({
                    'priority': 'high',
                    'category': 'CPU',
                    'action': 'Consider scaling up CPU resources or optimizing high-CPU processes'
                })
            
            if latest['memory.usage'] > 85:
                recommendations.append({
                    'priority': 'high',
                    'category': 'Memory',
                    'action': 'Investigate potential memory leaks and consider increasing memory allocation'
                })
            
            if latest['disk.usage'] > 90:
                recommendations.append({
                    'priority': 'high',
                    'category': 'Storage',
                    'action': 'Implement cleanup procedures or expand storage capacity'
                })
            
            # Alert-based recommendations
            if any(a['severity'] == 'critical' for a in alerts):
                recommendations.append({
                    'priority': 'critical',
                    'category': 'System',
                    'action': 'Address critical alerts immediately to prevent system degradation'
                })
            
            # Optimization-based recommendations
            if not optimizations:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Optimization',
                    'action': 'Consider implementing automated optimization strategies'
                })
            
            # Generate HTML
            if not recommendations:
                return "<p>No specific recommendations at this time.</p>"
            
            rec_html = """
            <table>
                <tr>
                    <th>Priority</th>
                    <th>Category</th>
                    <th>Recommended Action</th>
                </tr>
            """
            
            for rec in sorted(recommendations, key=lambda x: x['priority']):
                rec_html += f"""
                <tr class="{rec['priority']}">
                    <td>{rec['priority'].upper()}</td>
                    <td>{rec['category']}</td>
                    <td>{rec['action']}</td>
                </tr>
                """
            
            rec_html += "</table>"
            return rec_html

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return "Error generating recommendations"

    def _calculate_summary_stats(self, metrics_data: List[Dict]) -> Dict:
        """Calculate summary statistics"""
        try:
            df = pd.DataFrame(metrics_data)
            return {
                'cpu_avg': df['cpu.usage'].mean(),
                'memory_avg': df['memory.usage'].mean(),
                'disk_avg': df['disk.usage'].mean(),
                'network_avg': df['network.load'].mean(),
                'health_score_avg': df['analysis.system_health'].mean()
            }
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}

    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        try:
            recent = series.tail(5)
            slope = np.polyfit(range(len(recent)), recent, 1)[0]
            
            if abs(slope) < 0.1:
                return "→"
            elif slope > 0:
                return "↗"
            else:
                return "↘"
        except Exception:
            return "→"

    def _get_status_class(self, value: float) -> str:
        """Get status class based on value"""
        if value >= 90:
            return "critical"
        elif value >= 80:
            return "warning"
        else:
            return "normal"

    def _calculate_optimization_impact(self, optimization: Dict) -> str:
        """Calculate and format optimization impact"""
        try:
            before = optimization['metrics_before']
            after = optimization['metrics_after']
            
            impacts = []
            for metric in ['cpu', 'memory', 'disk']:
                if f'{metric}.usage' in before and f'{metric}.usage' in after:
                    change = before[f'{metric}.usage'] - after[f'{metric}.usage']
                    if abs(change) > 1:
                        impacts.append(f"{metric.upper()}: {change:+.1f}%")
            
            return ", ".join(impacts) if impacts else "No significant impact"
        except Exception:
            return "Impact not available"

    def _embed_charts(self, charts: Dict[str, str]) -> str:
        """Embed charts in HTML"""
        html = ""
        for chart_name, chart_path in charts.items():
            if chart_path.endswith('.html'):
                with open(chart_path, 'r') as f:
                    html += f"<div class='chart'>{f.read()}</div>"
            else:
                html += f"<div class='chart'><img src='{chart_path}' alt='{chart_name}'></div>"
        return html

    def _generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML report from template"""
        try:
            template_loader = jinja2.FileSystemLoader(searchpath=str(self.template_dir))
            template_env = jinja2.Environment(loader=template_loader)
            template = template_env.get_template('report_template.html')
            return template.render(**report_data)
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return ""

async def main():
    # Example usage
    try:
        # Create sample data
        metrics_data = [
            {
                'timestamp': datetime.now().isoformat(),
                'cpu.usage': 75.5,
                'memory.usage': 85.2,
                'disk.usage': 65.8,
                'network.load': 45.3,
                'analysis.system_health': 85.0
            }
        ]
        
        alerts = [
            {
                'timestamp': datetime.now().isoformat(),
                'severity': 'warning',
                'resource': 'Memory',
                'message': 'High memory usage detected'
            }
        ]
        
        optimizations = [
            {
                'timestamp': datetime.now().isoformat(),
                'strategy': 'Resource Balancing',
                'metrics_before': {'cpu.usage': 85.5},
                'metrics_after': {'cpu.usage': 75.5}
            }
        ]
        
        # Generate report
        generator = ReportGenerator()
        report_path = generator.generate_report(metrics_data, alerts, optimizations)
        print(f"Report generated at: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

