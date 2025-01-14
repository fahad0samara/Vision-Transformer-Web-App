import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from fpdf import FPDF
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Tuple, Union
import xlsxwriter

logger = logging.getLogger(__name__)

class PerformanceReport:
    """Generates performance reports in various formats."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.report_dir = 'reports'
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
            
        # Set up plotting style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        
    def generate_report(self, metrics: List[Dict], format: str = 'pdf', 
                       report_name: Optional[str] = None) -> str:
        """Generate a performance report in the specified format."""
        if not metrics:
            raise ValueError("No metrics data provided")
            
        if not report_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f'performance_report_{timestamp}'
            
        try:
            if format.lower() == 'pdf':
                return self._generate_pdf_report(metrics, report_name)
            elif format.lower() == 'excel':
                return self._generate_excel_report(metrics, report_name)
            elif format.lower() == 'html':
                return self._generate_html_report(metrics, report_name)
            else:
                raise ValueError(f"Unsupported report format: {format}")
        except Exception as e:
            logger.error(f"Error generating {format} report: {str(e)}")
            raise
            
    def _generate_pdf_report(self, metrics: List[Dict], report_name: str) -> str:
        """Generate a PDF report with performance metrics and visualizations."""
        try:
            # Create figures
            self._create_cpu_memory_plot(metrics)
            cpu_mem_plot = f'{self.report_dir}/cpu_memory_plot.png'
            plt.savefig(cpu_mem_plot)
            plt.close()
            
            self._create_network_plot(metrics)
            network_plot = f'{self.report_dir}/network_plot.png'
            plt.savefig(network_plot)
            plt.close()
            
            if any(m.get('gpu') for m in metrics):
                self._create_gpu_plot(metrics)
                gpu_plot = f'{self.report_dir}/gpu_plot.png'
                plt.savefig(gpu_plot)
                plt.close()
            
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Performance Report', ln=True, align='C')
            pdf.ln(10)
            
            # Add timestamp
            pdf.set_font('Arial', '', 12)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pdf.cell(0, 10, f'Generated on: {timestamp}', ln=True)
            pdf.ln(10)
            
            # Add plots
            pdf.image(cpu_mem_plot, x=10, w=190)
            pdf.ln(5)
            pdf.image(network_plot, x=10, w=190)
            
            if any(m.get('gpu') for m in metrics):
                pdf.ln(5)
                pdf.image(gpu_plot, x=10, w=190)
            
            # Add summary statistics
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Summary Statistics', ln=True)
            pdf.ln(5)
            
            stats = self._calculate_summary_stats(metrics)
            pdf.set_font('Arial', '', 12)
            for key, value in stats.items():
                pdf.cell(0, 10, f'{key}: {value}', ln=True)
            
            # Save PDF
            output_path = f'{self.report_dir}/{report_name}.pdf'
            pdf.output(output_path)
            
            # Clean up temporary plot files
            os.remove(cpu_mem_plot)
            os.remove(network_plot)
            if any(m.get('gpu') for m in metrics):
                os.remove(gpu_plot)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise
            
    def _generate_excel_report(self, metrics: List[Dict], report_name: str) -> str:
        """Generate an Excel report with performance metrics."""
        try:
            output_path = f'{self.report_dir}/{report_name}.xlsx'
            
            # Convert metrics to pandas DataFrame
            df = pd.json_normalize(metrics)
            
            # Create Excel writer
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            
            # Write main metrics sheet
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Create summary sheet
            stats = self._calculate_summary_stats(metrics)
            pd.DataFrame(list(stats.items()), columns=['Metric', 'Value']).to_excel(
                writer, sheet_name='Summary', index=False)
            
            # Add charts
            workbook = writer.book
            chart_sheet = workbook.add_worksheet('Charts')
            
            # CPU Usage Chart
            cpu_chart = workbook.add_chart({'type': 'line'})
            cpu_chart.add_series({
                'name': 'CPU Usage',
                'categories': '=Raw Data!$A:$A',
                'values': '=Raw Data!$B:$B'
            })
            chart_sheet.insert_chart('A1', cpu_chart)
            
            # Memory Usage Chart
            mem_chart = workbook.add_chart({'type': 'line'})
            mem_chart.add_series({
                'name': 'Memory Usage',
                'categories': '=Raw Data!$A:$A',
                'values': '=Raw Data!$C:$C'
            })
            chart_sheet.insert_chart('A16', mem_chart)
            
            writer.close()
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating Excel report: {str(e)}")
            raise
            
    def _generate_html_report(self, metrics: List[Dict], report_name: str) -> str:
        """Generate an HTML report with interactive visualizations."""
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('CPU & Memory Usage', 'Network Traffic', 'Disk Usage'),
                vertical_spacing=0.1
            )
            
            # Add CPU trace
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['cpu']['percent'] for m in metrics],
                    name='CPU Usage %'
                ),
                row=1, col=1
            )
            
            # Add Memory trace
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['memory']['percent'] for m in metrics],
                    name='Memory Usage %'
                ),
                row=1, col=1
            )
            
            # Add Network traces
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['network']['bytes_sent'] for m in metrics],
                    name='Bytes Sent'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['network']['bytes_recv'] for m in metrics],
                    name='Bytes Received'
                ),
                row=2, col=1
            )
            
            # Add Disk Usage trace
            fig.add_trace(
                go.Scatter(
                    x=[m['timestamp'] for m in metrics],
                    y=[m['disk']['percent'] for m in metrics],
                    name='Disk Usage %'
                ),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=1200,
                title_text='System Performance Metrics',
                showlegend=True
            )
            
            # Save HTML file
            output_path = f'{self.report_dir}/{report_name}.html'
            fig.write_html(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise
            
    def _create_cpu_memory_plot(self, metrics: List[Dict]) -> None:
        """Create CPU and Memory usage plot."""
        timestamps = [m['timestamp'] for m in metrics]
        cpu_usage = [m['cpu']['percent'] for m in metrics]
        memory_usage = [m['memory']['percent'] for m in metrics]
        
        plt.figure()
        plt.plot(timestamps, cpu_usage, label='CPU Usage %')
        plt.plot(timestamps, memory_usage, label='Memory Usage %')
        plt.title('CPU and Memory Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Usage %')
        plt.legend()
        plt.grid(True)
        
    def _create_network_plot(self, metrics: List[Dict]) -> None:
        """Create network traffic plot."""
        timestamps = [m['timestamp'] for m in metrics]
        bytes_sent = [m['network']['bytes_sent'] for m in metrics]
        bytes_recv = [m['network']['bytes_recv'] for m in metrics]
        
        plt.figure()
        plt.plot(timestamps, bytes_sent, label='Bytes Sent')
        plt.plot(timestamps, bytes_recv, label='Bytes Received')
        plt.title('Network Traffic Over Time')
        plt.xlabel('Time')
        plt.ylabel('Bytes')
        plt.legend()
        plt.grid(True)
        
    def _create_gpu_plot(self, metrics: List[Dict]) -> None:
        """Create GPU usage plot if GPU metrics are available."""
        timestamps = [m['timestamp'] for m in metrics]
        
        plt.figure()
        for gpu_idx in range(len(metrics[0]['gpu'])):
            memory_usage = [m['gpu'][gpu_idx]['memory_allocated'] 
                          for m in metrics if m['gpu']]
            plt.plot(timestamps, memory_usage, 
                    label=f'GPU {gpu_idx} Memory Usage')
        
        plt.title('GPU Memory Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Memory (bytes)')
        plt.legend()
        plt.grid(True)
        
    def _calculate_summary_stats(self, metrics: List[Dict]) -> Dict:
        """Calculate summary statistics from metrics data."""
        stats = {
            'Average CPU Usage (%)': np.mean([m['cpu']['percent'] for m in metrics]),
            'Max CPU Usage (%)': max(m['cpu']['percent'] for m in metrics),
            'Average Memory Usage (%)': np.mean([m['memory']['percent'] for m in metrics]),
            'Max Memory Usage (%)': max(m['memory']['percent'] for m in metrics),
            'Average Disk Usage (%)': np.mean([m['disk']['percent'] for m in metrics]),
            'Total Network Traffic (MB)': (
                sum(m['network']['bytes_sent'] + m['network']['bytes_recv'] 
                    for m in metrics) / 1024 / 1024
            )
        }
        
        # Add GPU stats if available
        if any(m.get('gpu') for m in metrics):
            for gpu_idx in range(len(metrics[0]['gpu'])):
                gpu_memory = [m['gpu'][gpu_idx]['memory_allocated'] 
                            for m in metrics if m['gpu']]
                stats[f'Average GPU {gpu_idx} Memory (MB)'] = (
                    np.mean(gpu_memory) / 1024 / 1024
                )
                stats[f'Max GPU {gpu_idx} Memory (MB)'] = (
                    max(gpu_memory) / 1024 / 1024
                )
        
        return {k: round(v, 2) if isinstance(v, float) else v 
                for k, v in stats.items()}

# Global report generator instance
report_generator = PerformanceReport()
