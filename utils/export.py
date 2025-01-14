import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import plotly.io as pio
from fpdf import FPDF
import xlsxwriter
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    """Generates comprehensive reports in various formats."""
    
    def __init__(self, output_dir='exports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different export types
        self.dirs = {
            'excel': os.path.join(output_dir, 'excel'),
            'pdf': os.path.join(output_dir, 'pdf'),
            'json': os.path.join(output_dir, 'json'),
            'csv': os.path.join(output_dir, 'csv'),
            'plots': os.path.join(output_dir, 'plots')
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def generate_excel_report(self, data: Dict[str, Any], job_id: str) -> str:
        """Generate detailed Excel report with multiple sheets."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'report_{job_id}_{timestamp}.xlsx'
        filepath = os.path.join(self.dirs['excel'], filename)
        
        # Create Excel writer
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        workbook = writer.book
        
        # Summary sheet
        summary_data = {
            'Job ID': [job_id],
            'Total Images': [len(data['results'])],
            'Success Rate': [self._calculate_success_rate(data['results'])],
            'Average Confidence': [self._calculate_avg_confidence(data['results'])],
            'Processing Time': [data.get('processing_time', 'N/A')]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Detailed results
        results_df = pd.DataFrame(data['results'])
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Create charts
        chart_sheet = workbook.add_worksheet('Charts')
        
        # Confidence distribution chart
        confidences = [r['confidence'] for r in data['results'] if 'confidence' in r]
        self._create_excel_chart(workbook, chart_sheet, 'Confidence Distribution', 
                               'A1', confidences)
        
        # Class distribution chart
        classes = [r['predicted_class'] for r in data['results'] if 'predicted_class' in r]
        class_dist = pd.Series(classes).value_counts()
        self._create_excel_chart(workbook, chart_sheet, 'Class Distribution', 
                               'A15', class_dist.values.tolist(), 
                               categories=class_dist.index.tolist())
        
        writer.close()
        return filepath
    
    def generate_pdf_report(self, data: Dict[str, Any], job_id: str) -> str:
        """Generate PDF report with visualizations."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'report_{job_id}_{timestamp}.pdf'
        filepath = os.path.join(self.dirs['pdf'], filename)
        
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'Analysis Report - Job {job_id}', ln=True, align='C')
        
        # Summary section
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Summary', ln=True)
        pdf.set_font('Arial', '', 10)
        
        summary_items = [
            f"Total Images: {len(data['results'])}",
            f"Success Rate: {self._calculate_success_rate(data['results'])}%",
            f"Average Confidence: {self._calculate_avg_confidence(data['results']):.2f}%",
            f"Processing Time: {data.get('processing_time', 'N/A')}"
        ]
        
        for item in summary_items:
            pdf.cell(0, 8, item, ln=True)
        
        # Add visualizations
        plot_paths = self._generate_report_plots(data, job_id)
        for plot_path in plot_paths:
            pdf.add_page()
            pdf.image(plot_path, x=10, y=10, w=190)
        
        pdf.output(filepath)
        return filepath
    
    def export_to_json(self, data: Dict[str, Any], job_id: str) -> str:
        """Export results to JSON format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results_{job_id}_{timestamp}.json'
        filepath = os.path.join(self.dirs['json'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def export_to_csv(self, data: Dict[str, Any], job_id: str) -> str:
        """Export results to CSV format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results_{job_id}_{timestamp}.csv'
        filepath = os.path.join(self.dirs['csv'], filename)
        
        df = pd.DataFrame(data['results'])
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def _calculate_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate success rate from results."""
        if not results:
            return 0.0
        successful = sum(1 for r in results if r.get('success', False))
        return (successful / len(results)) * 100
    
    def _calculate_avg_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average confidence from results."""
        confidences = [r.get('confidence', 0) for r in results]
        return np.mean(confidences) if confidences else 0.0
    
    def _create_excel_chart(self, workbook: xlsxwriter.Workbook, 
                          worksheet: xlsxwriter.Worksheet, 
                          title: str, cell: str, 
                          data: List[float], 
                          categories: Optional[List[str]] = None):
        """Create a chart in Excel worksheet."""
        chart = workbook.add_chart({'type': 'column'})
        
        # Write data
        row = int(cell[1:])
        col = ord(cell[0]) - ord('A')
        
        worksheet.write_row(row, col, data)
        if categories:
            worksheet.write_column(row, col-1, categories)
        
        # Configure chart
        chart.add_series({
            'values': f'=Charts!${chr(col+ord("A"))}${row+1}:${chr(col+ord("A"))}${row+len(data)}',
            'categories': f'=Charts!${chr(col+ord("A")-1)}${row+1}:${chr(col+ord("A")-1)}${row+len(data)}' if categories else None,
            'name': title
        })
        
        chart.set_title({'name': title})
        worksheet.insert_chart(row + len(data) + 2, col, chart)
    
    def _generate_report_plots(self, data: Dict[str, Any], job_id: str) -> List[str]:
        """Generate plots for PDF report."""
        plot_paths = []
        
        # Confidence distribution
        plt.figure(figsize=(10, 6))
        confidences = [r.get('confidence', 0) for r in data['results']]
        sns.histplot(confidences, bins=20)
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        
        plot_path = os.path.join(self.dirs['plots'], f'confidence_dist_{job_id}.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        
        # Class distribution
        plt.figure(figsize=(12, 6))
        classes = [r.get('predicted_class', 'Unknown') for r in data['results']]
        sns.countplot(y=classes)
        plt.title('Class Distribution')
        plt.xlabel('Count')
        plt.ylabel('Class')
        
        plot_path = os.path.join(self.dirs['plots'], f'class_dist_{job_id}.png')
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        
        return plot_paths
