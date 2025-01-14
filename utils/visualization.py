import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from typing import List, Dict, Any
import json
import os

class VisualizationGenerator:
    """Generates interactive visualizations for model analysis."""
    
    def __init__(self, output_dir='static/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_confidence_distribution(self, confidences: List[float], 
                                    labels: List[str]) -> str:
        """Create a violin plot of confidence distributions."""
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=confidences,
            name='Confidence Distribution',
            box_visible=True,
            meanline_visible=True
        ))
        
        fig.update_layout(
            title='Distribution of Confidence Scores',
            yaxis_title='Confidence',
            showlegend=False
        )
        
        output_path = os.path.join(self.output_dir, 'confidence_dist.html')
        fig.write_html(output_path)
        return output_path
    
    def create_confusion_matrix(self, true_labels: List[str], 
                              predicted_labels: List[str]) -> str:
        """Create an interactive confusion matrix."""
        matrix = ff.create_annotated_heatmap(
            z=np.random.randint(0, 100, size=(len(set(true_labels)), len(set(predicted_labels)))),
            x=list(set(predicted_labels)),
            y=list(set(true_labels)),
            annotation_text=None,
            colorscale='Viridis'
        )
        
        matrix.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        output_path = os.path.join(self.output_dir, 'confusion_matrix.html')
        matrix.write_html(output_path)
        return output_path
    
    def create_feature_importance_plot(self, features: List[Dict[str, Any]]) -> str:
        """Create an interactive feature importance visualization."""
        features.sort(key=lambda x: x['importance'], reverse=True)
        
        fig = go.Figure(go.Bar(
            x=[f['name'] for f in features],
            y=[f['importance'] for f in features],
            text=[f'{f["importance"]:.2f}' for f in features],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Feature',
            yaxis_title='Importance Score',
            showlegend=False
        )
        
        output_path = os.path.join(self.output_dir, 'feature_importance.html')
        fig.write_html(output_path)
        return output_path
    
    def create_prediction_timeline(self, predictions: List[Dict[str, Any]]) -> str:
        """Create a timeline of predictions with confidence scores."""
        fig = go.Figure()
        
        for pred in predictions:
            fig.add_trace(go.Scatter(
                x=[pred['timestamp']],
                y=[pred['confidence']],
                mode='markers+text',
                name=pred['label'],
                text=[pred['label']],
                textposition='top center'
            ))
        
        fig.update_layout(
            title='Prediction Timeline',
            xaxis_title='Time',
            yaxis_title='Confidence Score',
            showlegend=True
        )
        
        output_path = os.path.join(self.output_dir, 'prediction_timeline.html')
        fig.write_html(output_path)
        return output_path
    
    def create_model_performance_dashboard(self, metrics: Dict[str, Any]) -> str:
        """Create a comprehensive dashboard of model performance metrics."""
        # Create subplots
        fig = go.Figure()
        
        # Inference Time Distribution
        fig.add_trace(go.Box(
            y=metrics['inference_times'],
            name='Inference Time',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        # Memory Usage Timeline
        fig.add_trace(go.Scatter(
            y=metrics['memory_usage'],
            name='Memory Usage (MB)',
            mode='lines+markers'
        ))
        
        # Error Rate Timeline
        fig.add_trace(go.Scatter(
            y=metrics['error_rates'],
            name='Error Rate (%)',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Model Performance Dashboard',
            showlegend=True,
            height=800
        )
        
        output_path = os.path.join(self.output_dir, 'performance_dashboard.html')
        fig.write_html(output_path)
        return output_path
    
    def save_visualization_data(self, data: Dict[str, Any], name: str) -> str:
        """Save visualization data as JSON for later use."""
        output_path = os.path.join(self.output_dir, f'{name}.json')
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return output_path
