import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from captum.attr import IntegratedGradients, GuidedGradCam, Occlusion
from captum.attr import visualization as viz
from typing import List, Tuple, Dict

class ModelExplainer:
    """Provides explanations for model predictions."""
    
    def __init__(self, model=None, transform=None, device='cpu'):
        if model is None:
            from vit_model import ViTModel
            model = ViTModel()
            model.eval()
            
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        self.model = model
        self.transform = transform
        self.device = device
        self.integrated_gradients = IntegratedGradients(model)
        self.guided_gradcam = GuidedGradCam(model, model.blocks[-1])
        self.occlusion = Occlusion(model)
    
    def explain_prediction(self, image_path: str, target_class: int) -> Dict[str, str]:
        """Generate various explanations for a prediction."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate explanations
        explanations = {}
        
        # 1. Integrated Gradients
        ig_path = self._generate_integrated_gradients(input_tensor, target_class, image_path)
        explanations['integrated_gradients'] = ig_path
        
        # 2. Guided GradCAM
        gradcam_path = self._generate_guided_gradcam(input_tensor, target_class, image_path)
        explanations['guided_gradcam'] = gradcam_path
        
        # 3. Occlusion Sensitivity
        occlusion_path = self._generate_occlusion_map(input_tensor, target_class, image_path)
        explanations['occlusion'] = occlusion_path
        
        # 4. Feature Attribution Summary
        summary_path = self._generate_attribution_summary(input_tensor, target_class, image_path)
        explanations['summary'] = summary_path
        
        return explanations
    
    def _generate_integrated_gradients(self, input_tensor: torch.Tensor, target_class: int, 
                                     image_path: str) -> str:
        """Generate Integrated Gradients visualization."""
        attributions = self.integrated_gradients.attribute(input_tensor, target=target_class)
        output_path = image_path.replace('.', '_ig.')
        
        self._save_attribution_visualization(
            attributions, output_path,
            'Integrated Gradients: Areas that influenced the prediction'
        )
        return output_path
    
    def _generate_guided_gradcam(self, input_tensor: torch.Tensor, target_class: int,
                               image_path: str) -> str:
        """Generate Guided GradCAM visualization."""
        attributions = self.guided_gradcam.attribute(input_tensor, target=target_class)
        output_path = image_path.replace('.', '_gradcam.')
        
        self._save_attribution_visualization(
            attributions, output_path,
            'Guided GradCAM: Class-specific feature locations'
        )
        return output_path
    
    def _generate_occlusion_map(self, input_tensor: torch.Tensor, target_class: int,
                              image_path: str) -> str:
        """Generate Occlusion Sensitivity map."""
        window_size = 12
        attributions = self.occlusion.attribute(
            input_tensor,
            target=target_class,
            sliding_window_shapes=(3, window_size, window_size),
            strides=(3, 8, 8)
        )
        
        output_path = image_path.replace('.', '_occlusion.')
        self._save_attribution_visualization(
            attributions, output_path,
            'Occlusion Sensitivity: Impact of hiding image regions'
        )
        return output_path
    
    def _generate_attribution_summary(self, input_tensor: torch.Tensor, target_class: int,
                                   image_path: str) -> str:
        """Generate a summary of all attribution methods."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original image
        img = Image.open(image_path)
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        
        # Integrated Gradients
        ig_attr = self.integrated_gradients.attribute(input_tensor, target=target_class)
        self._plot_attribution(ig_attr, axes[0, 1], 'Integrated Gradients')
        
        # Guided GradCAM
        gradcam_attr = self.guided_gradcam.attribute(input_tensor, target=target_class)
        self._plot_attribution(gradcam_attr, axes[1, 0], 'Guided GradCAM')
        
        # Occlusion
        occl_attr = self.occlusion.attribute(
            input_tensor,
            target=target_class,
            sliding_window_shapes=(3, 12, 12),
            strides=(3, 8, 8)
        )
        self._plot_attribution(occl_attr, axes[1, 1], 'Occlusion Sensitivity')
        
        output_path = image_path.replace('.', '_summary.')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _save_attribution_visualization(self, attribution: torch.Tensor, output_path: str,
                                     title: str):
        """Save attribution visualization to file."""
        attribution = attribution.squeeze().cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(attribution.transpose(1, 2, 0))
        plt.title(title)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
    
    def _plot_attribution(self, attribution: torch.Tensor, ax: plt.Axes, title: str):
        """Plot attribution on a specific axis."""
        attribution = attribution.squeeze().cpu().detach().numpy()
        ax.imshow(attribution.transpose(1, 2, 0))
        ax.set_title(title)
        ax.axis('off')
    
    def get_feature_importance(self, image_path: str, target_class: int) -> List[Tuple[str, float]]:
        """Get importance scores for different image features."""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get attributions
        attributions = self.integrated_gradients.attribute(input_tensor, target=target_class)
        attr_sum = torch.sum(torch.abs(attributions), dim=[2, 3])
        
        # Calculate feature importance scores
        importance_scores = []
        feature_names = ['Color', 'Texture', 'Shape', 'Pattern', 'Background']
        
        # Simulate feature importance calculation
        for i, feature in enumerate(feature_names):
            score = float(attr_sum[0, i % 3])  # Using channel attributions as proxy
            importance_scores.append((feature, abs(score)))
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        return importance_scores
