import torch
from PIL import Image
from torchvision import transforms
from typing import Dict, Union, List
import numpy as np
from vit_model import create_vit_model
from imagenet_labels import get_class_label, IMAGENET_CLASSES

class ImageProcessor:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_vit_model(pretrained=True).to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for the ViT model.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)

    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze an image and return structured results.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Dict: Analysis results including classifications and metadata
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Get model predictions
        with torch.no_grad():
            results = self.model.forward(image_tensor)
            
            # Get top 5 predictions and their confidence scores
            confidence_scores, predictions = torch.topk(results['confidence_scores'], k=5)
            
            # Get image size
            with Image.open(image_path) as img:
                original_size = f"{img.size[0]}x{img.size[1]}"
            
            # Convert to Python types and add class labels
            classification_results = [
                {
                    'class_id': int(pred),
                    'class_name': get_class_label(int(pred)),
                    'confidence': float(score),
                    'confidence_formatted': f"{float(score) * 100:.2f}%"
                }
                for pred, score in zip(predictions[0], confidence_scores[0])
            ]
            
            return {
                'classification_results': classification_results,
                'metadata': {
                    'model_version': '1.0',
                    'model_type': 'Vision Transformer (ViT)',
                    'original_image_size': original_size,
                    'processed_image_size': '224x224',
                    'device': str(self.device),
                    'framework': f'PyTorch {torch.__version__}',
                    'total_classes': len(IMAGENET_CLASSES)
                }
            }

    def batch_analyze_images(self, image_paths: List[str]) -> List[Dict]:
        """
        Analyze multiple images in batch.
        
        Args:
            image_paths (List[str]): List of image paths
            
        Returns:
            List[Dict]: List of analysis results for each image
        """
        results = []
        for image_path in image_paths:
            result = self.analyze_image(image_path)
            results.append(result)
        return results
