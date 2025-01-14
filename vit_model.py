import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import Dict, List, Tuple
import json

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.norm(x)
        
        # Use CLS token for classification
        cls_token_output = x[:, 0]
        logits = self.head(cls_token_output)
        
        # Calculate confidence scores using softmax
        confidence_scores = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'confidence_scores': confidence_scores,
            'features': cls_token_output
        }

    def analyze_image(self, image: torch.Tensor) -> Dict:
        """
        Analyze an input image and return structured results.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            Dict: Structured output containing classification results and confidence scores
        """
        with torch.no_grad():
            output = self.forward(image)
            
            # Get top 5 predictions and their confidence scores
            confidence_scores, predictions = torch.topk(output['confidence_scores'], k=5)
            
            results = {
                'classification_results': [
                    {
                        'class_id': pred.item(),
                        'confidence': score.item()
                    }
                    for pred, score in zip(predictions[0], confidence_scores[0])
                ],
                'metadata': {
                    'model_version': '1.0',
                    'timestamp': '2025-01-14T09:45:41+02:00',
                    'image_size': f"{image.shape[2]}x{image.shape[3]}"
                }
            }
            
            return results

# Create an alias for VisionTransformer as ViTModel for backwards compatibility
ViTModel = VisionTransformer

def create_vit_model(
    image_size: int = 224,
    num_classes: int = 1000,
    pretrained: bool = False
) -> VisionTransformer:
    """
    Create a Vision Transformer model with specified parameters.
    
    Args:
        image_size (int): Input image size
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load pretrained weights
        
    Returns:
        VisionTransformer: Initialized model
    """
    model = VisionTransformer(
        image_size=image_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    )
    
    if pretrained:
        try:
            state_dict = torch.load('vit_weights.pth')
            model.load_state_dict(state_dict)
        except:
            print("Warning: Could not load pretrained weights")
    
    return model
