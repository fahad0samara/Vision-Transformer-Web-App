import cv2
import numpy as np
import albumentations as A
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Dict, Any, Tuple
import os

class ImageAugmentor:
    """Provides advanced image augmentation capabilities."""
    
    def __init__(self, output_dir='augmented'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define augmentation pipelines
        self.basic_aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        
        self.weather_aug = A.Compose([
            A.RandomRain(p=0.3),
            A.RandomSnow(p=0.3),
            A.RandomFog(p=0.3),
            A.RandomSunFlare(p=0.3),
            A.RandomShadow(p=0.3),
        ])
        
        self.style_aug = A.Compose([
            A.ToSepia(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.RGBShift(p=0.3),
        ])
    
    def augment_image(self, image_path: str, augmentation_type: str = 'basic', 
                     num_variants: int = 5) -> List[str]:
        """Generate augmented variants of an image."""
        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Select augmentation pipeline
        if augmentation_type == 'weather':
            pipeline = self.weather_aug
        elif augmentation_type == 'style':
            pipeline = self.style_aug
        else:
            pipeline = self.basic_aug
        
        # Generate variants
        output_paths = []
        basename = os.path.splitext(os.path.basename(image_path))[0]
        
        for i in range(num_variants):
            augmented = pipeline(image=image)['image']
            output_path = os.path.join(
                self.output_dir, 
                f"{basename}_{augmentation_type}_{i}.jpg"
            )
            
            # Save augmented image
            cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            output_paths.append(output_path)
        
        return output_paths
    
    def generate_adversarial_examples(self, image_path: str, model: torch.nn.Module,
                                    epsilon: float = 0.01) -> Tuple[str, str]:
        """Generate adversarial examples using FGSM attack."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        # Create adversarial example
        input_tensor.requires_grad = True
        output = model(input_tensor)
        
        # Get the index of the max log-probability
        target = output.argmax(1)
        
        # Calculate loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Zero gradients
        model.zero_grad()
        
        # Calculate gradients
        loss.backward()
        
        # Create adversarial example
        data_grad = input_tensor.grad.data
        perturbed_image = input_tensor + epsilon * data_grad.sign()
        
        # Save original and perturbed images
        original_path = os.path.join(self.output_dir, 'original.jpg')
        perturbed_path = os.path.join(self.output_dir, 'adversarial.jpg')
        
        self._save_tensor_as_image(input_tensor.squeeze(), original_path)
        self._save_tensor_as_image(perturbed_image.squeeze(), perturbed_path)
        
        return original_path, perturbed_path
    
    def _save_tensor_as_image(self, tensor: torch.Tensor, path: str):
        """Save a tensor as an image."""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
        
        # Convert to PIL image and save
        image = transforms.ToPILImage()(tensor)
        image.save(path)
    
    def create_style_variants(self, image_path: str, styles: List[str] = None) -> Dict[str, str]:
        """Create different style variants of an image."""
        if styles is None:
            styles = ['vintage', 'dramatic', 'cyberpunk', 'noir']
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        style_paths = {}
        basename = os.path.splitext(os.path.basename(image_path))[0]
        
        for style in styles:
            if style == 'vintage':
                styled = self._apply_vintage_effect(image)
            elif style == 'dramatic':
                styled = self._apply_dramatic_effect(image)
            elif style == 'cyberpunk':
                styled = self._apply_cyberpunk_effect(image)
            elif style == 'noir':
                styled = self._apply_noir_effect(image)
            
            output_path = os.path.join(self.output_dir, f"{basename}_{style}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(styled, cv2.COLOR_RGB2BGR))
            style_paths[style] = output_path
        
        return style_paths
    
    def _apply_vintage_effect(self, image: np.ndarray) -> np.ndarray:
        """Apply vintage photo effect."""
        # Convert to float
        image_float = image.astype(float) / 255.0
        
        # Sepia effect
        sepia = np.array([[0.393, 0.769, 0.189],
                         [0.349, 0.686, 0.168],
                         [0.272, 0.534, 0.131]])
        sepia_image = image_float @ sepia.T
        sepia_image = np.clip(sepia_image, 0, 1)
        
        # Add vignette
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        
        # Apply vignette and convert back
        vintage = sepia_image * mask[..., np.newaxis]
        return (vintage * 255).astype(np.uint8)
    
    def _apply_dramatic_effect(self, image: np.ndarray) -> np.ndarray:
        """Apply dramatic photo effect."""
        # Increase contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        dramatic = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Add dark edges
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (50,50), (image.shape[1]-50, image.shape[0]-50), 255, -1)
        mask = cv2.GaussianBlur(mask, (21,21), 11)
        dramatic = dramatic * (mask[..., np.newaxis]/255)
        
        return dramatic.astype(np.uint8)
    
    def _apply_cyberpunk_effect(self, image: np.ndarray) -> np.ndarray:
        """Apply cyberpunk style effect."""
        # Split channels and enhance blue/pink
        b, g, r = cv2.split(image)
        b = cv2.addWeighted(b, 1.5, np.zeros_like(b), 0, 0)
        r = cv2.addWeighted(r, 1.5, np.zeros_like(r), 0, 0)
        
        # Merge channels with enhanced contrast
        cyberpunk = cv2.merge([b, g, r])
        cyberpunk = cv2.addWeighted(cyberpunk, 1.5, np.zeros_like(cyberpunk), 0, 0)
        
        return np.clip(cyberpunk, 0, 255).astype(np.uint8)
    
    def _apply_noir_effect(self, image: np.ndarray) -> np.ndarray:
        """Apply film noir effect."""
        # Convert to grayscale with high contrast
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        noir = clahe.apply(gray)
        
        # Add grain
        noise = np.random.normal(0, 15, noir.shape).astype(np.uint8)
        noir = cv2.add(noir, noise)
        
        # Convert back to RGB
        return cv2.cvtColor(noir, cv2.COLOR_GRAY2RGB)
