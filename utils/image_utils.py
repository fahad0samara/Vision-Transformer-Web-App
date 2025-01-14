import os
import cv2
import numpy as np
from PIL import Image, ExifTags
from datetime import datetime

def get_image_metadata(image_path):
    """Extract metadata from image."""
    try:
        with Image.open(image_path) as img:
            # Basic metadata
            metadata = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'created': datetime.fromtimestamp(os.path.getctime(image_path)).isoformat(),
                'modified': datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat()
            }
            
            # Try to get EXIF data
            try:
                exif = {
                    ExifTags.TAGS[k]: v
                    for k, v in img._getexif().items()
                    if k in ExifTags.TAGS
                } if img._getexif() else {}
                
                # Add relevant EXIF data
                if exif:
                    metadata.update({
                        'camera_make': exif.get('Make', 'Unknown'),
                        'camera_model': exif.get('Model', 'Unknown'),
                        'datetime_taken': exif.get('DateTimeOriginal', 'Unknown'),
                        'exposure_time': exif.get('ExposureTime', 'Unknown'),
                        'f_number': exif.get('FNumber', 'Unknown'),
                        'iso': exif.get('ISOSpeedRatings', 'Unknown'),
                        'focal_length': exif.get('FocalLength', 'Unknown')
                    })
            except:
                pass  # EXIF data not available
                
            return metadata
    except Exception as e:
        return {'error': str(e)}

def enhance_image(image_path, output_path=None):
    """Enhance image quality using PIL."""
    try:
        # Read image with PIL
        with Image.open(image_path) as img:
            # Convert to RGB mode
            img = img.convert('RGB')
            
            # Convert to numpy array for processing
            img_array = np.array(img)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge channels
            limg = cv2.merge((cl,a,b))
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            # Convert back to PIL Image
            enhanced_img = Image.fromarray(enhanced)
            
            # Save if output path provided
            if output_path:
                enhanced_img.save(output_path, quality=95)
                return output_path
            
            return image_path
            
    except Exception as e:
        raise ValueError(f"Error enhancing image: {str(e)}")

def analyze_image_quality(image_path):
    """Analyze image quality metrics."""
    try:
        # Try to open with PIL first
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(image_path)
        
        # Read with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image with OpenCV")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize metrics to 0-1 range
        metrics = {
            'brightness': float(min(max(brightness / 255.0, 0), 1)),
            'contrast': float(min(max(contrast / 128.0, 0), 1)),
            'sharpness': float(min(max(sharpness / 1000.0, 0), 1)),
            'size_bytes': os.path.getsize(image_path),
            'resolution': f"{img.shape[1]}x{img.shape[0]}"
        }

        return metrics
    except Exception as e:
        print(f"Error analyzing image quality: {e}")
        return {
            'brightness': 0,
            'contrast': 0,
            'sharpness': 0,
            'error': str(e)
        }

def get_improvement_suggestions(quality_assessment):
    """Get suggestions for improving image quality."""
    suggestions = []
    if not quality_assessment['is_bright_enough']:
        suggestions.append("Image is too dark. Try increasing exposure or brightness.")
    if not quality_assessment['has_good_contrast']:
        suggestions.append("Image has low contrast. Try adjusting contrast or using HDR.")
    if not quality_assessment['is_sharp']:
        suggestions.append("Image might be blurry. Try using a tripod or faster shutter speed.")
    return suggestions if suggestions else ["Image quality looks good!"]
