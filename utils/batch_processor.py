import os
import concurrent.futures
import threading
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import queue
import json

from utils.image_utils import enhance_image, analyze_image_quality, get_image_metadata
from utils.metrics import log_inference
from utils.cache import cached

@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    images: List[str]
    created_at: datetime
    status: str = 'pending'
    progress: int = 0
    results: List[Dict[str, Any]] = None
    error: str = None

class BatchProcessor:
    """Handles batch processing of images."""
    
    def __init__(self, max_workers=4, queue_size=100):
        self.max_workers = max_workers
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue = queue.Queue(maxsize=queue_size)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        self.results_dir = 'batch_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        try:
            # Initialize the model
            from vit_model import ViTModel
            import torch
            from torchvision import transforms
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = ViTModel().to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Warning: Could not initialize model: {e}")
            self.model = None
            self.transform = None
        
        # ... rest of the code remains the same ...
    
    def submit_job(self, images: List[str]) -> str:
        """Submit a new batch job."""
        job_id = datetime.now().strftime('%Y%m%d_%H%M%S_') + str(len(self.jobs))
        job = BatchJob(
            job_id=job_id,
            images=images,
            created_at=datetime.now()
        )
        self.jobs[job_id] = job
        self.job_queue.put(job)
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job."""
        job = self.jobs.get(job_id)
        if not job:
            return {'error': 'Job not found'}
        
        status_data = {
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'created_at': job.created_at.isoformat(),
            'error': job.error,
            'total_images': len(job.images),
            'processed_images': int(len(job.images) * job.progress / 100) if job.progress else 0
        }
        
        # Ensure results are properly formatted
        if hasattr(job, 'results') and job.results is not None:
            if isinstance(job.results, list):
                status_data['results'] = job.results
            elif isinstance(job.results, dict):
                status_data['results'] = list(job.results.values())
            else:
                status_data['results'] = []
        
        return status_data
    
    @cached(ttl=3600)
    def _process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image."""
        try:
            # Verify image exists and is readable
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            # Try to open with PIL first to verify it's a valid image
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(image_path)
            except Exception as e:
                raise ValueError(f"Invalid image file: {str(e)}")
            
            # Get metadata
            metadata = get_image_metadata(image_path)
            
            # Enhance image
            enhanced_path = enhance_image(
                image_path,
                os.path.join(self.results_dir, f"enhanced_{os.path.basename(image_path)}")
            )
            
            # Get quality metrics
            quality_info = analyze_image_quality(image_path)
            
            predictions = []
            
            # Run inference with ViT model if available
            if self.model is not None and self.transform is not None:
                try:
                    import torch
                    from imagenet_labels import IMAGENET_CLASSES
                    import torch.nn.functional as F
                    
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        # Get model output
                        output = self.model(input_tensor)
                        
                        # Get logits and apply softmax
                        logits = output['logits']
                        probabilities = F.softmax(logits, dim=1)[0]
                        
                        # Get top 5 predictions
                        top_probs, top_indices = torch.topk(probabilities, k=5)
                        
                        # Convert to predictions
                        predictions = []
                        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                            confidence = float(prob * 100)  # Convert to percentage
                            class_id = int(idx)
                            
                            predictions.append({
                                'class_id': class_id,
                                'class_name': IMAGENET_CLASSES.get(class_id, f"Unknown Object ({class_id})"),
                                'probability': min(confidence, 100.0)  # Cap at 100%
                            })
                        
                except Exception as e:
                    print(f"Warning: Model inference failed: {e}")
                    predictions = []
            
            return {
                'original_path': image_path,
                'enhanced_path': enhanced_path,
                'metadata': metadata,
                'quality_info': quality_info,
                'predictions': predictions,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return {
                'original_path': image_path,
                'error': str(e),
                'success': False
            }
    
    def _process_queue(self):
        """Process jobs from the queue."""
        while True:
            try:
                job = self.job_queue.get()
                self._process_job(job)
                self.job_queue.task_done()
            except Exception as e:
                print(f"Error processing job queue: {e}")
    
    @log_inference
    def _process_job(self, job: BatchJob):
        """Process a batch job."""
        try:
            job.status = 'processing'
            job.results = []  # Initialize results as an empty list
            total_images = len(job.images)
            
            for i, image_path in enumerate(job.images):
                try:
                    result = self._process_single_image(image_path)
                    job.results.append(result)  # Append each result to the list
                    job.progress = int((i + 1) / total_images * 100)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    job.results.append({
                        'original_path': image_path,
                        'error': str(e),
                        'success': False
                    })
            
            job.status = 'completed'
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            print(f"Job {job.job_id} failed: {e}")
    
    def _save_results(self, job: BatchJob):
        """Save job results to disk."""
        results_path = os.path.join(self.results_dir, f"{job.job_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'job_id': job.job_id,
                'created_at': job.created_at.isoformat(),
                'status': job.status,
                'total_images': len(job.images),
                'results': job.results
            }, f, indent=2)
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of active jobs."""
        return [
            self.get_job_status(job_id)
            for job_id, job in self.jobs.items()
            if job.status in ['pending', 'processing']
        ]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job."""
        job = self.jobs.get(job_id)
        if not job or job.status not in ['pending', 'processing']:
            return False
        
        job.status = 'cancelled'
        return True

# Global batch processor instance
batch_processor = BatchProcessor()
