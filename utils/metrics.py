import time
import psutil
import torch
from collections import deque
import threading
import logging
from datetime import datetime

class ModelMetrics:
    def __init__(self, window_size=100):
        self.inference_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.gpu_usage = deque(maxlen=window_size) if torch.cuda.is_available() else None
        self.request_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        
    def record_inference(self, start_time, success=True):
        """Record inference time and update metrics."""
        with self.lock:
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.request_count += 1
            if not success:
                self.error_count += 1
            
            # Record system metrics
            self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
            if self.gpu_usage is not None and torch.cuda.is_available():
                self.gpu_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
    
    def get_metrics(self):
        """Get current metrics."""
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            metrics = {
                'inference_stats': {
                    'avg_time': sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0,
                    'min_time': min(self.inference_times) if self.inference_times else 0,
                    'max_time': max(self.inference_times) if self.inference_times else 0,
                    'total_requests': self.request_count,
                    'error_rate': (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
                },
                'system_stats': {
                    'memory_usage_mb': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                    'uptime_seconds': uptime,
                    'requests_per_second': self.request_count / uptime if uptime > 0 else 0
                }
            }
            
            if self.gpu_usage is not None:
                metrics['system_stats']['gpu_usage_mb'] = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0
            
            return metrics

class MetricsLogger:
    def __init__(self, log_file='logs/metrics.log'):
        self.logger = logging.getLogger('metrics')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
    
    def log_metrics(self, metrics):
        """Log current metrics."""
        self.logger.info(f"Inference Stats: {metrics['inference_stats']}")
        self.logger.info(f"System Stats: {metrics['system_stats']}")

# Global metrics instance
metrics = ModelMetrics()
metrics_logger = MetricsLogger()

def log_inference(func):
    """Decorator to log inference metrics."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            metrics.record_inference(start_time, success=True)
            return result
        except Exception as e:
            metrics.record_inference(start_time, success=False)
            raise e
    return wrapper
