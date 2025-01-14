import psutil
import time
from typing import Dict, List
import threading
import queue
from prometheus_client import Counter, Gauge, Histogram
import logging

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitors system resources and model performance."""
    
    def __init__(self, interval=1):
        self.interval = interval
        self.metrics_queue = queue.Queue()
        self._running = False
        self.metrics_history = []
        self.max_history = 1000  # Keep last 1000 measurements
        self.has_gpu = False
        
        # Initialize Prometheus metrics
        self.inference_time = Histogram(
            'model_inference_time_seconds',
            'Time spent on model inference',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        self.inference_time.observe(0.0)  # Initialize with 0
        
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Current system memory usage'
        )
        self.memory_usage.set(0)  # Initialize with 0
        
        self.gpu_usage = Gauge(
            'gpu_memory_usage_bytes',
            'Current GPU memory usage',
            ['device']
        )
        self.gpu_usage.labels(device='0').set(0)  # Initialize with 0
        
        self.batch_size = Histogram(
            'batch_size',
            'Size of processed batches',
            buckets=[1, 5, 10, 20, 50, 100]
        )
        self.batch_size.observe(1)  # Initialize with 1
        
        self.processed_images = Counter(
            'processed_images_total',
            'Total number of processed images'
        )
        # Counter starts at 0 by default
        
        self.errors = Counter(
            'processing_errors_total',
            'Total number of processing errors',
            ['error_type']
        )
        # Counter starts at 0 by default
        
        # Check if GPU is available
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            if not self.has_gpu:
                logger.info("No GPU available, running in CPU mode")
        except ImportError:
            logger.info("PyTorch not available, running in CPU mode")
    
    def start(self):
        """Start the monitoring thread."""
        if not self._running:
            self._running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("System monitoring started")
    
    def stop(self):
        """Stop the monitoring thread."""
        self._running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = self.collect_metrics()
                self.metrics_queue.put(metrics)
                self.update_prometheus_metrics(metrics)
                
                # Store in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
    
    def collect_metrics(self) -> Dict:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Helper function to safely get Prometheus metric value
            def get_metric_value(metric, default=0):
                try:
                    if metric is None:
                        return default
                    if hasattr(metric, '_value'):
                        return float(metric._value._value if hasattr(metric._value, '_value') else default)
                    return default
                except:
                    return default
            
            # Helper function to safely get histogram buckets
            def get_histogram_buckets(histogram, default=None):
                try:
                    if histogram is None or not hasattr(histogram, '_buckets'):
                        return default or []
                    return [float(getattr(bucket, '_value', 0)) for bucket in histogram._buckets]
                except:
                    return default or []
            
            metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                    'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                },
                'network': self._get_network_stats(),
                'processes': len(psutil.pids()),
                'processed_images': get_metric_value(self.processed_images._value),
                'avg_inference_time': get_metric_value(getattr(self.inference_time, 'count', None)),
                'inference_history': get_histogram_buckets(self.inference_time),
                'error_count': get_metric_value(self.errors._value),
                'batch_sizes': get_histogram_buckets(self.batch_size)
            }
            
            # Add GPU metrics only if GPU is available
            if self.has_gpu:
                try:
                    gpu_metrics = self._get_gpu_metrics()
                    metrics['gpu'] = gpu_metrics if gpu_metrics else []
                except Exception as e:
                    logger.warning(f"Failed to get GPU metrics: {str(e)}")
                    metrics['gpu'] = []
            else:
                metrics['gpu'] = []
            
            return metrics
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            return {
                'timestamp': time.time(),
                'cpu': {'percent': 0, 'count': 0, 'frequency': 0},
                'memory': {'total': 0, 'available': 0, 'percent': 0, 'used': 0},
                'disk': {'total': 0, 'used': 0, 'free': 0, 'percent': 0},
                'network': {},
                'processes': 0,
                'processed_images': 0,
                'avg_inference_time': 0,
                'inference_history': [],
                'error_count': 0,
                'batch_sizes': [],
                'gpu': []
            }
    
    def _get_network_stats(self) -> Dict:
        """Get network interface statistics."""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'error_in': net_io.errin,
            'error_out': net_io.errout,
            'drop_in': net_io.dropin,
            'drop_out': net_io.dropout
        }
    
    def _get_gpu_metrics(self) -> List[Dict]:
        """Get GPU metrics if available."""
        if not self.has_gpu:
            return []
            
        import torch
        metrics = []
        for i in range(torch.cuda.device_count()):
            metrics.append({
                'device': i,
                'name': torch.cuda.get_device_name(i),
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i),
                'utilization': None  # NVIDIA-SMI would be needed for this
            })
        return metrics
    
    def update_prometheus_metrics(self, metrics: Dict):
        """Update Prometheus metrics."""
        self.memory_usage.set(metrics['memory']['used'])
        
        if metrics.get('gpu'):
            for gpu in metrics['gpu']:
                self.gpu_usage.labels(device=str(gpu['device'])).set(gpu['memory_allocated'])
    
    def record_inference(self, duration: float):
        """Record a model inference duration."""
        self.inference_time.observe(duration)
    
    def record_batch(self, size: int):
        """Record a processed batch size."""
        self.batch_size.observe(size)
        self.processed_images.inc(size)
    
    def record_error(self, error_type: str):
        """Record a processing error."""
        self.errors.labels(error_type=error_type).inc()
    
    def get_latest_metrics(self):
        """Get the most recent metrics."""
        try:
            if not self.metrics_queue.empty():
                return self.metrics_queue.get_nowait()
            return self.collect_metrics()
        except Exception as e:
            logger.error(f"Error getting latest metrics: {str(e)}")
            return None
    
    def get_metrics_history(self, timeframe: str = '1h') -> List[Dict]:
        """Get historical metrics for the specified timeframe."""
        try:
            if not self.metrics_history:
                return []
                
            current_time = time.time()
            if timeframe.endswith('h'):
                hours = int(timeframe[:-1])
                cutoff_time = current_time - (hours * 3600)
            elif timeframe.endswith('m'):
                minutes = int(timeframe[:-1])
                cutoff_time = current_time - (minutes * 60)
            else:
                raise ValueError(f"Invalid timeframe format: {timeframe}")
            
            return [m for m in self.metrics_history if m['timestamp'] >= cutoff_time]
        except Exception as e:
            logger.error(f"Error getting metrics history: {str(e)}")
            return []

# Global monitor instance
system_monitor = SystemMonitor()
system_monitor.start()
