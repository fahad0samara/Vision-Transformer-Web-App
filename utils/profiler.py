import cProfile
import pstats
import io
import time
import torch
import numpy as np
from typing import Dict, List, Callable, Any
from functools import wraps
import logging
from utils.monitoring import system_monitor

logger = logging.getLogger(__name__)

class ModelProfiler:
    """Profiles model performance and resource usage."""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats = []
        self.current_profile = None
    
    def start_profile(self):
        """Start profiling."""
        self.profiler.enable()
        self.current_profile = {
            'start_time': time.time(),
            'memory_start': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
    
    def stop_profile(self) -> Dict:
        """Stop profiling and return statistics."""
        self.profiler.disable()
        end_time = time.time()
        
        # Get profiler stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Calculate memory usage
        memory_end = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_diff = memory_end - self.current_profile['start_time']
        
        # Compile stats
        stats = {
            'duration': end_time - self.current_profile['start_time'],
            'memory_used': memory_diff,
            'profile_output': s.getvalue()
        }
        
        self.stats.append(stats)
        self.current_profile = None
        return stats
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start_profile()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                stats = self.stop_profile()
                logger.info(f"Function {func.__name__} profiling stats:\n"
                          f"Duration: {stats['duration']:.2f}s\n"
                          f"Memory used: {stats['memory_used']/1024/1024:.2f}MB")
        return wrapper
    
    def profile_batch(self, batch_size: int, func: Callable, *args, **kwargs) -> Dict:
        """Profile function performance with different batch sizes."""
        results = []
        
        for size in range(1, batch_size + 1):
            self.start_profile()
            try:
                func(*args, **kwargs)
                stats = self.stop_profile()
                results.append({
                    'batch_size': size,
                    'duration': stats['duration'],
                    'memory_used': stats['memory_used']
                })
            except Exception as e:
                logger.error(f"Error profiling batch size {size}: {str(e)}")
        
        return self._analyze_batch_results(results)
    
    def _analyze_batch_results(self, results: List[Dict]) -> Dict:
        """Analyze batch profiling results."""
        durations = [r['duration'] for r in results]
        memory_usage = [r['memory_used'] for r in results]
        batch_sizes = [r['batch_size'] for r in results]
        
        return {
            'optimal_batch_size': self._find_optimal_batch_size(results),
            'statistics': {
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'mean_memory': np.mean(memory_usage),
                'std_memory': np.std(memory_usage)
            },
            'per_batch': results
        }
    
    def _find_optimal_batch_size(self, results: List[Dict]) -> int:
        """Find optimal batch size based on throughput and memory usage."""
        throughputs = []
        for r in results:
            images_per_second = r['batch_size'] / r['duration']
            memory_per_image = r['memory_used'] / r['batch_size']
            # Score combines throughput and memory efficiency
            score = images_per_second / (1 + np.log1p(memory_per_image))
            throughputs.append((r['batch_size'], score))
        
        # Return batch size with highest score
        return max(throughputs, key=lambda x: x[1])[0]
    
    def export_profile(self, filepath: str):
        """Export profiling data to file."""
        stats = pstats.Stats(self.profiler)
        stats.dump_stats(filepath)
    
    def memory_profile(self, func: Callable) -> Callable:
        """Decorator to profile memory usage."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            
            try:
                result = func(*args, **kwargs)
                end_mem = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()
                
                logger.info(f"Memory Profile - {func.__name__}:\n"
                          f"Peak memory: {peak_mem/1024/1024:.2f}MB\n"
                          f"Memory change: {(end_mem-start_mem)/1024/1024:.2f}MB")
                return result
            except Exception as e:
                logger.error(f"Error in memory profile: {str(e)}")
                raise
        return wrapper

# Global profiler instance
model_profiler = ModelProfiler()

def profile_inference(func: Callable) -> Callable:
    """Decorator to profile model inference."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            system_monitor.record_inference(duration)
            return result
        except Exception as e:
            system_monitor.record_error('inference_error')
            raise
    return wrapper
