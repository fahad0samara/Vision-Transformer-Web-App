import functools
import time
from collections import OrderedDict
import json
import os
import hashlib
import numpy as np

class LRUCache:
    """Least Recently Used (LRU) cache implementation."""
    
    def __init__(self, capacity=128):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        """Get item from cache."""
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        """Put item in cache."""
        # Convert numpy bool_ to Python bool
        if hasattr(value, 'item'):
            value = value.item()
        
        if key in self.cache:
            # Move to end
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)

class DiskCache:
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir='cache', max_size_mb=500):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key):
        """Get cache file path for key."""
        hash_key = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.json")
    
    def get(self, key):
        """Get item from cache."""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    if time.time() < data['expiry']:
                        return data['value']
                    os.remove(cache_path)
            except:
                pass
        return None
    
    def put(self, key, value, ttl=3600):
        """Put item in cache with Time-To-Live in seconds."""
        cache_path = self._get_cache_path(key)
        data = {
            'value': value,
            'expiry': time.time() + ttl
        }
        
        # Check cache size
        self._cleanup_if_needed()
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def _cleanup_if_needed(self):
        """Clean up old cache files if total size exceeds max_size."""
        total_size = 0
        files = []
        
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            size = os.path.getsize(filepath)
            total_size += size
            files.append((filepath, size, os.path.getctime(filepath)))
        
        if total_size > (self.max_size_mb * 1024 * 1024):
            # Sort by creation time (oldest first)
            files.sort(key=lambda x: x[2])
            
            # Remove old files until under limit
            for filepath, size, _ in files:
                os.remove(filepath)
                total_size -= size
                if total_size <= (self.max_size_mb * 1024 * 1024):
                    break

def cached(ttl=3600, max_size=128):
    """Decorator for caching function results."""
    memory_cache = LRUCache(max_size)
    disk_cache = DiskCache()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try memory cache first
            result = memory_cache.get(key)
            if result is not None:
                return result
            
            # Try disk cache
            result = disk_cache.get(key)
            if result is not None:
                memory_cache.put(key, result)
                return result
            
            # Calculate result
            result = func(*args, **kwargs)
            
            # Store in both caches
            memory_cache.put(key, result)
            disk_cache.put(key, result, ttl)
            
            return result
        return wrapper
    return decorator
