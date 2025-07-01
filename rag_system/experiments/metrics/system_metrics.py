# import time
# import psutil
# import logging
# from typing import Dict, Any, Optional
# import threading
# import os

# logger = logging.getLogger(__name__)

# class SystemMetrics:
#     """Calculates system-level performance metrics for the RAG system.
    
#     Tracks metrics like memory usage, CPU utilization, processing times,
#     and throughput metrics across the entire pipeline.
#     """
    
#     def __init__(self):
#         """Initialize the SystemMetrics calculator."""
#         self.process = psutil.Process(os.getpid())
#         self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
#         self.lock = threading.Lock()
#         logger.info("SystemMetrics initialized")
    
#     def calculate_metrics(self, 
#                          total_time: float,
#                          retrieval_time: float,
#                          generation_time: float,
#                          web_time: float = 0.0,
#                          response_length: Optional[int] = None) -> Dict[str, float]:
#         """Calculate comprehensive system metrics.
        
#         Args:
#             total_time: Total pipeline execution time in seconds
#             retrieval_time: Time spent on retrieval in seconds
#             generation_time: Time spent on text generation in seconds
#             web_time: Time spent on web search in seconds
#             response_length: Length of generated response in tokens/characters
            
#         Returns:
#             Dictionary containing calculated system metrics
#         """
#         with self.lock:
#             try:
#                 metrics = {}
                
#                 # Basic timing metrics
#                 metrics['total_pipeline_time'] = total_time
#                 metrics['retrieval_time_ratio'] = retrieval_time / total_time if total_time > 0 else 0
#                 metrics['generation_time_ratio'] = generation_time / total_time if total_time > 0 else 0
#                 metrics['web_time_ratio'] = web_time / total_time if total_time > 0 else 0
                
#                 # Memory metrics
#                 current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
#                 metrics['memory_usage'] = current_memory
#                 metrics['memory_increase'] = current_memory - self.initial_memory
                
#                 try:
#                     memory_percent = self.process.memory_percent()
#                     metrics['memory_percent'] = memory_percent
#                 except psutil.AccessDenied:
#                     logger.warning("Cannot access memory percentage")
#                     metrics['memory_percent'] = 0.0
                
#                 # CPU metrics
#                 try:
#                     cpu_percent = self.process.cpu_percent()
#                     metrics['cpu_utilization'] = cpu_percent
#                 except psutil.AccessDenied:
#                     logger.warning("Cannot access CPU percentage")
#                     metrics['cpu_utilization'] = 0.0
                
#                 # System-wide metrics
#                 try:
#                     system_memory = psutil.virtual_memory()
#                     metrics['system_memory_used_percent'] = system_memory.percent
#                     metrics['system_memory_available_mb'] = system_memory.available / 1024 / 1024
                    
#                     system_cpu = psutil.cpu_percent(interval=0.1)
#                     metrics['system_cpu_percent'] = system_cpu
#                 except Exception as e:
#                     logger.warning(f"Cannot access system-wide metrics: {e}")
                
#                 # GPU utilization (if available)
#                 metrics['gpu_utilization'] = self._get_gpu_utilization()
                
#                 # Throughput metrics
#                 if response_length and total_time > 0:
#                     metrics['tokens_per_second'] = response_length / total_time
#                     metrics['processing_efficiency'] = response_length / (total_time * 100)  # tokens per 100ms
#                 else:
#                     metrics['tokens_per_second'] = 0.0
#                     metrics['processing_efficiency'] = 0.0
                
#                 # Pipeline efficiency metrics
#                 processing_time = retrieval_time + generation_time + web_time
#                 overhead_time = total_time - processing_time
#                 metrics['pipeline_overhead'] = overhead_time if overhead_time > 0 else 0.0
#                 metrics['pipeline_efficiency'] = processing_time / total_time if total_time > 0 else 0.0
                
#                 # Cache metrics (placeholder - would need actual cache implementation)
#                 metrics['cache_hit_rate'] = self._calculate_cache_hit_rate()
                
#                 # Resource utilization score (composite metric)
#                 metrics['resource_utilization_score'] = self._calculate_resource_score(metrics)
                
#                 logger.debug(f"Calculated system metrics: {list(metrics.keys())}")
#                 return metrics
                
#             except Exception as e:
#                 logger.error(f"Error calculating system metrics: {e}")
#                 return {
#                     'total_pipeline_time': total_time,
#                     'error_in_system_metrics': 1.0
#                 }
    
#     def _get_gpu_utilization(self) -> float:
#         """Get GPU utilization if available.
        
#         Returns:
#             GPU utilization percentage (0-100) or 0 if not available
#         """
#         try:
#             import GPUtil
#             gpus = GPUtil.getGPUs()
#             if gpus:
#                 return gpus[0].load * 100  # Convert to percentage
#         except ImportError:
#             pass
#         except Exception as e:
#             logger.debug(f"Cannot get GPU utilization: {e}")
        
#         # Alternative method using nvidia-ml-py
#         try:
#             import pynvml
#             pynvml.nvmlInit()
#             handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#             utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
#             return float(utilization.gpu)
#         except ImportError:
#             pass
#         except Exception as e:
#             logger.debug(f"Cannot get GPU utilization via pynvml: {e}")
        
#         return 0.0  # No GPU available or accessible
    
#     def _calculate_cache_hit_rate(self) -> float:
#         """Calculate cache hit rate.
        
#         This is a placeholder implementation. In a real system, you would
#         track actual cache hits and misses.
        
#         Returns:
#             Cache hit rate as a percentage (0-100)
#         """
#         # Placeholder implementation
#         # In a real system, you would track:
#         # - Number of cache hits
#         # - Number of cache misses
#         # - Return hits / (hits + misses) * 100
#         return 0.0
    
#     def _calculate_resource_score(self, metrics: Dict[str, float]) -> float:
#         """Calculate a composite resource utilization score.
        
#         Args:
#             metrics: Dictionary of calculated metrics
            
#         Returns:
#             Resource utilization score (0-100, lower is better)
#         """
#         try:
#             # Weighted combination of different resource metrics
#             memory_weight = 0.4
#             cpu_weight = 0.3
#             time_weight = 0.2
#             gpu_weight = 0.1
            
#             memory_score = min(metrics.get('memory_percent', 0), 100)
#             cpu_score = min(metrics.get('cpu_utilization', 0), 100)
#             time_score = min(metrics.get('total_pipeline_time', 0) * 10, 100)  # Normalize time
#             gpu_score = min(metrics.get('gpu_utilization', 0), 100)
            
#             composite_score = (
#                 memory_score * memory_weight +
#                 cpu_score * cpu_weight +
#                 time_score * time_weight +
#                 gpu_score * gpu_weight
#             )
            
#             return composite_score
            
#         except Exception as e:
#             logger.warning(f"Error calculating resource score: {e}")
#             return 0.0
    
#     def get_system_info(self) -> Dict[str, Any]:
#         """Get static system information.
        
#         Returns:
#             Dictionary containing system information
#         """
#         try:
#             info = {}
            
#             # CPU information
#             info['cpu_count'] = psutil.cpu_count()
#             info['cpu_count_logical'] = psutil.cpu_count(logical=True)
            
#             # Memory information
#             memory = psutil.virtual_memory()
#             info['total_memory_gb'] = memory.total / (1024**3)
            
#             # Disk information
#             disk = psutil.disk_usage('/')
#             info['total_disk_gb'] = disk.total / (1024**3)
#             info['free_disk_gb'] = disk.free / (1024**3)
            
#             # Process information
#             info['process_id'] = self.process.pid
#             info['process_name'] = self.process.name()
            
#             return info
            
#         except Exception as e:
#             logger.error(f"Error getting system info: {e}")
#             return {}
    
#     def reset_baseline(self):
#         """Reset the memory baseline for relative measurements."""
#         try:
#             self.initial_memory = self.process.memory_info().rss / 1024 / 1024
#             logger.info(f"Memory baseline reset to {self.initial_memory:.2f} MB")
#         except Exception as e:
#             logger.error(f"Error resetting baseline: {e}")
    
#     def log_system_status(self):
#         """Log current system status for debugging."""
#         try:
#             memory_info = self.process.memory_info()
#             logger.info(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
#             logger.info(f"CPU percent: {self.process.cpu_percent():.2f}%")
            
#             system_memory = psutil.virtual_memory()
#             logger.info(f"System memory: {system_memory.percent:.2f}% used")
            
#         except Exception as e:
#             logger.error(f"Error logging system status: {e}")

# # Example usage:
# # system_metrics = SystemMetrics()
# # metrics = system_metrics.calculate_metrics(
# #     total_time=2.5,
# #     retrieval_time=0.8,
# #     generation_time=1.2,
# #     web_time=0.3,
# #     response_length=150
# # )







# import time
# import psutil
# import logging
# from typing import Dict, Any, Optional
# import threading
# import os

# logger = logging.getLogger(__name__)

# class SystemMetrics:
#     """Calculates system-level performance metrics for the RAG system.
    
#     Tracks memory usage, CPU utilization, and processing times.
#     """
    
#     def __init__(self):
#         """Initialize the SystemMetrics calculator."""
#         self.process = psutil.Process(os.getpid())
#         self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
#         self.lock = threading.Lock()
#         logger.info("SystemMetrics initialized")
    
#     def calculate_metrics(self, 
#                          total_time: float,
#                          retrieval_time: float,
#                          generation_time: float,
#                          web_time: float = 0.0,
#                          response_length: Optional[int] = None) -> Dict[str, float]:
#         """Calculate system metrics.
        
#         Args:
#             total_time: Total pipeline execution time in seconds
#             retrieval_time: Time spent on retrieval in seconds
#             generation_time: Time spent on text generation in seconds
#             web_time: Time spent on web search in seconds
#             response_length: Length of generated response in tokens/characters
            
#         Returns:
#             Dictionary containing calculated system metrics
#         """
#         with self.lock:
#             try:
#                 metrics = {
#                     'total_pipeline_time': total_time,
#                     'retrieval_time': retrieval_time,
#                     'generation_time': generation_time,
#                     'web_time': web_time,
#                     'memory_usage_mb': self.process.memory_info().rss / 1024 / 1024,
#                     'memory_increase_mb': (self.process.memory_info().rss / 1024 / 1024) - self.initial_memory,
#                     'cpu_utilization_percent': self.process.cpu_percent(),
#                     'system_memory_used_percent': psutil.virtual_memory().percent,
#                     'gpu_utilization_percent': self._get_gpu_utilization(),
#                     'tokens_per_second': (response_length / total_time if total_time > 0 else 0.0) if response_length else 0.0,
#                     'cache_hit_rate': self._calculate_cache_hit_rate()
#                 }
#                 return metrics
#             except Exception as e:
#                 logger.error(f"Error calculating system metrics: {e}")
#                 return {
#                     'total_pipeline_time': total_time,
#                     'error_in_system_metrics': 1.0
#                 }
    
#     def _get_gpu_utilization(self) -> float:
#         """Get GPU utilization if available, otherwise return 0.0."""
#         try:
#             import GPUtil
#             gpus = GPUtil.getGPUs()
#             return gpus[0].load * 100 if gpus else 0.0
#         except ImportError:
#             pass
#         except Exception as e:
#             logger.debug(f"Cannot get GPU utilization: {e}")
#         return 0.0
    
#     def _calculate_cache_hit_rate(self) -> float:
#         """Placeholder for cache hit rate calculation. Implement based on your cache system."""
#         return 0.0  # Replace with actual cache hit rate logic
    
#     def get_system_info(self) -> Dict[str, Any]:
#         """Get static system information.
        
#         Returns:
#             Dictionary containing system information
#         """
#         try:
#             memory = psutil.virtual_memory()
#             return {
#                 'cpu_count': psutil.cpu_count(),
#                 'total_memory_gb': memory.total / (1024**3),
#                 'free_memory_gb': memory.available / (1024**3),
#                 'process_id': self.process.pid
#             }
#         except Exception as e:
#             logger.error(f"Error getting system info: {e}")
#             return {}
    
#     def reset_baseline(self):
#         """Reset the memory baseline for relative measurements."""
#         try:
#             self.initial_memory = self.process.memory_info().rss / 1024 / 1024
#             logger.info(f"Memory baseline reset to {self.initial_memory:.2f} MB")
#         except Exception as e:
#             logger.error(f"Error resetting baseline: {e}")













import psutil
import logging
import time
from typing import Dict, Optional, Any
import numpy as np
from collections import defaultdict

# Optional GPU support
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, pynvml.NVMLError):
    GPU_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)

class SystemMetrics:
    """Calculates system-level metrics for the RAG pipeline."""
    
    def __init__(self):
        """Initialize SystemMetrics with process handle and baseline measurements."""
        self.process = psutil.Process()
        self.baseline_memory = self._get_memory_usage()
        self.cache_stats = defaultdict(int)  # Simple cache tracking
        logger.info("SystemMetrics initialized")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent(interval=None)
        except Exception as e:
            logger.warning(f"Could not get CPU usage: {e}")
            return 0.0
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU utilization and memory metrics."""
        gpu_metrics = {
            'gpu_utilization': 0.0,
            'gpu_memory_used': 0.0,
            'gpu_memory_total': 0.0,
            'gpu_temperature': 0.0
        }
        
        if not GPU_AVAILABLE or not pynvml:
            return gpu_metrics
        
        try:
            # Get first GPU (index 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_metrics['gpu_utilization'] = float(util.gpu)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_metrics['gpu_memory_used'] = mem_info.used / (1024 * 1024)  # MB
            gpu_metrics['gpu_memory_total'] = mem_info.total / (1024 * 1024)  # MB
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_metrics['gpu_temperature'] = float(temp)
            except pynvml.NVMLError:
                pass  # Temperature not available on all GPUs
                
        except pynvml.NVMLError as e:
            logger.warning(f"GPU metrics unavailable: {e}")
        
        return gpu_metrics
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (simple word-based approximation)."""
        if not text:
            return 0
        # Rough approximation: 1 token ≈ 0.75 words
        word_count = len(text.split())
        return int(word_count / 0.75)
    
    def update_cache_stats(self, cache_hit: bool):
        """Update cache statistics."""
        if cache_hit:
            self.cache_stats['hits'] += 1
        else:
            self.cache_stats['misses'] += 1
    
    def calculate_metrics(self, 
                         total_time: float, 
                         retrieval_time: float, 
                         generation_time: float, 
                         web_time: float, 
                         response: Optional[str] = None,
                         query: Optional[str] = None,
                         context: Optional[str] = None) -> Dict[str, float]:
        """Calculate comprehensive system metrics for a single query run.
        
        Args:
            total_time: Total pipeline execution time in seconds.
            retrieval_time: Time taken for retrieval in seconds.
            generation_time: Time taken for generation in seconds.
            web_time: Time taken for web search in seconds.
            response: Optional generated response for token calculations.
            query: Optional input query for token calculations.
            context: Optional context used for generation.
        
        Returns:
            Dictionary of calculated system metrics.
        """
        try:
            metrics = {}
            
            # === Time Metrics ===
            metrics['total_pipeline_time'] = total_time
            metrics['retrieval_time'] = retrieval_time
            metrics['generation_time'] = generation_time
            metrics['web_time'] = web_time
            
            # Time distribution percentages
            if total_time > 0:
                metrics['retrieval_time_pct'] = (retrieval_time / total_time) * 100
                metrics['generation_time_pct'] = (generation_time / total_time) * 100
                metrics['web_time_pct'] = (web_time / total_time) * 100
            else:
                metrics['retrieval_time_pct'] = 0.0
                metrics['generation_time_pct'] = 0.0
                metrics['web_time_pct'] = 0.0
            
            # === Memory Metrics ===
            current_memory = self._get_memory_usage()
            metrics['memory_usage'] = current_memory
            metrics['memory_increase'] = max(0, current_memory - self.baseline_memory)
            
            # === CPU Metrics ===
            metrics['cpu_usage'] = self._get_cpu_usage()
            
            # === GPU Metrics ===
            gpu_metrics = self._get_gpu_metrics()
            metrics.update(gpu_metrics)
            
            # === Token Metrics ===
            if response and generation_time > 0:
                token_count = self._estimate_tokens(response)
                metrics['response_tokens'] = token_count
                metrics['tokens_per_second'] = token_count / generation_time
            else:
                metrics['response_tokens'] = 0
                metrics['tokens_per_second'] = 0.0
            
            if query:
                metrics['query_tokens'] = self._estimate_tokens(query)
            else:
                metrics['query_tokens'] = 0
            
            if context:
                metrics['context_tokens'] = self._estimate_tokens(context)
            else:
                metrics['context_tokens'] = 0
            
            # === Throughput Metrics ===
            if total_time > 0:
                metrics['queries_per_second'] = 1.0 / total_time
            else:
                metrics['queries_per_second'] = 0.0
            
            # === Cache Metrics ===
            total_cache_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            if total_cache_requests > 0:
                metrics['cache_hit_rate'] = self.cache_stats['hits'] / total_cache_requests
            else:
                metrics['cache_hit_rate'] = 0.0
            
            metrics['cache_hits'] = self.cache_stats['hits']
            metrics['cache_misses'] = self.cache_stats['misses']
            
            # === Efficiency Metrics ===
            # Efficiency as tokens per second per MB of memory
            if current_memory > 0 and metrics['tokens_per_second'] > 0:
                metrics['memory_efficiency'] = metrics['tokens_per_second'] / current_memory
            else:
                metrics['memory_efficiency'] = 0.0
            
            # === Resource Utilization Score ===
            # Combined score of CPU and GPU utilization (0-100)
            cpu_util = min(metrics['cpu_usage'], 100.0)  # Cap at 100%
            gpu_util = min(metrics['gpu_utilization'], 100.0)  # Cap at 100%
            metrics['resource_utilization_score'] = (cpu_util + gpu_util) / 2.0
            
            logger.debug(f"System metrics calculated: {list(metrics.keys())}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating system metrics: {e}")
            return {
                'error_occurred': 1.0, 
                'total_pipeline_time': total_time,
                'memory_usage': self._get_memory_usage(),
                'cpu_usage': self._get_cpu_usage()
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        try:
            info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
                'gpu_available': GPU_AVAILABLE
            }
            
            if GPU_AVAILABLE and pynvml:
                try:
                    gpu_count = pynvml.nvmlDeviceGetCount()
                    info['gpu_count'] = gpu_count
                    if gpu_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        info['gpu_name'] = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                except pynvml.NVMLError:
                    info['gpu_count'] = 0
            else:
                info['gpu_count'] = 0
            
            return info
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {'error': str(e)}
    
    def reset_cache_stats(self):
        """Reset cache statistics."""
        self.cache_stats.clear()
        logger.debug("Cache statistics reset")