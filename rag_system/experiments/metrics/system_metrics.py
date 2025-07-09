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