"""
Performance profiling and benchmarking for inference.
"""
import time
import numpy as np
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProfileStats:
    """Statistics from profiling inference."""
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_samples_per_sec: float
    total_time_sec: float
    num_samples: int


class InferenceProfiler:
    """Profile inference latency and throughput."""
    
    def __init__(self, model, device: torch.device, warmup_iterations: int = 10):
        """
        Initialize profiler.
        
        Args:
            model: Model wrapper (e.g., KeywordModel)
            device: torch.device
            warmup_iterations: Number of warmup runs before profiling
        """
        self.model = model
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.latencies = []
    
    def warmup(self, sample_feat: torch.Tensor):
        """Run warmup iterations."""
        for _ in range(self.warmup_iterations):
            _ = self.model.predict(sample_feat)
    
    def profile_single(self, feat: torch.Tensor, num_runs: int = 100) -> ProfileStats:
        """
        Profile single-sample inference latency.
        
        Args:
            feat: Sample MFCC features (C, T)
            num_runs: Number of profiling iterations
            
        Returns:
            ProfileStats with latency metrics
        """
        self.warmup(feat)
        latencies = []
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_total = time.perf_counter()
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.model.predict(feat)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        
        end_total = time.perf_counter()
        total_time = end_total - start_total
        
        latencies_arr = np.array(latencies)
        return ProfileStats(
            latency_mean_ms=float(np.mean(latencies_arr)),
            latency_std_ms=float(np.std(latencies_arr)),
            latency_min_ms=float(np.min(latencies_arr)),
            latency_max_ms=float(np.max(latencies_arr)),
            latency_p95_ms=float(np.percentile(latencies_arr, 95)),
            latency_p99_ms=float(np.percentile(latencies_arr, 99)),
            throughput_samples_per_sec=num_runs / total_time,
            total_time_sec=total_time,
            num_samples=num_runs
        )
    
    def profile_batch(self, features: List[torch.Tensor]) -> ProfileStats:
        """
        Profile batch inference throughput.
        
        Args:
            features: List of MFCC features
            
        Returns:
            ProfileStats with throughput metrics
        """
        # Warmup
        self.warmup(features[0])
        
        latencies = []
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_total = time.perf_counter()
        for feat in features:
            start = time.perf_counter()
            _ = self.model.predict(feat)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        end_total = time.perf_counter()
        total_time = end_total - start_total
        
        latencies_arr = np.array(latencies)
        return ProfileStats(
            latency_mean_ms=float(np.mean(latencies_arr)),
            latency_std_ms=float(np.std(latencies_arr)),
            latency_min_ms=float(np.min(latencies_arr)),
            latency_max_ms=float(np.max(latencies_arr)),
            latency_p95_ms=float(np.percentile(latencies_arr, 95)),
            latency_p99_ms=float(np.percentile(latencies_arr, 99)),
            throughput_samples_per_sec=len(features) / total_time,
            total_time_sec=total_time,
            num_samples=len(features)
        )
    
    def print_stats(self, stats: ProfileStats, title: str = "Profile Results"):
        """Print profiling statistics."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Samples:          {stats.num_samples}")
        print(f"Total time:       {stats.total_time_sec:.3f}s")
        print(f"Throughput:       {stats.throughput_samples_per_sec:.1f} samples/sec")
        print(f"\nLatency (ms):")
        print(f"  Mean ± Std:     {stats.latency_mean_ms:.2f} ± {stats.latency_std_ms:.2f}")
        print(f"  Min / Max:      {stats.latency_min_ms:.2f} / {stats.latency_max_ms:.2f}")
        print(f"  P95 / P99:      {stats.latency_p95_ms:.2f} / {stats.latency_p99_ms:.2f}")
        print(f"{'='*60}\n")


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }
