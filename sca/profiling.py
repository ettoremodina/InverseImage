"""
Simple profiling utilities for performance analysis.
"""

import time
from functools import wraps
from collections import defaultdict
from typing import Dict, List
import atexit


class Profiler:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.stats: Dict[str, Dict] = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0.0,
            'times': []
        })
        self.enabled = True
        atexit.register(self.print_stats)
    
    def record(self, name: str, elapsed: float):
        if not self.enabled:
            return
        self.stats[name]['calls'] += 1
        self.stats[name]['total_time'] += elapsed
        if len(self.stats[name]['times']) < 1000:
            self.stats[name]['times'].append(elapsed)
    
    def print_stats(self):
        if not self.stats:
            return
        
        print("\n" + "=" * 70)
        print("PERFORMANCE PROFILING RESULTS")
        print("=" * 70)
        
        sorted_stats = sorted(
            self.stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        
        print(f"{'Function':<35} {'Calls':>10} {'Total(s)':>10} {'Avg(ms)':>10}")
        print("-" * 70)
        
        for name, data in sorted_stats:
            calls = data['calls']
            total = data['total_time']
            avg_ms = (total / calls * 1000) if calls > 0 else 0
            print(f"{name:<35} {calls:>10} {total:>10.3f} {avg_ms:>10.3f}")
        
        print("=" * 70)
    
    def reset(self):
        self.stats.clear()


profiler = Profiler()


def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        profiler.record(func.__qualname__, elapsed)
        return result
    return wrapper


class profile_block:
    def __init__(self, name: str):
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        profiler.record(self.name, elapsed)
