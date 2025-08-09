#!/usr/bin/env python3
"""
Comprehensive Benchmarking Module for PQC and Classical Algorithms
Implements detailed benchmarking for ML-KEM, ML-DSA, and their classical counterparts
"""

import asyncio
import time
import statistics
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import threading
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import from the package structure
try:
    from pqc_core import CryptoBenchmark, OQSCrypto
except ImportError:
    print("Warning: pqc_core not found. Using mock implementations.")
    # Mock implementations for development
    class MockCryptoBenchmark:
        def __init__(self, name, key_size=None):
            self.name = name
            self.key_size = key_size or 256
            self.metrics = {}
        
        def measure_operation(self, op, size):
            return {'latency_ms': np.random.uniform(1, 10)}
        
        def run_benchmark(self, iterations, data_sizes):
            return {'mock': 'data'}
    
    class MockOQSCrypto:
        def __init__(self, name):
            self.name = name
            self.key_size = 512
            self.metrics = {}
        
        def measure_operation(self, op, size):
            return {'latency_ms': np.random.uniform(1, 10)}
        
        def run_benchmark(self, iterations, data_sizes):
            return {'mock': 'data'}
    
    CryptoBenchmark = MockCryptoBenchmark
    OQSCrypto = MockOQSCrypto

@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results"""
    algorithm_name: str
    operation: str
    data_size: int
    iterations: int
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_99: Tuple[float, float]
    timestamp: float

@dataclass
class ResourceMetrics:
    """Data class for storing resource utilization metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    network_io_bytes: Optional[int] = None
    disk_io_bytes: Optional[int] = None

class SystemMonitor:
    """Monitor system resources during benchmarks"""
    
    def __init__(self, sampling_interval_ms: int = 100):
        self.sampling_interval_ms = sampling_interval_ms
        self.metrics: List[ResourceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get CPU and memory metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                metric = ResourceMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    memory_available_mb=memory.available / (1024 * 1024)
                )
                
                self.metrics.append(metric)
                
                # Sleep for sampling interval
                time.sleep(self.sampling_interval_ms / 1000.0)
                
            except Exception as e:
                logging.error(f"Error in system monitoring: {e}")
                break
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average resource metrics"""
        if not self.metrics:
            return {}
        
        return {
            'avg_cpu_percent': np.mean([m.cpu_percent for m in self.metrics]),
            'avg_memory_percent': np.mean([m.memory_percent for m in self.metrics]),
            'avg_memory_used_mb': np.mean([m.memory_used_mb for m in self.metrics]),
            'max_cpu_percent': np.max([m.cpu_percent for m in self.metrics]),
            'max_memory_percent': np.max([m.memory_percent for m in self.metrics])
        }

class ComprehensiveBenchmarker:
    """Comprehensive benchmarking for PQC and classical algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results: List[BenchmarkResult] = []
        
        # Initialize algorithms based on configuration
        self.algorithms = self._initialize_algorithms()
        
        # System monitor
        self.system_monitor = SystemMonitor(
            sampling_interval_ms=config.get('benchmarks', {}).get('resource_test', {}).get('sampling_interval_ms', 100)
        )
    
    def _initialize_algorithms(self) -> Dict[str, Any]:
        """Initialize all algorithms from configuration"""
        algorithms = {}
        
        # Classical algorithms
        classical_configs = self.config.get('algorithms', {}).get('classical', [])
        for algo_config in classical_configs:
            name = algo_config['name']
            key_size = algo_config['key_size']
            
            if 'RSA' in name:
                algorithms[name] = CryptoBenchmark(name, key_size)
            elif 'ECDSA' in name or 'ECDH' in name:
                algorithms[name] = CryptoBenchmark(name, key_size)
            elif 'AES' in name:
                algorithms[name] = CryptoBenchmark(name, key_size)
        
        # PQC algorithms
        pqc_configs = self.config.get('algorithms', {}).get('pqc', [])
        for algo_config in pqc_configs:
            name = algo_config['name']
            algorithms[name] = OQSCrypto(name)
        
        self.logger.info(f"Initialized {len(algorithms)} algorithms: {list(algorithms.keys())}")
        return algorithms
    
    async def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive benchmarks for all algorithms"""
        self.logger.info("Starting comprehensive benchmarking")
        
        benchmark_results = {
            'latency_benchmarks': {},
            'throughput_benchmarks': {},
            'resource_benchmarks': {},
            'stress_tests': {},
            'statistical_analysis': {},
            'comparison_analysis': {}
        }
        
        # Run latency benchmarks
        benchmark_results['latency_benchmarks'] = await self._run_latency_benchmarks()
        
        # Run throughput benchmarks
        benchmark_results['throughput_benchmarks'] = await self._run_throughput_benchmarks()
        
        # Run resource benchmarks
        benchmark_results['resource_benchmarks'] = await self._run_resource_benchmarks()
        
        # Run stress tests
        benchmark_results['stress_tests'] = await self._run_stress_tests()
        
        # Perform statistical analysis
        benchmark_results['statistical_analysis'] = self._perform_statistical_analysis()
        
        # Perform comparison analysis
        benchmark_results['comparison_analysis'] = self._perform_comparison_analysis()
        
        self.logger.info("Comprehensive benchmarking completed")
        return benchmark_results
    
    async def _run_latency_benchmarks(self) -> Dict[str, Any]:
        """Run detailed latency benchmarks"""
        self.logger.info("Running latency benchmarks")
        
        latency_config = self.config.get('benchmarks', {}).get('latency_test', {})
        iterations = latency_config.get('iterations', 1000)
        data_sizes = latency_config.get('data_sizes', [64, 1024, 4096, 16384])
        warmup_iterations = latency_config.get('warmup_iterations', 100)
        
        results = {}
        
        for algo_name, algorithm in self.algorithms.items():
            self.logger.info(f"Benchmarking latency for {algo_name}")
            results[algo_name] = {}
            
            for data_size in data_sizes:
                # Warmup phase
                for _ in range(warmup_iterations):
                    try:
                        algorithm.measure_operation("warmup", data_size)
                    except Exception as e:
                        self.logger.warning(f"Warmup failed for {algo_name}: {e}")
                
                # Actual benchmark
                latencies = []
                for i in range(iterations):
                    try:
                        start_time = time.perf_counter()
                        algorithm.measure_operation("benchmark", data_size)
                        end_time = time.perf_counter()
                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)
                        
                        if i % 100 == 0:
                            self.logger.debug(f"Progress: {i}/{iterations} for {algo_name} size {data_size}")
                    
                    except Exception as e:
                        self.logger.error(f"Benchmark iteration failed for {algo_name}: {e}")
                
                if latencies:
                    # Calculate statistics
                    result = self._calculate_latency_statistics(latencies, algo_name, "latency", data_size, iterations)
                    results[algo_name][data_size] = asdict(result)
                    self.results.append(result)
        
        return results
    
    async def _run_throughput_benchmarks(self) -> Dict[str, Any]:
        """Run throughput benchmarks"""
        self.logger.info("Running throughput benchmarks")
        
        throughput_config = self.config.get('benchmarks', {}).get('throughput_test', {})
        batch_sizes = throughput_config.get('batch_sizes', [100, 1000, 10000])
        duration_seconds = throughput_config.get('duration_seconds', 60)
        concurrent_operations = throughput_config.get('concurrent_operations', [1, 4, 8])
        
        results = {}
        
        for algo_name, algorithm in self.algorithms.items():
            self.logger.info(f"Benchmarking throughput for {algo_name}")
            results[algo_name] = {}
            
            for batch_size in batch_sizes:
                for concurrency in concurrent_operations:
                    self.logger.debug(
                        f"Throughput config -> algo={algo_name}, batch={batch_size}, concurrency={concurrency}, duration={duration_seconds}s"
                    )
                    throughput_result = await self._measure_throughput(
                        algorithm, algo_name, batch_size, duration_seconds, concurrency
                    )
                    results[algo_name][f"batch_{batch_size}_concurrent_{concurrency}"] = throughput_result
        
        return results
    
    async def _measure_throughput(self, algorithm, algo_name: str, batch_size: int, 
                                duration_seconds: int, concurrency: int) -> Dict[str, Any]:
        """Measure throughput for a specific configuration"""
        operations_completed = 0
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        last_log = start_time
        
        async def worker():
            nonlocal operations_completed
            while time.perf_counter() < end_time:
                try:
                    for _ in range(batch_size):
                        algorithm.measure_operation("throughput", 1024)
                        operations_completed += 1
                    # Cooperative yield to avoid starving the event loop
                    await asyncio.sleep(0)
                except Exception as e:
                    self.logger.error(f"Throughput worker error: {e}")
        
        # Run concurrent workers
        workers = [worker() for _ in range(concurrency)]
        # Periodically log progress while awaiting completion
        wait_task = asyncio.gather(*workers)
        while not wait_task.done():
            now = time.perf_counter()
            if now - last_log >= 5.0:  # log every 5 seconds
                elapsed = now - start_time
                self.logger.info(
                    f"Throughput progress [{algo_name} | batch={batch_size} | conc={concurrency}] - elapsed={elapsed:.1f}s ops={operations_completed}"
                )
                last_log = now
            await asyncio.sleep(0.2)
        
        actual_duration = time.perf_counter() - start_time
        throughput_ops_per_sec = operations_completed / actual_duration
        
        return {
            'operations_completed': operations_completed,
            'duration_seconds': actual_duration,
            'throughput_ops_per_sec': throughput_ops_per_sec,
            'batch_size': batch_size,
            'concurrency': concurrency
        }
    
    async def _run_resource_benchmarks(self) -> Dict[str, Any]:
        """Run resource utilization benchmarks"""
        self.logger.info("Running resource benchmarks")
        
        resource_config = self.config.get('benchmarks', {}).get('resource_test', {})
        iterations = 1000
        
        results = {}
        
        for algo_name, algorithm in self.algorithms.items():
            self.logger.info(f"Benchmarking resources for {algo_name}")
            
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            # Run intensive operations
            for i in range(iterations):
                try:
                    algorithm.measure_operation("resource_test", 4096)
                    
                    if i % 100 == 0:
                        self.logger.debug(f"Resource benchmark progress: {i}/{iterations} for {algo_name}")
                
                except Exception as e:
                    self.logger.error(f"Resource benchmark failed for {algo_name}: {e}")
            
            # Stop monitoring and get results
            self.system_monitor.stop_monitoring()
            resource_metrics = self.system_monitor.get_average_metrics()
            
            results[algo_name] = resource_metrics
        
        return results
    
    async def _run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests"""
        self.logger.info("Running stress tests")
        
        stress_config = self.config.get('benchmarks', {}).get('stress_test', {})
        max_connections = stress_config.get('max_concurrent_connections', 1000)
        ramp_up_time = stress_config.get('ramp_up_time_seconds', 60)
        hold_time = stress_config.get('hold_time_seconds', 300)
        
        results = {}
        
        for algo_name, algorithm in self.algorithms.items():
            self.logger.info(f"Running stress test for {algo_name}")
            
            stress_result = await self._run_stress_test_for_algorithm(
                algorithm, algo_name, max_connections, ramp_up_time, hold_time
            )
            results[algo_name] = stress_result
        
        return results
    
    async def _run_stress_test_for_algorithm(self, algorithm, algo_name: str, 
                                           max_connections: int, ramp_up_time: int, 
                                           hold_time: int) -> Dict[str, Any]:
        """Run stress test for a specific algorithm"""
        start_time = time.perf_counter()
        operations_completed = 0
        errors = 0
        
        # Ramp up phase
        ramp_up_end = start_time + ramp_up_time
        connections = 0
        
        while time.perf_counter() < ramp_up_end and connections < max_connections:
            try:
                algorithm.measure_operation("stress_test", 1024)
                operations_completed += 1
                connections += 1
                await asyncio.sleep(ramp_up_time / max_connections)
            except Exception as e:
                errors += 1
                self.logger.error(f"Stress test error: {e}")
        
        # Hold phase
        hold_end = ramp_up_end + hold_time
        while time.perf_counter() < hold_end:
            try:
                algorithm.measure_operation("stress_test", 1024)
                operations_completed += 1
                await asyncio.sleep(0.001)  # Small delay
            except Exception as e:
                errors += 1
        
        total_time = time.perf_counter() - start_time
        
        return {
            'total_operations': operations_completed,
            'total_errors': errors,
            'error_rate': errors / max(operations_completed, 1),
            'total_time_seconds': total_time,
            'throughput_ops_per_sec': operations_completed / total_time,
            'max_connections_reached': connections
        }
    
    def _calculate_latency_statistics(self, latencies: List[float], algorithm_name: str, 
                                    operation: str, data_size: int, iterations: int) -> BenchmarkResult:
        """Calculate comprehensive latency statistics"""
        latencies_array = np.array(latencies)
        
        # Basic statistics
        mean_latency = np.mean(latencies_array)
        median_latency = np.median(latencies_array)
        std_latency = np.std(latencies_array)
        min_latency = np.min(latencies_array)
        max_latency = np.max(latencies_array)
        
        # Percentiles
        p95_latency = np.percentile(latencies_array, 95)
        p99_latency = np.percentile(latencies_array, 99)
        
        # Throughput
        throughput_ops_per_sec = 1000.0 / mean_latency if mean_latency > 0 else 0
        
        # Confidence intervals
        confidence_95 = stats.t.interval(0.95, len(latencies_array)-1, 
                                       loc=mean_latency, scale=stats.sem(latencies_array))
        confidence_99 = stats.t.interval(0.99, len(latencies_array)-1, 
                                       loc=mean_latency, scale=stats.sem(latencies_array))
        
        return BenchmarkResult(
            algorithm_name=algorithm_name,
            operation=operation,
            data_size=data_size,
            iterations=iterations,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            std_latency_ms=std_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_ops_per_sec=throughput_ops_per_sec,
            memory_usage_mb=0.0,  # Would be calculated from system monitor
            cpu_usage_percent=0.0,  # Would be calculated from system monitor
            confidence_interval_95=confidence_95,
            confidence_interval_99=confidence_99,
            timestamp=time.time()
        )
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        self.logger.info("Performing statistical analysis")
        
        analysis = {
            'algorithm_comparisons': {},
            'performance_trends': {},
            'outlier_analysis': {},
            'correlation_analysis': {}
        }
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        if df.empty:
            return analysis
        
        # Algorithm comparisons
        for data_size in df['data_size'].unique():
            size_data = df[df['data_size'] == data_size]
            
            # ANOVA test for algorithm differences
            algorithms = size_data['algorithm_name'].unique()
            if len(algorithms) > 1:
                groups = [size_data[size_data['algorithm_name'] == algo]['mean_latency_ms'].values 
                         for algo in algorithms]
                f_stat, p_value = stats.f_oneway(*groups)
                
                analysis['algorithm_comparisons'][f'data_size_{data_size}'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < 0.05,
                    'algorithms_tested': list(algorithms)
                }
        
        # Performance trends
        for algo in df['algorithm_name'].unique():
            algo_data = df[df['algorithm_name'] == algo]
            if len(algo_data) > 1:
                # Linear regression for performance vs data size
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    algo_data['data_size'], algo_data['mean_latency_ms']
                )
                
                analysis['performance_trends'][algo] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'scalability_factor': slope
                }
        
        return analysis
    
    def _perform_comparison_analysis(self) -> Dict[str, Any]:
        """Perform comparison analysis between PQC and classical algorithms"""
        self.logger.info("Performing comparison analysis")
        
        comparison = {
            'pqc_vs_classical': {},
            'security_level_comparison': {},
            'performance_overhead': {},
            'recommendations': {}
        }
        
        # Group algorithms by type
        classical_algorithms = [name for name in self.algorithms.keys() 
                              if any(x in name for x in ['RSA', 'ECDSA', 'ECDH', 'AES'])]
        pqc_algorithms = [name for name in self.algorithms.keys() 
                         if any(x in name for x in ['ML-KEM', 'ML-DSA'])]
        
        # Compare PQC vs Classical
        for data_size in [64, 1024, 4096, 16384]:
            classical_results = [r for r in self.results 
                               if r.algorithm_name in classical_algorithms and r.data_size == data_size]
            pqc_results = [r for r in self.results 
                          if r.algorithm_name in pqc_algorithms and r.data_size == data_size]
            
            if classical_results and pqc_results:
                classical_avg = np.mean([r.mean_latency_ms for r in classical_results])
                pqc_avg = np.mean([r.mean_latency_ms for r in pqc_results])
                
                overhead_percent = ((pqc_avg - classical_avg) / classical_avg) * 100
                
                comparison['pqc_vs_classical'][f'data_size_{data_size}'] = {
                    'classical_avg_ms': classical_avg,
                    'pqc_avg_ms': pqc_avg,
                    'overhead_percent': overhead_percent,
                    'acceptable_overhead': overhead_percent < 100  # 2x overhead threshold
                }
        
        return comparison
    
    def generate_visualizations(self, output_dir: str = "results"):
        """Generate comprehensive visualizations"""
        self.logger.info("Generating visualizations")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(result) for result in self.results])
        
        if df.empty:
            self.logger.warning("No results to visualize")
            return
        
        # 1. Latency comparison chart
        self._create_latency_comparison_chart(df, output_path)
        
        # 2. Throughput analysis
        self._create_throughput_analysis_chart(df, output_path)
        
        # 3. Performance trends
        self._create_performance_trends_chart(df, output_path)
        
        # 4. Algorithm comparison heatmap
        self._create_algorithm_comparison_heatmap(df, output_path)
        
        # 5. Interactive dashboard
        self._create_interactive_dashboard(df, output_path)
    
    def _create_latency_comparison_chart(self, df: pd.DataFrame, output_path: Path):
        """Create latency comparison chart"""
        plt.figure(figsize=(15, 10))
        
        # Group by data size
        for data_size in sorted(df['data_size'].unique()):
            plt.subplot(2, 2, list(sorted(df['data_size'].unique())).index(data_size) + 1)
            
            size_data = df[df['data_size'] == data_size]
            
            # Separate classical and PQC
            classical_data = size_data[size_data['algorithm_name'].str.contains('RSA|ECDSA|ECDH|AES')]
            pqc_data = size_data[size_data['algorithm_name'].str.contains('ML-KEM|ML-DSA')]
            
            if not classical_data.empty:
                plt.bar(classical_data['algorithm_name'], classical_data['mean_latency_ms'], 
                       alpha=0.7, label='Classical', color='blue')
            
            if not pqc_data.empty:
                plt.bar(pqc_data['algorithm_name'], pqc_data['mean_latency_ms'], 
                       alpha=0.7, label='PQC', color='red')
            
            plt.title(f'Latency Comparison - Data Size: {data_size} bytes')
            plt.ylabel('Mean Latency (ms)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_throughput_analysis_chart(self, df: pd.DataFrame, output_path: Path):
        """Create throughput analysis chart"""
        plt.figure(figsize=(12, 8))
        
        # Group by algorithm
        for algo in df['algorithm_name'].unique():
            algo_data = df[df['algorithm_name'] == algo]
            plt.plot(algo_data['data_size'], algo_data['throughput_ops_per_sec'], 
                    marker='o', label=algo, linewidth=2)
        
        plt.title('Throughput Analysis by Algorithm')
        plt.xlabel('Data Size (bytes)')
        plt.ylabel('Throughput (ops/sec)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'throughput_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_trends_chart(self, df: pd.DataFrame, output_path: Path):
        """Create performance trends chart"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        metrics = ['mean_latency_ms', 'p95_latency_ms', 'throughput_ops_per_sec', 'std_latency_ms']
        titles = ['Mean Latency', '95th Percentile Latency', 'Throughput', 'Latency Standard Deviation']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            pivot_data = df.pivot(index='data_size', columns='algorithm_name', values=metric)
            pivot_data.plot(kind='line', marker='o', ax=axes[i])
            axes[i].set_title(title)
            axes[i].set_xlabel('Data Size (bytes)')
            axes[i].set_xscale('log')
            if 'latency' in metric:
                axes[i].set_ylabel('Latency (ms)')
            else:
                axes[i].set_ylabel('Throughput (ops/sec)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_algorithm_comparison_heatmap(self, df: pd.DataFrame, output_path: Path):
        """Create algorithm comparison heatmap"""
        # Create pivot table for mean latency
        pivot_data = df.pivot(index='algorithm_name', columns='data_size', values='mean_latency_ms')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Mean Latency (ms)'})
        plt.title('Algorithm Performance Heatmap')
        plt.xlabel('Data Size (bytes)')
        plt.ylabel('Algorithm')
        plt.tight_layout()
        plt.savefig(output_path / 'algorithm_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_dashboard(self, df: pd.DataFrame, output_path: Path):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latency Comparison', 'Throughput Analysis', 
                          'Performance Trends', 'Algorithm Heatmap'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Latency comparison (bar chart)
        for data_size in [1024, 4096]:
            size_data = df[df['data_size'] == data_size]
            fig.add_trace(
                go.Bar(x=size_data['algorithm_name'], y=size_data['mean_latency_ms'],
                      name=f'{data_size} bytes', showlegend=True),
                row=1, col=1
            )
        
        # 2. Throughput analysis (scatter)
        for algo in df['algorithm_name'].unique():
            algo_data = df[df['algorithm_name'] == algo]
            fig.add_trace(
                go.Scatter(x=algo_data['data_size'], y=algo_data['throughput_ops_per_sec'],
                          mode='lines+markers', name=algo),
                row=1, col=2
            )
        
        # 3. Performance trends (scatter)
        for algo in df['algorithm_name'].unique():
            algo_data = df[df['algorithm_name'] == algo]
            fig.add_trace(
                go.Scatter(x=algo_data['data_size'], y=algo_data['mean_latency_ms'],
                          mode='lines+markers', name=algo, showlegend=False),
                row=2, col=1
            )
        
        # 4. Heatmap
        pivot_data = df.pivot(index='algorithm_name', columns='data_size', values='mean_latency_ms')
        fig.add_trace(
            go.Heatmap(z=pivot_data.values, x=pivot_data.columns, y=pivot_data.index,
                      colorscale='YlOrRd', showscale=True),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Quantum-Resilient Cryptography Performance Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Algorithm", row=1, col=1)
        fig.update_yaxes(title_text="Mean Latency (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Data Size (bytes)", row=1, col=2)
        fig.update_yaxes(title_text="Throughput (ops/sec)", row=1, col=2)
        fig.update_xaxes(title_text="Data Size (bytes)", row=2, col=1)
        fig.update_yaxes(title_text="Mean Latency (ms)", row=2, col=1)
        fig.update_xaxes(title_text="Data Size (bytes)", row=2, col=2)
        fig.update_yaxes(title_text="Algorithm", row=2, col=2)
        
        # Save interactive dashboard
        fig.write_html(output_path / 'interactive_dashboard.html')
    
    def save_results(self, output_dir: str = "results"):
        """Save all benchmark results to files"""
        self.logger.info(f"Saving results to {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save raw results as JSON
        results_dict = [asdict(result) for result in self.results]
        with open(output_path / 'benchmark_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(results_dict)
        df.to_csv(output_path / 'benchmark_results.csv', index=False)
        
        # Save as Excel with multiple sheets
        with pd.ExcelWriter(output_path / 'benchmark_results.xlsx') as writer:
            df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Summary statistics
            summary = df.groupby('algorithm_name').agg({
                'mean_latency_ms': ['mean', 'std', 'min', 'max'],
                'throughput_ops_per_sec': ['mean', 'std', 'min', 'max']
            }).round(4)
            summary.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Performance comparison
            comparison = df.pivot_table(
                index='data_size', 
                columns='algorithm_name', 
                values='mean_latency_ms'
            )
            comparison.to_excel(writer, sheet_name='Performance_Comparison')
        
        self.logger.info("Results saved successfully")

async def main():
    """Main entry point for benchmarking"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize benchmarker
    benchmarker = ComprehensiveBenchmarker(config)
    
    # Run comprehensive benchmarks
    results = await benchmarker.run_comprehensive_benchmarks()
    
    # Generate visualizations
    benchmarker.generate_visualizations()
    
    # Save results
    benchmarker.save_results()
    
    print("Benchmarking completed successfully!")
    print(f"Results saved to: results/")
    print(f"Visualizations saved to: results/")

if __name__ == "__main__":
    asyncio.run(main())
