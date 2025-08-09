#!/usr/bin/env python3
"""
Simple script to run PQC and Classical algorithm benchmarks
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Main function to run benchmarks"""
    print("🚀 Starting Quantum-Resilient Cryptography Benchmarking")
    print("=" * 60)
    
    try:
        # Import the benchmarking module
        from python_orchestrator.benchmarking import ComprehensiveBenchmarker
        import yaml
        
        # Load configuration
        config_path = Path('config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✅ Configuration loaded successfully")
        else:
            print("⚠️  Configuration file not found, using defaults")
            config = {
                'framework': {'log_level': 'INFO', 'output_directory': 'results'},
                'benchmarks': {
                    'latency_test': {'iterations': 100, 'data_sizes': [64, 1024, 4096]},
                    'throughput_test': {'batch_sizes': [100, 1000], 'duration_seconds': 60}
                }
            }
        
        # Initialize benchmarker
        print("🔧 Initializing benchmarker...")
        benchmarker = ComprehensiveBenchmarker(config)
        
        # Run comprehensive benchmarks
        print("📊 Running comprehensive benchmarks...")
        results = asyncio.run(benchmarker.run_comprehensive_benchmarks())
        
        # Generate visualizations
        print("📈 Generating visualizations...")
        benchmarker.generate_visualizations()
        
        # Save results
        print("💾 Saving results...")
        benchmarker.save_results()
        
        print("\n✅ Benchmarking completed successfully!")
        print("📁 Results saved to: results/")
        print("📊 Visualizations saved to: results/")
        print("📋 Check the following files:")
        print("   - results/benchmark_results.json")
        print("   - results/benchmark_results.csv")
        print("   - results/benchmark_results.xlsx")
        print("   - results/interactive_dashboard.html")
        print("   - results/*.png (various charts)")
        
        # Print summary
        print("\n📋 Benchmark Summary:")
        if hasattr(benchmarker, 'results') and benchmarker.results:
            print(f"   - Total benchmark results: {len(benchmarker.results)}")
            algorithms = set(r.algorithm_name for r in benchmarker.results)
            print(f"   - Algorithms tested: {len(algorithms)}")
            print(f"   - Algorithms: {', '.join(sorted(algorithms))}")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
