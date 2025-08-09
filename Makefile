# Quantum-Resilient Cryptography Benchmarking Framework
# Makefile for easy execution of benchmarks and analysis

.PHONY: help install test benchmark analyze clean results

# Default target
help:
	@echo "Quantum-Resilient Cryptography Benchmarking Framework"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install Python dependencies"
	@echo "  test        - Run all tests"
	@echo "  benchmark   - Run comprehensive benchmarks"
	@echo "  analyze     - Run analysis and generate visualizations"
	@echo "  objective3  - Run Objective 3 (Comprehensive Benchmarking)"
	@echo "  objective4  - Run Objective 4 (Performance Comparison)"
	@echo "  notebook    - Start Jupyter notebook for interactive analysis"
	@echo "  clean       - Clean generated files"
	@echo "  results     - Show benchmark results summary"
	@echo "  help        - Show this help message"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests completed"

# Run comprehensive benchmarks
benchmark:
	@echo "🚀 Running comprehensive benchmarks..."
	python run_benchmarks.py
	@echo "✅ Benchmarks completed"

# Run analysis and generate visualizations
analyze:
	@echo "📊 Running analysis and generating visualizations..."
	python -c "
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from python_orchestrator.benchmarking import ComprehensiveBenchmarker
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

benchmarker = ComprehensiveBenchmarker(config)
benchmarker.generate_visualizations()
print('✅ Analysis completed')
"
	@echo "✅ Analysis completed"

# Run Objective 3 (Comprehensive Benchmarking)
objective3:
	@echo "🎯 Running Objective 3: Comprehensive Benchmarking"
	python src/python_orchestrator/main.py --objective 3
	@echo "✅ Objective 3 completed"

# Run Objective 4 (Performance Comparison)
objective4:
	@echo "🎯 Running Objective 4: Performance Comparison"
	python src/python_orchestrator/main.py --objective 4
	@echo "✅ Objective 4 completed"

# Start Jupyter notebook
notebook:
	@echo "📓 Starting Jupyter notebook..."
	jupyter notebook notebooks/benchmark_analysis.ipynb

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf results/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/python_orchestrator/__pycache__/
	rm -rf src/real_world/__pycache__/
	rm -rf src/reporting/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf tests/unit/__pycache__/
	rm -rf tests/benchmarks/__pycache__/
	rm -rf tests/integration/__pycache__/
	rm -f *.log
	@echo "✅ Clean completed"

# Show benchmark results summary
results:
	@echo "📊 Benchmark Results Summary"
	@echo "============================"
	@if [ -f "results/benchmark_results.json" ]; then \
		echo "✅ Benchmark results found"; \
		echo "📁 Results directory: results/"; \
		echo "📄 Files generated:"; \
		ls -la results/ 2>/dev/null || echo "   No results directory found"; \
	else \
		echo "❌ No benchmark results found"; \
		echo "💡 Run 'make benchmark' to generate results"; \
	fi

# Run full pipeline (install, test, benchmark, analyze)
all: install test benchmark analyze results

# Quick benchmark (minimal iterations for testing)
quick-benchmark:
	@echo "⚡ Running quick benchmark (reduced iterations)..."
	@cp config.yaml config.yaml.backup
	@sed -i 's/iterations: 1000/iterations: 10/g' config.yaml
	@sed -i 's/duration_seconds: 300/duration_seconds: 10/g' config.yaml
	python run_benchmarks.py
	@mv config.yaml.backup config.yaml
	@echo "✅ Quick benchmark completed"

# Show system information
info:
	@echo "🔍 System Information"
	@echo "===================="
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "Installed packages:"
	@pip list | grep -E "(pandas|numpy|matplotlib|plotly|scipy)"
	@echo ""
	@echo "Available algorithms:"
	@python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
classical = [algo['name'] for algo in config['algorithms']['classical']]
pqc = [algo['name'] for algo in config['algorithms']['pqc']]
print(f'Classical: {len(classical)} algorithms')
print(f'PQC: {len(pqc)} algorithms')
print(f'Total: {len(classical) + len(pqc)} algorithms')
"

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -f docker/Dockerfile -t pqc-benchmark .

docker-run:
	@echo "🐳 Running benchmarks in Docker..."
	docker run -v $(PWD)/results:/app/results pqc-benchmark

# Kubernetes targets
k8s-deploy:
	@echo "☸️  Deploying to Kubernetes..."
	kubectl apply -f k8s/

k8s-status:
	@echo "☸️  Kubernetes deployment status:"
	kubectl get pods -n quantumresilient

k8s-logs:
	@echo "☸️  Kubernetes logs:"
	kubectl logs -f deployment/quantumresilient -n quantumresilient

# Development targets
format:
	@echo "🎨 Formatting code..."
	black src/ tests/
	flake8 src/ tests/

lint:
	@echo "🔍 Running linter..."
	flake8 src/ tests/
	mypy src/

# Documentation
docs:
	@echo "📚 Building documentation..."
	cd docs && make html
	@echo "✅ Documentation built in docs/_build/html/"

# Performance profiling
profile:
	@echo "📈 Running performance profiling..."
	python -m cProfile -o profile.stats run_benchmarks.py
	@echo "✅ Profiling completed. Use 'python -c \"import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)\"' to view results"

# Memory profiling
memory-profile:
	@echo "🧠 Running memory profiling..."
	python -m memory_profiler run_benchmarks.py
	@echo "✅ Memory profiling completed"
