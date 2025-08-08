.PHONY: help install test lint format docs build clean run-all run-objective-1 run-objective-2 run-objective-3 run-objective-4 run-objective-5 docker-build docker-run k8s-deploy

# Default target
help:
	@echo "QuantumResilient Framework - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  docs        - Build documentation"
	@echo "  build       - Build Rust core"
	@echo "  clean       - Clean build artifacts"
	@echo ""
	@echo "Research Framework:"
	@echo "  run-all     - Run all research objectives"
	@echo "  run-objective-1 - Run algorithm selection analysis"
	@echo "  run-objective-2 - Run framework development"
	@echo "  run-objective-3 - Run comprehensive benchmarking"
	@echo "  run-objective-4 - Run performance comparison"
	@echo "  run-objective-5 - Generate engineering recommendations"
	@echo ""
	@echo "Deployment:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker"
	@echo "  k8s-deploy   - Deploy to Kubernetes"

# Install dependencies
install:
	pip install -e ".[dev,docs]"
	cd src/rust_core && cargo build --release

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Build documentation
docs:
	cd docs && make html

# Build Rust core
build:
	cd src/rust_core && cargo build --release

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	cd src/rust_core && cargo clean
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/

# Run all research objectives
run-all:
	python src/python_orchestrator/main.py --output-dir results --log-level INFO

# Run individual objectives
run-objective-1:
	python src/python_orchestrator/main.py --objective 1 --output-dir results --log-level INFO

run-objective-2:
	python src/python_orchestrator/main.py --objective 2 --output-dir results --log-level INFO

run-objective-3:
	python src/python_orchestrator/main.py --objective 3 --output-dir results --log-level INFO

run-objective-4:
	python src/python_orchestrator/main.py --objective 4 --output-dir results --log-level INFO

run-objective-5:
	python src/python_orchestrator/main.py --objective 5 --output-dir results --log-level INFO

# Docker commands
docker-build:
	docker build -f docker/Dockerfile -t quantumresilient .

docker-run:
	docker run -v $(PWD)/results:/app/results quantumresilient

# Kubernetes deployment
k8s-deploy:
	kubectl apply -f k8s/

# Development setup
dev-setup: install build test lint

# Full research run
research-run: run-all
	python src/reporting/research_reporter.py

# Quick test run
quick-test:
	python src/python_orchestrator/main.py --objective 1 --output-dir test_results --log-level WARNING
