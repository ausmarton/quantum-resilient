# QuantumResilient Performance Research Framework

A comprehensive research framework for analyzing Quantum-Resilient Cryptography performance in Anti-Money Laundering (AML) systems. This hybrid Rust+Python framework implements all research objectives with real-world applicability.

## 🎯 Research Objectives

This framework addresses five key research objectives:

1. **Algorithm Selection Criteria** - Establish criteria for selecting PQC algorithms relevant to real-time data streaming
2. **Modular Framework Development** - Develop a modular framework for real-time data streaming systems
3. **Comprehensive Benchmarking** - Benchmark selected PQC algorithms against classical encryption
4. **Performance Comparison** - Compare PQC performance against existing classical encryption techniques
5. **Engineering Recommendations** - Provide engineering recommendations for optimized quantum-resilient real-time data pipelines

## 🏗️ Architecture

### Hybrid Rust + Python Design

- **Rust Core**: High-performance cryptographic operations and system metrics
- **Python Orchestrator**: Research framework coordination and analysis
- **Real-world Integration**: AML transaction processing simulation
- **Comprehensive Reporting**: Automated report generation and visualization

### Key Components

```
quantumresilient/
├── src/
│   ├── rust_core/           # Rust cryptographic core
│   ├── python_orchestrator/ # Python research framework
│   ├── real_world/          # AML pipeline simulation
│   └── reporting/           # Report generation
├── docker/                  # Containerization
├── k8s/                     # Kubernetes deployment
├── tests/                   # Comprehensive testing
└── results/                 # Research outputs
```

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+
- Python 3.11+
- Docker
- Kubernetes (optional)

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd quantumresilient

# Install Python dependencies
pip install -r requirements.txt

# Build Rust core (optional - mock implementations available)
cd src/rust_core
cargo build --release
cd ../..

# Run the research framework
python src/python_orchestrator/main.py
```

### Docker Deployment

```bash
# Build and run with Docker
docker build -f docker/Dockerfile -t quantumresilient .
docker run -v $(pwd)/results:/app/results quantumresilient

# Or use docker-compose
docker-compose -f docker/docker-compose.yml up
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n quantumresilient
kubectl logs -f deployment/quantumresilient -n quantumresilient
```

## 📊 Research Framework

### Objective 1: Algorithm Selection

```python
# Run algorithm selection analysis
python src/python_orchestrator/main.py --objective 1
```

**Deliverables:**
- Security analysis of PQC algorithms
- Performance characteristics assessment
- Implementation maturity evaluation
- Algorithm selection recommendations

### Objective 2: Framework Development

```python
# Run framework development tests
python src/python_orchestrator/main.py --objective 2
```

**Deliverables:**
- Modular framework for real-time data streaming
- Integration tests with AML systems
- Performance validation
- Framework compatibility assessment

### Objective 3: Comprehensive Benchmarking

```python
# Run comprehensive benchmarks
python src/python_orchestrator/main.py --objective 3
```

**Deliverables:**
- Latency benchmarks across data sizes
- Throughput analysis
- Resource utilization metrics
- Statistical analysis with confidence intervals

### Objective 4: Performance Comparison

```python
# Run comparative analysis
python src/python_orchestrator/main.py --objective 4
```

**Deliverables:**
- PQC vs Classical performance comparison
- Security characteristics analysis
- Cost implications assessment
- Migration feasibility analysis

### Objective 5: Engineering Recommendations

```python
# Generate engineering recommendations
python src/python_orchestrator/main.py --objective 5
```

**Deliverables:**
- Algorithm selection recommendations
- Implementation strategy guidance
- Performance optimization strategies
- Migration planning framework
- Risk mitigation strategies

## 🔬 AML Integration

The framework includes real-world AML transaction processing simulation:

```python
# Run AML simulation
python src/real_world/aml_pipeline.py
```

**Features:**
- Realistic transaction data generation
- Risk assessment integration
- Compliance checking
- Performance monitoring
- Regulatory reporting

## 📈 Reporting and Visualization

### Automated Report Generation

```python
# Generate comprehensive reports
python src/reporting/research_reporter.py
```

**Report Types:**
- Individual objective reports
- Executive summary
- Technical documentation
- Performance visualizations
- Statistical analysis

### Visualization Examples

- Latency comparison charts
- Throughput analysis graphs
- Resource utilization plots
- Statistical significance testing
- Confidence interval analysis

## 🧪 Testing

### Unit Tests

```bash
# Run unit tests
python -m pytest tests/unit/ -v
```

### Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/ -v
```

### Benchmark Tests

```bash
# Run benchmark tests
python -m pytest tests/benchmarks/ -v
```

## 🐳 Deployment Options

### Local Development

```bash
# Run locally with Python
python src/python_orchestrator/main.py --objective 1

# Run with Docker
docker run -v $(pwd)/results:/app/results pqc-research
```

### Cloud Deployment (GCP)

```bash
# Deploy to GKE
kubectl apply -f k8s/gcp/

# Monitor deployment
kubectl get pods -n pqc-research
kubectl logs -f deployment/pqc-research -n pqc-research
```

### Production Scaling

```bash
# Scale deployment
kubectl scale deployment pqc-research --replicas=5 -n pqc-research

# Monitor performance
kubectl top pods -n pqc-research
```

## 📋 Configuration

### Algorithm Configuration

```yaml
# config/algorithms.yaml
classical:
  - name: RSA-2048
    key_size: 2048
  - name: AES-256
    key_size: 256
pqc:
  - name: Kyber512
    variant: Kyber512
  - name: Dilithium2
    variant: Dilithium2
```

### Benchmark Configuration

```yaml
# config/benchmarks.yaml
latency_test:
  iterations: 1000
  data_sizes: [64, 1024, 4096, 16384]
throughput_test:
  batch_sizes: [100, 1000, 10000]
  duration_seconds: 300
```

## 📊 Results and Outputs

### Research Results

All results are saved in structured JSON format:

```json
{
  "objective_1": {
    "security_analysis": {...},
    "performance_analysis": {...},
    "recommendations": {...}
  },
  "objective_2": {
    "framework_components": {...},
    "integration_tests": {...}
  },
  ...
}
```

### Generated Reports

- `reports/objective_1_report.json` - Algorithm selection analysis
- `reports/objective_2_report.json` - Framework development report
- `reports/objective_3_report.json` - Benchmarking results
- `reports/objective_4_report.json` - Comparative analysis
- `reports/objective_5_report.json` - Engineering recommendations
- `reports/executive_summary_report.json` - Executive summary
- `reports/technical_report.json` - Comprehensive technical report

### Visualizations

- `reports/benchmark_visualization.png` - Performance comparison charts
- `reports/comparison_visualization.png` - Classical vs PQC analysis

## 🔧 Development

### Adding New Algorithms

1. **Rust Core**: Add algorithm implementation in `src/rust_core/src/lib.rs`
2. **Python Integration**: Update algorithm registry in `src/python_orchestrator/main.py`
3. **Testing**: Add tests in `tests/unit/` and `tests/benchmarks/`

### Extending Benchmarks

1. **Add Benchmark Type**: Implement in `src/python_orchestrator/main.py`
2. **Update Configuration**: Add to `config/benchmarks.yaml`
3. **Generate Reports**: Update `src/reporting/research_reporter.py`

### Custom AML Integration

1. **Transaction Types**: Extend `AMLTransaction` in `src/real_world/aml_pipeline.py`
2. **Risk Models**: Implement custom risk assessment logic
3. **Compliance Rules**: Add regulatory compliance checks

## 📚 Documentation

### API Documentation

- **Rust Core API**: `docs/rust_core_api.md`
- **Python Framework API**: `docs/python_framework_api.md`
- **AML Integration API**: `docs/aml_integration_api.md`

### Deployment Guides

- **Local Development**: `docs/local_development.md`
- **Docker Deployment**: `docs/docker_deployment.md`
- **Kubernetes Deployment**: `docs/kubernetes_deployment.md`
- **GCP Production**: `docs/gcp_production.md`

### Research Methodology

- **Experimental Design**: `docs/experimental_design.md`
- **Statistical Analysis**: `docs/statistical_analysis.md`
- **Performance Metrics**: `docs/performance_metrics.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Development Guidelines

- Follow Rust and Python coding standards
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Open Quantum Safe (OQS) project for PQC implementations
- NIST for PQC standardization efforts
- Financial industry partners for AML use case validation

## 📞 Support

For questions and support:

- **Research Questions**: Contact the research team
- **Technical Issues**: Open an issue on GitHub
- **Deployment Help**: Check the deployment documentation

---

**Note**: This framework is designed for research purposes and should be thoroughly tested before use in production environments.
