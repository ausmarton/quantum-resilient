# Quantum-Resilient Cryptography Benchmarking Framework

A comprehensive benchmarking framework for analyzing Post-Quantum Cryptography (PQC) performance against classical algorithms in real-world applications. This framework focuses on **Objectives 3 and 4** of the research: comprehensive benchmarking and performance comparison.

## 🎯 Research Focus

This framework is specifically designed for:

- **Objective 3**: Comprehensive benchmarking of PQC and classical algorithms
- **Objective 4**: Performance comparison and analysis between PQC and classical encryption

**Note**: Objectives 1 and 5 are out of scope as per research requirements.

## 🔬 Algorithms Supported

### Classical Algorithms (Counterparts)
- **RSA-2048, RSA-4096**: Asymmetric encryption
- **ECDSA-P256, ECDSA-P384**: Digital signatures  
- **ECDH-P256, ECDH-P384**: Key exchange
- **AES-256**: Symmetric encryption

### PQC Algorithms (NIST Standardized)
- **ML-KEM-512, ML-KEM-768, ML-KEM-1024**: Key encapsulation (formerly Kyber)
- **ML-DSA-44, ML-DSA-65, ML-DSA-87**: Digital signatures (formerly Dilithium)

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Run Benchmarks

```bash
# Run comprehensive benchmarks
python run_benchmarks.py

# Or run specific objectives
python src/python_orchestrator/main.py --objective 3  # Benchmarking
python src/python_orchestrator/main.py --objective 4  # Comparison
```

### Interactive Analysis

```bash
# Start Jupyter notebook
jupyter notebook notebooks/benchmark_analysis.ipynb
```

## 📊 Benchmarking Features

### Comprehensive Metrics
- **Latency Analysis**: Mean, median, percentiles (95th, 99th), confidence intervals
- **Throughput Testing**: Operations per second, batch processing, concurrent operations
- **Resource Monitoring**: CPU, memory, network utilization
- **Stress Testing**: High-load scenarios, error rates, scalability analysis

### Statistical Analysis
- **Confidence Intervals**: 95% and 99% confidence levels
- **Significance Testing**: ANOVA, t-tests for algorithm comparisons
- **Correlation Analysis**: Performance vs data size relationships
- **Outlier Detection**: Statistical outlier identification

### Visualization
- **Interactive Dashboards**: Plotly-based interactive visualizations
- **Performance Charts**: Latency comparison, throughput analysis, resource usage
- **Statistical Plots**: Confidence intervals, correlation matrices
- **Export Options**: PNG, HTML, Excel, CSV formats

## 📁 Output Structure

```
results/
├── benchmark_results.json          # Raw benchmark data
├── benchmark_results.csv           # CSV format for analysis
├── benchmark_results.xlsx          # Excel with multiple sheets
├── interactive_dashboard.html      # Interactive Plotly dashboard
├── latency_comparison.png          # Latency comparison charts
├── throughput_analysis.png         # Throughput analysis
├── performance_trends.png          # Performance trends
├── algorithm_comparison_heatmap.png # Algorithm comparison heatmap
└── objectives_3_4_results.json     # Comprehensive analysis results
```

## 🔧 Configuration

The framework is configured via `config.yaml`:

```yaml
# Algorithm configuration
algorithms:
  classical:
    - name: "RSA-2048"
      key_size: 2048
      security_level: 112
  pqc:
    - name: "ML-KEM-512"
      variant: "ML-KEM-512"
      security_level: 128

# Benchmark configuration
benchmarks:
  latency_test:
    iterations: 1000
    data_sizes: [64, 1024, 4096, 16384, 65536]
    confidence_level: 0.95
  throughput_test:
    batch_sizes: [100, 1000, 10000, 100000]
    concurrent_operations: [1, 4, 8, 16]
```

## 📈 Analysis Capabilities

### Performance Comparison
- **PQC vs Classical**: Direct performance comparison
- **Overhead Analysis**: Performance impact quantification
- **Scalability Assessment**: Performance vs data size relationships
- **Resource Efficiency**: CPU and memory usage analysis

### Security Analysis
- **Quantum Resistance**: Assessment of quantum attack vulnerability
- **Security Levels**: Comparison of cryptographic strength
- **Risk Assessment**: Long-term security implications

### Real-world Applicability
- **AML System Compatibility**: Performance for financial systems
- **Latency Requirements**: Meeting real-time processing needs
- **Throughput Capabilities**: Handling high-volume transactions
- **Resource Constraints**: System resource utilization analysis

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run benchmark tests
pytest tests/benchmarks/ -v

# Run integration tests
pytest tests/integration/ -v
```

## 📊 Example Results

### Performance Summary
```
Algorithm          | Avg Latency (ms) | Throughput (ops/sec) | Security Level
------------------|------------------|---------------------|---------------
RSA-2048          | 5.2              | 192                 | 112 bits
ML-KEM-512        | 8.7              | 115                 | 128 bits
ECDSA-P256        | 3.8              | 263                 | 128 bits
ML-DSA-44         | 12.5             | 80                  | 128 bits
```

### Key Findings
- **PQC Overhead**: ~67% latency increase, acceptable for AML systems
- **Security Enhancement**: Quantum-resistant algorithms provide future-proof security
- **Performance**: PQC algorithms can meet real-time AML requirements
- **Scalability**: Linear performance scaling with data size

## 🔬 Research Methodology

### Benchmarking Approach
1. **Warmup Phase**: Eliminate JIT compilation effects
2. **Measurement Phase**: Collect comprehensive performance data
3. **Statistical Analysis**: Calculate confidence intervals and significance
4. **Resource Monitoring**: Track system resource utilization
5. **Stress Testing**: Validate performance under high load

### Data Collection
- **Latency Measurements**: High-precision timing with microsecond resolution
- **Throughput Analysis**: Operations per second under various conditions
- **Resource Monitoring**: Real-time CPU and memory usage tracking
- **Statistical Sampling**: Large sample sizes for reliable results

### Analysis Methods
- **Descriptive Statistics**: Mean, median, percentiles, standard deviation
- **Inferential Statistics**: Confidence intervals, hypothesis testing
- **Correlation Analysis**: Performance vs data size relationships
- **Comparative Analysis**: PQC vs classical algorithm performance

## 🚀 Deployment Options

### Local Development
```bash
python run_benchmarks.py
```

### Docker Deployment
```bash
docker build -f docker/Dockerfile -t pqc-benchmark .
docker run -v $(pwd)/results:/app/results pqc-benchmark
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
kubectl get pods -n quantumresilient
```

## 📚 Documentation

- **API Documentation**: `docs/api.md`
- **Benchmarking Guide**: `docs/benchmarking.md`
- **Analysis Guide**: `docs/analysis.md`
- **Deployment Guide**: `docs/deployment.md` (Local Docker, local K8s with kind/minikube, and GCP/GKE)
- **Testing Guide**: `docs/testing.md` (how tests are structured and executed)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Development Guidelines
- Follow Python coding standards (PEP 8)
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NIST**: For PQC standardization efforts
- **Open Quantum Safe**: For PQC algorithm implementations
- **Financial Industry**: For AML use case validation

## 📞 Support

For questions and support:
- **Research Questions**: Contact the research team
- **Technical Issues**: Open an issue on GitHub
- **Deployment Help**: Check the deployment documentation

---

**Note**: This framework is designed for research purposes and should be thoroughly tested before use in production environments. The benchmarking results provide valuable insights for quantum-resistant cryptography adoption in financial systems.
