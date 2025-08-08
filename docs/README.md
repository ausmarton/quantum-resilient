# QuantumResilient Framework Documentation

This directory contains comprehensive documentation for the QuantumResilient Framework.

## Documentation Structure

- **Architecture Documentation**: Complete system architecture and design
- **API Documentation**: Detailed API reference for all components
- **Deployment Guides**: Step-by-step deployment instructions
- **Research Methodology**: Experimental design and statistical analysis
- **User Guides**: Tutorials and usage examples

## Core Documentation

### Architecture Documentation
- **`ARCHITECTURE.md`** - Comprehensive system architecture and design documentation
  - Hybrid Rust+Python architecture overview
  - Component interactions and data flow
  - Deployment strategies (Local, Docker, Kubernetes)
  - Security and scalability considerations
  - Technical implementation details

## Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Documentation Sections

### Architecture Documentation
- `ARCHITECTURE.md` - Complete system architecture and design

### API Documentation
- `rust_core_api.md` - Rust core API reference
- `python_framework_api.md` - Python framework API reference
- `aml_integration_api.md` - AML integration API reference

### Deployment Guides
- `local_development.md` - Local development setup
- `docker_deployment.md` - Docker deployment guide
- `kubernetes_deployment.md` - Kubernetes deployment guide
- `gcp_production.md` - GCP production deployment

### Research Methodology
- `experimental_design.md` - Experimental design documentation
- `statistical_analysis.md` - Statistical analysis methods
- `performance_metrics.md` - Performance metrics documentation

## Contributing to Documentation

1. Follow the existing documentation structure
2. Use clear, concise language
3. Include code examples where appropriate
4. Update this README when adding new documentation
5. Ensure architecture documentation stays current with implementation
