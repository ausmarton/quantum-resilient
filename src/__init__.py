"""
QuantumResilient Framework

A comprehensive research framework for analyzing Quantum-Resilient Cryptography 
performance in Anti-Money Laundering (AML) systems.

This package provides:
- Hybrid Rust+Python architecture for high-performance cryptographic operations
- Comprehensive benchmarking of quantum-resilient algorithms
- Real-world AML transaction processing simulation
- Automated reporting and analysis capabilities
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Import main components for easy access
try:
    from .python_orchestrator.main import PQCResearchFramework
    from .real_world.aml_pipeline import RealWorldAMLPipeline, AMLDataGenerator
    from .reporting.research_reporter import ResearchReporter
except ImportError:
    # Handle case where dependencies are not available
    pass

__all__ = [
    "PQCResearchFramework",
    "RealWorldAMLPipeline", 
    "AMLDataGenerator",
    "ResearchReporter",
]
