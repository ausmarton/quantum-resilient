"""
Real-world Integration Package

This package contains real-world AML transaction processing simulation and integration
components for the QuantumResilient Framework.
"""

from .aml_pipeline import RealWorldAMLPipeline, AMLDataGenerator, AMLTransaction

__all__ = ["RealWorldAMLPipeline", "AMLDataGenerator", "AMLTransaction"]
