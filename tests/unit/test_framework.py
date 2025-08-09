"""
Unit tests for the QuantumResilient Framework
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the framework components
try:
    from src.python_orchestrator.main import QuantumResilientFramework
    from src.real_world.aml_pipeline import RealWorldAMLPipeline, AMLDataGenerator
    from src.reporting.research_reporter import ResearchReporter
except ImportError:
    pytest.skip("Framework components not available", allow_module_level=True)


class TestQuantumResilientFramework:
    """Test cases for the main research framework"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            'log_level': 'INFO',
            'output_directory': 'test_results',
            'benchmark_iterations': 10,
            'data_sizes': [64, 1024]
        }
        self.framework = QuantumResilientFramework(self.config)
    
    def test_framework_initialization(self):
        """Test that the framework initializes correctly"""
        assert self.framework is not None
        assert hasattr(self.framework, 'algorithms')
        assert len(self.framework.algorithms) == 4  # RSA-2048, AES-256, Kyber512, Dilithium2
    
    def test_algorithm_analysis(self):
        """Test algorithm analysis methods"""
        # Test security analysis
        security_result = self.framework._analyze_security('RSA-2048', Mock())
        assert security_result['algorithm_type'] == 'Classical'
        assert security_result['quantum_resistance'] is False
        
        # Test performance analysis
        performance_result = self.framework._analyze_performance('Kyber512', Mock())
        assert 'avg_latency_ms' in performance_result
        assert 'throughput_ops_per_sec' in performance_result
    
    def test_algorithm_recommendations(self):
        """Test algorithm recommendation generation"""
        analysis_results = {
            'performance_analysis': {
                'RSA-2048': {'avg_latency_ms': 3.0},
                'Kyber512': {'avg_latency_ms': 8.0}
            }
        }
        recommendations = self.framework._generate_algorithm_recommendations(analysis_results)
        assert 'primary_recommendations' in recommendations
        assert 'secondary_recommendations' in recommendations
        assert 'risk_assessment' in recommendations
    
    # Objective 1 is out of scope in the updated framework; test Objective 3 path exists
    @pytest.mark.asyncio
    async def test_objective_3_execution(self):
        """Test Objective 3 execution"""
        result = await self.framework.run_objective_3_benchmarking()
        assert 'latency_benchmarks' in result
        assert 'throughput_benchmarks' in result
        assert 'resource_benchmarks' in result
    
    def test_synthetic_transaction_generation(self):
        """Test synthetic transaction generation"""
        transactions = self.framework._generate_synthetic_transactions(5)
        assert len(transactions) == 5
        for tx in transactions:
            assert hasattr(tx, 'id')
            assert hasattr(tx, 'amount')
            assert hasattr(tx, 'sender')
            assert hasattr(tx, 'recipient')


class TestRealWorldAMLPipeline:
    """Test cases for the real-world AML pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = {
            'risk_threshold': 0.7,
            'max_latency_ms': 100,
            'batch_size': 1
        }
        self.pipeline = RealWorldAMLPipeline('Kyber512', self.config)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline is not None
        assert self.pipeline.crypto_algorithm == 'Kyber512'
        assert self.pipeline.risk_threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_transaction_processing(self):
        """Test transaction processing"""
        from src.real_world.aml_pipeline import AMLTransaction
        
        # Create a test transaction
        tx = AMLTransaction(
            transaction_id="TEST_TX_001",
            timestamp=1234567890.0,
            amount=1000.0,
            currency="USD",
            sender_account="ACC_001",
            recipient_account="ACC_002",
            sender_country="US",
            recipient_country="UK",
            transaction_type="transfer",
            risk_indicators={'high_value': False, 'cross_border': True},
            customer_risk_score=0.3,
            regulatory_flags=[]
        )
        
        result = await self.pipeline.process_transaction(tx)
        assert 'processing_time_ms' in result
        assert 'risk_score' in result
        assert 'alert_generated' in result
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        metrics = self.pipeline.get_performance_metrics()
        # Should return empty dict if no transactions processed
        assert isinstance(metrics, dict)


class TestAMLDataGenerator:
    """Test cases for AML data generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = {'risk_threshold': 0.7}
        self.generator = AMLDataGenerator(self.config)
    
    def test_transaction_generation(self):
        """Test transaction generation"""
        tx = self.generator.generate_transaction()
        assert hasattr(tx, 'transaction_id')
        assert hasattr(tx, 'amount')
        assert hasattr(tx, 'currency')
        assert hasattr(tx, 'risk_indicators')
        assert hasattr(tx, 'customer_risk_score')
    
    def test_risk_indicator_generation(self):
        """Test risk indicator generation"""
        risk_score = 0.8
        indicators = self.generator._generate_risk_indicators(risk_score)
        assert isinstance(indicators, dict)
        assert 'high_value' in indicators
        assert 'cross_border' in indicators
        assert 'structured_transactions' in indicators


class TestResearchReporter:
    """Test cases for the research reporter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sample_results = {
            'objective_1': {
                'security_analysis': {'RSA-2048': {'algorithm_type': 'Classical'}},
                'performance_analysis': {'RSA-2048': {'avg_latency_ms': 5.0}},
                'recommendations': {'primary_recommendations': ['AES-256']}
            },
            'objective_2': {
                'framework_components': {'RSA-2048': {'framework_compatibility': True}}
            }
        }
        self.reporter = ResearchReporter(self.sample_results, 'test_reports')
    
    def test_reporter_initialization(self):
        """Test reporter initialization"""
        assert self.reporter is not None
        assert self.reporter.results == self.sample_results
    
    def test_report_generation(self):
        """Test report generation"""
        reports = self.reporter.generate_all_reports()
        assert 'objective_1' in reports
        assert 'objective_2' in reports
        assert 'executive_summary' in reports
        assert 'technical_report' in reports
    
    def test_executive_summary_generation(self):
        """Test executive summary generation"""
        summary = self.reporter._generate_executive_summary()
        assert 'title' in summary
        assert 'research_objectives' in summary
        assert 'key_findings' in summary
        assert 'recommendations' in summary


if __name__ == "__main__":
    pytest.main([__file__])
