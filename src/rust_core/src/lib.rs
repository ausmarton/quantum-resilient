use pyo3::prelude::*;
use std::time::Instant;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use sysinfo::{System, SystemExt, CpuExt};
use chrono::Utc;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PerformanceMetrics {
    pub latency_ms: f64,
    pub memory_delta_bytes: i64,
    pub cpu_percent: f64,
    pub timestamp: f64,
    pub operation_name: String,
    pub data_size: usize,
    pub algorithm_name: String,
}

#[pyclass]
pub struct CryptoBenchmark {
    metrics: HashMap<String, Vec<PerformanceMetrics>>,
    algorithm_name: String,
    key_size: u32,
    system: System,
}

#[pymethods]
impl CryptoBenchmark {
    #[new]
    fn new(algorithm_name: String, key_size: u32) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        CryptoBenchmark {
            metrics: HashMap::new(),
            algorithm_name,
            key_size,
            system,
        }
    }

    fn measure_operation(&mut self, operation_name: &str, data_size: usize) -> PyResult<f64> {
        let start_time = Instant::now();
        let start_memory = self.get_memory_usage();
        let start_cpu = self.get_cpu_usage();

        // Simulate cryptographic operation based on algorithm and data size
        self.simulate_crypto_operation(operation_name, data_size);

        let end_time = Instant::now();
        let end_memory = self.get_memory_usage();
        let end_cpu = self.get_cpu_usage();

        let latency = (end_time - start_time).as_millis() as f64;
        let memory_delta = end_memory - start_memory;
        let cpu_usage = (start_cpu + end_cpu) / 2.0;

        let metrics = PerformanceMetrics {
            latency_ms: latency,
            memory_delta_bytes: memory_delta,
            cpu_percent: cpu_usage,
            timestamp: start_time.elapsed().as_secs_f64(),
            operation_name: operation_name.to_string(),
            data_size,
            algorithm_name: self.algorithm_name.clone(),
        };

        self.metrics.entry(operation_name.to_string())
            .or_insert_with(Vec::new)
            .push(metrics);

        Ok(latency)
    }

    fn get_metrics_count(&self) -> PyResult<usize> {
        let total = self.metrics.values().map(|v| v.len()).sum();
        Ok(total)
    }

    fn run_benchmark(&mut self, iterations: usize, data_sizes: Vec<usize>) -> PyResult<f64> {
        let mut total_latency = 0.0;
        let mut total_operations = 0;
        
        for size in data_sizes {
            for i in 0..iterations {
                if i % 100 == 0 {
                    println!("Progress: {}/{} for size {}", i, iterations, size);
                }
                
                total_latency += self.measure_operation("encryption", size)?;
                total_latency += self.measure_operation("decryption", size)?;
                total_operations += 2;
            }
        }
        
        Ok(total_latency / total_operations as f64)
    }

    fn get_statistics(&self) -> PyResult<HashMap<String, f64>> {
        let mut stats = HashMap::new();
        
        for (operation, metrics) in &self.metrics {
            if !metrics.is_empty() {
                let latencies: Vec<f64> = metrics.iter().map(|m| m.latency_ms).collect();
                let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
                let min = latencies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = latencies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                
                stats.insert(format!("{}_mean_ms", operation), mean);
                stats.insert(format!("{}_min_ms", operation), min);
                stats.insert(format!("{}_max_ms", operation), max);
                stats.insert(format!("{}_count", operation), metrics.len() as f64);
            }
        }
        
        Ok(stats)
    }

    fn simulate_crypto_operation(&self, operation_name: &str, data_size: usize) {
        let base_duration = match self.algorithm_name.as_str() {
            "RSA-2048" => match operation_name {
                "key_generation" => 100,
                "encryption" => 5 + (data_size / 100),
                "decryption" => 10 + (data_size / 50),
                _ => 1,
            },
            "AES-256" => match operation_name {
                "key_generation" => 1,
                "encryption" => 2 + (data_size / 1000),
                "decryption" => 2 + (data_size / 1000),
                _ => 1,
            },
            "Kyber512" => match operation_name {
                "key_generation" => 50,
                "encryption" => 8 + (data_size / 200),
                "decryption" => 12 + (data_size / 150),
                _ => 1,
            },
            "Dilithium2" => match operation_name {
                "key_generation" => 80,
                "signing" => 15 + (data_size / 100),
                "verification" => 20 + (data_size / 80),
                _ => 1,
            },
            _ => 1,
        };
        
        std::thread::sleep(std::time::Duration::from_micros(base_duration.try_into().unwrap()));
    }

    fn get_memory_usage(&mut self) -> i64 {
        self.system.refresh_memory();
        self.system.used_memory() as i64
    }

    fn get_cpu_usage(&mut self) -> f64 {
        self.system.refresh_cpu();
        self.system.global_cpu_info().cpu_usage().into()
    }
}

#[pyclass]
pub struct PipelineSimulator {
    crypto_benchmark: CryptoBenchmark,
    transactions_processed: u64,
    processing_times: Vec<f64>,
    system: System,
}

#[pymethods]
impl PipelineSimulator {
    #[new]
    fn new(algorithm_name: String, key_size: u32) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        PipelineSimulator {
            crypto_benchmark: CryptoBenchmark::new(algorithm_name, key_size),
            transactions_processed: 0,
            processing_times: Vec::new(),
            system,
        }
    }

    fn process_transaction(&mut self, transaction_id: String, amount: f64, sender: String, recipient: String) -> PyResult<f64> {
        let start_time = Instant::now();
        
        // Simulate transaction processing with crypto operations
        let tx_data_size = transaction_id.len() + sender.len() + recipient.len();
        
        self.crypto_benchmark.measure_operation("transaction_encryption", tx_data_size)?;
        self.crypto_benchmark.measure_operation("risk_assessment", 1024)?;
        
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.processing_times.push(processing_time);
        self.transactions_processed += 1;

        Ok(processing_time)
    }

    fn get_statistics(&self) -> PyResult<HashMap<String, f64>> {
        if self.processing_times.is_empty() {
            return Ok(HashMap::new());
        }

        let sum: f64 = self.processing_times.iter().sum();
        let mean = sum / self.processing_times.len() as f64;
        let min = self.processing_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.processing_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut stats = HashMap::new();
        stats.insert("mean_processing_time_ms".to_string(), mean);
        stats.insert("min_processing_time_ms".to_string(), min);
        stats.insert("max_processing_time_ms".to_string(), max);
        stats.insert("total_transactions".to_string(), self.transactions_processed as f64);
        stats.insert("throughput_tps".to_string(), self.transactions_processed as f64 / (sum / 1000.0));

        Ok(stats)
    }

    fn get_system_metrics(&mut self) -> PyResult<HashMap<String, f64>> {
        self.system.refresh_all();
        
        let mut metrics = HashMap::new();
        metrics.insert("cpu_percent".to_string(), self.system.global_cpu_info().cpu_usage().into());
        metrics.insert("memory_percent".to_string(), (self.system.used_memory() as f64 / self.system.total_memory() as f64) * 100.0);
        metrics.insert("memory_used_bytes".to_string(), self.system.used_memory() as f64 * 1024.0);
        metrics.insert("memory_total_bytes".to_string(), self.system.total_memory() as f64 * 1024.0);
        metrics.insert("timestamp".to_string(), Utc::now().timestamp_millis() as f64);
        
        Ok(metrics)
    }
}

#[pyclass]
pub struct OQSCrypto {
    algorithm_name: String,
    key_size: u32,
    metrics: HashMap<String, Vec<PerformanceMetrics>>,
}

#[pymethods]
impl OQSCrypto {
    #[new]
    fn new(algorithm_name: String) -> Self {
        let key_size = match algorithm_name.as_str() {
            "Kyber512" => 512,
            "Kyber768" => 768,
            "Kyber1024" => 1024,
            "Dilithium2" => 256,
            "Dilithium3" => 384,
            "Dilithium5" => 512,
            _ => 256,
        };
        
        OQSCrypto {
            algorithm_name,
            key_size,
            metrics: HashMap::new(),
        }
    }

    fn generate_keys(&mut self) -> PyResult<(Vec<u8>, Vec<u8>)> {
        let start_time = Instant::now();
        
        // Simulate PQC key generation with realistic characteristics
        let public_key_size = match self.algorithm_name.as_str() {
            "Kyber512" => 800,
            "Kyber768" => 1184,
            "Kyber1024" => 1568,
            "Dilithium2" => 1312,
            "Dilithium3" => 1952,
            "Dilithium5" => 2592,
            _ => 800,
        };
        
        let secret_key_size = match self.algorithm_name.as_str() {
            "Kyber512" => 1632,
            "Kyber768" => 2400,
            "Kyber1024" => 3168,
            "Dilithium2" => 2528,
            "Dilithium3" => 4000,
            "Dilithium5" => 4864,
            _ => 1632,
        };
        
        // Simulate key generation time
        let key_gen_time = match self.algorithm_name.as_str() {
            "Kyber512" => 50,
            "Kyber768" => 75,
            "Kyber1024" => 100,
            "Dilithium2" => 80,
            "Dilithium3" => 120,
            "Dilithium5" => 160,
            _ => 50,
        };
        
        std::thread::sleep(std::time::Duration::from_millis(key_gen_time.try_into().unwrap()));
        
        let public_key = vec![0u8; public_key_size];
        let secret_key = vec![0u8; secret_key_size];
        
        let latency = start_time.elapsed().as_millis() as f64;
        self.record_metric("key_generation", latency, 0);
        
        Ok((public_key, secret_key))
    }

    fn encrypt(&mut self, data: &[u8], public_key: &[u8]) -> PyResult<Vec<u8>> {
        let start_time = Instant::now();
        
        // Simulate PQC encryption
        let ciphertext_size = match self.algorithm_name.as_str() {
            "Kyber512" => 768,
            "Kyber768" => 1088,
            "Kyber1024" => 1568,
            _ => 768,
        };
        
        let encryption_time = 5 + (data.len() / 100);
        std::thread::sleep(std::time::Duration::from_millis(encryption_time.try_into().unwrap()));
        
        let ciphertext = vec![0u8; ciphertext_size];
        
        let latency = start_time.elapsed().as_millis() as f64;
        self.record_metric("encryption", latency, data.len());
        
        Ok(ciphertext)
    }

    fn decrypt(&mut self, ciphertext: &[u8], secret_key: &[u8]) -> PyResult<Vec<u8>> {
        let start_time = Instant::now();
        
        // Simulate PQC decryption
        let decryption_time = 8 + (ciphertext.len() / 80);
        std::thread::sleep(std::time::Duration::from_millis(decryption_time.try_into().unwrap()));
        
        let shared_secret = vec![0u8; 32];
        
        let latency = start_time.elapsed().as_millis() as f64;
        self.record_metric("decryption", latency, ciphertext.len());
        
        Ok(shared_secret)
    }

    fn sign(&mut self, data: &[u8], secret_key: &[u8]) -> PyResult<Vec<u8>> {
        let start_time = Instant::now();
        
        // Simulate PQC signing
        let signature_size = match self.algorithm_name.as_str() {
            "Dilithium2" => 2701,
            "Dilithium3" => 3366,
            "Dilithium5" => 4886,
            _ => 2701,
        };
        
        let signing_time = 10 + (data.len() / 50);
        std::thread::sleep(std::time::Duration::from_millis(signing_time.try_into().unwrap()));
        
        let signature = vec![0u8; signature_size];
        
        let latency = start_time.elapsed().as_millis() as f64;
        self.record_metric("signing", latency, data.len());
        
        Ok(signature)
    }

    fn verify(&mut self, data: &[u8], signature: &[u8], public_key: &[u8]) -> PyResult<bool> {
        let start_time = Instant::now();
        
        // Simulate PQC verification
        let verification_time = 15 + (data.len() / 40);
        std::thread::sleep(std::time::Duration::from_millis(verification_time.try_into().unwrap()));
        
        let latency = start_time.elapsed().as_millis() as f64;
        self.record_metric("verification", latency, data.len());
        
        Ok(true) // Simulate successful verification
    }

    fn get_metrics_count(&self) -> PyResult<usize> {
        let total = self.metrics.values().map(|v| v.len()).sum();
        Ok(total)
    }

    fn record_metric(&mut self, operation_name: &str, latency_ms: f64, data_size: usize) {
        let metric = PerformanceMetrics {
            latency_ms,
            memory_delta_bytes: 0, // Simplified for simulation
            cpu_percent: 0.0, // Simplified for simulation
            timestamp: Utc::now().timestamp_millis() as f64,
            operation_name: operation_name.to_string(),
            data_size,
            algorithm_name: self.algorithm_name.clone(),
        };
        
        self.metrics.entry(operation_name.to_string())
            .or_insert_with(Vec::new)
            .push(metric);
    }
}

#[pymodule]
fn pqc_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CryptoBenchmark>()?;
    m.add_class::<PipelineSimulator>()?;
    m.add_class::<OQSCrypto>()?;
    Ok(())
}
