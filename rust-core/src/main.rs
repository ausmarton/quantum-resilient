//! PQC Benchmark Framework Binary
//!
//! This is the main entry point for running cryptographic benchmarks
//! comparing Post-Quantum Cryptography (PQC) with classical algorithms.

use clap::Parser;
use rust_core::{
    get_adapter, init_tracing, load_scenario, supported_adapters, supported_operations,
    JsonlWriter, Metrics, Pipeline, SysInfoSampler,
};
use rust_core::telemetry::start_metrics_server;
use tracing::info;

/// PQC Benchmark Framework - Compare PQC vs classical cryptography performance
#[derive(Parser, Debug)]
#[command(name = "pqc-bench")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the scenario YAML file
    #[arg(long)]
    scenario: String,
}

#[tokio::main]
async fn main() {
    println!("Starting PQC Benchmark Framework...");

    // Parse command line arguments
    let args = Args::parse();

    // Load scenario from YAML file
    let scenario = match load_scenario(&args.scenario) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    println!("Loaded scenario: {}", scenario.id);

    // Initialize tracing
    init_tracing("pqc-bench");

    // Validate adapter
    let adapter = match get_adapter(&scenario.algorithm.adapter) {
        Ok(a) => a,
        Err(_) => {
            eprintln!(
                "Error: Unknown adapter '{}'. Supported adapters: {}",
                scenario.algorithm.adapter,
                supported_adapters().join(", ")
            );
            std::process::exit(1);
        }
    };

    println!("Using adapter: {}", adapter.name());

    // Validate operation
    let operation = &scenario.algorithm.operation;
    if !supported_operations().contains(&operation.as_str()) {
        eprintln!(
            "Error: Unknown operation '{}'. Supported operations: {}",
            operation,
            supported_operations().join(", ")
        );
        std::process::exit(1);
    }

    println!("Running operation: {}", operation);

    // Create results directory
    let jsonl_path = scenario.jsonl_output_path();
    if let Some(parent) = std::path::Path::new(&jsonl_path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Warning: Failed to create results directory: {}", e);
            }
        }
    }

    // Initialize metrics
    let metrics = match Metrics::new() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error: Failed to initialize metrics: {}", e);
            std::process::exit(1);
        }
    };

    // Start metrics server
    let prometheus_endpoint = &scenario.metrics.prometheus_endpoint;
    println!(
        "Starting Prometheus metrics server on {}",
        prometheus_endpoint
    );
    let _metrics_handle = start_metrics_server(prometheus_endpoint, metrics.clone()).await;

    // Initialize JSONL writer
    let jsonl_writer = match JsonlWriter::new(&jsonl_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error: Failed to create JSONL writer: {}", e);
            std::process::exit(1);
        }
    };

    // Initialize system sampler
    let sampler = SysInfoSampler::new();

    // Create and run pipeline
    let pipeline = Pipeline::new();

    info!(
        "Starting pipeline: {} events at {} msg/s for {} seconds",
        scenario.workload.duration_sec as u64 * scenario.workload.msgs_per_sec as u64,
        scenario.workload.msgs_per_sec,
        scenario.workload.duration_sec
    );

    let stats = match pipeline
        .run_async(&scenario, adapter.clone(), metrics.clone(), jsonl_writer.clone(), sampler)
        .await
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Pipeline execution failed: {:?}", e);
            std::process::exit(1);
        }
    };

    // Print summary
    println!();
    println!("========================================");
    println!("Run complete: {} events processed", stats.events_processed);
    println!("Duration: {:.2}s", stats.duration.as_secs_f64());
    println!("Average latency: {:.2} μs", stats.avg_latency_us);
    println!("Throughput: {:.2} ops/sec", stats.events_processed as f64 / stats.duration.as_secs_f64());
    println!("JSONL output: {}", jsonl_writer.path());
    println!(
        "Metrics available at: http://{}/metrics",
        prometheus_endpoint
    );
    println!("========================================");
    println!();
    println!("Done.");
}
