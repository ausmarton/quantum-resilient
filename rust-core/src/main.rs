//! PQC Benchmark Framework Binary
//!
//! This is the main entry point for running cryptographic benchmarks
//! comparing Post-Quantum Cryptography (PQC) with classical algorithms.

use clap::Parser;
use rust_core::{
    get_adapter, init_tracing, load_scenario, supported_adapters, supported_operations,
    ControlPlaneState, ExecutionMode, JsonlWriter, Metrics, Pipeline,
    SysInfoSampler, start_control_plane_server,
};
use rust_core::telemetry::start_metrics_server;
use tokio::signal;
use tracing::info;

/// PQC Benchmark Framework - Compare PQC vs classical cryptography performance
#[derive(Parser, Debug)]
#[command(name = "pqc-bench")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the scenario YAML file
    #[arg(long)]
    scenario: String,

    /// Control plane server port (default: 6060)
    #[arg(long, default_value = "6060")]
    control_port: u16,
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
    if let Some(ref desc) = scenario.description {
        println!("Description: {}", desc);
    }

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

    // Print execution configuration
    let exec_mode = match scenario.execution.mode {
        ExecutionMode::Single => "single",
        ExecutionMode::FixedPool => "fixed_pool",
        ExecutionMode::Elastic => "elastic",
    };
    println!("Execution mode: {}", exec_mode);
    println!("Queue capacity: {}", scenario.execution.queue_capacity);

    match scenario.execution.mode {
        ExecutionMode::Single => {
            println!("Workers: 1 (single)");
        }
        ExecutionMode::FixedPool => {
            println!("Workers: {} (fixed)", scenario.execution.workers);
        }
        ExecutionMode::Elastic => {
            println!(
                "Workers: 1-{} (elastic)",
                scenario.execution.max_workers
            );
        }
    }

    // Print workload configuration
    let workload_pattern = match scenario.workload.pattern {
        rust_core::WorkloadPattern::Constant => "constant",
        rust_core::WorkloadPattern::Burst => "burst",
        rust_core::WorkloadPattern::Ramp => "ramp",
        rust_core::WorkloadPattern::Trace => "trace",
    };
    println!("Workload pattern: {}", workload_pattern);

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

    // Create shared execution state (will be used by both control plane and pipeline)
    let execution_state = rust_core::pipeline::ExecutionState::new(scenario.execution.queue_capacity);

    // Create control plane state
    let control_state = ControlPlaneState::new(metrics.clone(), exec_mode)
        .with_execution_state(execution_state.clone());
    control_state.set_metrics_server_running(true);

    // Start control plane server
    let control_addr = format!("0.0.0.0:{}", args.control_port);
    println!("Starting control plane server on {}", control_addr);
    let _control_handle = start_control_plane_server(&control_addr, control_state).await;

    // Create and run pipeline
    let pipeline = Pipeline::new();

    info!(
        "Starting pipeline: workload={}, execution_mode={}, duration={}s",
        workload_pattern,
        exec_mode,
        scenario.workload.duration_sec
    );

    println!();
    println!("========================================");
    println!("Pipeline starting...");
    println!("Prometheus: http://{}/metrics", prometheus_endpoint);
    println!("Control plane: http://127.0.0.1:{}/healthz", args.control_port);
    println!("========================================");
    println!();

    // Set up shutdown signal handler
    let shutdown_state = execution_state.clone();
    let shutdown_handle = tokio::spawn(async move {
        // Wait for either SIGTERM or Ctrl+C
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                info!("Received Ctrl+C, initiating shutdown...");
            }
            _ = terminate => {
                info!("Received SIGTERM, initiating shutdown...");
            }
            _ = async {
                // Also check for shutdown flag from control plane
                loop {
                    if shutdown_state.is_shutdown_requested() {
                        break;
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            } => {
                info!("Shutdown requested via control plane...");
            }
        }

        shutdown_state.request_shutdown();
    });

    // Run the pipeline with shared execution state
    let stats = match pipeline
        .run_async(
            &scenario,
            adapter.clone(),
            metrics.clone(),
            jsonl_writer.clone(),
            sampler,
            Some(execution_state.clone()),
        )
        .await
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Pipeline execution failed: {:?}", e);
            std::process::exit(1);
        }
    };

    // Cancel the shutdown handler since pipeline completed naturally
    shutdown_handle.abort();

    // Print summary
    println!();
    println!("========================================");
    println!("Run complete: {} events processed", stats.events_processed);
    println!("Total events planned: {}", stats.total_events);
    println!("Duration: {:.2}s", stats.duration.as_secs_f64());
    println!("Average latency: {:.2} μs", stats.avg_latency_us);
    println!(
        "Throughput: {:.2} ops/sec",
        stats.events_processed as f64 / stats.duration.as_secs_f64()
    );
    println!("Max active workers: {}", stats.max_active_workers);
    println!("JSONL output: {}", jsonl_writer.path());
    println!(
        "Metrics available at: http://{}/metrics",
        prometheus_endpoint
    );
    println!("========================================");
    println!();
    println!("Done.");
}
