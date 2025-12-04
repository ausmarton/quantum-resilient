//! PQC Benchmark Framework Binary
//!
//! This is the main entry point for running cryptographic benchmarks
//! comparing Post-Quantum Cryptography (PQC) with classical algorithms.
//!
//! Supports both standalone operation and distributed experiments via orchestrator.

use clap::Parser;
use rust_core::{
    get_adapter, init_tracing, load_scenario, supported_adapters, supported_operations,
    ControlPlaneState, ExecutionMode, JsonlWriter, Metrics, OrchestrationState, Pipeline,
    SysInfoSampler, start_control_plane_server,
};
use rust_core::telemetry::start_metrics_server;
use std::env;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::signal;
use tracing::{info, warn};

/// PQC Benchmark Framework - Compare PQC vs classical cryptography performance
#[derive(Parser, Debug)]
#[command(name = "pqc-bench")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the scenario YAML file (overrides QR_SCENARIO_PATH env var)
    #[arg(long)]
    scenario: Option<String>,

    /// Control plane server port (default: 6060, or QR_CONTROL_PLANE_PORT env var)
    #[arg(long)]
    control_port: Option<u16>,
}

/// Orchestrator configuration from environment
struct OrchestratorConfig {
    address: Option<String>,
    experiment_id: Option<String>,
    enforce_timesync: bool,
    pod_name: String,
    pod_ip: String,
}

impl OrchestratorConfig {
    fn from_env() -> Self {
        Self {
            address: env::var("QR_ORCHESTRATOR_ADDRESS").ok(),
            experiment_id: env::var("QR_EXPERIMENT_ID").ok(),
            enforce_timesync: env::var("QR_ENFORCE_TIMESYNC")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            pod_name: env::var("POD_NAME").unwrap_or_else(|_| "unknown".to_string()),
            pod_ip: env::var("POD_IP").unwrap_or_else(|_| "127.0.0.1".to_string()),
        }
    }

    fn is_orchestrated(&self) -> bool {
        self.address.is_some() && self.experiment_id.is_some()
    }
}

/// Registers this worker with the orchestrator
async fn register_with_orchestrator(
    config: &OrchestratorConfig,
    orchestration_state: &OrchestrationState,
) -> Result<(u32, u64, u64), String> {
    let address = config.address.as_ref().ok_or("No orchestrator address")?;
    let experiment_id = config.experiment_id.as_ref().ok_or("No experiment ID")?;

    let url = format!("{}/experiment/{}/register", address, experiment_id);
    
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "podName": config.pod_name,
        "podIp": config.pod_ip
    });

    info!("Registering with orchestrator at {}", url);

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Failed to register: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("Registration failed: HTTP {}", resp.status()));
    }

    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct RegisterResponse {
        worker_id: u32,
        global_start_unix_ns: u64,
        orchestrator_time_ns: u64,
    }

    let response: RegisterResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    // Store registration in orchestration state
    orchestration_state.set_registered(response.worker_id, response.global_start_unix_ns);

    Ok((response.worker_id, response.global_start_unix_ns, response.orchestrator_time_ns))
}

/// Notifies orchestrator that this worker is ready
async fn notify_ready(
    config: &OrchestratorConfig,
    worker_id: u32,
) -> Result<(), String> {
    let address = config.address.as_ref().ok_or("No orchestrator address")?;
    let experiment_id = config.experiment_id.as_ref().ok_or("No experiment ID")?;

    let url = format!("{}/experiment/{}/ready", address, experiment_id);
    
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "workerId": worker_id,
        "workerTimeNs": now_ns
    });

    client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Failed to notify ready: {}", e))?;

    Ok(())
}

/// Notifies orchestrator that this worker has completed
async fn notify_completed(
    config: &OrchestratorConfig,
    worker_id: u32,
) -> Result<(), String> {
    let address = config.address.as_ref().ok_or("No orchestrator address")?;
    let experiment_id = config.experiment_id.as_ref().ok_or("No experiment ID")?;

    let url = format!("{}/experiment/{}/completed", address, experiment_id);
    
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "workerId": worker_id
    });

    client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Failed to notify completed: {}", e))?;

    Ok(())
}

/// Waits for the global start signal and synchronizes start time
async fn wait_for_start_signal(
    orchestration_state: &OrchestrationState,
    max_wait_secs: u64,
) -> Result<(), String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(max_wait_secs);

    // Wait for start signal
    while !orchestration_state.is_start_signal_received() {
        if std::time::Instant::now() > deadline {
            return Err("Timeout waiting for start signal".to_string());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Wait until global start time
    let global_start_ns = orchestration_state.get_global_start_ns();
    if global_start_ns > 0 {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        if global_start_ns > now_ns {
            let wait_ns = global_start_ns - now_ns;
            let wait_duration = Duration::from_nanos(wait_ns);
            info!("Waiting {:.2}ms for synchronized start", wait_duration.as_secs_f64() * 1000.0);
            tokio::time::sleep(wait_duration).await;
        }
    }

    Ok(())
}

/// Resolves the scenario path from CLI argument or environment variable
fn resolve_scenario_path(cli_arg: Option<String>) -> Result<String, String> {
    // Priority: CLI argument > QR_SCENARIO_PATH env var
    if let Some(path) = cli_arg {
        return Ok(path);
    }

    if let Ok(env_path) = env::var("QR_SCENARIO_PATH") {
        if !env_path.is_empty() {
            return Ok(env_path);
        }
    }

    Err(
        "No scenario specified. Use --scenario <path> or set QR_SCENARIO_PATH environment variable."
            .to_string(),
    )
}

/// Resolves the control plane port from CLI argument or environment variable
fn resolve_control_port(cli_arg: Option<u16>) -> u16 {
    // Priority: CLI argument > QR_CONTROL_PLANE_PORT env var > default 6060
    if let Some(port) = cli_arg {
        return port;
    }

    if let Ok(env_port) = env::var("QR_CONTROL_PLANE_PORT") {
        if let Ok(port) = env_port.parse::<u16>() {
            return port;
        }
    }

    6060 // default
}

#[tokio::main]
async fn main() {
    println!("Starting PQC Benchmark Framework...");

    // Parse command line arguments
    let args = Args::parse();

    // Resolve scenario path from CLI or environment
    let scenario_path = match resolve_scenario_path(args.scenario) {
        Ok(path) => path,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    println!("Scenario path: {}", scenario_path);

    // Load scenario from YAML file
    let scenario = match load_scenario(&scenario_path) {
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

    // Create orchestration state for distributed experiments
    let orchestration_state = OrchestrationState::new();

    // Create control plane state
    let control_state = ControlPlaneState::new(metrics.clone(), exec_mode)
        .with_execution_state(execution_state.clone())
        .with_orchestration(orchestration_state.clone());
    control_state.set_metrics_server_running(true);

    // Start control plane server
    let control_port = resolve_control_port(args.control_port);
    let control_addr = format!("0.0.0.0:{}", control_port);
    println!("Starting control plane server on {}", control_addr);
    let _control_handle = start_control_plane_server(&control_addr, control_state).await;

    // Check for orchestrator configuration
    let orch_config = OrchestratorConfig::from_env();
    let worker_id: Option<u32>;

    if orch_config.is_orchestrated() {
        println!("Orchestrated mode detected");
        println!("  Orchestrator: {}", orch_config.address.as_ref().unwrap());
        println!("  Experiment: {}", orch_config.experiment_id.as_ref().unwrap());
        println!("  Pod: {} ({})", orch_config.pod_name, orch_config.pod_ip);

        // Register with orchestrator
        match register_with_orchestrator(&orch_config, &orchestration_state).await {
            Ok((wid, global_start, orch_time)) => {
                worker_id = Some(wid);
                println!("Registered as worker {}", wid);

                // Check time drift
                let local_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
                let drift_ns = (local_time as i64 - orch_time as i64).abs();
                let drift_ms = drift_ns as f64 / 1_000_000.0;

                if drift_ms > 5.0 {
                    if orch_config.enforce_timesync {
                        warn!("Time drift of {:.2}ms detected (threshold: 5ms)", drift_ms);
                    }
                    println!("Warning: Time drift of {:.2}ms detected", drift_ms);
                }

                // Notify orchestrator we're ready
                if let Err(e) = notify_ready(&orch_config, wid).await {
                    warn!("Failed to notify ready: {}", e);
                }
                orchestration_state.set_ready();

                // Wait for start signal
                println!("Waiting for start signal from orchestrator...");
                if let Err(e) = wait_for_start_signal(&orchestration_state, 300).await {
                    eprintln!("Error waiting for start signal: {}", e);
                    std::process::exit(1);
                }
                println!("Start signal received, beginning experiment");
            }
            Err(e) => {
                eprintln!("Failed to register with orchestrator: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        worker_id = None;
        println!("Standalone mode (no orchestrator configured)");
    }

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
    println!("Control plane: http://127.0.0.1:{}/healthz", control_port);
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

    // Notify orchestrator of completion if in orchestrated mode
    if let Some(wid) = worker_id {
        if let Err(e) = notify_completed(&orch_config, wid).await {
            warn!("Failed to notify completion: {}", e);
        }
        println!("Notified orchestrator of completion");
    }

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
