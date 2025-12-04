//! Quantum-Resilient Orchestrator
//!
//! Manages distributed multi-pod benchmark experiments across a Kubernetes cluster.
//! Coordinates worker pods, aggregates results, and uploads to object storage.

use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod aggregator;
mod api;
mod controller;
mod coordinator;
mod k8s_client;
mod storage;

use controller::ExperimentController;

/// Quantum-Resilient Experiment Orchestrator
#[derive(Parser, Debug)]
#[command(name = "qr-orchestrator")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// HTTP API listen address
    #[arg(long, env = "QR_ORCH_LISTEN_ADDR", default_value = "0.0.0.0:7070")]
    listen_addr: String,

    /// Kubernetes namespace for experiments
    #[arg(long, env = "QR_ORCH_NAMESPACE", default_value = "default")]
    namespace: String,

    /// Worker image to use for experiments
    #[arg(long, env = "QR_WORKER_IMAGE", default_value = "localhost/pqc-bench:latest")]
    worker_image: String,

    /// Storage backend URI (s3://bucket/prefix or gs://bucket/prefix)
    #[arg(long, env = "QR_STORAGE_URI")]
    storage_uri: Option<String>,

    /// Local results directory
    #[arg(long, env = "QR_LOCAL_RESULTS_DIR", default_value = "/tmp/qr-orchestrator")]
    local_results_dir: String,

    /// Maximum allowed time drift between workers (nanoseconds)
    #[arg(long, env = "QR_MAX_TIME_DRIFT_NS", default_value = "5000000")]
    max_time_drift_ns: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting Quantum-Resilient Orchestrator...");

    let args = Args::parse();

    // Create results directory
    tokio::fs::create_dir_all(&args.local_results_dir).await?;

    // Initialize Kubernetes client
    let k8s_client = k8s_client::K8sClient::new(&args.namespace).await?;
    info!("Connected to Kubernetes cluster");

    // Create the experiment controller
    let controller = ExperimentController::new(
        k8s_client,
        args.worker_image.clone(),
        args.namespace.clone(),
        args.local_results_dir.clone(),
        args.storage_uri.clone(),
        args.max_time_drift_ns,
    );

    let controller = Arc::new(controller);

    // Build the API router
    let app = api::create_router(controller.clone());

    // Parse listen address
    let addr: SocketAddr = args.listen_addr.parse()?;
    info!("Starting HTTP API server on {}", addr);

    // Start the server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

