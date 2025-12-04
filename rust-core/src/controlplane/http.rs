//! Control Plane HTTP Server
//!
//! Provides HTTP endpoints for Kubernetes liveness, readiness probes,
//! worker status, graceful shutdown, and orchestrator coordination.

use crate::pipeline::ExecutionState;
use crate::telemetry::Metrics;
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, warn};

/// Orchestration state for distributed experiments
#[derive(Clone)]
pub struct OrchestrationState {
    /// Worker ID assigned by orchestrator
    pub worker_id: Arc<AtomicU32>,
    /// Global start timestamp in nanoseconds
    pub global_start_ns: Arc<AtomicU64>,
    /// Whether this worker is registered with orchestrator
    pub registered: Arc<AtomicBool>,
    /// Whether start signal has been received
    pub start_signal_received: Arc<AtomicBool>,
    /// Whether worker is ready to start
    pub ready_for_start: Arc<AtomicBool>,
}

impl OrchestrationState {
    pub fn new() -> Self {
        Self {
            worker_id: Arc::new(AtomicU32::new(0)),
            global_start_ns: Arc::new(AtomicU64::new(0)),
            registered: Arc::new(AtomicBool::new(false)),
            start_signal_received: Arc::new(AtomicBool::new(false)),
            ready_for_start: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn set_registered(&self, worker_id: u32, global_start_ns: u64) {
        self.worker_id.store(worker_id, Ordering::SeqCst);
        self.global_start_ns.store(global_start_ns, Ordering::SeqCst);
        self.registered.store(true, Ordering::SeqCst);
    }

    pub fn set_ready(&self) {
        self.ready_for_start.store(true, Ordering::SeqCst);
    }

    pub fn set_start_signal(&self, global_start_ns: u64) {
        self.global_start_ns.store(global_start_ns, Ordering::SeqCst);
        self.start_signal_received.store(true, Ordering::SeqCst);
    }

    pub fn is_start_signal_received(&self) -> bool {
        self.start_signal_received.load(Ordering::Relaxed)
    }

    pub fn get_global_start_ns(&self) -> u64 {
        self.global_start_ns.load(Ordering::Relaxed)
    }

    pub fn get_worker_id(&self) -> u32 {
        self.worker_id.load(Ordering::Relaxed)
    }
}

impl Default for OrchestrationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared state for the control plane
pub struct ControlPlaneState {
    /// Execution state from the pipeline
    pub execution_state: Option<ExecutionState>,
    /// Metrics instance
    pub metrics: Metrics,
    /// Whether the metrics server is running
    pub metrics_server_running: Arc<AtomicBool>,
    /// Execution mode string
    pub execution_mode: String,
    /// Orchestration state for distributed experiments
    pub orchestration: OrchestrationState,
}

impl Clone for ControlPlaneState {
    fn clone(&self) -> Self {
        Self {
            execution_state: self.execution_state.clone(),
            metrics: self.metrics.clone(),
            metrics_server_running: self.metrics_server_running.clone(),
            execution_mode: self.execution_mode.clone(),
            orchestration: self.orchestration.clone(),
        }
    }
}

impl ControlPlaneState {
    pub fn new(metrics: Metrics, execution_mode: &str) -> Self {
        Self {
            execution_state: None,
            metrics,
            metrics_server_running: Arc::new(AtomicBool::new(false)),
            execution_mode: execution_mode.to_string(),
            orchestration: OrchestrationState::new(),
        }
    }

    pub fn with_execution_state(mut self, state: ExecutionState) -> Self {
        self.execution_state = Some(state);
        self
    }

    pub fn with_orchestration(mut self, orchestration: OrchestrationState) -> Self {
        self.orchestration = orchestration;
        self
    }

    pub fn set_metrics_server_running(&self, running: bool) {
        self.metrics_server_running.store(running, Ordering::SeqCst);
    }

    pub fn orchestration(&self) -> &OrchestrationState {
        &self.orchestration
    }
}

/// Response for /healthz endpoint
#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

/// Response for /readyz endpoint
#[derive(Serialize)]
struct ReadyResponse {
    ready: bool,
    metrics_server_running: bool,
    pipeline_started: bool,
    scenario_loaded: bool,
}

/// Response for /workers endpoint
#[derive(Serialize)]
struct WorkersResponse {
    mode: String,
    active_workers: usize,
    queue_length: usize,
    queue_capacity: usize,
}

/// Response for /shutdown endpoint
#[derive(Serialize)]
struct ShutdownResponse {
    status: String,
    message: String,
}

/// Request for /register endpoint (from orchestrator)
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RegisterRequest {
    worker_id: u32,
    global_start_unix_ns: u64,
}

/// Response for /register endpoint
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RegisterResponse {
    registered: bool,
    worker_id: u32,
}

/// Response for /ready_for_start endpoint
#[derive(Serialize)]
struct ReadyForStartResponse {
    ready: bool,
}

/// Request for /start_signal endpoint
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StartSignalRequest {
    global_start_unix_ns: u64,
}

/// Response for /start_signal endpoint
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StartSignalResponse {
    acknowledged: bool,
    worker_id: u32,
}

/// Starts the control plane HTTP server
///
/// Listens on the specified address and provides:
/// - GET /healthz - Liveness probe
/// - GET /readyz - Readiness probe
/// - GET /workers - Worker status
/// - POST /shutdown - Graceful shutdown
pub async fn start_control_plane_server(
    addr: &str,
    state: ControlPlaneState,
) -> tokio::task::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
    let addr: SocketAddr = addr.parse().expect("Invalid control plane server address");

    info!("Starting control plane server on {}", addr);

    tokio::spawn(async move {
        let listener = TcpListener::bind(addr).await?;

        loop {
            let (stream, remote_addr) = listener.accept().await?;
            let io = TokioIo::new(stream);
            let state_clone = state.clone();

            tokio::spawn(async move {
                let service = service_fn(move |req| {
                    let state = state_clone.clone();
                    async move { handle_request(req, state, remote_addr).await }
                });

                if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                    warn!("Error serving control plane connection: {:?}", err);
                }
            });
        }
    })
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    state: ControlPlaneState,
    _remote_addr: SocketAddr,
) -> Result<Response<Full<Bytes>>, Infallible> {
    use http_body_util::BodyExt;

    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // Collect body for POST requests
    let body_bytes = if method == Method::POST {
        match req.collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(e) => {
                warn!("Failed to read request body: {}", e);
                return Ok(json_response(
                    StatusCode::BAD_REQUEST,
                    &serde_json::json!({"error": "Failed to read body"}),
                ));
            }
        }
    } else {
        Bytes::new()
    };

    let response = match (method.clone(), path.as_str()) {
        // Health check - always returns OK if server is running
        (Method::GET, "/healthz") => {
            let body = HealthResponse {
                status: "ok".to_string(),
            };
            json_response(StatusCode::OK, &body)
        }

        // Readiness check - indicates if the service is ready to receive traffic
        (Method::GET, "/readyz") => {
            let (metrics_running, pipeline_started, scenario_loaded) =
                if let Some(ref exec_state) = state.execution_state {
                    (
                        state.metrics_server_running.load(Ordering::Relaxed),
                        exec_state.pipeline_started.load(Ordering::Relaxed),
                        exec_state.scenario_loaded.load(Ordering::Relaxed),
                    )
                } else {
                    (state.metrics_server_running.load(Ordering::Relaxed), false, false)
                };

            let ready = metrics_running && scenario_loaded;

            let body = ReadyResponse {
                ready,
                metrics_server_running: metrics_running,
                pipeline_started,
                scenario_loaded,
            };

            let status = if ready {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };

            json_response(status, &body)
        }

        // Worker status
        (Method::GET, "/workers") => {
            let (active_workers, queue_length, queue_capacity) =
                if let Some(ref exec_state) = state.execution_state {
                    (
                        exec_state.active_workers.load(Ordering::Relaxed),
                        exec_state.queue_length.load(Ordering::Relaxed),
                        exec_state.queue_capacity,
                    )
                } else {
                    (
                        state.metrics.get_active_workers(),
                        state.metrics.get_queue_length(),
                        state.metrics.get_queue_capacity(),
                    )
                };

            let body = WorkersResponse {
                mode: state.execution_mode.clone(),
                active_workers,
                queue_length,
                queue_capacity,
            };

            json_response(StatusCode::OK, &body)
        }

        // Graceful shutdown
        (Method::POST, "/shutdown") | (Method::GET, "/shutdown") => {
            info!("Shutdown requested via control plane");

            if let Some(ref exec_state) = state.execution_state {
                exec_state.request_shutdown();

                let body = ShutdownResponse {
                    status: "accepted".to_string(),
                    message: "Shutdown initiated. Pipeline will drain and exit.".to_string(),
                };

                json_response(StatusCode::ACCEPTED, &body)
            } else {
                let body = ShutdownResponse {
                    status: "error".to_string(),
                    message: "No execution state available".to_string(),
                };

                json_response(StatusCode::INTERNAL_SERVER_ERROR, &body)
            }
        }

        // Orchestrator registration - called during distributed experiment setup
        (Method::POST, "/register") => {
            match serde_json::from_slice::<RegisterRequest>(&body_bytes) {
                Ok(request) => {
                    info!(
                        "Registering with orchestrator: worker_id={}, global_start={}",
                        request.worker_id, request.global_start_unix_ns
                    );

                    state
                        .orchestration
                        .set_registered(request.worker_id, request.global_start_unix_ns);

                    let body = RegisterResponse {
                        registered: true,
                        worker_id: request.worker_id,
                    };

                    json_response(StatusCode::OK, &body)
                }
                Err(e) => {
                    warn!("Failed to parse register request: {}", e);
                    json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({"error": e.to_string()}),
                    )
                }
            }
        }

        // Ready for start check - called by orchestrator to check if worker is ready
        (Method::GET, "/ready_for_start") => {
            let ready = state.orchestration.ready_for_start.load(Ordering::Relaxed);
            let body = ReadyForStartResponse { ready };
            json_response(StatusCode::OK, &body)
        }

        // Start signal - orchestrator signals worker to begin
        (Method::POST, "/start_signal") => {
            match serde_json::from_slice::<StartSignalRequest>(&body_bytes) {
                Ok(request) => {
                    info!(
                        "Received start signal: global_start_ns={}",
                        request.global_start_unix_ns
                    );

                    state.orchestration.set_start_signal(request.global_start_unix_ns);

                    let body = StartSignalResponse {
                        acknowledged: true,
                        worker_id: state.orchestration.get_worker_id(),
                    };

                    json_response(StatusCode::OK, &body)
                }
                Err(e) => {
                    warn!("Failed to parse start_signal request: {}", e);
                    json_response(
                        StatusCode::BAD_REQUEST,
                        &serde_json::json!({"error": e.to_string()}),
                    )
                }
            }
        }

        // Not found
        _ => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(
                r#"{"error":"Not Found"}"#.to_string(),
            )))
            .unwrap(),
    };

    Ok(response)
}

fn json_response<T: Serialize>(status: StatusCode, body: &T) -> Response<Full<Bytes>> {
    let json = serde_json::to_string(body).unwrap_or_else(|_| "{}".to_string());
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(json)))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_plane_state_new() {
        let metrics = Metrics::new().unwrap();
        let state = ControlPlaneState::new(metrics, "single");
        assert_eq!(state.execution_mode, "single");
        assert!(state.execution_state.is_none());
    }

    #[test]
    fn test_control_plane_state_with_execution() {
        let metrics = Metrics::new().unwrap();
        let exec_state = ExecutionState::new(1000);
        let state = ControlPlaneState::new(metrics, "fixed_pool").with_execution_state(exec_state);
        assert!(state.execution_state.is_some());
    }

    #[test]
    fn test_health_response_serialize() {
        let response = HealthResponse {
            status: "ok".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("ok"));
    }

    #[test]
    fn test_ready_response_serialize() {
        let response = ReadyResponse {
            ready: true,
            metrics_server_running: true,
            pipeline_started: true,
            scenario_loaded: true,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("true"));
    }

    #[test]
    fn test_workers_response_serialize() {
        let response = WorkersResponse {
            mode: "elastic".to_string(),
            active_workers: 4,
            queue_length: 100,
            queue_capacity: 2000,
        };
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("elastic"));
        assert!(json.contains("4"));
    }
}


