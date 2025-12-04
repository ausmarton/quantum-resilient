//! Control Plane HTTP Server
//!
//! Provides HTTP endpoints for Kubernetes liveness, readiness probes,
//! worker status, and graceful shutdown.

use crate::pipeline::ExecutionState;
use crate::telemetry::Metrics;
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::Serialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, warn};

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
}

impl Clone for ControlPlaneState {
    fn clone(&self) -> Self {
        Self {
            execution_state: self.execution_state.clone(),
            metrics: self.metrics.clone(),
            metrics_server_running: self.metrics_server_running.clone(),
            execution_mode: self.execution_mode.clone(),
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
        }
    }

    pub fn with_execution_state(mut self, state: ExecutionState) -> Self {
        self.execution_state = Some(state);
        self
    }

    pub fn set_metrics_server_running(&self, running: bool) {
        self.metrics_server_running.store(running, Ordering::SeqCst);
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
    let response = match (req.method(), req.uri().path()) {
        // Health check - always returns OK if server is running
        (&Method::GET, "/healthz") => {
            let body = HealthResponse {
                status: "ok".to_string(),
            };
            json_response(StatusCode::OK, &body)
        }

        // Readiness check - indicates if the service is ready to receive traffic
        (&Method::GET, "/readyz") => {
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
        (&Method::GET, "/workers") => {
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
        (&Method::POST, "/shutdown") | (&Method::GET, "/shutdown") => {
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


