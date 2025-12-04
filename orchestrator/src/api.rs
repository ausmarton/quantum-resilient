//! Orchestrator REST API
//!
//! Provides HTTP endpoints for experiment management.

use crate::controller::{
    AggregationSummary, CollectResult, ControllerError, CreateExperimentRequest, Experiment,
    ExperimentController, ExperimentStatus,
};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

/// Shared state for API handlers
type AppState = Arc<ExperimentController>;

/// Create the API router
pub fn create_router(controller: Arc<ExperimentController>) -> Router {
    Router::new()
        // Health endpoints
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        // Experiment management
        .route("/experiment", post(create_experiment))
        .route("/experiments", get(list_experiments))
        .route("/experiment/:id", get(get_experiment))
        .route("/experiment/:id", axum::routing::delete(delete_experiment))
        .route("/experiment/:id/status", get(get_experiment_status))
        .route("/experiment/:id/start", post(start_experiment))
        .route("/experiment/:id/stop", post(stop_experiment))
        .route("/experiment/:id/collect", post(collect_results))
        // Worker registration (called by workers)
        .route("/experiment/:id/register", post(register_worker))
        .route("/experiment/:id/ready", post(mark_worker_ready))
        .route("/experiment/:id/completed", post(mark_worker_completed))
        // Metrics
        .route("/metrics", get(metrics))
        .with_state(controller)
}

/// Health check endpoint
async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// Readiness check endpoint
async fn readyz(State(controller): State<AppState>) -> impl IntoResponse {
    // Could check K8s connectivity here
    Json(serde_json::json!({
        "ready": true,
        "experiments_count": controller.list_experiments().len()
    }))
}

/// Create a new experiment
async fn create_experiment(
    State(controller): State<AppState>,
    Json(request): Json<CreateExperimentRequest>,
) -> Result<(StatusCode, Json<ExperimentResponse>), ApiError> {
    info!("API: Create experiment request: {}", request.experiment_id);
    
    let experiment = controller.create_experiment(request).await?;
    
    Ok((StatusCode::CREATED, Json(ExperimentResponse::from(experiment))))
}

/// List all experiments
async fn list_experiments(State(controller): State<AppState>) -> impl IntoResponse {
    let experiments: Vec<ExperimentResponse> = controller
        .list_experiments()
        .into_iter()
        .map(ExperimentResponse::from)
        .collect();
    
    Json(experiments)
}

/// Get a specific experiment
async fn get_experiment(
    State(controller): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ExperimentStatus>, ApiError> {
    let status = controller.get_status(&id)?;
    Ok(Json(status))
}

/// Delete an experiment
async fn delete_experiment(
    State(controller): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    controller.delete_experiment(&id).await?;
    Ok(Json(serde_json::json!({ "deleted": true, "id": id })))
}

/// Get experiment status
async fn get_experiment_status(
    State(controller): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ExperimentStatus>, ApiError> {
    let status = controller.get_status(&id)?;
    Ok(Json(status))
}

/// Start an experiment
async fn start_experiment(
    State(controller): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    info!("API: Start experiment: {}", id);
    controller.start_experiment(&id).await?;
    Ok(Json(serde_json::json!({ "started": true, "id": id })))
}

/// Stop an experiment
async fn stop_experiment(
    State(controller): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    info!("API: Stop experiment: {}", id);
    controller.stop_experiment(&id).await?;
    Ok(Json(serde_json::json!({ "stopped": true, "id": id })))
}

/// Collect and aggregate results
async fn collect_results(
    State(controller): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<CollectResult>, ApiError> {
    info!("API: Collect results for experiment: {}", id);
    let result = controller.collect_results(&id).await?;
    Ok(Json(result))
}

/// Worker registration request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RegisterRequest {
    pod_name: String,
    pod_ip: String,
}

/// Worker registration response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RegisterResponse {
    worker_id: u32,
    global_start_unix_ns: u64,
    orchestrator_time_ns: u64,
}

/// Register a worker with an experiment
async fn register_worker(
    State(controller): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, ApiError> {
    info!(
        "API: Worker registration for experiment {}: pod={}",
        id, request.pod_name
    );
    
    let (worker_id, global_start) = controller.register_worker(&id, &request.pod_name, &request.pod_ip)?;
    let orchestrator_time = controller.current_timestamp_ns();
    
    Ok(Json(RegisterResponse {
        worker_id,
        global_start_unix_ns: global_start,
        orchestrator_time_ns: orchestrator_time,
    }))
}

/// Worker ready notification
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReadyRequest {
    worker_id: u32,
    worker_time_ns: u64,
}

/// Mark a worker as ready
async fn mark_worker_ready(
    State(controller): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<ReadyRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Get experiment coordinator
    let experiments = controller.list_experiments();
    let experiment = experiments
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| ControllerError::ExperimentNotFound(id.clone()))?;
    
    experiment.coordinator.mark_worker_ready(request.worker_id, request.worker_time_ns);
    
    Ok(Json(serde_json::json!({ "ready": true })))
}

/// Worker completion notification
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompletedRequest {
    worker_id: u32,
}

/// Mark a worker as completed
async fn mark_worker_completed(
    State(controller): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<CompletedRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let experiments = controller.list_experiments();
    let experiment = experiments
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| ControllerError::ExperimentNotFound(id.clone()))?;
    
    experiment.coordinator.mark_worker_completed(request.worker_id);
    
    Ok(Json(serde_json::json!({ "completed": true })))
}

/// Prometheus metrics endpoint
async fn metrics() -> impl IntoResponse {
    // Basic metrics - could be expanded
    let metrics = r#"# HELP qr_orchestrator_experiments_total Total number of experiments
# TYPE qr_orchestrator_experiments_total counter
qr_orchestrator_experiments_total 0
"#;
    (
        StatusCode::OK,
        [("Content-Type", "text/plain; charset=utf-8")],
        metrics,
    )
}

/// Simplified experiment response for API
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExperimentResponse {
    id: String,
    replicas: u32,
    phase: String,
    created_at: String,
    started_at: Option<String>,
    completed_at: Option<String>,
}

impl From<Experiment> for ExperimentResponse {
    fn from(exp: Experiment) -> Self {
        Self {
            id: exp.id,
            replicas: exp.replicas,
            phase: exp.phase.to_string(),
            created_at: exp.created_at.to_rfc3339(),
            started_at: exp.started_at.map(|t| t.to_rfc3339()),
            completed_at: exp.completed_at.map(|t| t.to_rfc3339()),
        }
    }
}

/// API error handling
struct ApiError(ControllerError);

impl From<ControllerError> for ApiError {
    fn from(err: ControllerError) -> Self {
        ApiError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self.0 {
            ControllerError::ExperimentNotFound(_) => (StatusCode::NOT_FOUND, self.0.to_string()),
            ControllerError::InvalidState { .. } => (StatusCode::CONFLICT, self.0.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()),
        };

        let body = serde_json::json!({
            "error": message
        });

        (status, Json(body)).into_response()
    }
}

