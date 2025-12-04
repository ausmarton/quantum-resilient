//! Control Plane Module
//!
//! This module provides HTTP endpoints for health checks, readiness probes,
//! runtime status for Kubernetes deployment, and orchestrator coordination.

pub mod http;

pub use http::{start_control_plane_server, ControlPlaneState, OrchestrationState};

