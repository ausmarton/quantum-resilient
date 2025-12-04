//! Control Plane Module
//!
//! This module provides HTTP endpoints for health checks, readiness probes,
//! and runtime status for Kubernetes deployment.

pub mod http;

pub use http::{start_control_plane_server, ControlPlaneState};


