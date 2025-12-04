//! Tracing Setup
//!
//! Initializes tracing infrastructure for distributed tracing and logging.

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initializes the tracing subscriber with console output
///
/// Sets up a tracing subscriber that outputs spans and events to stdout.
/// The subscriber uses the RUST_LOG environment variable for filtering,
/// defaulting to "info" level.
pub fn init_tracing(service_name: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_file(false)
                .with_line_number(false)
                .with_ansi(true),
        );

    if tracing::subscriber::set_global_default(subscriber).is_err() {
        // Subscriber already set, ignore
    }

    tracing::info!(service = service_name, "Tracing initialized");
}

/// Creates a span for a crypto operation
#[macro_export]
macro_rules! crypto_span {
    ($algorithm:expr, $operation:expr, $event_id:expr, $payload_size:expr) => {
        tracing::info_span!(
            "crypto_operation",
            algorithm = $algorithm,
            operation = $operation,
            event_id = $event_id,
            payload_size = $payload_size
        )
    };
}

pub use crypto_span;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_tracing() {
        // Just verify it doesn't panic
        init_tracing("test-service");
    }
}

