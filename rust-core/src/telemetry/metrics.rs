//! Prometheus Metrics Server
//!
//! Provides Prometheus metrics collection and HTTP exposition.

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, Opts, Registry,
    TextEncoder,
};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Prometheus metrics container
#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    registry: Registry,
    latency_histogram: Histogram,
    ops_total: CounterVec,
    current_rps: Gauge,
    memory_bytes: Gauge,
    events_processed: Counter,
    // NEW: Queue metrics
    queue_length: Gauge,
    queue_capacity: Gauge,
    queue_delay_histogram: Histogram,
    // NEW: Worker metrics
    active_workers: Gauge,
    worker_events: GaugeVec,
}

impl Metrics {
    /// Creates a new Metrics instance with registered metrics
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();

        // Histogram for operation latency in microseconds
        let latency_histogram = Histogram::with_opts(
            HistogramOpts::new(
                "pqc_operation_latency_us",
                "Latency of cryptographic operations in microseconds",
            )
            .buckets(vec![
                0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0,
                100000.0,
            ]),
        )?;
        registry.register(Box::new(latency_histogram.clone()))?;

        // Counter for total operations
        let ops_total = CounterVec::new(
            Opts::new("pqc_ops_total", "Total number of cryptographic operations"),
            &["algorithm", "operation", "success"],
        )?;
        registry.register(Box::new(ops_total.clone()))?;

        // Gauge for current RPS
        let current_rps = Gauge::new("pqc_current_rps", "Current operations per second")?;
        registry.register(Box::new(current_rps.clone()))?;

        // Gauge for memory usage
        let memory_bytes = Gauge::new("pqc_memory_bytes", "Current process memory usage in bytes")?;
        registry.register(Box::new(memory_bytes.clone()))?;

        // Counter for events processed
        let events_processed =
            Counter::new("pqc_events_processed_total", "Total events processed")?;
        registry.register(Box::new(events_processed.clone()))?;

        // NEW: Queue length gauge
        let queue_length = Gauge::new("pqc_queue_length", "Current number of events in queue")?;
        registry.register(Box::new(queue_length.clone()))?;

        // NEW: Queue capacity gauge
        let queue_capacity = Gauge::new("pqc_queue_capacity", "Maximum queue capacity")?;
        registry.register(Box::new(queue_capacity.clone()))?;

        // NEW: Queue delay histogram (queueing delay in microseconds)
        let queue_delay_histogram = Histogram::with_opts(
            HistogramOpts::new(
                "pqc_queue_delay_us",
                "Time spent waiting in queue in microseconds",
            )
            .buckets(vec![
                1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0,
                500000.0, 1000000.0,
            ]),
        )?;
        registry.register(Box::new(queue_delay_histogram.clone()))?;

        // NEW: Active workers gauge
        let active_workers = Gauge::new("pqc_active_workers", "Current number of active workers")?;
        registry.register(Box::new(active_workers.clone()))?;

        // NEW: Worker events counter per worker
        let worker_events = GaugeVec::new(
            Opts::new("pqc_worker_events", "Events processed per worker"),
            &["worker_id"],
        )?;
        registry.register(Box::new(worker_events.clone()))?;

        Ok(Self {
            inner: Arc::new(MetricsInner {
                registry,
                latency_histogram,
                ops_total,
                current_rps,
                memory_bytes,
                events_processed,
                queue_length,
                queue_capacity,
                queue_delay_histogram,
                active_workers,
                worker_events,
            }),
        })
    }

    /// Observes a latency measurement
    pub fn observe_latency(&self, _algo: &str, _op: &str, us: f64) {
        self.inner.latency_histogram.observe(us);
    }

    /// Increments the operation counter
    pub fn inc_ops(&self, algo: &str, op: &str, success: bool) {
        let success_str = if success { "true" } else { "false" };
        self.inner
            .ops_total
            .with_label_values(&[algo, op, success_str])
            .inc();
        self.inner.events_processed.inc();
    }

    /// Sets the current RPS gauge
    pub fn set_current_rps(&self, rps: f64) {
        self.inner.current_rps.set(rps);
    }

    /// Sets the memory bytes gauge
    pub fn set_memory_bytes(&self, bytes: u64) {
        self.inner.memory_bytes.set(bytes as f64);
    }

    /// Returns the total events processed
    pub fn events_processed(&self) -> u64 {
        self.inner.events_processed.get() as u64
    }

    // NEW: Queue metrics methods

    /// Sets the current queue length
    pub fn set_queue_length(&self, len: usize) {
        self.inner.queue_length.set(len as f64);
    }

    /// Gets the current queue length
    pub fn get_queue_length(&self) -> usize {
        self.inner.queue_length.get() as usize
    }

    /// Sets the queue capacity
    pub fn set_queue_capacity(&self, capacity: usize) {
        self.inner.queue_capacity.set(capacity as f64);
    }

    /// Gets the queue capacity
    pub fn get_queue_capacity(&self) -> usize {
        self.inner.queue_capacity.get() as usize
    }

    /// Observes a queue delay measurement in microseconds
    pub fn observe_queue_delay(&self, us: f64) {
        self.inner.queue_delay_histogram.observe(us);
    }

    // NEW: Worker metrics methods

    /// Sets the number of active workers
    pub fn set_active_workers(&self, count: usize) {
        self.inner.active_workers.set(count as f64);
    }

    /// Gets the number of active workers
    pub fn get_active_workers(&self) -> usize {
        self.inner.active_workers.get() as usize
    }

    /// Increments events processed for a specific worker
    pub fn inc_worker_events(&self, worker_id: usize) {
        self.inner
            .worker_events
            .with_label_values(&[&worker_id.to_string()])
            .inc();
    }

    /// Gathers all metrics as Prometheus text format
    pub fn gather(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.inner.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new().expect("Failed to create default metrics")
    }
}

/// Starts the Prometheus metrics HTTP server
///
/// Returns a JoinHandle that can be used to await the server
pub async fn start_metrics_server(
    addr: &str,
    metrics: Metrics,
) -> tokio::task::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>> {
    let addr: SocketAddr = addr.parse().expect("Invalid metrics server address");

    tokio::spawn(async move {
        let listener = TcpListener::bind(addr).await?;

        loop {
            let (stream, _) = listener.accept().await?;
            let io = TokioIo::new(stream);
            let metrics_clone = metrics.clone();

            tokio::spawn(async move {
                let service = service_fn(move |req| {
                    let metrics = metrics_clone.clone();
                    async move { handle_request(req, metrics).await }
                });

                if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                    eprintln!("Error serving connection: {:?}", err);
                }
            });
        }
    })
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    metrics: Metrics,
) -> Result<Response<Full<Bytes>>, Infallible> {
    let response = match req.uri().path() {
        "/metrics" => {
            let body = metrics.gather();
            Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/plain; charset=utf-8")
                .body(Full::new(Bytes::from(body)))
                .unwrap()
        }
        "/health" => Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::from("OK")))
            .unwrap(),
        _ => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from("Not Found")))
            .unwrap(),
    };

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new().unwrap();
        assert!(metrics.gather().contains("pqc_operation_latency_us"));
    }

    #[test]
    fn test_metrics_observe_latency() {
        let metrics = Metrics::new().unwrap();
        metrics.observe_latency("noop", "sign", 100.0);
        metrics.observe_latency("noop", "sign", 200.0);

        let output = metrics.gather();
        assert!(output.contains("pqc_operation_latency_us"));
    }

    #[test]
    fn test_metrics_inc_ops() {
        let metrics = Metrics::new().unwrap();
        metrics.inc_ops("noop", "sign", true);
        metrics.inc_ops("noop", "sign", false);

        let output = metrics.gather();
        assert!(output.contains("pqc_ops_total"));
    }

    #[test]
    fn test_metrics_gauges() {
        let metrics = Metrics::new().unwrap();
        metrics.set_current_rps(100.0);
        metrics.set_memory_bytes(1024 * 1024);

        let output = metrics.gather();
        assert!(output.contains("pqc_current_rps"));
        assert!(output.contains("pqc_memory_bytes"));
    }

    #[test]
    fn test_queue_metrics() {
        let metrics = Metrics::new().unwrap();
        metrics.set_queue_length(100);
        metrics.set_queue_capacity(2000);
        metrics.observe_queue_delay(500.0);

        let output = metrics.gather();
        assert!(output.contains("pqc_queue_length"));
        assert!(output.contains("pqc_queue_capacity"));
        assert!(output.contains("pqc_queue_delay_us"));

        assert_eq!(metrics.get_queue_length(), 100);
        assert_eq!(metrics.get_queue_capacity(), 2000);
    }

    #[test]
    fn test_worker_metrics() {
        let metrics = Metrics::new().unwrap();
        metrics.set_active_workers(4);
        metrics.inc_worker_events(0);
        metrics.inc_worker_events(1);

        let output = metrics.gather();
        assert!(output.contains("pqc_active_workers"));
        assert!(output.contains("pqc_worker_events"));

        assert_eq!(metrics.get_active_workers(), 4);
    }
}
