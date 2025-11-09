use crate::{MetricsCollector, OperationMetrics};
use prometheus::{Encoder, Gauge, Histogram, HistogramOpts, Opts, Registry, TextEncoder};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tiny_http::{Response, Server};

pub struct JsonFileMetricsCollector {
	file: Arc<Mutex<std::fs::File>>,
}

impl JsonFileMetricsCollector {
	pub fn new(path: PathBuf) -> std::io::Result<Self> {
		let file = OpenOptions::new().create(true).append(true).open(path)?;
		Ok(Self { file: Arc::new(Mutex::new(file)) })
	}
}

impl MetricsCollector for JsonFileMetricsCollector {
	fn record(&self, metrics: &OperationMetrics) {
		if let Ok(mut f) = self.file.lock() {
			if let Ok(line) = serde_json::to_string(metrics) {
				let _ = writeln!(f, "{}", line);
			}
		}
	}
}

pub struct PrometheusMetricsCollector {
	registry: Registry,
	latency_seconds: Histogram,
	cpu_user_seconds: Gauge,
	cpu_system_seconds: Gauge,
	max_rss_bytes: Gauge,
}

impl PrometheusMetricsCollector {
	pub fn new() -> Self {
		let registry = Registry::new();
		let latency_seconds = Histogram::with_opts(
			HistogramOpts::new("crypto_operation_latency_seconds", "Operation latency in seconds")
				.buckets(vec![
					0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
				]),
		)
		.unwrap();
		let cpu_user_seconds = Gauge::with_opts(Opts::new("crypto_cpu_user_seconds", "User CPU time for process")).unwrap();
		let cpu_system_seconds = Gauge::with_opts(Opts::new("crypto_cpu_system_seconds", "System CPU time for process")).unwrap();
		let max_rss_bytes = Gauge::with_opts(Opts::new("crypto_max_rss_bytes", "Max resident set size in bytes")).unwrap();
		registry.register(Box::new(latency_seconds.clone())).ok();
		registry.register(Box::new(cpu_user_seconds.clone())).ok();
		registry.register(Box::new(cpu_system_seconds.clone())).ok();
		registry.register(Box::new(max_rss_bytes.clone())).ok();
		Self { registry, latency_seconds, cpu_user_seconds, cpu_system_seconds, max_rss_bytes }
	}

	pub fn registry(&self) -> &Registry {
		&self.registry
	}

	pub fn start_http_server(self: Arc<Self>, addr: &str) -> std::thread::JoinHandle<()> {
		let addr_string = addr.to_string();
		std::thread::spawn(move || {
			let server = Server::http(&addr_string).expect("failed to start metrics server");
			for request in server.incoming_requests() {
				if request.url() == "/metrics" {
					let encoder = TextEncoder::new();
					let metric_families = self.registry.gather();
					let mut buffer = Vec::new();
					if let Err(_e) = encoder.encode(&metric_families, &mut buffer) {
						let _ = request.respond(Response::from_string("encode error").with_status_code(500));
						continue;
					}
					let response = Response::from_data(buffer).with_status_code(200);
					let _ = request.respond(response);
				} else {
					let _ = request.respond(Response::from_string("not found").with_status_code(404));
				}
			}
		})
	}
}

impl MetricsCollector for PrometheusMetricsCollector {
	fn record(&self, metrics: &OperationMetrics) {
		// Prometheus prefers seconds; convert from micros
		let seconds = (metrics.latency_micros as f64) / 1_000_000.0;
		self.latency_seconds.observe(seconds);
		if let Some(us) = metrics.cpu_user_micros {
			self.cpu_user_seconds.set((us as f64) / 1_000_000.0);
		}
		if let Some(us) = metrics.cpu_system_micros {
			self.cpu_system_seconds.set((us as f64) / 1_000_000.0);
		}
		if let Some(bytes) = metrics.max_rss_bytes {
			self.max_rss_bytes.set(bytes as f64);
		}
	}
}


