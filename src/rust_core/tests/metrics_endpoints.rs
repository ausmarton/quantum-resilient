use rust_core::metrics::PrometheusMetricsCollector;
use rust_core::{MetricsCollector, OperationKind, OperationMetrics};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn get_free_port() -> u16 {
	let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
	let port = listener.local_addr().unwrap().port();
	drop(listener);
	port
}

fn http_get(host: &str, port: u16, path: &str) -> String {
	let mut stream = TcpStream::connect((host, port)).expect("connect");
	let req = format!("GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", path, host);
	stream.write_all(req.as_bytes()).unwrap();
	let mut buf = String::new();
	stream.read_to_string(&mut buf).unwrap();
	buf
}

#[test]
fn prometheus_endpoint_exposes_metrics() {
	let collector = Arc::new(PrometheusMetricsCollector::new());
	// record one event
	let evt = OperationMetrics {
		timestamp_seconds_utc: Some(chrono::Utc::now()),
		operation: OperationKind::Keygen,
		latency_micros: 100,
		attempts: Some(1),
		error: None,
		cpu_user_micros: None,
		cpu_system_micros: None,
		max_rss_bytes: None,
		algorithm: Some("test".into()),
		parameter_set: None,
		public_key_bytes: None,
		secret_key_bytes: None,
		signature_bytes: None,
		ciphertext_bytes: None,
		storage_overhead_pct: None,
		keygen_time_ms: Some(0.1),
		encapsulate_time_ms: None,
		decapsulate_time_ms: None,
		encrypt_time_ms: None,
		decrypt_time_ms: None,
		sign_time_ms: None,
		verify_time_ms: None,
		throughput_ops_per_sec: None,
		avg_cpu_percent: None,
		avg_memory_mb: None,
		disk_io_bytes: None,
		net_tx_bytes: None,
		net_rx_bytes: None,
	};
	collector.record(&evt);
	let port = get_free_port();
	let addr = format!("127.0.0.1:{}", port);
	let handle = {
		let c = collector.clone();
		c.start_http_server(&addr)
	};
	// wait briefly for server
	thread::sleep(Duration::from_millis(200));
	let body = http_get("127.0.0.1", port, "/metrics");
	// stop server thread by drop (test end)
	handle.thread().unpark();
	assert!(body.contains("crypto_operation_latency_seconds"), "metrics body missing histogram");
}


