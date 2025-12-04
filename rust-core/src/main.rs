//! PQC Benchmark Framework Binary
//!
//! This is the main entry point for running cryptographic benchmarks
//! comparing Post-Quantum Cryptography (PQC) with classical algorithms.

use clap::Parser;
use rust_core::{
    get_adapter, load_scenario, supported_adapters, supported_operations, Pipeline,
};

/// PQC Benchmark Framework - Compare PQC vs classical cryptography performance
#[derive(Parser, Debug)]
#[command(name = "pqc-bench")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the scenario YAML file
    #[arg(long)]
    scenario: String,
}

fn main() {
    println!("Starting PQC Benchmark Framework...");

    // Parse command line arguments
    let args = Args::parse();

    // Load scenario from YAML file
    let scenario = match load_scenario(&args.scenario) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    println!("Loaded scenario: {}", scenario.id);

    // Instantiate the appropriate crypto adapter from registry
    let adapter = match get_adapter(&scenario.algorithm.adapter) {
        Ok(a) => a,
        Err(_) => {
            eprintln!(
                "Error: Unknown adapter '{}'. Supported adapters: {}",
                scenario.algorithm.adapter,
                supported_adapters().join(", ")
            );
            std::process::exit(1);
        }
    };

    println!("Using adapter: {}", adapter.name());

    // Validate operation
    let operation = &scenario.algorithm.operation;
    if !supported_operations().contains(&operation.as_str()) {
        eprintln!(
            "Error: Unknown operation '{}'. Supported operations: {}",
            operation,
            supported_operations().join(", ")
        );
        std::process::exit(1);
    }

    println!("Running operation: {}", operation);

    // Generate dummy payload based on workload config
    let payload = vec![0u8; scenario.workload.msg_size_bytes];

    // Calculate total number of operations
    let total_ops = scenario.workload.duration_sec as u32 * scenario.workload.msgs_per_sec;

    // Run timed operations
    for i in 1..=total_ops {
        match Pipeline::run_timed_operation(adapter.as_ref(), operation, &payload) {
            Ok(duration_us) => {
                println!("Event {}: {} μs", i, duration_us);
            }
            Err(e) => {
                eprintln!("Error during operation {}: {}", i, e);
                std::process::exit(1);
            }
        }
    }

    println!("Done.");
}
