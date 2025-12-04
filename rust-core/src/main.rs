//! PQC Benchmark Framework Binary
//!
//! This is the main entry point for running cryptographic benchmarks
//! comparing Post-Quantum Cryptography (PQC) with classical algorithms.

use clap::Parser;
use rust_core::{load_scenario, CryptoAdapter, NoOpCryptoAdapter, Pipeline};

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

    // Instantiate the appropriate crypto adapter
    let adapter: Box<dyn CryptoAdapter> = match scenario.algorithm.adapter.as_str() {
        "noop" => Box::new(NoOpCryptoAdapter::new()),
        unknown => {
            eprintln!("Error: Unknown adapter '{}'. Supported: noop", unknown);
            std::process::exit(1);
        }
    };

    println!("Using adapter: {}", adapter.name());

    // Initialize the benchmark pipeline
    let mut pipeline = Pipeline::new();

    // Initialize pipeline resources
    if let Err(e) = pipeline.init() {
        eprintln!("Failed to initialize pipeline: {:?}", e);
        std::process::exit(1);
    }

    println!("Pipeline ready — running warm-up...");

    // Run the benchmark pipeline
    match pipeline.run() {
        Ok(()) => {
            println!("Pipeline OK ({})", adapter.name());
        }
        Err(e) => {
            eprintln!("Pipeline execution failed: {:?}", e);
            std::process::exit(1);
        }
    }

    // Cleanup
    if let Err(e) = pipeline.shutdown() {
        eprintln!("Warning: Pipeline shutdown error: {:?}", e);
    }
}
