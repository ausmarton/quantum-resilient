//! PQC Benchmark Framework Binary
//!
//! This is the main entry point for running cryptographic benchmarks
//! comparing Post-Quantum Cryptography (PQC) with classical algorithms.

use rust_core::Pipeline;

fn main() {
    println!("Starting PQC Benchmark Framework...");

    // Initialize the benchmark pipeline
    let mut pipeline = Pipeline::new();

    // Initialize pipeline resources
    if let Err(e) = pipeline.init() {
        eprintln!("Failed to initialize pipeline: {:?}", e);
        std::process::exit(1);
    }

    // Run the benchmark pipeline
    match pipeline.run() {
        Ok(()) => {
            println!("pipeline OK");
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

