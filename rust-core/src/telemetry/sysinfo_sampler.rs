//! System Info Sampler
//!
//! Provides CPU and memory sampling for the current process using sysinfo.

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use std::sync::Mutex;

/// System information sampler for CPU and memory metrics
pub struct SysInfoSampler {
    system: Mutex<System>,
    pid: Pid,
}

impl SysInfoSampler {
    /// Creates a new sampler for the current process
    pub fn new() -> Self {
        let mut system = System::new();
        let pid = Pid::from_u32(std::process::id());
        
        // Initial refresh to populate process data
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            ProcessRefreshKind::everything(),
        );

        Self {
            system: Mutex::new(system),
            pid,
        }
    }

    /// Samples current CPU and memory usage
    ///
    /// Returns (cpu_user_seconds, rss_bytes) for the current process.
    /// cpu_user_seconds is an approximation based on CPU usage percentage.
    pub fn sample(&self) -> (f64, u64) {
        let mut system = self.system.lock().unwrap();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            ProcessRefreshKind::everything(),
        );

        if let Some(process) = system.process(self.pid) {
            let cpu_usage = process.cpu_usage() as f64 / 100.0; // Convert percentage to fraction
            let memory_bytes = process.memory(); // Returns bytes

            (cpu_usage, memory_bytes)
        } else {
            (0.0, 0)
        }
    }

    /// Returns just the RSS memory in bytes
    pub fn memory_rss_bytes(&self) -> u64 {
        let (_, rss) = self.sample();
        rss
    }

    /// Returns CPU usage as a fraction (0.0 to 1.0+)
    pub fn cpu_usage(&self) -> f64 {
        let (cpu, _) = self.sample();
        cpu
    }
}

impl Default for SysInfoSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SysInfoSampler {
    fn clone(&self) -> Self {
        // Create a new sampler instance since System is not Clone
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_creation() {
        let sampler = SysInfoSampler::new();
        let (cpu, memory) = sampler.sample();

        // Memory should be non-zero for a running process
        assert!(memory > 0, "Memory should be non-zero");

        // CPU can be 0 if sampled too quickly
        assert!(cpu >= 0.0, "CPU should be non-negative");
    }

    #[test]
    fn test_sampler_memory() {
        let sampler = SysInfoSampler::new();
        let memory = sampler.memory_rss_bytes();
        assert!(memory > 0);
    }
}
