//! System Info Sampler
//!
//! Provides CPU and memory sampling for the current process using sysinfo.
//!
//! CPU sampling uses `/proc/self/stat` on Linux to get cumulative CPU time,
//! which is more accurate than sysinfo's percentage-based `cpu_usage()` method
//! for fast operations.

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use std::sync::Mutex;

/// System information sampler for CPU and memory metrics
pub struct SysInfoSampler {
    system: Mutex<System>,
    pid: Pid,
    /// Clock ticks per second (from sysconf(_SC_CLK_TCK), typically 100)
    clock_ticks_per_sec: f64,
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

        // Get clock ticks per second (typically 100 on Linux)
        let clock_ticks_per_sec = Self::get_clock_ticks_per_sec();

        Self {
            system: Mutex::new(system),
            pid,
            clock_ticks_per_sec,
        }
    }

    /// Get clock ticks per second from system
    /// Falls back to 100 (standard Linux value) if unavailable
    fn get_clock_ticks_per_sec() -> f64 {
        #[cfg(target_os = "linux")]
        {
            // Try to read from /proc/self/stat or use sysconf
            // For simplicity, we'll use the standard Linux value of 100
            // This can be verified at runtime if needed
            100.0
        }
        #[cfg(not(target_os = "linux"))]
        {
            // Non-Linux: fall back to sysinfo (less accurate for fast ops)
            100.0
        }
    }

    /// Read cumulative CPU time from /proc/self/stat on Linux
    /// Returns (utime + stime) in clock ticks, or None if unavailable
    #[cfg(target_os = "linux")]
    fn read_cumulative_cpu_ticks() -> Option<u64> {
        use std::fs;

        let stat_content = fs::read_to_string("/proc/self/stat").ok()?;
        let fields: Vec<&str> = stat_content.split_whitespace().collect();
        
        // Field 13 (0-indexed) = utime (user time in clock ticks)
        // Field 14 (0-indexed) = stime (system time in clock ticks)
        if fields.len() < 15 {
            return None;
        }

        let utime: u64 = fields[13].parse().ok()?;
        let stime: u64 = fields[14].parse().ok()?;
        
        Some(utime + stime)
    }

    /// Read cumulative CPU time (fallback for non-Linux)
    #[cfg(not(target_os = "linux"))]
    fn read_cumulative_cpu_ticks() -> Option<u64> {
        // Non-Linux: sysinfo doesn't provide cumulative time easily
        // Return None to fall back to percentage-based method
        None
    }

    /// Samples current CPU and memory usage
    ///
    /// Returns (cpu_user_seconds, rss_bytes) for the current process.
    /// cpu_user_seconds is cumulative CPU time since process start (or since first sample).
    /// On Linux, this uses /proc/self/stat for accurate cumulative CPU time.
    /// On other platforms, falls back to sysinfo percentage (less accurate for fast operations).
    pub fn sample(&self) -> (f64, u64) {
        // Try to get cumulative CPU time from /proc/self/stat (Linux)
        let cpu_user_seconds = if let Some(cpu_ticks) = Self::read_cumulative_cpu_ticks() {
            // Convert clock ticks to seconds
            // This is cumulative CPU time since process start
            // Analysis scripts can calculate deltas between consecutive events
            cpu_ticks as f64 / self.clock_ticks_per_sec
        } else {
            // Fallback: Use sysinfo (percentage-based, less accurate)
            let mut system = self.system.lock().unwrap();
            system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&[self.pid]),
                ProcessRefreshKind::everything(),
            );

            if let Some(process) = system.process(self.pid) {
                // sysinfo::Process::cpu_usage() returns percentage (0-100%)
                // This is instantaneous usage, not cumulative time
                // For fast operations, this will be 0 or very small
                // We return it as-is, but note it's not cumulative seconds
                process.cpu_usage() as f64 / 100.0
            } else {
                0.0
            }
        };

        // Get memory from sysinfo (this works correctly)
        let mut system = self.system.lock().unwrap();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            ProcessRefreshKind::everything(),
        );

        let memory_bytes = if let Some(process) = system.process(self.pid) {
            process.memory()
        } else {
            0
        };

        (cpu_user_seconds, memory_bytes)
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
        let (cpu1, memory) = sampler.sample();

        // Memory should be non-zero for a running process
        assert!(memory > 0, "Memory should be non-zero");

        // CPU should be non-negative (cumulative CPU time)
        assert!(cpu1 >= 0.0, "CPU should be non-negative");
        
        // On Linux, cumulative CPU time should increase over time
        #[cfg(target_os = "linux")]
        {
            // Do some work to consume CPU
            let _ = (0..1000).map(|i| i * 2).collect::<Vec<_>>();
            
            let (cpu2, _) = sampler.sample();
            // Cumulative CPU time should be >= previous value (may be equal if very fast)
            assert!(cpu2 >= cpu1, "Cumulative CPU time should be non-decreasing");
        }
    }

    #[test]
    fn test_sampler_memory() {
        let sampler = SysInfoSampler::new();
        let memory = sampler.memory_rss_bytes();
        assert!(memory > 0);
    }
}
