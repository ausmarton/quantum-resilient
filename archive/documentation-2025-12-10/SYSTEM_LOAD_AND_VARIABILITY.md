# System Load and Variability Guide

This guide explains how system load affects benchmark results and how the framework accounts for variability to ensure academic rigor.

## Impact of System Load by Environment

### Native Mode (Local Machine)

**High Impact** - Directly affected by system load:
- ✅ **CPU contention**: Other processes compete for CPU time
- ✅ **Memory pressure**: Can cause swapping, affecting performance
- ✅ **I/O contention**: Disk/network activity can slow operations
- ✅ **Thermal throttling**: High system load can trigger CPU throttling

**Recommendations:**
- Close unnecessary applications (browser, IDE, etc.)
- Disable background services if possible
- Use `nice` or `ionice` to prioritize benchmarks
- Monitor system load during runs

### Minikube Mode (Local Container)

**Medium Impact** - Some isolation, but still affected:
- ⚠️ **CPU sharing**: Containers share host CPU, but with some isolation
- ⚠️ **Memory limits**: Container memory limits help, but host memory pressure still matters
- ✅ **I/O isolation**: Better than native, but still shares host I/O
- ⚠️ **Host system load**: High host load can affect container performance

**Recommendations:**
- Close heavy applications (browser, IDE)
- Monitor container resource usage
- Consider using resource limits in Minikube

### GCP Mode (Cloud VMs)

**Low Impact** - Best isolation:
- ✅ **Dedicated VMs**: Each experiment gets dedicated resources
- ✅ **Resource guarantees**: CPU and memory are guaranteed
- ✅ **Network isolation**: Dedicated network resources
- ⚠️ **Noisy neighbor**: Rare, but possible in shared infrastructure

**Recommendations:**
- No need to close local applications (runs in cloud)
- Use dedicated machine types for consistency
- Monitor VM metrics during runs

## How Variability is Accounted For

### 1. Multiple Runs (5 per Configuration)

The experiment matrix specifies **5 runs per configuration**. This is the primary mechanism for handling variability:

- **Statistical power**: 5 runs provide sufficient data for statistical analysis
- **Outlier detection**: Extreme values from system load spikes are identified
- **Confidence intervals**: Multiple runs enable calculation of confidence intervals
- **Distribution analysis**: Can analyze distribution shape, not just mean

**Example:**
```
Configuration: rsa2048, 256B payload, 100 msg/s
Run 1: 150 μs (low system load)
Run 2: 152 μs (low system load)
Run 3: 180 μs (browser opened during run)
Run 4: 151 μs (low system load)
Run 5: 153 μs (low system load)

Mean: 157 μs
Std Dev: 12.5 μs
Outlier: Run 3 (can be identified and handled statistically)
```

### 2. System Metrics Captured

Each event captures system state:
- `cpu_user_time_us`: CPU usage at time of operation
- `memory_rss_bytes`: Memory usage at time of operation
- `queue_delay_us`: Queue delay (indicates system load)

This allows:
- **Post-hoc analysis**: Identify runs with high system load
- **Correlation analysis**: Correlate latency with system metrics
- **Filtering**: Optionally filter high-load events from analysis

### 3. Statistical Methods

The analysis pipeline uses:
- **Percentiles (p50, p95, p99)**: Less sensitive to outliers than mean
- **Confidence intervals**: Account for variability across runs
- **Robust statistics**: Median-based measures are less affected by outliers
- **Distribution analysis**: CDFs show full distribution, not just point estimates

### 4. System Metadata

Each experiment captures:
- CPU model and frequency
- Kernel version
- System load at start (if available)
- Hardware configuration

This enables:
- **Reproducibility**: Document exact system state
- **Comparison**: Compare across different system configurations
- **Analysis**: Identify if system load correlates with results

## Best Practices by Environment

### Native Mode: Recommended Setup

**Before running:**
```bash
# Check current system load
uptime
top -bn1 | head -5

# Close heavy applications
# - Browser (especially with many tabs)
# - IDE (if not needed)
# - Other development tools

# Set process priority (optional)
nice -n -10 ./run_full_scale_data_collection.sh --env native
```

**During runs:**
- Avoid opening new applications
- Avoid heavy I/O operations (file copies, downloads)
- Monitor system load: `watch -n 1 'uptime; free -h'`

**What you CAN keep open:**
- ✅ Terminal/SSH sessions
- ✅ Light text editors
- ✅ System monitoring tools
- ✅ Background services (if not CPU-intensive)

**What to CLOSE:**
- ❌ Web browsers (especially with many tabs)
- ❌ IDEs (if not actively needed)
- ❌ Video players
- ❌ Large file operations
- ❌ Compilation jobs
- ❌ Virtual machines (if running)

### Minikube Mode: Recommended Setup

**Before running:**
```bash
# Check Minikube resource allocation
minikube status
kubectl top nodes

# Close heavy applications (less critical than native)
# - Browser (if many tabs)
# - Heavy IDEs
```

**During runs:**
- Monitor container resources: `kubectl top pods`
- Avoid heavy host I/O operations
- Less critical than native mode, but still recommended

### GCP Mode: Recommended Setup

**No local restrictions needed:**
- Runs in cloud, isolated from your laptop
- Can use your laptop normally during GCP runs
- Only consideration: Network bandwidth for result download

## Monitoring System Load

### During Native Runs

Create a monitoring script:

```bash
#!/bin/bash
# monitor_system_load.sh

while true; do
    clear
    echo "=== System Load Monitor ===" 
    date
    echo ""
    echo "Load Average:"
    uptime
    echo ""
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | head -1
    echo ""
    echo "Memory:"
    free -h
    echo ""
    echo "Top Processes:"
    ps aux --sort=-%cpu | head -6
    sleep 5
done
```

Run in separate terminal:
```bash
./monitor_system_load.sh
```

### Analyzing System Load in Results

After collection, analyze system load impact:

```python
import pandas as pd
import json

# Load experiment data
df = pd.read_json('results/native/rsa2048_p256_r100_run1_*/merged/merged.jsonl', lines=True)

# Check CPU usage distribution
print("CPU Usage Statistics:")
print(df['cpu_user_time_us'].describe())

# Identify high-load events
high_load = df[df['cpu_user_time_us'] > df['cpu_user_time_us'].quantile(0.95)]
print(f"\nHigh-load events: {len(high_load)} ({len(high_load)/len(df)*100:.1f}%)")

# Correlate with latency
correlation = df['latency_us'].corr(df['cpu_user_time_us'])
print(f"\nLatency-CPU correlation: {correlation:.3f}")
```

## Handling Variability in Analysis

### Option 1: Include All Data (Recommended)

**Default approach**: Include all runs, even those with higher system load.

**Rationale:**
- Represents real-world variability
- Statistical methods (percentiles, CIs) handle outliers
- Multiple runs provide robustness

**Analysis:**
- Use percentiles (p50, p95, p99) instead of just mean
- Calculate confidence intervals
- Report both mean and median

### Option 2: Filter High-Load Events

If system load is extreme, you can filter:

```python
# Filter events with CPU usage > 95th percentile
cpu_threshold = df['cpu_user_time_us'].quantile(0.95)
filtered_df = df[df['cpu_user_time_us'] <= cpu_threshold]

# Re-analyze with filtered data
# ... statistical analysis ...
```

**When to use:**
- System load is extreme (e.g., >50% CPU from other processes)
- You want to measure "best case" performance
- Document the filtering in your methodology

**Important**: Always document if you filter data!

### Option 3: Separate Analysis by Load

Analyze high-load and low-load runs separately:

```python
# Separate by median CPU usage
median_cpu = df['cpu_user_time_us'].median()
low_load = df[df['cpu_user_time_us'] <= median_cpu]
high_load = df[df['cpu_user_time_us'] > median_cpu]

# Compare results
print("Low load latency:", low_load['latency_us'].mean())
print("High load latency:", high_load['latency_us'].mean())
```

## Academic Rigor Considerations

### What to Document

In your dissertation, document:

1. **System Configuration:**
   - CPU model and frequency
   - Memory size
   - Operating system version
   - Kernel version

2. **System Load:**
   - Average system load during runs
   - Any applications running
   - Process priorities used

3. **Variability Handling:**
   - Number of runs per configuration (5)
   - Statistical methods used (percentiles, CIs)
   - Any data filtering (if applied)

4. **Reproducibility:**
   - Exact commands used
   - System state documentation
   - Random seed values (for reproducibility)

### Example Methodology Section

```
System Load and Variability

Benchmarks were run on a dedicated development machine with the following 
specifications: [CPU model, memory, OS]. To minimize variability from system 
load, we:

1. Closed unnecessary applications (browser, IDE) during native runs
2. Used process prioritization (nice -10) for benchmark processes
3. Ran 5 independent runs per configuration to account for residual variability
4. Captured system metrics (CPU, memory) with each event for post-hoc analysis

Statistical analysis used percentiles (p50, p95, p99) and confidence intervals 
to account for variability. No data filtering was applied, as statistical 
methods (robust statistics, confidence intervals) adequately handle outliers 
from occasional system load spikes.

For GCP runs, experiments used dedicated VMs with guaranteed resources, 
providing better isolation from system load variability.
```

## Quick Reference

| Environment | Close Apps? | Monitor Load? | Impact Level |
|-------------|-------------|--------------|--------------|
| **Native** | ✅ Yes (browser, IDE) | ✅ Yes | High |
| **Minikube** | ⚠️ Recommended | ⚠️ Optional | Medium |
| **GCP** | ❌ No | ❌ No | Low |

## Summary

✅ **Multiple runs (5 per config)** handle most variability  
✅ **Statistical methods** (percentiles, CIs) are robust to outliers  
✅ **System metrics captured** enable post-hoc analysis  
✅ **Native mode**: Close heavy apps, monitor load  
✅ **Minikube**: Less critical, but still recommended  
✅ **GCP**: No local restrictions needed  

The framework is designed to handle variability through multiple runs and robust statistics. While minimizing system load is good practice (especially for native mode), the statistical approach ensures results are valid even with some variability.

