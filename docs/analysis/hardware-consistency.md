# Hardware Consistency Analysis: Cross-Environment Comparisons

## Critical Issue: Hardware Differences Across Environments

### Current Hardware Configuration

| Environment | CPU | CPU Count | Memory | Notes |
|------------|-----|-----------|--------|-------|
| **Native** | AMD RYZEN AI MAX+ PRO 395 | Many cores | 94 GB | Local machine |
| **Minikube** | AMD RYZEN AI MAX+ PRO 395 | Same as native | 94 GB | Same machine, containerized |
| **GCP** | Intel Xeon @ 2.80GHz | 2 vCPUs | ~8 GB | n2-standard-2 VM |

### The Problem

**Hardware differences confound environment comparisons:**

1. **CPU Architecture**: AMD vs Intel
   - Different instruction sets
   - Different performance characteristics
   - Different cryptographic acceleration (if any)

2. **CPU Count**: Many cores (native/minikube) vs 2 vCPUs (GCP)
   - Native/minikube can use more parallelism
   - GCP is more constrained
   - Affects throughput capabilities

3. **Memory**: 94 GB (native/minikube) vs ~8 GB (GCP)
   - Native/minikube have more headroom
   - GCP may experience memory pressure
   - Affects caching behavior

4. **Performance Baseline**: Different absolute performance
   - Cannot directly compare absolute latencies
   - Relative differences are confounded by hardware

---

## What This Means for Analysis

### ❌ Cannot Do (Invalid Comparisons)

1. **Direct Absolute Comparisons**
   - ❌ "GCP latency is X μs vs native Y μs" (hardware confounds)
   - ❌ "GCP is X% slower than native" (hardware + environment)
   - ❌ Direct throughput comparisons (CPU count differs)

2. **Unnormalized Performance Claims**
   - ❌ "Containerization adds X% overhead" (if comparing native vs GCP)
   - ❌ "Cloud deployment degrades performance by Y%" (hardware confounds)

### ✅ Can Do (Valid Comparisons)

1. **Native vs Minikube (Same Hardware)**
   - ✅ "Containerization adds X% overhead" (same hardware)
   - ✅ "Minikube latency is Y% higher than native" (valid)
   - ✅ Direct comparison of absolute values (same hardware)

2. **Relative Performance Patterns**
   - ✅ "Algorithm A is X% faster than B in native"
   - ✅ "Algorithm A is Y% faster than B in GCP"
   - ✅ Compare relative rankings across environments

3. **Normalized Comparisons**
   - ✅ Normalize by CPU count (throughput per CPU)
   - ✅ Normalize by baseline performance (relative to native)
   - ✅ Compare relative changes (percentiles, distributions)

4. **Deployment Context Analysis**
   - ✅ "Performance in cloud deployment context (GCP n2-standard-2)"
   - ✅ "Performance in containerized context (Minikube)"
   - ✅ Frame as "real-world deployment scenarios" not "isolated environment impact"

---

## Solutions and Framing Strategies

### Strategy 1: Frame as "Deployment Context" (Recommended)

**Reframe comparisons to include hardware as part of deployment context:**

Instead of:
> "Containerization adds X% overhead"

Say:
> "Containerized deployment (Minikube on local hardware) adds X% overhead compared to native execution"

And:
> "Cloud deployment (GCP n2-standard-2) shows Y μs latency, representing a typical production deployment scenario"

**Benefits:**
- ✅ Acknowledges hardware differences
- ✅ Still provides useful information
- ✅ Reflects real-world deployment scenarios
- ✅ No normalization needed

**Limitations:**
- ⚠️ Cannot isolate "pure" environment effect
- ⚠️ Hardware differences are part of the story

### Strategy 2: Normalize for Hardware Differences

**Normalize metrics to account for hardware:**

1. **Throughput Normalization**
   ```python
   # Normalize throughput by CPU count
   normalized_throughput = throughput / cpu_count
   
   # Compare normalized values
   native_norm = native_throughput / native_cpu_count
   gcp_norm = gcp_throughput / gcp_cpu_count
   ```

2. **Baseline Normalization**
   ```python
   # Normalize by native baseline (for same algorithm/config)
   relative_latency = (env_latency - native_latency) / native_latency
   ```

3. **CPU Model Normalization**
   - Use CPU benchmarks to estimate relative performance
   - Normalize by estimated CPU performance ratio
   - More complex, requires CPU benchmarks

**Benefits:**
- ✅ Isolates environment effect from hardware
- ✅ More precise comparisons
- ✅ Can make "pure" environment claims

**Limitations:**
- ⚠️ Normalization assumptions may not hold
- ⚠️ More complex analysis
- ⚠️ May not reflect real-world scenarios

### Strategy 3: Separate Hardware Analysis

**Analyze hardware impact separately:**

1. **Same Environment, Different Hardware**
   - Run native on different machines
   - Compare hardware impact within same environment

2. **Hardware Impact Analysis**
   - Document hardware differences
   - Analyze how hardware affects results
   - Separate hardware effect from environment effect

**Benefits:**
- ✅ Most rigorous approach
- ✅ Can isolate both effects
- ✅ Provides comprehensive understanding

**Limitations:**
- ⚠️ Requires additional experiments
- ⚠️ More complex analysis
- ⚠️ May not be feasible with current data

---

## Recommended Approach for Your Dissertation

### Primary Comparisons (Valid)

1. **Native vs Minikube** ✅
   - Same hardware, different execution context
   - Can make direct comparisons
   - Can claim "containerization overhead"
   - **Data**: Native complete, Minikube partial

2. **Algorithm Comparisons Within Environment** ✅
   - "RSA vs Kyber in native"
   - "RSA vs Kyber in GCP"
   - Relative rankings are valid
   - **Data**: Native complete, GCP partial

### Secondary Comparisons (Frame Appropriately)

1. **Native vs GCP** ⚠️
   - Frame as "deployment context comparison"
   - Acknowledge hardware differences
   - Focus on relative patterns, not absolute values
   - **Data**: Native complete, GCP partial

2. **Minikube vs GCP** ⚠️
   - Frame as "containerized local vs cloud deployment"
   - Acknowledge hardware differences
   - Focus on deployment context differences
   - **Data**: Both partial

### What to Document

1. **Hardware Specifications**
   ```markdown
   ## Experimental Setup
   
   ### Native Environment
   - CPU: AMD RYZEN AI MAX+ PRO 395
   - Memory: 94 GB
   - OS: Linux (kernel version)
   
   ### Minikube Environment
   - Same hardware as native
   - Container runtime: Podman
   - Kubernetes: Minikube
   
   ### GCP Environment
   - Machine type: n2-standard-2
   - CPU: Intel Xeon @ 2.80GHz (2 vCPUs)
   - Memory: ~8 GB
   - Region: europe-west2
   ```

2. **Comparison Limitations**
   ```markdown
   ## Comparison Limitations
   
   Direct comparisons between native and GCP are confounded by hardware 
   differences (AMD Ryzen vs Intel Xeon, different CPU counts, different 
   memory). Therefore:
   
   - Native vs Minikube comparisons isolate containerization overhead
   - Native vs GCP comparisons reflect deployment context (hardware + environment)
   - Algorithm comparisons within each environment are valid
   ```

3. **Framing in Results**
   ```markdown
   ## Results
   
   ### Containerization Overhead (Native vs Minikube)
   Minikube shows X% higher latency than native, indicating containerization 
   overhead. This comparison is valid as both use identical hardware.
   
   ### Cloud Deployment Performance (GCP)
   GCP deployment shows Y μs latency on n2-standard-2 VMs, representing a 
   typical production deployment scenario. Direct comparison to native is 
   confounded by hardware differences, but relative algorithm rankings are 
   consistent across environments.
   ```

---

## Implementation: Hardware Metadata Capture

### Current Status

✅ **GCP**: Captures hardware metadata (`cloud_metadata.json`)
- CPU model, CPU count, memory
- Machine type, region
- Instance ID, cluster name

⚠️ **Minikube**: Captures container metadata (`container_metadata.json`)
- Node name, pod name
- Kernel version, arch
- **Missing**: CPU model, CPU count, memory

❌ **Native**: No hardware metadata captured
- **Missing**: CPU model, CPU count, memory
- **Missing**: System specifications

### Recommendations

1. **Capture Native Hardware Metadata**
   ```bash
   # Add to native run script
   cat > results/native/$EXP_ID/hardware_metadata.json << EOF
   {
     "type": "native",
     "cpu_model": "$(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)",
     "cpu_count": $(nproc),
     "memory_total_kb": $(grep MemTotal /proc/meminfo | awk '{print $2}'),
     "kernel_version": "$(uname -r)",
     "arch": "$(uname -m)"
   }
   EOF
   ```

2. **Enhance Minikube Metadata**
   ```bash
   # Add to minikube worker job
   # Capture host hardware (from node)
   CPU_MODEL=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
   CPU_COUNT=$(nproc)
   MEMORY_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
   ```

3. **Add Hardware Consistency Check**
   ```bash
   # Check hardware consistency before comparisons
   python3 analysis/check_hardware_consistency.py \
       --native results/native/*/hardware_metadata.json \
       --minikube results/minikube/*/container_metadata.json \
       --gcp results/gcp/*/cloud_metadata.json
   ```

---

## Analysis Script Modifications

### Current Analysis Scripts

**Issue**: Analysis scripts don't account for hardware differences

**Example**: `analysis/compare_all_environments.py`
- Directly compares absolute values
- No normalization
- No hardware awareness

### Recommended Modifications

1. **Add Hardware Metadata Loading**
   ```python
   def load_hardware_metadata(exp_dir: Path) -> dict:
       """Load hardware metadata for experiment."""
       metadata_path = exp_dir / "hardware_metadata.json"
       if not metadata_path.exists():
           metadata_path = exp_dir / "cloud_metadata.json"
       if not metadata_path.exists():
           metadata_path = exp_dir / "container_metadata.json"
       
       if metadata_path.exists():
           with open(metadata_path) as f:
               return json.load(f)
       return {}
   ```

2. **Add Normalization Options**
   ```python
   def normalize_throughput(throughput: float, cpu_count: int) -> float:
       """Normalize throughput by CPU count."""
       return throughput / cpu_count if cpu_count > 0 else throughput
   
   def normalize_by_baseline(value: float, baseline: float) -> float:
       """Normalize by baseline (relative change)."""
       return (value - baseline) / baseline if baseline > 0 else 0
   ```

3. **Add Hardware Warnings**
   ```python
   def check_hardware_compatibility(env_a: dict, env_b: dict) -> list[str]:
       """Check if hardware is compatible for direct comparison."""
       warnings = []
       
       if env_a.get('cpu_model') != env_b.get('cpu_model'):
           warnings.append("Different CPU models - direct comparison may be confounded")
       
       if env_a.get('cpu_count') != env_b.get('cpu_count'):
           warnings.append("Different CPU counts - normalize throughput by CPU count")
       
       if abs(env_a.get('memory_total_kb', 0) - env_b.get('memory_total_kb', 0)) > 0.1 * env_a.get('memory_total_kb', 1):
           warnings.append("Significant memory difference - may affect caching behavior")
       
       return warnings
   ```

4. **Update Comparison Functions**
   ```python
   def compare_environments(env_a: EnvironmentMetrics, env_b: EnvironmentMetrics, 
                           hardware_a: dict, hardware_b: dict,
                           normalize: bool = False) -> Comparison:
       """Compare environments with hardware awareness."""
       
       # Check hardware compatibility
       warnings = check_hardware_compatibility(hardware_a, hardware_b)
       
       if warnings and not normalize:
           print("WARNING: Hardware differences detected:")
           for warning in warnings:
               print(f"  - {warning}")
           print("Consider using --normalize flag")
       
       # Perform comparison
       if normalize:
           # Normalize metrics
           normalized_a = normalize_metrics(env_a, hardware_a)
           normalized_b = normalize_metrics(env_b, hardware_b)
           return compare_normalized(normalized_a, normalized_b)
       else:
           # Direct comparison (with warnings)
           return compare_direct(env_a, env_b, warnings)
   ```

---

## Summary and Recommendations

### Key Findings

1. **Hardware Differences Exist**
   - Native/Minikube: AMD Ryzen, many cores, 94GB RAM
   - GCP: Intel Xeon, 2 vCPUs, ~8GB RAM
   - **Impact**: Confounds direct comparisons

2. **Valid Comparisons**
   - ✅ Native vs Minikube (same hardware)
   - ✅ Algorithm comparisons within environment
   - ✅ Relative performance patterns

3. **Invalid Comparisons**
   - ❌ Direct native vs GCP (hardware confounds)
   - ❌ Unnormalized absolute comparisons

### Recommendations

1. **Frame Comparisons Appropriately**
   - Use "deployment context" framing
   - Acknowledge hardware differences
   - Focus on relative patterns

2. **Capture Hardware Metadata**
   - Add native hardware capture
   - Enhance minikube metadata
   - Document all hardware specs

3. **Update Analysis Scripts**
   - Add hardware awareness
   - Add normalization options
   - Add hardware compatibility checks

4. **Document Limitations**
   - Clearly state hardware differences
   - Explain comparison limitations
   - Frame results appropriately

### Next Steps

1. ✅ **Immediate**: Document hardware differences in dissertation
2. ✅ **Immediate**: Frame comparisons as "deployment context"
3. ⚠️ **Short-term**: Add hardware metadata capture for native
4. ⚠️ **Short-term**: Update analysis scripts with hardware awareness
5. 📋 **Long-term**: Consider normalization if needed for specific claims

---

## Example: Correct Framing in Dissertation

### ❌ Incorrect Framing

> "GCP deployment shows 50% higher latency than native execution, indicating 
> significant cloud deployment overhead."

**Problem**: Confounds hardware differences with environment differences.

### ✅ Correct Framing

> "GCP deployment (n2-standard-2 VMs) shows 150 μs p95 latency, compared to 
> 100 μs in native execution (AMD Ryzen). This difference reflects both 
> hardware differences (Intel Xeon vs AMD Ryzen, 2 vCPUs vs many cores) and 
> cloud deployment overhead. However, relative algorithm rankings remain 
> consistent across environments, suggesting that algorithmic performance 
> characteristics are preserved despite hardware and deployment context 
> differences."

**Better**: Focus on what's comparable

> "Containerized execution (Minikube) shows 15% higher latency than native 
> execution on identical hardware, isolating the containerization overhead. 
> Cloud deployment (GCP n2-standard-2) shows 150 μs latency, representing a 
> typical production deployment scenario. While direct comparison to native is 
> confounded by hardware differences, the relative performance rankings of 
> algorithms (RSA < Kyber < Dilithium) remain consistent across all 
> environments, indicating that algorithmic characteristics are preserved 
> regardless of deployment context."

