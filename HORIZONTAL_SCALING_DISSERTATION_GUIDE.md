# Horizontal Scaling: Analysis and Dissertation Integration Guide

## Executive Summary

**Question**: How do horizontal scaling experiments work in analysis, and how do we incorporate findings into the dissertation when native doesn't support scaling?

**Answer**:
- ✅ **Analysis Framework**: Fully supports scaling analysis with dedicated plots and metrics
- ✅ **Native Limitation**: Not a problem - frame as "deployment context analysis" rather than "scaling comparison"
- ✅ **Dissertation Strategy**: Use native as baseline, Minikube for orchestration overhead, GCP for production scaling
- 📊 **Outputs**: 4 scaling plots + metrics JSON + dissertation-ready insights

---

## How Scaling Analysis Works

### 1. Data Collection

**Scaling experiments** (replicas 2, 4, 8):
- Run on **Minikube** and **GCP** only (native skips automatically)
- Use separate scenario IDs: `{algorithm}_p{payload}_r{rate}_run{N}_{hash}_r{replicas}`
- Store results in separate directories: `results/{env}/{scenario_id}_r{replicas}/`

**Baseline experiments** (replica 1):
- Run on **all environments** (native, minikube, gcp)
- Already included in your 459 baseline experiments
- Used as baseline for scaling comparisons

### 2. Analysis Pipeline

**Automatic Analysis** (when `--replicas 1,2,4,8` is used):
```bash
# Phase 9: Replica Scaling Analysis
python3 analysis/plot_replica_scaling.py \
    --index final-results/index.json \
    --output final-results/figures/scaling
```

**Manual Analysis** (if needed):
```bash
# After collecting all data
python3 analysis/plot_replica_scaling.py \
    --index final-results/index.json \
    --output final-results/figures/scaling
```

### 3. Generated Outputs

**Plots Generated** (4 total):

1. **`throughput_scaling.png`**
   - Left: Throughput (ops/s) vs Number of Replicas
   - Right: Speedup (×) vs Number of Replicas (with ideal linear line)
   - Shows: How throughput scales with replicas

2. **`latency_scaling.png`**
   - Left: P95 Latency (μs) vs Number of Replicas
   - Right: Latency Ratio (normalized to 1 replica)
   - Shows: How latency degrades with scaling

3. **`scaling_efficiency.png`**
   - Bar chart: Efficiency (speedup / replicas) vs Replicas
   - Ideal = 1.0 (100% efficiency)
   - Shows: How efficiently resources are used

4. **`interference_heatmap.png`**
   - Heatmap: Latency increase (%) vs Replicas (per algorithm, per environment)
   - Shows: Resource contention and interference effects

**Metrics JSON** (`scaling_metrics.json`):
```json
{
  "algorithms": {
    "kyber512": {
      "minikube": {
        "replica_counts": [1, 2, 4, 8],
        "throughput": {
          "mean": [500, 950, 1800, 3200],
          "speedup": [1.0, 1.9, 3.6, 6.4]
        },
        "efficiency": [1.0, 0.95, 0.90, 0.80],
        "interference_factor": [0.0, 0.05, 0.12, 0.25]
      },
      "gcp": { ... }
    }
  }
}
```

---

## Handling Native Limitation in Dissertation

### Strategy 1: Deployment Context Analysis (Recommended)

**Frame**: Native as baseline, Minikube/GCP as deployment contexts

**Dissertation Structure**:

#### Chapter: "Deployment Context Analysis"

**Section 1: Baseline Performance (Native)**
- "Native execution provides baseline algorithmic performance without deployment overhead"
- Use native data for:
  - Algorithm comparison (PQC vs classical)
  - Relative performance characteristics
  - Latency distributions
  - Throughput capabilities

**Section 2: Containerization Overhead (Minikube)**
- "Minikube tests containerization and orchestration overhead on a single machine"
- Use Minikube scaling data for:
  - Container runtime overhead
  - Kubernetes orchestration costs
  - Resource contention on shared resources
  - **Note**: "Single-node cluster limits true horizontal scaling insights"

**Section 3: Production Scaling (GCP)**
- "GCP provides true horizontal scaling across multiple nodes"
- Use GCP scaling data for:
  - Production-like deployment scenarios
  - True horizontal scaling behavior
  - Network latency between nodes
  - Load distribution across cluster

**Key Insight**:
> "Native execution establishes baseline algorithmic performance. Minikube reveals containerization overhead, while GCP demonstrates production-scale horizontal scaling capabilities. Together, these environments provide a comprehensive view of PQC algorithm performance from development to production deployment."

### Strategy 2: Three-Tier Analysis

**Frame**: Development → Staging → Production

**Dissertation Structure**:

1. **Development (Native)**
   - Single-process execution
   - No deployment overhead
   - Pure algorithm performance

2. **Staging (Minikube)**
   - Containerized, single-node
   - Orchestration overhead visible
   - Limited scaling (resource contention)

3. **Production (GCP)**
   - Multi-node cluster
   - True horizontal scaling
   - Production-like deployment

**Key Insight**:
> "The three-tier analysis (native → minikube → GCP) demonstrates how PQC algorithm performance characteristics translate from development to production, with each tier revealing different aspects of deployment complexity."

### Strategy 3: Algorithmic vs. Deployment Performance

**Frame**: Separate algorithmic insights from deployment insights

**Dissertation Structure**:

**Part A: Algorithmic Performance (Native)**
- All 459 baseline experiments
- Algorithm comparison
- Relative performance characteristics
- Statistical significance

**Part B: Deployment Performance (Minikube + GCP)**
- Scaling experiments (replicas 2, 4, 8)
- Containerization overhead
- Horizontal scaling efficiency
- Production deployment guidance

**Key Insight**:
> "Native execution isolates algorithmic performance, while Minikube and GCP experiments reveal deployment-specific characteristics. This separation allows for clear recommendations: algorithm selection (from native) and deployment strategy (from scaling experiments)."

---

## What the Analysis Reveals

### 1. Throughput Scaling

**What it shows**:
- How throughput increases with replicas
- Whether scaling is linear, sub-linear, or super-linear
- Saturation points (where adding replicas stops helping)

**Dissertation Language**:
> "Throughput scaling analysis reveals that [algorithm] achieves [X]× speedup with [N] replicas, indicating [linear/sub-linear/super-linear] scaling behavior. Efficiency metrics show [Y]% resource utilization, suggesting [algorithm] is [well-suited/limited] for horizontal scaling in production deployments."

### 2. Latency Degradation

**What it shows**:
- How latency increases with replicas (interference effects)
- Whether latency remains acceptable at scale
- Resource contention patterns

**Dissertation Language**:
> "Latency analysis shows that [algorithm] maintains [X]% latency increase with [N] replicas, remaining within acceptable bounds for [use case]. The interference heatmap reveals [algorithm] exhibits [low/medium/high] resource contention, indicating [suitable/unsuitable] for high-replica deployments."

### 3. Scaling Efficiency

**What it shows**:
- Efficiency = speedup / replicas (ideal = 1.0)
- How well resources are utilized
- Diminishing returns with more replicas

**Dissertation Language**:
> "Scaling efficiency analysis demonstrates that [algorithm] achieves [X]% efficiency with [N] replicas, indicating [excellent/good/poor] resource utilization. Efficiency drops to [Y]% at [M] replicas, suggesting an optimal replica count of [N] for this algorithm."

### 4. Environment Comparison

**What it shows**:
- Minikube vs. GCP scaling behavior
- Orchestration overhead (Minikube)
- True horizontal scaling (GCP)

**Dissertation Language**:
> "Environment comparison reveals that Minikube scaling is limited by single-node resource contention, achieving [X]× speedup with [N] replicas. GCP demonstrates true horizontal scaling, achieving [Y]× speedup with [N] replicas, indicating [algorithm] scales effectively in production multi-node deployments."

---

## Dissertation Chapter Structure

### Recommended Structure

#### Chapter 5: Results and Analysis

**5.1 Algorithmic Performance (Native Baseline)**
- All 459 baseline experiments
- Algorithm comparison (PQC vs classical)
- Statistical significance
- Relative performance characteristics

**5.2 Deployment Context Analysis**
- **5.2.1 Containerization Overhead (Minikube)**
  - Baseline comparison (native vs minikube)
  - Orchestration overhead metrics
  - Single-node scaling limitations
  
- **5.2.2 Production Scaling (GCP)**
  - Baseline comparison (native vs GCP)
  - Horizontal scaling experiments (replicas 2, 4, 8)
  - Throughput scaling curves
  - Latency degradation analysis
  - Scaling efficiency metrics

**5.3 Cross-Environment Insights**
- Native as baseline reference
- Minikube for orchestration overhead
- GCP for production scaling
- Deployment recommendations

#### Chapter 6: Discussion

**6.1 Algorithm Selection Guidelines**
- Based on native performance (algorithmic characteristics)
- Relative performance comparison
- Use case recommendations

**6.2 Deployment Strategy Guidelines**
- Based on scaling experiments (Minikube + GCP)
- Horizontal scaling recommendations
- Replica count optimization
- Production deployment considerations

**6.3 Limitations and Future Work**
- Native limitation: "Native execution does not support horizontal scaling, as it runs a single-process binary. Scaling analysis is therefore limited to containerized environments (Minikube) and cloud deployments (GCP)."
- Minikube limitation: "Minikube scaling is limited to single-node resource contention, not true horizontal scaling across nodes."
- GCP as production proxy: "GCP experiments provide production-like scaling insights, though actual production deployments may exhibit additional factors (network topology, load balancing, etc.)."

---

## Example Dissertation Text

### Introduction to Scaling Analysis

> "This study evaluates PQC algorithm performance across three deployment contexts: native execution (baseline), containerized single-node (Minikube), and cloud multi-node (GCP). Native execution provides baseline algorithmic performance without deployment overhead, enabling pure algorithm comparison. Minikube experiments reveal containerization and orchestration overhead on a single machine, while GCP experiments demonstrate true horizontal scaling across multiple nodes in a production-like environment.
>
> Horizontal scaling experiments (replicas 2, 4, 8) are conducted on Minikube and GCP only, as native execution does not support multi-process replication. This design allows for comprehensive analysis of deployment-specific characteristics while maintaining native as a baseline reference for algorithmic performance."

### Results Presentation

> "Figure X shows throughput scaling for [algorithm] across Minikube and GCP environments. Minikube achieves [X]× speedup with 4 replicas, limited by single-node resource contention. GCP demonstrates true horizontal scaling, achieving [Y]× speedup with 4 replicas, indicating effective multi-node deployment.
>
> Table Y presents scaling efficiency metrics (speedup / replicas) for all algorithms. [Algorithm] achieves [X]% efficiency with 4 replicas on GCP, indicating excellent resource utilization. Efficiency drops to [Y]% at 8 replicas, suggesting an optimal replica count of 4 for this algorithm in production deployments."

### Discussion

> "The three-tier analysis (native → minikube → GCP) reveals distinct performance characteristics at each deployment level. Native execution establishes baseline algorithmic performance, showing [algorithm] is [X]% faster than [baseline]. Minikube experiments reveal [Y]% containerization overhead, while GCP scaling experiments demonstrate [algorithm] scales effectively to [N] replicas with [Z]% efficiency.
>
> For production deployment, we recommend [algorithm] with [N] replicas based on scaling efficiency analysis. Native performance data guides algorithm selection, while GCP scaling data informs deployment strategy."

---

## Analysis Workflow

### Step 1: Collect Baseline Data (All Environments)

```bash
# Native (459 experiments)
./run_full_scale_data_collection.sh --env native

# Minikube (459 experiments)
./run_full_scale_data_collection.sh --env minikube

# GCP (459 experiments)
./run_full_scale_data_collection.sh --env gcp --project <project> --bucket <bucket>
```

### Step 2: Collect Scaling Data (Minikube + GCP Only)

```bash
# Run scaling experiments
./run_all_experiments.sh \
  --envs minikube,gcp \
  --replicas 1,2,4,8 \
  --project <project> \
  --bucket <bucket> \
  --skip-generation \
  --matrix orchestration/experiment_matrix.yaml
```

**Note**: This runs replicas 2, 4, 8 (replica 1 already in baseline).

### Step 3: Generate Scaling Analysis

```bash
# Automatic (if using run_all_experiments.sh with --replicas)
# Or manual:
python3 analysis/plot_replica_scaling.py \
    --index final-results/index.json \
    --output final-results/figures/scaling
```

### Step 4: Integrate into Dissertation

1. **Use native data** for:
   - Algorithm comparison sections
   - Relative performance analysis
   - Statistical significance tests
   - Baseline performance metrics

2. **Use Minikube scaling data** for:
   - Containerization overhead discussion
   - Single-node resource contention analysis
   - Orchestration overhead metrics

3. **Use GCP scaling data** for:
   - Production scaling recommendations
   - Horizontal scaling efficiency
   - Replica count optimization
   - Deployment strategy guidelines

---

## Key Takeaways for Dissertation

### ✅ What You CAN Claim

1. **Algorithmic Performance** (from native):
   - "Algorithm X is Y% faster than baseline Z"
   - "Statistical analysis shows significant differences (p < 0.05)"
   - "Native execution provides baseline algorithmic performance"

2. **Deployment Overhead** (from Minikube):
   - "Containerization adds X% overhead compared to native"
   - "Minikube scaling is limited by single-node resource contention"
   - "Orchestration overhead is Y% for [algorithm]"

3. **Production Scaling** (from GCP):
   - "GCP experiments demonstrate true horizontal scaling"
   - "Algorithm X achieves Y× speedup with N replicas"
   - "Scaling efficiency is Z% with N replicas"
   - "Optimal replica count for [algorithm] is N based on efficiency analysis"

### ❌ What You CANNOT Claim

1. **Native Scaling**:
   - ❌ "Native supports horizontal scaling" (it doesn't)
   - ❌ "Native scaling experiments show..." (they don't exist)

2. **Minikube as Production**:
   - ❌ "Minikube represents production scaling" (single-node limitation)
   - ❌ "Minikube scaling is equivalent to GCP" (different architectures)

### ⚠️ What You CAN Claim with Caveats

1. **Native as Baseline**:
   - ✅ "Native execution provides baseline algorithmic performance **for comparison with deployment contexts**"
   - ✅ "Native data establishes **algorithmic characteristics** that scale to deployment environments"

2. **Three-Tier Analysis**:
   - ✅ "The three-tier analysis (native → minikube → GCP) provides **comprehensive deployment context insights**"
   - ✅ "Native serves as **baseline reference**, while Minikube and GCP reveal **deployment-specific characteristics**"

---

## Summary

### Analysis Capabilities

- ✅ **Fully automated**: Scaling analysis runs automatically with `--replicas` flag
- ✅ **Comprehensive outputs**: 4 plots + metrics JSON + dissertation-ready insights
- ✅ **Environment-aware**: Handles missing native scaling data gracefully
- ✅ **Dissertation-ready**: Provides clear framing strategies for native limitation

### Dissertation Strategy

1. **Frame native as baseline** (not missing scaling data)
2. **Use Minikube for orchestration overhead** (not true scaling)
3. **Use GCP for production scaling** (true horizontal scaling)
4. **Separate algorithmic vs. deployment insights**

### Key Message

> "Native execution provides baseline algorithmic performance. Minikube reveals containerization overhead, while GCP demonstrates production-scale horizontal scaling. Together, these environments provide comprehensive insights from development to production deployment."

---

## Next Steps

1. **Run baseline experiments** (all 459 per environment)
2. **Run scaling experiments** (Minikube + GCP, replicas 2, 4, 8)
3. **Generate scaling analysis** (automatic or manual)
4. **Review generated plots** (4 scaling plots + metrics JSON)
5. **Integrate into dissertation** (use framing strategies above)

The analysis framework handles the native limitation automatically - you just need to frame it appropriately in your dissertation text.

