# Data Interpretation Framework

**Purpose**: Template and framework for creating comprehensive data interpretation supporting research claims.

---

## Structure

### 1. Executive Summary
- Overall findings
- Key performance differences
- Statistical significance summary
- Practical implications

### 2. Algorithm Performance Analysis

#### 2.1 Native Baseline Performance
**Data Sources**:
- `final-results/aggregated_stats.json` (native environment)
- `final-results/hypothesis_tests.json` (PQC vs Classical comparisons)

**Key Claims to Support**:
1. "PQC key generation incurs 1-3μs overhead"
2. "Dilithium signature generation comparable to ECDSA"
3. "Large effect sizes (Cohen's d > 1.2, p < 0.001)"

**Interpretation Template**:
```
Algorithm X demonstrated [latency] mean latency (p50: [value], p95: [value], p99: [value]) 
for [operation] operations. Statistical analysis using [test] revealed [significant/not significant] 
differences compared to [baseline] (p = [value], Cohen's d = [value], [interpretation] effect size).
The [practical/negligible] impact on user-perceived performance suggests [implication].
```

#### 2.2 Payload Size Impact
**Data Sources**:
- Aggregated stats grouped by payload size
- Payload size effect size calculations

**Interpretation Template**:
```
Performance scales [linearly/non-linearly] with payload size, with [X]% increase per KB.
Algorithm [X] shows [better/worse] scaling characteristics than [baseline].
```

#### 2.3 Workload Pattern Impact
**Data Sources**:
- Constant vs Burst pattern comparisons
- Queue delay analysis

**Interpretation Template**:
```
Burst patterns increase latency by [X]% compared to constant patterns.
Queue delay accounts for [X]% of total latency under [conditions].
```

### 3. Environment Comparison Analysis

#### 3.1 Containerization Overhead (Minikube)
**Data Sources**:
- `final-results/aggregated_stats.json` (environment_deltas)
- Native vs Minikube effect sizes

**Interpretation Template**:
```
Containerization adds [X]% overhead (mean: [value]μs, CI: [low]-[high]μs) compared to native execution.
Statistical analysis shows [significant/not significant] differences (p = [value]).
The overhead is [acceptable/notable] for [use case].
```

#### 3.2 Cloud Deployment (GCP)
**Data Sources**:
- Native vs GCP comparisons
- Variability analysis

**Interpretation Template**:
```
GCP deployment shows [X]% overhead with [high/moderate/low] variability (std: [value]).
The cloud environment introduces [quantify] additional latency compared to native.
```

### 4. Horizontal Scaling Analysis

**Data Sources**:
- Scaling experiment results
- Throughput vs replicas
- Latency degradation

**Interpretation Template**:
```
Algorithm [X] achieves [Y]× speedup with [N] replicas, demonstrating [Z]% scaling efficiency.
Latency increases by [X]% with [N] replicas due to [reason].
Optimal replica count for [algorithm] is [N] based on [criteria].
```

### 5. Statistical Significance Summary

**Data Sources**:
- `final-results/hypothesis_tests.json`
- All p-values with Holm-Bonferroni correction
- Effect sizes with confidence intervals

**Template**:
```
Of [N] comparisons, [X] showed statistically significant differences (p < 0.05 after correction).
Effect sizes ranged from [min] to [max] (Cohen's d), with [X]% showing large effects (d > 0.8).
Key findings:
- [Comparison 1]: [significant/not significant], d = [value] ([interpretation])
- [Comparison 2]: [significant/not significant], d = [value] ([interpretation])
```

### 6. Size and Bandwidth Analysis

**Data Sources**:
- Ciphertext/signature size measurements
- Size inflation calculations

**Template**:
```
PQC algorithms show [2-45]× size inflation compared to classical algorithms.
- [Algorithm X]: [X]× increase ([size] vs [baseline_size])
- [Algorithm Y]: [Y]× increase ([size] vs [baseline_size])
This represents [significant/moderate] bandwidth overhead for [use case].
```

### 7. Practical Implications

**Template Sections**:
- Algorithm selection recommendations
- Deployment strategy guidelines
- Use case suitability
- Limitations and trade-offs

---

## Data Extraction Commands

### Extract Performance Metrics
```bash
# From aggregated_stats.json
python3 -c "
import json
data = json.load(open('final-results/aggregated_stats.json'))
for stat in data['aggregated']:
    if stat['environment'] == 'native':
        print(f\"{stat['algorithm']}: p95={stat['p95']['mean']:.2f}μs, throughput={stat['throughput']['mean']:.0f} ops/s\")
"
```

### Extract Effect Sizes
```bash
# From hypothesis_tests.json
python3 -c "
import json
data = json.load(open('final-results/hypothesis_tests.json'))
for test in data.get('tests', []):
    if test.get('effect_size', {}).get('cohens_d', 0) > 0.8:
        print(f\"{test['comparison_id']}: d={test['effect_size']['cohens_d']:.2f}\")
"
```

### Extract Environment Deltas
```bash
# From aggregated_stats.json
python3 -c "
import json
data = json.load(open('final-results/aggregated_stats.json'))
for delta in data.get('environment_deltas', []):
    if delta.get('native_to_minikube_pct'):
        print(f\"{delta['algorithm']}: Minikube overhead = {delta['native_to_minikube_pct']:.1f}%\")
"
```

---

## Claim Verification Checklist

For each claim in `docs/REQUIREMENTS_SPECIFICATION.md`:

- [ ] Data available to support claim
- [ ] Statistical test performed
- [ ] Effect size calculated
- [ ] Visualization created (if needed)
- [ ] Interpretation written
- [ ] Claim verified against data

---

**Status**: Framework ready - populate once analysis complete
