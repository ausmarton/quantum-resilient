# Data Validation Scripts

## Overview

Two comprehensive scripts have been created to validate and manage experiment data:

1. **`validate_dissertation_data.sh`** - Comprehensive validation for dissertation claims support
2. **`remove_unusable_data.sh`** - Identify and remove unusable experiments for re-run

---

## 1. validate_dissertation_data.sh

### Purpose
Validates all collected experiment data against dissertation requirements to ensure:
- Data format compliance (all required fields)
- Data quality (no errors, valid values)
- Completeness (all runs, all experiments)
- Statistical validity (sufficient runs per configuration)
- Dissertation claims support (all required data points present)

### Usage

```bash
# Full validation for all environments
./scripts/validate_dissertation_data.sh

# Validate specific environment
./scripts/validate_dissertation_data.sh --env gcp

# Generate detailed JSON report
./scripts/validate_dissertation_data.sh --env gcp --output validation_report.json

# List unusable experiments
./scripts/validate_dissertation_data.sh --env gcp --list-unusable

# Fail on issues (for CI/CD)
./scripts/validate_dissertation_data.sh --env gcp --fail-on-issues
```

### What It Checks

#### Data Format Compliance
- ✅ All required fields present (12 fields from REQUIREMENTS_SPECIFICATION.md)
- ✅ Field types correct (u64, u128, f64)
- ✅ Timestamps valid ISO 8601 format
- ✅ Event IDs sequential

#### Data Quality
- ✅ Error rate analysis (identifies experiments with errors)
- ✅ Latency values in expected range
- ✅ Timestamps monotonic
- ✅ CPU/memory values reasonable
- ✅ No missing values in critical fields

#### Completeness
- ✅ Expected vs actual experiment counts
- ✅ Run completeness (5 runs per configuration)
- ✅ Scaling experiments (3 runs per replica)
- ✅ Cross-environment consistency

#### Statistical Validity
- ✅ Sufficient runs per configuration (5 for baseline, 3 for scaling)
- ✅ Large sample sizes (thousands of events)
- ✅ Statistical power assessment

#### Dissertation Claims Support
- ✅ Algorithm comparison data available
- ✅ Statistical rigor (sufficient runs)
- ✅ Resource utilization data (CPU, memory)
- ✅ Queue delay analysis data
- ✅ Payload size impact data

### Output

**Console Output:**
- Summary statistics
- Field coverage report
- Error analysis
- Algorithm/operation status
- Run completeness
- Unusable experiments list (if `--list-unusable`)

**JSON Report** (if `--output` specified):
- Complete validation results
- All statistics and metrics
- Unusable experiments with details
- Issues and recommendations

### Example Output

```
================================================================================
DISSERTATION DATA VALIDATION REPORT
================================================================================

Total Experiments: 115
Total Runs: 533
Total Events: 47,130,669
Experiments with Errors: 23
Fit for Purpose: ❌ NO

================================================================================
UNUSABLE EXPERIMENTS (Need Re-run)
================================================================================

hybrid_kyber_dilithium_p1024_r2000_burst_89bf3162:
  Reason: Has errors in data
  Runs: ['run-1', 'run-2', 'run-3', 'run-4', 'run-5']
  Errors: {'Internal error: No keypair available for KEM operation': 6}
```

---

## 2. remove_unusable_data.sh

### Purpose
Identifies and removes unusable experiment data so they can be re-run. Supports:
- Dry-run mode (preview without removing)
- Error-only removal (keep incomplete runs)
- Incomplete-only removal (keep errors)
- Confirmation prompts (safety)

### Usage

```bash
# Dry run to see what would be removed
./scripts/remove_unusable_data.sh --env gcp --dry-run --error-only

# Remove only experiments with errors
./scripts/remove_unusable_data.sh --env gcp --error-only

# Remove all unusable data (errors + incomplete)
./scripts/remove_unusable_data.sh --env gcp

# Force removal without confirmation
./scripts/remove_unusable_data.sh --env gcp --error-only --force
```

### What It Removes

**Experiments with Errors:**
- All runs have errors in data
- Cannot be used for analysis
- Need to be re-run with fixed code

**Experiments with Insufficient Runs:**
- Less than 3 runs (statistically invalid)
- Less than 5 runs for non-scaling experiments
- Cannot support statistical analysis

### Safety Features

- **Dry-run mode**: Preview what would be removed
- **Confirmation prompts**: Require user confirmation (default)
- **Force mode**: Skip confirmation (use with caution)
- **Selective removal**: Choose error-only or incomplete-only

### Example Output

```
[INFO] Identifying unusable experiments...
[WARN] Found 23 unusable experiment(s)

==================================================================================
EXPERIMENTS TO BE REMOVED
==================================================================================

1. hybrid_kyber_dilithium_p1024_r2000_burst_89bf3162
   Reasons: Has errors: {'Internal error: No keypair available for KEM operation': 6}
   Runs: 5
   Total events: 450,214
   Errors: {'Internal error: No keypair available for KEM operation': 6}
   Directory: results/gcp/hybrid_kyber_dilithium_p1024_r2000_burst_89bf3162

[INFO] DRY RUN - No data will be removed
```

---

## Validation Criteria

### Required Fields (from REQUIREMENTS_SPECIFICATION.md)

All experiments must have:
1. `run_id` - Experiment run identifier
2. `scenario_id` - Scenario identifier
3. `event_id` - Event sequence number
4. `timestamp_utc_iso` - UTC timestamp (ISO 8601)
5. `timestamp_monotonic_ns` - Monotonic timestamp (nanoseconds)
6. `operation` - Operation type (sign, encrypt, etc.)
7. `algorithm` - Algorithm name
8. `latency_ns` - Operation latency (nanoseconds)
9. `payload_size_bytes` - Payload size
10. `cpu_user_seconds` - CPU usage (cumulative)
11. `memory_rss_bytes` - Memory usage (RSS)
12. `rng_seed` - RNG seed for reproducibility

### Optional but Important Fields

- `queue_delay_ns` - Queue delay (for queue delay analysis)
- `worker_id` - Worker identifier (for multi-worker analysis)
- `ciphertext_size_bytes` - Ciphertext size (for encryption operations)
- `signature_size_bytes` - Signature size (for signature operations)
- `error` - Error message (for error rate analysis)

### Statistical Validity Requirements

- **Baseline experiments**: 5 runs per configuration
- **Scaling experiments**: 3 runs per replica
- **Minimum events**: 100 events per experiment (for percentiles)
- **Statistical power**: ~80% for medium effects, ~95% for large effects

### Dissertation Claims Requirements

The validation ensures data supports:
1. Algorithm performance comparison
2. Environment comparison
3. Horizontal scaling analysis
4. Statistical rigor (hypothesis tests, effect sizes)
5. Resource utilization analysis
6. Queue delay analysis
7. Payload size impact analysis
8. Workload pattern impact analysis
9. Error rate analysis

---

## Workflow

### After Data Collection

1. **Validate data quality:**
   ```bash
   ./scripts/validate_dissertation_data.sh --env gcp --output gcp_validation.json
   ```

2. **Review validation report:**
   - Check field coverage (should be 100%)
   - Check error rates (should be 0% for usable data)
   - Check completeness (should match expected counts)
   - Review unusable experiments list

3. **Remove unusable data (if needed):**
   ```bash
   # Preview what would be removed
   ./scripts/remove_unusable_data.sh --env gcp --dry-run --error-only
   
   # Actually remove (after reviewing)
   ./scripts/remove_unusable_data.sh --env gcp --error-only
   ```

4. **Re-run failed experiments:**
   - Submit jobs again for removed experiments
   - Verify new data with validation script

5. **Final validation:**
   ```bash
   ./scripts/validate_dissertation_data.sh --env gcp --fail-on-issues
   ```

---

## Integration with Analysis Pipeline

The validation scripts ensure data is ready for:
- Statistical analysis (`compute_statistics.py`)
- Aggregation (`aggregate_results.py`)
- Hypothesis testing
- Visualization generation
- Dissertation claims support

---

## Troubleshooting

### "No unusable experiments found" but you know there are errors

- Check if errors are in the data (may have been fixed)
- Verify file paths are correct
- Check environment filter matches your data location

### "Output file exists but is empty"

- Python script may have failed silently
- Check Python dependencies
- Verify file permissions

### "Experiments with errors: 0" but validation shows errors

- Validation script checks all events, remove script checks first event only
- Some experiments may have errors in later events
- Use validation script's `--list-unusable` for comprehensive list

---

## Notes

- Both scripts support native, minikube, and GCP directory structures
- Validation is comprehensive and checks all dissertation requirements
- Removal is safe with dry-run and confirmation by default
- Scripts can be run multiple times safely
