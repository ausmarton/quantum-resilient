#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh - Master orchestration script
#
# Executes all benchmarking scenarios across all environments (native, Minikube,
# GCP), runs multiple repeats, collects data, and produces dissertation-ready
# final-results/ directory.
#
# Usage:
#   ./run_all_experiments.sh \
#     --project <gcp-project> \
#     --bucket <gcs-bucket> \
#     --matrix orchestration/experiment_matrix.yaml \
#     --envs native,minikube,gcp
#
# Requirements:
#   - Python 3.10+ with analysis dependencies
#   - For native: Rust toolchain
#   - For Minikube: Minikube + Podman
#   - For GCP: gcloud, Terraform, Podman
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED_SCENARIOS_DIR="$SCRIPT_DIR/generated-scenarios"
RESULTS_BASE="$SCRIPT_DIR/results"

# Default values
MATRIX="$SCRIPT_DIR/orchestration/experiment_matrix.yaml"
ENVS="native"
PROJECT=""
BUCKET=""
REGION="us-central1"
PARALLEL_JOBS=1
REPLICAS="1"  # Comma-separated list: 1,2,4,8
SKIP_GENERATION=false
SKIP_NATIVE=false
SKIP_MINIKUBE=false
SKIP_GCP=false
SKIP_ANALYSIS=false
SKIP_SCALING=false
DRY_RUN=false
CONTINUE_ON_ERROR=true
MAX_RETRIES=2
SMOKE_TEST=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Tracking
TOTAL_SCENARIOS=0
COMPLETED_SCENARIOS=0
FAILED_SCENARIOS=0
MASTER_INDEX=()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}\n"
}

log_phase() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  PHASE: $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Execute all benchmarking scenarios across environments.

OPTIONS:
    --matrix PATH           Experiment matrix YAML (default: orchestration/experiment_matrix.yaml)
    --envs LIST             Comma-separated environments: native,minikube,gcp (default: native)
    --project ID            GCP project ID (required for gcp env)
    --bucket NAME           GCS bucket name (required for gcp env)
    --region REGION         GCP region (default: us-central1)
    --parallel N            Parallel jobs per environment (default: 1)
    --replicas LIST         Comma-separated replica counts for scaling tests: 1,2,4,8
    --skip-generation       Skip scenario generation
    --skip-native           Skip native experiments
    --skip-minikube         Skip Minikube experiments
    --skip-gcp              Skip GCP experiments
    --skip-analysis         Skip final analysis
    --skip-scaling          Skip scaling analysis
    --dry-run               Show what would be executed
    --continue-on-error     Continue if individual experiments fail (default: true)
    --max-retries N         Max retries per failed experiment (default: 2)
    --smoke-test            Enable smoke-test mode (reduced scale, minimal cost)
    -h, --help              Show this help message

EXAMPLE:
    # Standard experiments
    $0 --envs native,minikube,gcp \\
       --project my-gcp-project \\
       --bucket pqc-bench-results \\
       --matrix orchestration/experiment_matrix.yaml

    # Scaling experiments with multiple replica counts
    $0 --envs minikube,gcp \\
       --replicas 1,2,4,8 \\
       --project my-gcp-project \\
       --bucket pqc-bench-results
EOF
    exit 1
}

# Progress tracking
update_progress() {
    local current=$1
    local total=$2
    local env=$3
    local scenario=$4
    local pct=$((current * 100 / total))
    echo -e "${CYAN}[${pct}%]${NC} [${env}] ${scenario}"
}

# Run single experiment with retries
run_experiment() {
    local env=$1
    local scenario_path=$2
    local scenario_id=$3
    local output_dir=$4
    local replicas=${5:-1}
    local retries=0
    
    while [[ $retries -le $MAX_RETRIES ]]; do
        local exit_code=0
        
        case $env in
            native)
                # Native doesn't support replicas
                "$SCRIPT_DIR/run_local.sh" \
                    --scenario "$scenario_path" \
                    --out "$output_dir" \
                    --duration 30 \
                    $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") 2>&1 || exit_code=$?
                ;;
            minikube)
                "$SCRIPT_DIR/run_minikube.sh" \
                    --scenario "$scenario_path" \
                    --out "$output_dir" \
                    --replicas "$replicas" \
                    --exp-id "$scenario_id" \
                    $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") 2>&1 || exit_code=$?
                ;;
            gcp)
                "$SCRIPT_DIR/deploy_gcp.sh" \
                    --scenario "$scenario_path" \
                    --exp-id "$scenario_id" \
                    --project "$PROJECT" \
                    --bucket "$BUCKET" \
                    --region "$REGION" \
                    --replicas "$replicas" \
                    $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "") 2>&1 || exit_code=$?
                
                if [[ $exit_code -eq 0 ]]; then
                    "$SCRIPT_DIR/fetch_and_analyse_from_gcs.sh" \
                        --exp-id "$scenario_id" \
                        --bucket "$BUCKET" \
                        --out "$output_dir" 2>&1 || exit_code=$?
                fi
                ;;
        esac
        
        if [[ $exit_code -eq 0 ]]; then
            return 0
        fi
        
        retries=$((retries + 1))
        if [[ $retries -le $MAX_RETRIES ]]; then
            log_warn "Experiment failed, retrying ($retries/$MAX_RETRIES)..."
            sleep 5
        fi
    done
    
    return 1
}

# Add entry to master index
add_to_index() {
    local scenario_id=$1
    local env=$2
    local algorithm=$3
    local payload=$4
    local rate=$5
    local output_dir=$6
    local status=$7
    local replicas=${8:-1}
    
    MASTER_INDEX+=("{\"scenario_id\":\"$scenario_id\",\"environment\":\"$env\",\"algorithm\":\"$algorithm\",\"payload_size\":$payload,\"rate\":$rate,\"replicas\":$replicas,\"output_dir\":\"$output_dir\",\"status\":\"$status\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}")
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --matrix)
            MATRIX="$2"
            shift 2
            ;;
        --envs)
            ENVS="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --replicas)
            REPLICAS="$2"
            shift 2
            ;;
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --skip-native)
            SKIP_NATIVE=true
            shift
            ;;
        --skip-minikube)
            SKIP_MINIKUBE=true
            shift
            ;;
        --skip-gcp)
            SKIP_GCP=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --skip-scaling)
            SKIP_SCALING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ ! -f "$MATRIX" ]]; then
    log_error "Matrix file not found: $MATRIX"
    exit 1
fi

# Check if GCP env is requested but missing credentials
if [[ "$ENVS" == *"gcp"* ]]; then
    if [[ -z "$PROJECT" ]]; then
        log_error "GCP environment requested but --project not provided"
        exit 1
    fi
    if [[ -z "$BUCKET" ]]; then
        log_error "GCP environment requested but --bucket not provided"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
START_TIME=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo -e "${MAGENTA}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║      PQC Benchmark - Complete Experiment Orchestration Suite          ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Matrix: $MATRIX"
log_info "Environments: $ENVS"
log_info "Replicas: $REPLICAS"
[[ "$SMOKE_TEST" == "true" ]] && log_info "Mode: SMOKE-TEST (reduced scale, minimal cost)"
log_info "Started: $START_ISO"

# Parse replicas into array
IFS=',' read -ra REPLICA_ARRAY <<< "$REPLICAS"

if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "DRY RUN MODE - No experiments will be executed"
fi

# =============================================================================
# Phase 1: Generate Scenarios
# =============================================================================
log_phase "1. Scenario Generation"

if [[ "$SKIP_GENERATION" == "true" ]]; then
    log_warn "Skipping scenario generation (--skip-generation)"
else
    log_info "Generating scenarios from matrix..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        python3 "$SCRIPT_DIR/orchestration/generate_scenarios.py" \
            --matrix "$MATRIX" \
            --output "$GENERATED_SCENARIOS_DIR" \
            --dry-run \
            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "")
    else
        python3 "$SCRIPT_DIR/orchestration/generate_scenarios.py" \
            --matrix "$MATRIX" \
            --output "$GENERATED_SCENARIOS_DIR" \
            $([ "$SMOKE_TEST" == "true" ] && echo "--smoke-test" || echo "")
    fi
    
    log_success "Scenarios generated"
fi

# Count scenarios
if [[ -f "$GENERATED_SCENARIOS_DIR/manifest.json" ]]; then
    TOTAL_SCENARIOS=$(python3 -c "import json; print(json.load(open('$GENERATED_SCENARIOS_DIR/manifest.json'))['total_scenarios'])")
    log_info "Total scenarios: $TOTAL_SCENARIOS"
fi

# =============================================================================
# Phase 2: Create Output Directories
# =============================================================================
log_phase "2. Initialize Output Directories"

mkdir -p "$FINAL_RESULTS_DIR/figures"
mkdir -p "$FINAL_RESULTS_DIR/stats"
mkdir -p "$FINAL_RESULTS_DIR/tables"
mkdir -p "$RESULTS_BASE/native"
mkdir -p "$RESULTS_BASE/minikube"
mkdir -p "$RESULTS_BASE/gcp"

log_success "Output directories created"

# =============================================================================
# Phase 3: Execute Experiments
# =============================================================================
log_phase "3. Execute Experiments"

# Set final results directory based on smoke-test mode
if [[ "$SMOKE_TEST" == "true" ]]; then
    FINAL_RESULTS_DIR="$SCRIPT_DIR/final-results-smoke"
    REPLICAS="1"  # Force replicas to 1 in smoke-test mode
else
    FINAL_RESULTS_DIR="$SCRIPT_DIR/final-results"
fi

# Parse environments
IFS=',' read -ra ENV_ARRAY <<< "$ENVS"

for env in "${ENV_ARRAY[@]}"; do
    # Skip if flagged
    case $env in
        native)
            [[ "$SKIP_NATIVE" == "true" ]] && { log_warn "Skipping native"; continue; }
            ;;
        minikube)
            [[ "$SKIP_MINIKUBE" == "true" ]] && { log_warn "Skipping minikube"; continue; }
            ;;
        gcp)
            [[ "$SKIP_GCP" == "true" ]] && { log_warn "Skipping gcp"; continue; }
            ;;
    esac
    
    log_step "Environment: ${env^^}"
    
    # Read scenarios from manifest
    if [[ ! -f "$GENERATED_SCENARIOS_DIR/manifest.json" ]]; then
        log_error "Scenario manifest not found. Run without --skip-generation first."
        exit 1
    fi
    
    # Process scenarios
    scenario_count=0
    env_completed=0
    env_failed=0
    
    # Extract scenarios using Python for reliable JSON parsing
    scenarios=$(python3 -c "
import json
with open('$GENERATED_SCENARIOS_DIR/manifest.json') as f:
    manifest = json.load(f)
for s in manifest['scenarios']:
    print(f\"{s['id']}|{s['path']}|{s['algorithm']}|{s['payload_size']}|{s['rate']}\")
")
    
    while IFS='|' read -r scenario_id scenario_path algorithm payload rate; do
        scenario_count=$((scenario_count + 1))
        
        # Iterate over replica counts
        for replica_count in "${REPLICA_ARRAY[@]}"; do
            # For native, only run with 1 replica
            if [[ "$env" == "native" ]] && [[ "$replica_count" -gt 1 ]]; then
                continue
            fi
            
            # In smoke-test mode, only run with 1 replica
            if [[ "$SMOKE_TEST" == "true" ]] && [[ "$replica_count" -gt 1 ]]; then
                continue
            fi
            
            # Generate unique output dir and ID for scaling experiments
            if [[ "$replica_count" -gt 1 ]]; then
                output_dir="$RESULTS_BASE/$env/${scenario_id}_r${replica_count}"
                run_scenario_id="${scenario_id}_r${replica_count}"
            else
                output_dir="$RESULTS_BASE/$env/$scenario_id"
                run_scenario_id="$scenario_id"
            fi
            
            update_progress $scenario_count $TOTAL_SCENARIOS "$env" "$run_scenario_id"
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "  Would run: $run_scenario_id (replicas: $replica_count)"
                add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "dry_run" "$replica_count"
                continue
            fi
            
            # Check if already completed
            if [[ -f "$output_dir/stats/summary.json" ]] || [[ -f "$output_dir/merged/merged.jsonl" ]]; then
                log_info "  Skipping (already completed): $run_scenario_id"
                add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "cached" "$replica_count"
                env_completed=$((env_completed + 1))
                continue
            fi
            
            # Run experiment with replica count
            if run_experiment "$env" "$scenario_path" "$run_scenario_id" "$output_dir" "$replica_count"; then
                log_success "  Completed: $run_scenario_id (replicas: $replica_count)"
                add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "success" "$replica_count"
                env_completed=$((env_completed + 1))
                COMPLETED_SCENARIOS=$((COMPLETED_SCENARIOS + 1))
            else
                log_error "  Failed: $run_scenario_id"
                add_to_index "$run_scenario_id" "$env" "$algorithm" "$payload" "$rate" "$output_dir" "failed" "$replica_count"
                env_failed=$((env_failed + 1))
                FAILED_SCENARIOS=$((FAILED_SCENARIOS + 1))
                
                if [[ "$CONTINUE_ON_ERROR" != "true" ]]; then
                    log_error "Stopping due to failure (use --continue-on-error to ignore)"
                    exit 1
                fi
            fi
        done
        
    done <<< "$scenarios"
    
    log_info "Environment $env: $env_completed completed, $env_failed failed"
done

# =============================================================================
# Phase 4: Write Master Index
# =============================================================================
log_phase "4. Write Master Index"

# Write index.json
INDEX_FILE="$FINAL_RESULTS_DIR/index.json"
{
    echo "{"
    echo "  \"generated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"matrix_file\": \"$MATRIX\","
    echo "  \"environments\": [\"${ENV_ARRAY[*]// /\", \"}\"],"
    echo "  \"total_scenarios\": $TOTAL_SCENARIOS,"
    echo "  \"completed_scenarios\": $COMPLETED_SCENARIOS,"
    echo "  \"failed_scenarios\": $FAILED_SCENARIOS,"
    echo "  \"experiments\": ["
    
    first=true
    for entry in "${MASTER_INDEX[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo ","
        fi
        echo -n "    $entry"
    done
    
    echo ""
    echo "  ]"
    echo "}"
} > "$INDEX_FILE"

log_success "Master index: $INDEX_FILE"

# =============================================================================
# Phase 5: Statistical Aggregation
# =============================================================================
log_phase "5. Statistical Aggregation"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping aggregation"
else
    log_info "Aggregating results across experiments..."
    
    python3 "$SCRIPT_DIR/analysis/aggregate_results.py" \
        --index "$INDEX_FILE" \
        --output "$FINAL_RESULTS_DIR" 2>&1 || log_warn "Aggregation completed with warnings"
    
    log_success "Aggregation complete"
fi

# =============================================================================
# Phase 6: Generate Combined Figures
# =============================================================================
log_phase "6. Generate Combined Figures"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping figure generation"
else
    log_info "Generating combined CDF plots..."
    python3 "$SCRIPT_DIR/analysis/plot_combined_cdfs.py" \
        --index "$INDEX_FILE" \
        --output "$FINAL_RESULTS_DIR/figures" 2>&1 || log_warn "CDF plots completed with warnings"
    
    log_info "Generating scaling curves..."
    python3 "$SCRIPT_DIR/analysis/plot_scaling_curves.py" \
        --index "$INDEX_FILE" \
        --output "$FINAL_RESULTS_DIR/figures" 2>&1 || log_warn "Scaling plots completed with warnings"
    
    log_success "Figures generated"
fi

# =============================================================================
# Phase 7: Hypothesis Testing
# =============================================================================
log_phase "7. Hypothesis Testing"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping hypothesis tests"
else
    log_info "Running statistical hypothesis tests..."
    log_info "  - Kolmogorov-Smirnov test (distribution shape)"
    log_info "  - Mann-Whitney U test (distribution location)"
    log_info "  - Welch's t-test (mean difference)"
    log_info "  - Cohen's d with 95% CI (effect size)"
    log_info "  - Holm-Bonferroni correction (multiple comparisons)"
    
    python3 "$SCRIPT_DIR/analysis/hypothesis_tests.py" \
        --index "$INDEX_FILE" \
        --matrix "$MATRIX" \
        --output "$FINAL_RESULTS_DIR" 2>&1 || log_warn "Hypothesis tests completed with warnings"
    
    # Report results
    if [[ -f "$FINAL_RESULTS_DIR/hypothesis_tests.json" ]]; then
        TOTAL_TESTS=$(python3 -c "import json; print(json.load(open('$FINAL_RESULTS_DIR/hypothesis_tests.json'))['total_comparisons'])" 2>/dev/null || echo "?")
        SIG_TESTS=$(python3 -c "import json; print(json.load(open('$FINAL_RESULTS_DIR/hypothesis_tests.json'))['significant_comparisons'])" 2>/dev/null || echo "?")
        log_info "  Total comparisons: $TOTAL_TESTS"
        log_info "  Significant (α=0.05, corrected): $SIG_TESTS"
    fi
    
    log_success "Hypothesis tests complete"
fi

# =============================================================================
# Phase 8: Build Final Report (PDF)
# =============================================================================
log_phase "8. Build Final Report (PDF)"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping report generation"
else
    log_info "Building dissertation-ready PDF report..."
    
    python3 "$SCRIPT_DIR/analysis/build_final_report.py" \
        --results-dir "$FINAL_RESULTS_DIR" \
        --output "$FINAL_RESULTS_DIR/report.pdf" 2>&1 || log_warn "Report generation completed with warnings"
    
    if [[ -f "$FINAL_RESULTS_DIR/report.pdf" ]]; then
        REPORT_SIZE=$(du -h "$FINAL_RESULTS_DIR/report.pdf" | cut -f1)
        log_success "PDF report generated: $FINAL_RESULTS_DIR/report.pdf ($REPORT_SIZE)"
    else
        log_warn "PDF generation may have failed (reportlab required)"
    fi
fi

# =============================================================================
# Phase 9: Replica Scaling Analysis
# =============================================================================
log_phase "9. Replica Scaling Analysis"

if [[ "$SKIP_ANALYSIS" == "true" ]] || [[ "$SKIP_SCALING" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    log_warn "Skipping scaling analysis"
else
    # Check if we have scaling experiments (replicas > 1)
    HAS_SCALING=false
    for r in "${REPLICA_ARRAY[@]}"; do
        if [[ "$r" -gt 1 ]]; then
            HAS_SCALING=true
            break
        fi
    done
    
    if [[ "$HAS_SCALING" == "true" ]]; then
        log_info "Generating replica scaling plots..."
        
        mkdir -p "$FINAL_RESULTS_DIR/figures/scaling"
        
        python3 "$SCRIPT_DIR/analysis/plot_replica_scaling.py" \
            --index "$INDEX_FILE" \
            --output "$FINAL_RESULTS_DIR/figures/scaling" 2>&1 || log_warn "Scaling plots completed with warnings"
        
        # List generated files
        if [[ -d "$FINAL_RESULTS_DIR/figures/scaling" ]]; then
            SCALING_FILES=$(ls "$FINAL_RESULTS_DIR/figures/scaling/"*.png 2>/dev/null | wc -l)
            log_success "Generated $SCALING_FILES scaling figures"
        fi
    else
        log_info "No scaling experiments (replicas > 1) detected, skipping"
    fi
fi

# =============================================================================
# Final Summary
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                    EXPERIMENT SUITE COMPLETE                          ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log_info "Duration: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
log_info "Total scenarios: $TOTAL_SCENARIOS"
log_info "Completed: $COMPLETED_SCENARIOS"
log_info "Failed: $FAILED_SCENARIOS"
echo ""

log_info "Final results location:"
echo "  $FINAL_RESULTS_DIR/"
echo ""

log_info "Key outputs:"
[[ -f "$FINAL_RESULTS_DIR/index.json" ]] && echo "  ├── index.json (master experiment index)"
[[ -f "$FINAL_RESULTS_DIR/aggregated_stats.json" ]] && echo "  ├── aggregated_stats.json"
[[ -f "$FINAL_RESULTS_DIR/aggregated_stats.csv" ]] && echo "  ├── aggregated_stats.csv"
[[ -f "$FINAL_RESULTS_DIR/hypothesis_tests.json" ]] && echo "  ├── hypothesis_tests.json (statistical tests)"
[[ -f "$FINAL_RESULTS_DIR/hypothesis_table.csv" ]] && echo "  ├── hypothesis_table.csv"
[[ -f "$FINAL_RESULTS_DIR/hypothesis_interpretation.txt" ]] && echo "  ├── hypothesis_interpretation.txt"
[[ -d "$FINAL_RESULTS_DIR/figures" ]] && echo "  ├── figures/"
[[ -d "$FINAL_RESULTS_DIR/figures/scaling" ]] && echo "  │   └── scaling/ (throughput, latency, efficiency)"
[[ -d "$FINAL_RESULTS_DIR/stats" ]] && echo "  ├── stats/"
[[ -d "$FINAL_RESULTS_DIR/tables" ]] && echo "  ├── tables/"
[[ -f "$FINAL_RESULTS_DIR/report.pdf" ]] && echo "  └── report.pdf (dissertation-ready)"
echo ""

if [[ $FAILED_SCENARIOS -gt 0 ]]; then
    log_warn "$FAILED_SCENARIOS experiment(s) failed. Check logs for details."
fi

log_success "Done!"

exit 0

