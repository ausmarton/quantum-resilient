#!/usr/bin/env bash
# =============================================================================
# cleanup_gcp_resources.sh - Safely remove all GCP resources created by PQC benchmarks
#
# Removes (ephemeral resources):
# - GKE clusters
# - Node pools
# - Persistent disks
# - Service accounts
# - Artifact Registry repositories
# - Load balancers
# - Forwarding rules
# - Static IPs
# - Firewall rules (non-default)
# - Orphaned PVs/PVCs
#
# Preserves (persistent resources):
# - GCS bucket (stores experiment results)
#
# Usage:
#   ./scripts/cleanup_gcp_resources.sh --project <project> --region <region> [--cluster-name <name>]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Default values
PROJECT=""
REGION=""
CLUSTER_NAME=""
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --cluster-name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$PROJECT" ]]; then
    log_error "Missing required argument: --project"
    exit 1
fi

if [[ -z "$REGION" ]]; then
    log_error "Missing required argument: --region"
    exit 1
fi

log_info "Cleaning up GCP resources in project '$PROJECT', region '$REGION'"

# Set project
gcloud config set project "$PROJECT" >/dev/null 2>&1 || true

# Global array to track clusters being deleted
declare -a CLUSTERS_TO_DELETE=()

# Function to delete GKE clusters
cleanup_clusters() {
    log_info "Checking for GKE clusters..."
    
    if [[ -n "$CLUSTER_NAME" ]]; then
        CLUSTERS_TO_DELETE=("$CLUSTER_NAME")
    else
        # Find all clusters matching our naming pattern
        CLUSTERS_TO_DELETE=($(gcloud container clusters list \
            --project "$PROJECT" \
            --region "$REGION" \
            --format="value(name)" 2>/dev/null | grep -E "pqc-(bench-gke|smoke-test)" || true))
    fi
    
    for cluster in "${CLUSTERS_TO_DELETE[@]}"; do
        if [[ -z "$cluster" ]]; then
            continue
        fi
        
        log_info "Deleting cluster: $cluster"
        
        # Check if cluster exists first
        if ! gcloud container clusters describe "$cluster" \
            --project "$PROJECT" \
            --region "$REGION" >/dev/null 2>&1; then
            log_info "Cluster '$cluster' does not exist (may already be deleted)"
            continue
        fi
        
        # Wait for cluster to finish provisioning before attempting deletion
        log_info "Checking cluster status before deletion..."
        CLUSTER_STATUS=$(gcloud container clusters describe "$cluster" \
            --project "$PROJECT" \
            --region "$REGION" \
            --format="value(status)" 2>/dev/null || echo "UNKNOWN")
        
        if [[ "$CLUSTER_STATUS" == "PROVISIONING" ]]; then
            log_warn "Cluster '$cluster' is still PROVISIONING. Waiting for it to finish before deletion..."
            MAX_WAIT=600  # 10 minutes
            ELAPSED=0
            while [[ $ELAPSED -lt $MAX_WAIT ]]; do
                CLUSTER_STATUS=$(gcloud container clusters describe "$cluster" \
                    --project "$PROJECT" \
                    --region "$REGION" \
                    --format="value(status)" 2>/dev/null || echo "UNKNOWN")
                
                if [[ "$CLUSTER_STATUS" == "RUNNING" ]]; then
                    log_info "Cluster '$cluster' is now RUNNING, proceeding with deletion"
                    break
                elif [[ "$CLUSTER_STATUS" == "STOPPING" || "$CLUSTER_STATUS" == "ERROR" ]]; then
                    log_info "Cluster '$cluster' is in $CLUSTER_STATUS state, proceeding with cleanup"
                    break
                fi
                
                if [[ $((ELAPSED % 30)) -eq 0 ]]; then
                    log_info "Still waiting for cluster to finish provisioning... ($ELAPSED seconds elapsed, status: $CLUSTER_STATUS)"
                fi
                sleep 10
                ELAPSED=$((ELAPSED + 10))
            done
            
            if [[ "$CLUSTER_STATUS" == "PROVISIONING" ]]; then
                log_warn "Cluster '$cluster' is still PROVISIONING after $MAX_WAIT seconds. Attempting deletion anyway..."
            fi
        fi
        
        # Delete cluster synchronously (not async) to ensure it actually happens
        log_info "Initiating deletion of cluster '$cluster' (this will take 5-10 minutes)..."
        
        # Check for deletion protection first
        DELETION_PROTECTION=$(gcloud container clusters describe "$cluster" \
            --project "$PROJECT" \
            --region "$REGION" \
            --format="value(deletionProtection)" 2>/dev/null || echo "false")
        
        if [[ "$DELETION_PROTECTION" == "true" ]]; then
            log_error "Cluster has deletion protection enabled. Disabling it first..."
            if ! gcloud container clusters update "$cluster" \
                --project "$PROJECT" \
                --region "$REGION" \
                --no-deletion-protection \
                --quiet 2>&1; then
                log_error "Failed to disable deletion protection. You may need to do this manually:"
                echo "  gcloud container clusters update $cluster --project $PROJECT --region $REGION --no-deletion-protection"
                continue
            fi
            log_success "Deletion protection disabled"
        fi
        
        # Delete the cluster
        log_info "Deleting cluster '$cluster' (this will take 5-15 minutes)..."
        log_info "Executing: gcloud container clusters delete $cluster --project $PROJECT --region $REGION --quiet"
        
        # Use --quiet to avoid interactive prompts
        set +e  # Don't exit on error
        DELETE_OUTPUT=$(gcloud container clusters delete "$cluster" \
            --project "$PROJECT" \
            --region "$REGION" \
            --quiet \
            2>&1)
        DELETE_EXIT_CODE=$?
        set -e  # Re-enable exit on error
        
        log_info "Deletion command completed with exit code: $DELETE_EXIT_CODE"
        
        # Check if deletion was initiated (even if command returned error)
        STATUS=$(gcloud container clusters describe "$cluster" \
            --project "$PROJECT" \
            --region "$REGION" \
            --format="value(status)" 2>/dev/null || echo "UNKNOWN")
        
        if [[ "$STATUS" == "DELETING" ]]; then
            log_success "Cluster deletion initiated successfully (status: DELETING)"
        elif [[ "$STATUS" == "UNKNOWN" ]]; then
            log_success "Cluster appears to be deleted (cannot describe it)"
        elif [[ $DELETE_EXIT_CODE -eq 0 ]]; then
            log_success "Cluster deletion command completed successfully: $cluster"
        else
            log_error "Cluster deletion command failed (exit code: $DELETE_EXIT_CODE)"
            log_info "Command output:"
            echo "$DELETE_OUTPUT" | head -20
            
            log_info "Current cluster status: $STATUS"
            
            if [[ "$STATUS" == "RUNNING" ]]; then
                log_error "Cluster is still RUNNING - deletion did not start"
                log_error "Possible causes:"
                echo "  1. Permission denied (need container.clusters.delete role)"
                echo "  2. Cluster has active workloads preventing deletion"
                echo "  3. API quota limits or network issues"
                echo "  4. Deletion protection (should have been disabled above)"
                
                # Try to delete with --async flag as fallback
                log_info "Attempting async deletion as fallback..."
                if gcloud container clusters delete "$cluster" \
                    --project "$PROJECT" \
                    --region "$REGION" \
                    --async \
                    --quiet 2>&1; then
                    log_info "Async deletion initiated"
                    STATUS="DELETING"
                else
                    log_error "Async deletion also failed"
                fi
            else
                log_warn "Cluster status: $STATUS (may still be deleting)"
            fi
        fi
    done
    
    # Wait for clusters to be deleted
    if [[ ${#CLUSTERS_TO_DELETE[@]} -gt 0 ]]; then
        log_info "Waiting for clusters to be deleted (this may take 5-10 minutes)..."
        sleep 15  # Initial wait for deletion to start
        
        for cluster in "${CLUSTERS_TO_DELETE[@]}"; do
            if [[ -z "$cluster" ]]; then
                continue
            fi
            
            log_info "Waiting for cluster '$cluster' to be fully deleted..."
            # Wait up to 20 minutes (GKE cluster deletion can take 5-15 minutes)
            MAX_WAIT=120  # 120 iterations of 10 seconds = 20 minutes
            CLUSTER_DELETED=false
            for i in $(seq 1 $MAX_WAIT); do
                if ! gcloud container clusters describe "$cluster" \
                    --project "$PROJECT" \
                    --region "$REGION" >/dev/null 2>&1; then
                    log_success "Cluster '$cluster' deleted"
                    CLUSTER_DELETED=true
                    break
                fi
                
                # Check status
                STATUS=$(gcloud container clusters describe "$cluster" \
                    --project "$PROJECT" \
                    --region "$REGION" \
                    --format="value(status)" 2>/dev/null || echo "UNKNOWN")
                
                if [[ "$STATUS" == "UNKNOWN" ]]; then
                    log_success "Cluster '$cluster' appears to be deleted"
                    CLUSTER_DELETED=true
                    break
                fi
                
                if [[ $i -eq $MAX_WAIT ]]; then
                    log_warn "Cluster '$cluster' still exists after 20 minutes (status: $STATUS)"
                    log_warn "Cluster deletion may still be in progress. You can check status with:"
                    echo "  gcloud container clusters describe $cluster --project $PROJECT --region $REGION"
                elif [[ $((i % 6)) -eq 0 ]]; then
                    # Log progress every minute
                    log_info "Still waiting for cluster '$cluster' to be deleted... ($((i * 10))s elapsed, status: $STATUS)"
                fi
                sleep 10
            done
            
            if [[ "$CLUSTER_DELETED" != "true" ]]; then
                log_warn "Cluster '$cluster' deletion did not complete within timeout"
                log_info "Cluster deletion can take 5-15 minutes. It will continue in the background."
                log_info "You can check status with: gcloud container clusters list --project $PROJECT --region $REGION"
            fi
        done
        
        # Additional wait for associated resources to be cleaned up
        log_info "Waiting for associated resources (instances, disks) to be cleaned up..."
        sleep 30
    fi
}

# Function to delete compute instances (GKE nodes)
cleanup_compute_instances() {
    log_info "Checking for GKE compute instances..."
    
    # Find GKE instances that might be orphaned
    # Look for instances in all zones of the region
    INSTANCES=()
    for zone in $(gcloud compute zones list --project "$PROJECT" --filter="region:${REGION}" --format="value(name)" 2>/dev/null || echo ""); do
        if [[ -z "$zone" ]]; then
            continue
        fi
        ZONE_INSTANCES=($(gcloud compute instances list \
            --project "$PROJECT" \
            --filter="zone:${zone} AND name~'gke.*pqc'" \
            --format="value(name,zone)" 2>/dev/null || true))
        INSTANCES+=("${ZONE_INSTANCES[@]}")
    done
    
    if [[ ${#INSTANCES[@]} -eq 0 ]]; then
        log_info "No GKE compute instances found"
        return
    fi
    
    log_info "Found ${#INSTANCES[@]} GKE compute instance(s) to check"
    
    for instance_info in "${INSTANCES[@]}"; do
        if [[ -z "$instance_info" ]]; then
            continue
        fi
        # Parse name and zone (handle both space and tab separators)
        INSTANCE_NAME=$(echo "$instance_info" | awk '{print $1}')
        INSTANCE_ZONE=$(echo "$instance_info" | awk '{print $2}')
        
        if [[ -z "$INSTANCE_NAME" || -z "$INSTANCE_ZONE" ]]; then
            continue
        fi
        
        # Check if instance is part of a cluster that's being deleted
        INSTANCE_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
            --project "$PROJECT" \
            --zone "$INSTANCE_ZONE" \
            --format="value(status)" 2>/dev/null || echo "UNKNOWN")
        
        if [[ "$INSTANCE_STATUS" == "STOPPING" || "$INSTANCE_STATUS" == "TERMINATED" ]]; then
            log_info "Instance '$INSTANCE_NAME' is already stopping/terminated (part of cluster deletion)"
            continue
        fi
        
        log_info "Deleting GKE instance: $INSTANCE_NAME in zone $INSTANCE_ZONE (status: $INSTANCE_STATUS)"
        set +e
        DELETE_OUTPUT=$(gcloud compute instances delete "$INSTANCE_NAME" \
            --project "$PROJECT" \
            --zone "$INSTANCE_ZONE" \
            2>&1)
        DELETE_EXIT_CODE=$?
        set -e
        
        if [[ $DELETE_EXIT_CODE -eq 0 ]]; then
            log_success "Deleted instance: $INSTANCE_NAME"
        else
            # Check if instance still exists
            CURRENT_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
                --project "$PROJECT" \
                --zone "$INSTANCE_ZONE" \
                --format="value(status)" 2>/dev/null || echo "UNKNOWN")
            
            if [[ "$CURRENT_STATUS" == "UNKNOWN" ]]; then
                log_success "Instance '$INSTANCE_NAME' appears to be deleted"
            else
                log_warn "Failed to delete instance '$INSTANCE_NAME' (status: $CURRENT_STATUS)"
                log_info "Error output: $DELETE_OUTPUT"
            fi
        fi
    done
}

# Function to delete persistent disks
cleanup_disks() {
    log_info "Checking for persistent disks..."
    
    # Find disks that might be from our benchmarks
    # Exclude disks that are attached to GKE instances (they'll be deleted with the cluster)
    DISKS=($(gcloud compute disks list \
        --project "$PROJECT" \
        --filter="zone:${REGION}* AND name~'pqc|bench' AND -users:*" \
        --format="value(name,zone)" 2>/dev/null || true))
    
    if [[ ${#DISKS[@]} -eq 0 ]]; then
        log_info "No orphaned benchmark-related disks found (GKE-managed disks will be deleted with clusters)"
        return
    fi
    
    for disk_info in "${DISKS[@]}"; do
        if [[ -z "$disk_info" ]]; then
            continue
        fi
        # Parse name and zone
        DISK_NAME=$(echo "$disk_info" | cut -d' ' -f1)
        DISK_ZONE=$(echo "$disk_info" | cut -d' ' -f2)
        
        if [[ -z "$DISK_NAME" || -z "$DISK_ZONE" ]]; then
            continue
        fi
        
        log_info "Deleting orphaned disk: $DISK_NAME in zone $DISK_ZONE"
        if gcloud compute disks delete "$DISK_NAME" \
            --project "$PROJECT" \
            --zone "$DISK_ZONE" \
            --quiet 2>/dev/null; then
            log_success "Deleted disk: $DISK_NAME"
        else
            log_warn "Disk may not exist or already deleted: $DISK_NAME"
        fi
    done
}

# Function to delete forwarding rules and load balancers
cleanup_load_balancers() {
    log_info "Checking for forwarding rules and load balancers..."
    
    # Delete forwarding rules
    FORWARDING_RULES=($(gcloud compute forwarding-rules list \
        --project "$PROJECT" \
        --regions "$REGION" \
        --format="value(name)" 2>/dev/null || true))
    
    for rule in "${FORWARDING_RULES[@]}"; do
        if [[ -z "$rule" ]]; then
            continue
        fi
        log_info "Deleting forwarding rule: $rule"
        if gcloud compute forwarding-rules delete "$rule" \
            --project "$PROJECT" \
            --region "$REGION" \
            --quiet 2>/dev/null; then
            log_success "Deleted forwarding rule: $rule"
        else
            log_warn "Forwarding rule may not exist: $rule"
        fi
    done
    
    # Delete target pools (legacy)
    TARGET_POOLS=($(gcloud compute target-pools list \
        --project "$PROJECT" \
        --regions "$REGION" \
        --format="value(name)" 2>/dev/null || true))
    
    for pool in "${TARGET_POOLS[@]}"; do
        if [[ -z "$pool" ]]; then
            continue
        fi
        log_info "Deleting target pool: $pool"
        if gcloud compute target-pools delete "$pool" \
            --project "$PROJECT" \
            --region "$REGION" \
            --quiet 2>/dev/null; then
            log_success "Deleted target pool: $pool"
        else
            log_warn "Target pool may not exist: $pool"
        fi
    done
}

# Function to delete static IPs
cleanup_static_ips() {
    log_info "Checking for static IP addresses..."
    
    # Find static IPs in the region
    IPS=($(gcloud compute addresses list \
        --project "$PROJECT" \
        --filter="region:${REGION}" \
        --format="value(name)" 2>/dev/null || true))
    
    for ip in "${IPS[@]}"; do
        if [[ -z "$ip" ]]; then
            continue
        fi
        # Skip default IPs (they're usually named differently)
        if [[ "$ip" == *"default"* ]]; then
            continue
        fi
        log_info "Deleting static IP: $ip"
        if gcloud compute addresses delete "$ip" \
            --project "$PROJECT" \
            --region "$REGION" \
            --quiet 2>/dev/null; then
            log_success "Deleted static IP: $ip"
        else
            log_warn "Static IP may not exist: $ip"
        fi
    done
}

# Function to delete firewall rules (non-default)
cleanup_firewall_rules() {
    log_info "Checking for custom firewall rules..."
    
    # Find firewall rules that might be from our benchmarks
    RULES=($(gcloud compute firewall-rules list \
        --project "$PROJECT" \
        --filter="name~'pqc|bench' AND name!~'default'" \
        --format="value(name)" 2>/dev/null || true))
    
    for rule in "${RULES[@]}"; do
        if [[ -z "$rule" ]]; then
            continue
        fi
        log_info "Deleting firewall rule: $rule"
        if gcloud compute firewall-rules delete "$rule" \
            --project "$PROJECT" \
            --quiet 2>/dev/null; then
            log_success "Deleted firewall rule: $rule"
        else
            log_warn "Firewall rule may not exist: $rule"
        fi
    done
}

# Function to delete service accounts
cleanup_service_accounts() {
    log_info "Checking for PQC benchmark service accounts..."
    
    # Find service accounts that might be from our benchmarks
    # Terraform creates: qr-orchestrator, qr-worker
    # Legacy names: pqc.*bench, pqc.*smoke
    SERVICE_ACCOUNTS=($(gcloud iam service-accounts list \
        --project "$PROJECT" \
        --filter="email~'qr-orchestrator@' OR email~'qr-worker@' OR email~'pqc.*bench' OR email~'pqc.*smoke'" \
        --format="value(email)" 2>/dev/null || true))
    
    for sa_email in "${SERVICE_ACCOUNTS[@]}"; do
        if [[ -z "$sa_email" ]]; then
            continue
        fi
        log_info "Deleting service account: $sa_email"
        
        # Delete IAM bindings first (if any)
        # Note: We don't delete bindings to the bucket as it's persistent
        
        # Delete the service account
        if gcloud iam service-accounts delete "$sa_email" \
            --project "$PROJECT" \
            --quiet 2>/dev/null; then
            log_success "Deleted service account: $sa_email"
        else
            log_warn "Service account may not exist or may be in use: $sa_email"
        fi
    done
}

# Function to delete Artifact Registry repositories
cleanup_artifact_registry() {
    log_info "Checking for PQC benchmark Artifact Registry repositories..."
    
    # Find Artifact Registry repositories that might be from our benchmarks
    # Look in common locations (us-central1, europe-west2, etc.)
    AR_LOCATIONS=("us-central1" "europe-west2" "europe-west1" "us-east1")
    
    for location in "${AR_LOCATIONS[@]}"; do
        REPOS=($(gcloud artifacts repositories list \
            --project "$PROJECT" \
            --location "$location" \
            --filter="name~'pqc'" \
            --format="value(name)" 2>/dev/null || true))
        
        for repo in "${REPOS[@]}"; do
            if [[ -z "$repo" ]]; then
                continue
            fi
            log_info "Deleting Artifact Registry repository: $repo (location: $location)"
            
            if gcloud artifacts repositories delete "$repo" \
                --location "$location" \
                --project "$PROJECT" \
                --quiet 2>/dev/null; then
                log_success "Deleted Artifact Registry repository: $repo"
            else
                log_warn "Artifact Registry repository may not exist: $repo"
            fi
        done
    done
}

# Function to verify cleanup
verify_cleanup() {
    log_info "Verifying cleanup..."
    
    local ERRORS=0
    
    # Use the global CLUSTERS_TO_DELETE array if available, otherwise find clusters
    if [[ ${#CLUSTERS_TO_DELETE[@]} -eq 0 ]]; then
        if [[ -n "$CLUSTER_NAME" ]]; then
            CLUSTERS_TO_DELETE=("$CLUSTER_NAME")
        else
            CLUSTERS_TO_DELETE=($(gcloud container clusters list \
                --project "$PROJECT" \
                --region "$REGION" \
                --format="value(name)" 2>/dev/null | grep -E "pqc-(bench-gke|smoke-test)" || true))
        fi
    fi
    
    # Check clusters
    REMAINING_CLUSTERS=$(gcloud container clusters list \
        --project "$PROJECT" \
        --region "$REGION" \
        --format="value(name)" 2>/dev/null | grep -E "pqc-(bench-gke|smoke-test)" || true)
    
    if [[ -n "$REMAINING_CLUSTERS" ]]; then
        # Check if clusters are in "DELETING" state
        DELETING_CLUSTERS=""
        for cluster in $REMAINING_CLUSTERS; do
            STATUS=$(gcloud container clusters describe "$cluster" \
                --project "$PROJECT" \
                --region "$REGION" \
                --format="value(status)" 2>/dev/null || echo "UNKNOWN")
            
            if [[ "$STATUS" == "DELETING" ]]; then
                DELETING_CLUSTERS="$DELETING_CLUSTERS $cluster"
            else
                log_error "Cluster '$cluster' still exists (status: $STATUS)"
                ERRORS=$((ERRORS + 1))
            fi
        done
        
        if [[ -n "$DELETING_CLUSTERS" ]]; then
            log_info "Clusters in deletion process (this is normal, deletion takes 5-10 minutes):"
            echo "$DELETING_CLUSTERS"
        fi
    fi
    
    # Check instances (GKE instances are deleted as part of cluster deletion)
    # Only report if they're still there after cluster deletion
    REMAINING_INSTANCES=$(gcloud compute instances list \
        --project "$PROJECT" \
        --filter="zone:${REGION}* AND name~'gke.*pqc'" \
        --format="value(name)" 2>/dev/null || true)
    
    if [[ -n "$REMAINING_INSTANCES" ]]; then
        # Check if the cluster still exists - if not, instances should be gone soon
        CLUSTER_STILL_EXISTS=false
        for cluster in "${CLUSTERS_TO_DELETE[@]}"; do
            if [[ -n "$cluster" ]] && gcloud container clusters describe "$cluster" \
                --project "$PROJECT" \
                --region "$REGION" >/dev/null 2>&1; then
                CLUSTER_STILL_EXISTS=true
                break
            fi
        done
        
        if [[ "$CLUSTER_STILL_EXISTS" == "true" ]]; then
            log_info "GKE instances still exist (cluster deletion in progress - this is normal)"
        else
            log_warn "GKE instances found but cluster is deleted (may still be cleaning up):"
            echo "$REMAINING_INSTANCES"
        fi
    fi
    
    # Check disks (GKE-managed disks are deleted with the cluster)
    # Only check for orphaned disks (not attached to instances)
    REMAINING_DISKS=$(gcloud compute disks list \
        --project "$PROJECT" \
        --filter="zone:${REGION}* AND name~'pqc|bench' AND -users:*" \
        --format="value(name)" 2>/dev/null || true)
    
    if [[ -n "$REMAINING_DISKS" ]]; then
        log_error "Orphaned disks found (not attached to instances):"
        echo "$REMAINING_DISKS"
        ERRORS=$((ERRORS + 1))
    else
        # Check if there are any disks attached to instances (these will be deleted with instances)
        ATTACHED_DISKS=$(gcloud compute disks list \
            --project "$PROJECT" \
            --filter="zone:${REGION}* AND name~'pqc|bench' AND users:*" \
            --format="value(name)" 2>/dev/null || true)
        
        if [[ -n "$ATTACHED_DISKS" ]]; then
            log_info "Disks found but attached to instances (will be deleted with instances/cluster)"
        fi
    fi
    
    # Check forwarding rules
    REMAINING_FORWARDING_RULES=$(gcloud compute forwarding-rules list \
        --project "$PROJECT" \
        --regions "$REGION" \
        --format="value(name)" 2>/dev/null || true)
    
    if [[ -n "$REMAINING_FORWARDING_RULES" ]]; then
        log_warn "Remaining forwarding rules found:"
        echo "$REMAINING_FORWARDING_RULES"
    fi
    
    # Check static IPs
    REMAINING_IPS=$(gcloud compute addresses list \
        --project "$PROJECT" \
        --filter="region:${REGION} AND name!~'default'" \
        --format="value(name)" 2>/dev/null || true)
    
    if [[ -n "$REMAINING_IPS" ]]; then
        log_warn "Remaining static IPs found:"
        echo "$REMAINING_IPS"
    fi
    
    if [[ $ERRORS -eq 0 ]]; then
        log_success "Cleanup verification passed"
        return 0
    else
        log_error "Cleanup verification found $ERRORS issue(s)"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting GCP resource cleanup..."
    
    # First, delete clusters (this is the most important step)
    log_info "Step 1: Deleting GKE clusters..."
    cleanup_clusters
    
    # Wait a bit for cluster deletion to start
    sleep 10
    
    # Then clean up compute instances (GKE nodes) if they're orphaned
    log_info "Step 2: Cleaning up GKE compute instances..."
    cleanup_compute_instances
    
    # Then clean up other resources
    log_info "Step 3: Cleaning up service accounts and Artifact Registry..."
    cleanup_service_accounts
    cleanup_artifact_registry
    
    log_info "Step 4: Cleaning up load balancers, IPs, and firewall rules..."
    cleanup_load_balancers
    cleanup_static_ips
    cleanup_firewall_rules
    
    # Note: cleanup_disks is called after cluster deletion completes
    # because GKE-managed disks are deleted automatically with clusters
    
    # Wait for cluster deletion to complete before checking disks
    log_info "Step 5: Waiting for cluster deletion to complete..."
    
    # Verify cleanup (this will wait for clusters and check for orphaned resources)
    if verify_cleanup; then
        # Now check for any orphaned disks (not managed by GKE)
        log_info "Step 6: Checking for orphaned disks..."
        cleanup_disks
        
        # Final check for compute instances (in case they weren't deleted with cluster)
        log_info "Step 7: Final check for orphaned compute instances..."
        cleanup_compute_instances
        
        log_success "All resources cleaned up successfully"
        return 0
    else
        log_warn "Some resources may still be cleaning up."
        log_info "GKE cluster deletion can take 5-10 minutes. Resources will be automatically cleaned up."
        
        # Try to clean up compute instances even if cluster deletion is in progress
        log_info "Attempting to clean up compute instances that may be orphaned..."
        cleanup_compute_instances
        
        log_info "You can run this script again in a few minutes to verify cleanup is complete."
        return 1
    fi
}

main "$@"

