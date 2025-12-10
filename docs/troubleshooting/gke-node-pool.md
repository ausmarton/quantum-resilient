# Troubleshooting GKE Node Pool Creation Errors

## Overview

When GKE node pool creation fails with "ERROR" state, the error message doesn't always show the specific reason. This guide helps diagnose and fix common node pool creation issues.

## Diagnosing Node Pool Errors

### 1. Check the Actual Error in GCP Console

```bash
# Get the cluster name
CLUSTER_NAME=$(cd terraform/gke && terraform output -raw cluster_name 2>/dev/null || echo "pqc-bench-gke")

# Or for smoke test
CLUSTER_NAME="pqc-smoke-test"

# Describe the node pool to see the error
gcloud container node-pools describe pqc-bench-pool \
  --cluster "$CLUSTER_NAME" \
  --region us-central1 \
  --project <your-project>
```

### 2. Check via GCP Console

1. Go to: https://console.cloud.google.com/kubernetes/clusters
2. Click on your cluster
3. Click on "Node pools" tab
4. Click on "pqc-bench-pool"
5. Look at the "Status" and error messages

### 3. Check Cluster Status

Ensure the cluster itself is healthy:

```bash
gcloud container clusters describe pqc-smoke-test \
  --region us-central1 \
  --project <your-project> \
  --format="value(status)"
```

### 4. Check Logs

```bash
# Check GKE operation logs
gcloud logging read "resource.type=gke_cluster AND resource.labels.cluster_name=pqc-smoke-test" \
  --limit 50 \
  --project <your-project>
```

## Common Issues and Fixes

### Issue: Machine Type Not Available

**Error:** "Machine type 'e2-small' is not available in zone 'us-central1-a'"

**Fix:** Try a different machine type or zone:

```bash
# Check available machine types
gcloud compute machine-types list --filter="zone:us-central1-a" | grep e2

# Or use e2-medium instead
# Edit terraform/gke/main.tf, change e2-small to e2-medium
```

**Solution in Terraform:**
```hcl
node_config {
  machine_type = "e2-medium"  # More reliable than e2-small
  # ... rest of config
}
```

### Issue: Quota Exceeded

**Error:** "Quota 'IN_USE_ADDRESSES' exceeded"

**Fix:** Check quotas:

```bash
gcloud compute project-info describe --project <your-project> | grep -A 5 quota
```

**Solution:**
- Request quota increase in GCP Console
- Or reduce node count temporarily
- Or use a different region with available quota

### Issue: Service Account Permissions

**Error:** "Permission denied" or "Service account does not have required permissions"

**Fix:** For smoke test, we're using default service account (null). If still failing, ensure:

```bash
# Check if default compute service account exists
gcloud iam service-accounts list --project <your-project>
```

**Solution:**
- Ensure default compute service account has necessary permissions
- Or specify a custom service account with proper IAM roles

### Issue: Disk Size Too Small

**Error:** "Disk size must be at least 20GB"

**Fix:** Already fixed - changed from 10GB to 20GB minimum

**Solution in Terraform:**
```hcl
node_config {
  disk_size_gb = 20  # Minimum required
  disk_type    = "pd-standard"
  # ... rest of config
}
```

### Issue: Shielded VM Not Supported

**Error:** "Shielded VM features not supported on this machine type"

**Fix:** Already fixed - disabled shielded VM for smoke test

**Solution in Terraform:**
```hcl
node_config {
  # No shielded VM configuration
  # ... rest of config
}
```

## Simplified Configuration for Troubleshooting

If still failing, try the absolute minimum configuration:

```hcl
resource "google_container_node_pool" "primary" {
  name       = "pqc-bench-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  project    = var.project_id
  node_count = 1

  node_config {
    machine_type = "e2-medium"  # More reliable than e2-small
    disk_size_gb = 20
    disk_type    = "pd-standard"
    
    # Use default service account
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Minimal metadata
    metadata = {
      disable-legacy-endpoints = "true"
    }
    
    # No shielded VM
    # No custom service account
    # No workload identity
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
```

## Manual Node Pool Creation (for Testing)

If Terraform keeps failing, create manually to see the exact error:

```bash
gcloud container node-pools create pqc-bench-pool \
  --cluster pqc-smoke-test \
  --region us-central1 \
  --machine-type e2-medium \
  --disk-size 20 \
  --disk-type pd-standard \
  --num-nodes 1 \
  --project <your-project>
```

This will show the exact error message.

## Related Documentation

- [GCP Deployment Guide](../reference/gcp-deployment.md) - Complete GCP/GKE deployment guide
- [Terraform GKE Configuration](../../terraform/gke/) - Terraform configuration files
