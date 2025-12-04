#!/usr/bin/env python3
"""
Generate provenance metadata for an experiment.

Captures:
- Git commit hashes
- Cluster configuration
- Scenario YAML
- File checksums
- Timestamps
- Worker jitter statistics

Usage:
    python provenance.py --exp-id exp_001 --data-dir analysis/data/exp_001 --out research/output/exp_001/
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


def get_git_commit(repo_path: Optional[str] = None) -> str:
    """Get current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "rev-parse", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch(repo_path: Optional[str] = None) -> str:
    """Get current git branch."""
    try:
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_checksums(data_dir: Path) -> dict[str, str]:
    """Compute checksums for all relevant files."""
    checksums = {}
    
    # Check merged directory
    merged_dir = data_dir / "merged"
    if merged_dir.exists():
        for f in merged_dir.glob("*"):
            if f.is_file():
                checksums[f"merged/{f.name}"] = compute_sha256(f)
    
    # Check raw JSONL files
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        for f in raw_dir.glob("*.jsonl"):
            checksums[f"raw/{f.name}"] = compute_sha256(f)
    
    # Check stats
    stats_dir = data_dir / "stats"
    if stats_dir.exists():
        for f in stats_dir.glob("*"):
            if f.is_file():
                checksums[f"stats/{f.name}"] = compute_sha256(f)
    
    return checksums


def load_scenario_yaml(data_dir: Path) -> Optional[str]:
    """Load scenario YAML if available."""
    scenario_path = data_dir / "scenario.yaml"
    if scenario_path.exists():
        return scenario_path.read_text()
    
    # Try to find in fetch metadata
    metadata_path = data_dir / "fetch_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            return metadata.get("scenario_yaml")
        except (json.JSONDecodeError, KeyError):
            pass
    
    return None


def compute_worker_jitter_stats(data_dir: Path) -> Optional[dict]:
    """Compute worker start-time jitter statistics from merged data."""
    try:
        import pandas as pd
        
        merged_path = data_dir / "merged" / "merged.parquet"
        if not merged_path.exists():
            merged_path = data_dir / "merged" / "merged.jsonl"
        
        if not merged_path.exists():
            return None
        
        if merged_path.suffix == ".parquet":
            df = pd.read_parquet(merged_path)
        else:
            df = pd.read_json(merged_path, lines=True)
        
        if "worker_id" not in df.columns or "timestamp_utc_iso" not in df.columns:
            return None
        
        df["timestamp"] = pd.to_datetime(df["timestamp_utc_iso"])
        
        # Get first timestamp per worker
        first_timestamps = df.groupby("worker_id")["timestamp"].min()
        
        if len(first_timestamps) < 2:
            return {"worker_count": len(first_timestamps), "max_jitter_ms": 0, "mean_jitter_ms": 0}
        
        min_start = first_timestamps.min()
        jitter_ms = (first_timestamps - min_start).dt.total_seconds() * 1000
        
        return {
            "worker_count": len(first_timestamps),
            "max_jitter_ms": float(jitter_ms.max()),
            "mean_jitter_ms": float(jitter_ms.mean()),
            "std_jitter_ms": float(jitter_ms.std()),
            "first_timestamps": {int(k): str(v) for k, v in first_timestamps.to_dict().items()},
        }
    except Exception as e:
        print(f"Warning: Could not compute jitter stats: {e}")
        return None


def load_cluster_config(data_dir: Path) -> Optional[dict]:
    """Load cluster configuration if available."""
    config_path = data_dir / "cluster_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass
    
    # Try fetch metadata
    metadata_path = data_dir / "fetch_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            return metadata.get("cluster_config")
        except (json.JSONDecodeError, KeyError):
            pass
    
    return None


def load_stats_summary(data_dir: Path) -> Optional[dict]:
    """Load statistical summary if available."""
    stats_path = data_dir / "stats" / "summary.json"
    if stats_path.exists():
        try:
            return json.loads(stats_path.read_text())
        except json.JSONDecodeError:
            pass
    return None


def generate_provenance(
    experiment_id: str,
    data_dir: Path,
    output_dir: Path,
    storage_uri: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """Generate complete provenance metadata."""
    print(f"Generating provenance for {experiment_id}")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build provenance
    provenance: dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generator": "quantum-resilient-provenance",
        "generator_version": "1.0.0",
    }
    
    # Git information
    provenance["git_commit"] = get_git_commit()
    provenance["git_branch"] = get_git_branch()
    provenance["orchestrator_commit"] = get_git_commit()  # Same repo for now
    
    # Storage URI
    if storage_uri:
        provenance["storage_uri"] = storage_uri
    else:
        # Try to get from fetch metadata
        metadata_path = data_dir / "fetch_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                provenance["storage_uri"] = metadata.get("uri", "unknown")
            except json.JSONDecodeError:
                provenance["storage_uri"] = "unknown"
    
    # Scenario YAML
    scenario_yaml = load_scenario_yaml(data_dir)
    if scenario_yaml:
        provenance["scenario_yaml"] = scenario_yaml
    
    # Cluster configuration
    cluster_config = load_cluster_config(data_dir)
    if cluster_config:
        provenance["cluster_config"] = cluster_config
    
    # File checksums
    print("  Computing checksums...")
    provenance["checksums"] = compute_checksums(data_dir)
    print(f"    Computed {len(provenance['checksums'])} checksums")
    
    # Worker jitter statistics
    print("  Computing worker jitter stats...")
    jitter_stats = compute_worker_jitter_stats(data_dir)
    if jitter_stats:
        provenance["worker_jitter_stats"] = jitter_stats
        print(f"    Max jitter: {jitter_stats['max_jitter_ms']:.2f} ms")
    
    # Statistics summary (for quick reference)
    stats = load_stats_summary(data_dir)
    if stats:
        provenance["stats_summary"] = {
            "total_events": stats.get("total_events"),
            "latency_mean": stats.get("latency", {}).get("mean"),
            "latency_p99": stats.get("latency", {}).get("p99"),
            "throughput_mean": stats.get("throughput", {}).get("mean_msgs_per_sec"),
        }
    
    # Resource allocation (if available)
    if extra_metadata:
        provenance["extra"] = extra_metadata
    
    # Environment info
    provenance["environment"] = {
        "python_version": os.popen("python --version 2>&1").read().strip(),
        "hostname": os.uname().nodename,
        "os": f"{os.uname().sysname} {os.uname().release}",
    }
    
    # Save provenance
    provenance_path = output_dir / "provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    
    print(f"  Saved provenance to {provenance_path}")
    
    return provenance


def main():
    parser = argparse.ArgumentParser(description="Generate experiment provenance metadata")
    parser.add_argument("--exp-id", required=True, help="Experiment identifier")
    parser.add_argument("--data-dir", required=True, help="Path to experiment data directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--storage-uri", help="Storage URI (e.g., gs://bucket/path)")
    parser.add_argument("--extra-json", help="Path to JSON file with extra metadata")
    
    args = parser.parse_args()
    
    extra_metadata = None
    if args.extra_json:
        with open(args.extra_json) as f:
            extra_metadata = json.load(f)
    
    provenance = generate_provenance(
        experiment_id=args.exp_id,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.out),
        storage_uri=args.storage_uri,
        extra_metadata=extra_metadata,
    )
    
    print("\nProvenance generation complete!")
    print(f"  Checksums: {len(provenance.get('checksums', {}))}")
    print(f"  Git commit: {provenance.get('git_commit', 'unknown')[:12]}")


if __name__ == "__main__":
    main()

