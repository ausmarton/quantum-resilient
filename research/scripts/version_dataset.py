#!/usr/bin/env python3
"""
Dataset versioning for experiment results.

Computes deterministic checksums and manages semantic versions.

Usage:
    python version_dataset.py --exp-id exp_001 --data-dir analysis/data/exp_001 --version 1.0.0 --out research/output/exp_001/
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compute_directory_checksum(dir_path: Path, extensions: Optional[list[str]] = None) -> str:
    """Compute deterministic checksum for a directory."""
    sha256_hash = hashlib.sha256()
    
    files = sorted(dir_path.rglob("*"))
    for filepath in files:
        if not filepath.is_file():
            continue
        if extensions and filepath.suffix not in extensions:
            continue
        
        # Include relative path in hash for determinism
        rel_path = filepath.relative_to(dir_path)
        sha256_hash.update(str(rel_path).encode())
        sha256_hash.update(compute_sha256(filepath).encode())
    
    return sha256_hash.hexdigest()


def compute_dataset_checksums(data_dir: Path) -> dict[str, str]:
    """Compute checksums for all dataset components."""
    checksums = {}
    
    # Merged files
    merged_dir = data_dir / "merged"
    if merged_dir.exists():
        jsonl = merged_dir / "merged.jsonl"
        if jsonl.exists():
            checksums["merged.jsonl"] = compute_sha256(jsonl)
        
        parquet = merged_dir / "merged.parquet"
        if parquet.exists():
            checksums["merged.parquet"] = compute_sha256(parquet)
    
    # Statistics
    stats_dir = data_dir / "stats"
    if stats_dir.exists():
        summary = stats_dir / "summary.json"
        if summary.exists():
            checksums["summary.json"] = compute_sha256(summary)
        
        # Compute checksum for all stats files
        checksums["stats_dir"] = compute_directory_checksum(stats_dir)
    
    # Plots
    figures_dir = data_dir.parent.parent / "figures" / data_dir.name
    if not figures_dir.exists():
        figures_dir = data_dir / "figures"
    
    if figures_dir.exists():
        checksums["figures_dir"] = compute_directory_checksum(
            figures_dir, extensions=[".png", ".pdf", ".eps"]
        )
    
    # Raw files checksum
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        checksums["raw_dir"] = compute_directory_checksum(raw_dir, extensions=[".jsonl"])
    
    # Overall dataset checksum
    all_checksums = "".join(sorted(checksums.values()))
    checksums["dataset_overall"] = hashlib.sha256(all_checksums.encode()).hexdigest()
    
    return checksums


def parse_semantic_version(version: str) -> tuple[int, int, int]:
    """Parse semantic version string."""
    parts = version.lstrip("v").split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}. Use X.Y.Z")
    return int(parts[0]), int(parts[1]), int(parts[2])


def version_dataset(
    experiment_id: str,
    data_dir: Path,
    output_dir: Path,
    version: str = "1.0.0",
    changelog: str = "Initial release",
    previous_version_file: Optional[Path] = None,
) -> dict:
    """Generate dataset version metadata."""
    print(f"Versioning dataset for {experiment_id}")
    print(f"  Version: {version}")
    print(f"  Data directory: {data_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse version
    major, minor, patch = parse_semantic_version(version)
    
    # Compute checksums
    print("  Computing checksums...")
    checksums = compute_dataset_checksums(data_dir)
    print(f"    Computed {len(checksums)} checksums")
    
    # Build version metadata
    version_data = {
        "version": version,
        "version_parts": {
            "major": major,
            "minor": minor,
            "patch": patch,
        },
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checksums": checksums,
        "changelog": changelog,
    }
    
    # Load previous version for comparison
    if previous_version_file and previous_version_file.exists():
        try:
            with open(previous_version_file) as f:
                prev = json.load(f)
            
            prev_checksums = prev.get("checksums", {})
            changes = []
            
            for key, value in checksums.items():
                if key not in prev_checksums:
                    changes.append(f"Added: {key}")
                elif prev_checksums[key] != value:
                    changes.append(f"Modified: {key}")
            
            for key in prev_checksums:
                if key not in checksums:
                    changes.append(f"Removed: {key}")
            
            if changes:
                version_data["changes_from_previous"] = changes
                version_data["previous_version"] = prev.get("version")
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not load previous version: {e}")
    
    # File inventory
    inventory = []
    for key in checksums:
        if key.endswith("_dir"):
            continue
        inventory.append({
            "file": key,
            "checksum": checksums[key],
        })
    version_data["file_inventory"] = inventory
    
    # Determine version type
    if major > 0 and minor == 0 and patch == 0:
        version_data["version_type"] = "major"
        version_data["version_note"] = "Major release with breaking changes or new experiment type"
    elif minor > 0 and patch == 0:
        version_data["version_type"] = "minor"
        version_data["version_note"] = "Minor release with new statistics or analysis"
    else:
        version_data["version_type"] = "patch"
        version_data["version_note"] = "Patch release with corrections or documentation updates"
    
    # Save version file
    version_path = output_dir / "dataset_version.json"
    with open(version_path, "w") as f:
        json.dump(version_data, f, indent=2)
    
    print(f"  Saved version metadata to {version_path}")
    print(f"  Dataset checksum: {checksums.get('dataset_overall', 'N/A')[:16]}...")
    
    return version_data


def main():
    parser = argparse.ArgumentParser(description="Generate dataset version metadata")
    parser.add_argument("--exp-id", required=True, help="Experiment identifier")
    parser.add_argument("--data-dir", required=True, help="Path to experiment data directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--version", default="1.0.0", help="Semantic version (X.Y.Z)")
    parser.add_argument("--changelog", default="Initial release", help="Changelog entry")
    parser.add_argument("--previous", help="Path to previous version file for comparison")
    
    args = parser.parse_args()
    
    previous_version = Path(args.previous) if args.previous else None
    
    version_data = version_dataset(
        experiment_id=args.exp_id,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.out),
        version=args.version,
        changelog=args.changelog,
        previous_version_file=previous_version,
    )
    
    print(f"\nDataset version {version_data['version']} created!")
    print(f"  Type: {version_data.get('version_type', 'unknown')}")


if __name__ == "__main__":
    main()

