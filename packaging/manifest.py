#!/usr/bin/env python3
"""
Generate machine-readable manifest for experiment bundles.

The manifest describes all files, checksums, metadata, and provenance
information for a complete experiment artifact bundle.

Usage:
    python -m packaging.manifest exp_001 --data-dir analysis/data/exp_001 --research-dir research/output/exp_001
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_file_info(filepath: Path, base_path: Path, category: str = "data") -> dict:
    """Get file information including checksum."""
    return {
        "path": str(filepath.relative_to(base_path)),
        "sha256": compute_sha256(filepath),
        "size_bytes": filepath.stat().st_size,
        "category": category,
    }


def collect_files(
    data_dir: Path,
    research_dir: Path,
    figures_dir: Optional[Path] = None,
) -> tuple[list[dict], Path]:
    """Collect all files for the manifest."""
    files = []
    base_path = data_dir.parent.parent  # Go up to get common base
    
    # Data files
    merged_dir = data_dir / "merged"
    if merged_dir.exists():
        for f in merged_dir.glob("*"):
            if f.is_file():
                files.append(get_file_info(f, base_path, "data"))
    
    # Stats files
    stats_dir = data_dir / "stats"
    if stats_dir.exists():
        for f in stats_dir.glob("*"):
            if f.is_file():
                files.append(get_file_info(f, base_path, "statistics"))
    
    # Research outputs
    if research_dir.exists():
        # Provenance
        provenance = research_dir / "provenance.json"
        if provenance.exists():
            files.append(get_file_info(provenance, base_path, "metadata"))
        
        # Dataset version
        version = research_dir / "dataset_version.json"
        if version.exists():
            files.append(get_file_info(version, base_path, "metadata"))
        
        # Tables
        tables_dir = research_dir / "tables"
        if tables_dir.exists():
            for f in tables_dir.glob("*"):
                if f.is_file():
                    files.append(get_file_info(f, base_path, "tables"))
        
        # Figures
        figs_dir = research_dir / "figures"
        if figs_dir.exists():
            for f in figs_dir.rglob("*"):
                if f.is_file() and f.suffix in [".png", ".pdf", ".eps"]:
                    files.append(get_file_info(f, base_path, "figures"))
        
        # Reports
        for report in ["report.md", "report.tex"]:
            report_path = research_dir / report
            if report_path.exists():
                files.append(get_file_info(report_path, base_path, "reports"))
    
    # External figures directory
    if figures_dir and figures_dir.exists():
        for f in figures_dir.rglob("*"):
            if f.is_file() and f.suffix in [".png", ".pdf", ".eps"]:
                files.append(get_file_info(f, base_path, "figures"))
    
    return files, base_path


def collect_figures_metadata(research_dir: Path) -> list[dict]:
    """Collect figure metadata from manifest if available."""
    figures = []
    manifest_path = research_dir / "figures" / "manifest.json"
    
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                fig_manifest = json.load(f)
            figures = fig_manifest.get("figures", [])
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Fall back to scanning directory
    if not figures:
        figs_dir = research_dir / "figures"
        if figs_dir.exists():
            for png in figs_dir.glob("**/*.png"):
                figures.append({
                    "name": png.stem,
                    "path": str(png.relative_to(figs_dir)),
                    "caption": png.stem.replace("_", " ").title(),
                    "formats": ["png"],
                })
    
    return figures


def collect_tables_metadata(research_dir: Path) -> list[dict]:
    """Collect table metadata."""
    tables = []
    tables_dir = research_dir / "tables"
    
    if not tables_dir.exists():
        return tables
    
    # Find unique table names (without extension)
    table_names = set()
    for f in tables_dir.glob("*.tex"):
        table_names.add(f.stem)
    
    for name in sorted(table_names):
        table = {
            "name": name,
            "path_tex": f"tables/{name}.tex",
            "path_md": f"tables/{name}.md" if (tables_dir / f"{name}.md").exists() else "",
            "description": name.replace("_", " ").title(),
        }
        tables.append(table)
    
    return tables


def load_provenance(research_dir: Path) -> dict:
    """Load provenance data if available."""
    provenance_path = research_dir / "provenance.json"
    if provenance_path.exists():
        try:
            with open(provenance_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}


def load_stats(data_dir: Path) -> dict:
    """Load statistics summary if available."""
    stats_path = data_dir / "stats" / "summary.json"
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}


def load_dataset_version(research_dir: Path) -> tuple[str, str]:
    """Load dataset version and checksum."""
    version_path = research_dir / "dataset_version.json"
    if version_path.exists():
        try:
            with open(version_path) as f:
                data = json.load(f)
            return (
                data.get("version", "1.0.0"),
                data.get("checksums", {}).get("dataset_overall", "")
            )
        except json.JSONDecodeError:
            pass
    return "1.0.0", ""


def generate_manifest(
    experiment_id: str,
    data_dir: Path,
    research_dir: Path,
    output_dir: Path,
    figures_dir: Optional[Path] = None,
    storage_uri: Optional[str] = None,
    use_template: bool = False,
) -> dict:
    """Generate complete manifest for experiment bundle."""
    print(f"Generating manifest for {experiment_id}")
    print(f"  Data directory: {data_dir}")
    print(f"  Research directory: {research_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect files
    files, base_path = collect_files(data_dir, research_dir, figures_dir)
    print(f"  Collected {len(files)} files")
    
    # Load metadata
    provenance = load_provenance(research_dir)
    stats_raw = load_stats(data_dir)
    dataset_version, dataset_checksum = load_dataset_version(research_dir)
    
    # Build stats summary
    stats = {
        "total_events": stats_raw.get("total_events", 0),
        "duration_sec": stats_raw.get("throughput", {}).get("total_duration_sec", 0),
        "latency_mean_us": stats_raw.get("latency", {}).get("mean", 0),
        "latency_p50_us": stats_raw.get("latency", {}).get("p50", 0),
        "latency_p99_us": stats_raw.get("latency", {}).get("p99", 0),
        "throughput_mean": stats_raw.get("throughput", {}).get("mean_msgs_per_sec", 0),
        "throughput_max": stats_raw.get("throughput", {}).get("max_msgs_per_sec", 0),
    }
    
    # Build runner config from provenance
    cluster_config = provenance.get("cluster_config", {})
    runner_config = {
        "replicas": cluster_config.get("worker_replicas", 0),
        "node_pool": cluster_config.get("node_pool", "default"),
        "machine_type": cluster_config.get("machine_type", "n2-standard-4"),
        "autoscaling": cluster_config.get("autoscaling", False),
    }
    
    # Collect figures and tables metadata
    figures = collect_figures_metadata(research_dir)
    tables = collect_tables_metadata(research_dir)
    
    # Build manifest
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment_id": experiment_id,
        "experiment_timestamp_utc": provenance.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "bundle_created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": provenance.get("git_commit", get_git_commit()),
        "orchestrator_commit": provenance.get("orchestrator_commit", get_git_commit()),
        "scenario": provenance.get("scenario_yaml", ""),
        "runner_config": runner_config,
        "dataset_version": dataset_version,
        "dataset_checksum": dataset_checksum,
        "provenance_file": "metadata/provenance.json",
        "stats_summary": stats,
        "files": files,
        "figures": figures,
        "tables": tables,
        "reports": {
            "latex": "report/report.tex" if (research_dir / "report.tex").exists() else "",
            "markdown": "report/report.md" if (research_dir / "report.md").exists() else "",
        },
        "storage_uri": storage_uri or provenance.get("storage_uri", ""),
        "reproduction": {
            "command": f"python research/scripts/pipeline_runner.py --exp-id {experiment_id} --generate-all",
            "requirements": "analysis/requirements.txt",
            "python_version": ">=3.10",
        },
    }
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Saved manifest to {manifest_path}")
    print(f"  Files: {len(files)}, Figures: {len(figures)}, Tables: {len(tables)}")
    
    return manifest


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate experiment manifest")
    parser.add_argument("experiment_id", help="Experiment identifier")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--research-dir", required=True, help="Path to research output directory")
    parser.add_argument("--out", help="Output directory (default: packaging/output/<exp-id>)")
    parser.add_argument("--figures-dir", help="Additional figures directory")
    parser.add_argument("--storage-uri", help="Storage URI for reproduction")
    
    args = parser.parse_args()
    
    output_dir = Path(args.out) if args.out else Path("packaging/output") / args.experiment_id
    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    
    manifest = generate_manifest(
        experiment_id=args.experiment_id,
        data_dir=Path(args.data_dir),
        research_dir=Path(args.research_dir),
        output_dir=output_dir,
        figures_dir=figures_dir,
        storage_uri=args.storage_uri,
    )
    
    print(f"\nManifest generated successfully!")
    print(f"  Total files: {len(manifest['files'])}")


if __name__ == "__main__":
    main()

