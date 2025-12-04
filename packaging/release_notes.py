#!/usr/bin/env python3
"""
Generate release notes from experiment data.

Creates human-readable release notes with:
- Experiment summary
- Key results
- Environment details
- Reproduction instructions

Usage:
    python -m packaging.release_notes exp_001 --data-dir analysis/data/exp_001 --research-dir research/output/exp_001
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader

from rich.console import Console

console = Console()


def filesizeformat(value: int) -> str:
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def load_json_file(filepath: Path) -> Optional[dict]:
    """Load JSON file if it exists."""
    if filepath.exists():
        try:
            with open(filepath) as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return None


def generate_release_notes(
    experiment_id: str,
    data_dir: Path,
    research_dir: Path,
    output_dir: Path,
    templates_dir: Optional[Path] = None,
    description: Optional[str] = None,
    storage_uri: Optional[str] = None,
) -> Path:
    """Generate release notes from template."""
    console.print(f"[bold blue]Generating release notes for {experiment_id}[/bold blue]")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Jinja2
    if templates_dir is None:
        templates_dir = Path(__file__).parent / "templates"
    
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["filesizeformat"] = filesizeformat
    env.filters["tojson"] = lambda x: json.dumps(x, indent=2, default=str)
    
    template = env.get_template("release_notes.md.j2")
    
    # Load data
    provenance = load_json_file(research_dir / "provenance.json") or {}
    stats_raw = load_json_file(data_dir / "stats" / "summary.json") or {}
    manifest = load_json_file(research_dir / "manifest.json") or load_json_file(output_dir / "manifest.json") or {}
    version_data = load_json_file(research_dir / "dataset_version.json") or {}
    
    # Build stats
    stats = {
        "total_events": stats_raw.get("total_events", 0),
        "duration_sec": stats_raw.get("throughput", {}).get("total_duration_sec", 0),
        "latency_mean_us": stats_raw.get("latency", {}).get("mean", 0),
        "latency_p50_us": stats_raw.get("latency", {}).get("p50", 0),
        "latency_p99_us": stats_raw.get("latency", {}).get("p99", 0),
        "throughput_mean": stats_raw.get("throughput", {}).get("mean_msgs_per_sec", 0),
        "throughput_max": stats_raw.get("throughput", {}).get("max_msgs_per_sec", 0),
    }
    
    # Build runner config
    cluster_config = provenance.get("cluster_config", {})
    runner_config = {
        "replicas": cluster_config.get("worker_replicas", 0),
        "node_pool": cluster_config.get("node_pool", "default"),
        "machine_type": cluster_config.get("machine_type", "n2-standard-4"),
        "autoscaling": cluster_config.get("autoscaling", False),
    }
    
    # Get algorithms from stats
    algorithms = list(stats_raw.get("per_algorithm", {}).keys()) if stats_raw.get("per_algorithm") else []
    
    # Load effect sizes if available
    effect_sizes = []
    for es_path in output_dir.glob("**/effect_sizes*.json"):
        es_data = load_json_file(es_path)
        if es_data:
            if isinstance(es_data, list):
                effect_sizes.extend(es_data)
            elif "effect_sizes" in es_data:
                d = es_data["effect_sizes"]
                effect_sizes.append({
                    "comparison": f"{es_data.get('experiment_a', 'A')} vs {es_data.get('experiment_b', 'B')}",
                    "cohens_d": d.get("cohens_d", 0),
                    "interpretation": es_data.get("interpretation", {}).get("cohens_d", "unknown"),
                })
    
    # Get files list from manifest or generate
    files = manifest.get("files", [])
    if not files:
        # Generate from directories
        for f in data_dir.rglob("*"):
            if f.is_file():
                files.append({
                    "path": str(f.relative_to(data_dir.parent)),
                    "sha256": "checksum-not-computed",
                    "size_bytes": f.stat().st_size,
                })
    
    # Build context
    context: dict[str, Any] = {
        "experiment_id": experiment_id,
        "bundle_created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "git_commit": provenance.get("git_commit", "unknown"),
        "dataset_version": version_data.get("version", "1.0.0"),
        "dataset_checksum": version_data.get("checksums", {}).get("dataset_overall", ""),
        "description": description or provenance.get("description", ""),
        "stats": stats,
        "runner_config": runner_config,
        "algorithms": algorithms,
        "effect_sizes": effect_sizes,
        "files": files,
        "storage_uri": storage_uri or provenance.get("storage_uri", ""),
        "packaging_version": "1.0.0",
    }
    
    # Render template
    console.print("[cyan]Rendering release notes...[/cyan]")
    output = template.render(**context)
    
    # Save
    output_path = output_dir / "release_notes.md"
    with open(output_path, "w") as f:
        f.write(output)
    
    console.print(f"[green]Saved release notes to {output_path}[/green]")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate release notes")
    parser.add_argument("experiment_id", help="Experiment identifier")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--research-dir", required=True, help="Path to research output directory")
    parser.add_argument("--out", help="Output directory (default: packaging/output/<exp-id>)")
    parser.add_argument("--templates-dir", help="Custom templates directory")
    parser.add_argument("--description", help="Experiment description")
    parser.add_argument("--storage-uri", help="Storage URI for reproduction")
    
    args = parser.parse_args()
    
    output_dir = Path(args.out) if args.out else Path("packaging/output") / args.experiment_id
    templates_dir = Path(args.templates_dir) if args.templates_dir else None
    
    output_path = generate_release_notes(
        experiment_id=args.experiment_id,
        data_dir=Path(args.data_dir),
        research_dir=Path(args.research_dir),
        output_dir=output_dir,
        templates_dir=templates_dir,
        description=args.description,
        storage_uri=args.storage_uri,
    )
    
    console.print(f"\n[bold]Release notes generated: {output_path}[/bold]")


if __name__ == "__main__":
    main()

