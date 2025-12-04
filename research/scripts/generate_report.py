#!/usr/bin/env python3
"""
Generate experiment reports using Jinja2 templates.

Produces LaTeX and Markdown reports with:
- Experiment metadata
- Summary tables
- Figure references
- Effect size results

Usage:
    python generate_report.py --exp-id exp_001 --format tex --out research/output/exp_001/
    python generate_report.py --exp-id exp_001 --format md --out research/output/exp_001/
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader


def load_json_file(filepath: Path) -> Optional[dict]:
    """Load JSON file if it exists."""
    if filepath.exists():
        try:
            with open(filepath) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def load_tables(tables_dir: Path) -> dict[str, str]:
    """Load generated tables."""
    tables = {}
    
    if not tables_dir.exists():
        return tables
    
    for tex_file in tables_dir.glob("*.tex"):
        name = f"{tex_file.stem}_tex"
        tables[name] = tex_file.read_text()
    
    for md_file in tables_dir.glob("*.md"):
        name = f"{md_file.stem}_md"
        tables[name] = md_file.read_text()
    
    return tables


def load_figures(figures_dir: Path) -> list[dict]:
    """Load figure manifest."""
    manifest_path = figures_dir / "manifest.json"
    
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            return manifest.get("figures", [])
        except json.JSONDecodeError:
            pass
    
    # Fall back to scanning directory
    figures = []
    for png_file in figures_dir.glob("*.png"):
        figures.append({
            "name": png_file.stem,
            "path": str(png_file),
            "caption": png_file.stem.replace("_", " ").title(),
            "label": f"fig:{png_file.stem}",
            "title": png_file.stem.replace("_", " ").title(),
        })
    
    return figures


def generate_report(
    experiment_id: str,
    output_dir: Path,
    output_format: str = "tex",
    templates_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Path:
    """Generate report from template."""
    print(f"Generating {output_format.upper()} report for {experiment_id}")
    
    # Setup paths
    if templates_dir is None:
        templates_dir = Path(__file__).parent.parent / "templates"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    
    # Add custom filters
    env.filters["tojson"] = lambda x: json.dumps(x, indent=2, default=str)
    
    # Select template
    template_name = f"report.{output_format}.j2"
    try:
        template = env.get_template(template_name)
    except Exception as e:
        print(f"Error loading template {template_name}: {e}")
        raise
    
    # Load data
    provenance = {}
    stats = {}
    effect_sizes = []
    tables = {}
    figures = []
    dataset_version = "1.0.0"
    version_changelog = "Initial release"
    scenario_yaml = ""
    description = ""
    
    # Load provenance
    provenance_path = output_dir / "provenance.json"
    provenance = load_json_file(provenance_path) or {}
    
    # Load stats from data directory or output
    if data_dir:
        stats_path = data_dir / "stats" / "summary.json"
        stats = load_json_file(stats_path) or {}
    else:
        # Try output directory
        stats = load_json_file(output_dir / "stats" / "summary.json") or {}
    
    # Load effect sizes
    for es_path in output_dir.glob("**/effect_sizes*.json"):
        es_data = load_json_file(es_path)
        if es_data:
            if isinstance(es_data, list):
                effect_sizes.extend(es_data)
            else:
                effect_sizes.append(es_data)
    
    # Load tables
    tables_dir = output_dir / "tables"
    tables = load_tables(tables_dir)
    
    # Load figures
    figures_dir = output_dir / "figures"
    figures = load_figures(figures_dir)
    
    # Load dataset version
    version_path = output_dir / "dataset_version.json"
    version_data = load_json_file(version_path)
    if version_data:
        dataset_version = version_data.get("version", "1.0.0")
        version_changelog = version_data.get("changelog", "Initial release")
    
    # Get scenario YAML from provenance
    scenario_yaml = provenance.get("scenario_yaml", "# Scenario not available")
    
    # Build template context
    context = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "provenance": provenance,
        "stats": stats,
        "effect_sizes": effect_sizes,
        "tables": tables,
        "figures": figures,
        "dataset_version": dataset_version,
        "version_changelog": version_changelog,
        "scenario_yaml": scenario_yaml,
        "description": description or provenance.get("description", ""),
    }
    
    # Render template
    print("  Rendering template...")
    output = template.render(**context)
    
    # Save output
    output_filename = f"report.{output_format}"
    output_path = output_dir / output_filename
    with open(output_path, "w") as f:
        f.write(output)
    
    print(f"  Saved report to {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--exp-id", required=True, help="Experiment identifier")
    parser.add_argument(
        "--format",
        choices=["tex", "md"],
        default="tex",
        help="Output format (tex or md)"
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--templates-dir", help="Custom templates directory")
    parser.add_argument("--data-dir", help="Data directory (for stats)")
    
    args = parser.parse_args()
    
    templates_dir = Path(args.templates_dir) if args.templates_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else None
    
    output_path = generate_report(
        experiment_id=args.exp_id,
        output_dir=Path(args.out),
        output_format=args.format,
        templates_dir=templates_dir,
        data_dir=data_dir,
    )
    
    print(f"\nReport generated: {output_path}")


if __name__ == "__main__":
    main()

