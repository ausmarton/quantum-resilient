#!/usr/bin/env python3
"""
Export publication-ready dataset packages.

Creates clean folder structures suitable for:
- Dissertation appendices
- GitHub Releases
- Data repositories

Usage:
    python -m packaging.exporter exp_001 --data-dir analysis/data/exp_001 --research-dir research/output/exp_001
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def create_export_structure(
    experiment_id: str,
    data_dir: Path,
    research_dir: Path,
    output_dir: Path,
    figures_dir: Optional[Path] = None,
    lite: bool = False,
    include_raw: bool = False,
) -> Path:
    """Create clean export folder structure."""
    console.print(f"[bold blue]Creating export for {experiment_id}[/bold blue]")
    console.print(f"  Mode: {'Lite' if lite else 'Full'}")
    
    # Create export directory
    export_dir = output_dir / "export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    
    export_dir.mkdir(parents=True)
    
    # Create subdirectories
    (export_dir / "data").mkdir()
    (export_dir / "figures").mkdir()
    (export_dir / "tables").mkdir()
    (export_dir / "report").mkdir()
    (export_dir / "metadata").mkdir()
    
    files_copied = 0
    
    # Data files
    console.print("[cyan]Copying data files...[/cyan]")
    merged_dir = data_dir / "merged"
    if merged_dir.exists():
        # Always include parquet (efficient)
        parquet = merged_dir / "merged.parquet"
        if parquet.exists():
            shutil.copy2(parquet, export_dir / "data" / "merged.parquet")
            files_copied += 1
        
        # Include JSONL only if not lite mode
        if not lite:
            jsonl = merged_dir / "merged.jsonl"
            if jsonl.exists():
                shutil.copy2(jsonl, export_dir / "data" / "merged.jsonl")
                files_copied += 1
    
    # Summary statistics
    stats_file = data_dir / "stats" / "summary.json"
    if stats_file.exists():
        shutil.copy2(stats_file, export_dir / "data" / "summary.json")
        files_copied += 1
    
    # Raw files (only if explicitly requested and not lite)
    if include_raw and not lite:
        raw_dir = data_dir / "raw"
        if raw_dir.exists():
            raw_export = export_dir / "data" / "raw"
            raw_export.mkdir()
            for f in raw_dir.glob("*.jsonl"):
                shutil.copy2(f, raw_export / f.name)
                files_copied += 1
    
    # Figures
    console.print("[cyan]Copying figures...[/cyan]")
    src_figs = research_dir / "figures"
    if src_figs.exists():
        # Copy PNG files
        png_dir = src_figs / "png"
        if png_dir.exists():
            for f in png_dir.glob("*.png"):
                shutil.copy2(f, export_dir / "figures" / f.name)
                files_copied += 1
        
        # Copy PDF files (for LaTeX)
        pdf_dir = src_figs / "pdf"
        if pdf_dir.exists():
            for f in pdf_dir.glob("*.pdf"):
                shutil.copy2(f, export_dir / "figures" / f.name)
                files_copied += 1
        
        # Check for files directly in figures dir
        for f in src_figs.glob("*.png"):
            shutil.copy2(f, export_dir / "figures" / f.name)
            files_copied += 1
        for f in src_figs.glob("*.pdf"):
            shutil.copy2(f, export_dir / "figures" / f.name)
            files_copied += 1
    
    # Also check external figures directory
    if figures_dir and figures_dir.exists():
        for f in figures_dir.glob("*.png"):
            dst = export_dir / "figures" / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                files_copied += 1
        for f in figures_dir.glob("*.pdf"):
            dst = export_dir / "figures" / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                files_copied += 1
    
    # Tables
    console.print("[cyan]Copying tables...[/cyan]")
    tables_dir = research_dir / "tables"
    if tables_dir.exists():
        for f in tables_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, export_dir / "tables" / f.name)
                files_copied += 1
    
    # Reports
    console.print("[cyan]Copying reports...[/cyan]")
    for report in ["report.md", "report.tex"]:
        src = research_dir / report
        if src.exists():
            shutil.copy2(src, export_dir / "report" / report)
            files_copied += 1
    
    # Metadata
    console.print("[cyan]Copying metadata...[/cyan]")
    for meta in ["manifest.json", "provenance.json", "dataset_version.json"]:
        # Check research dir first, then output dir
        src = research_dir / meta
        if not src.exists():
            src = output_dir / meta
        if src.exists():
            shutil.copy2(src, export_dir / "metadata" / meta)
            files_copied += 1
    
    # Release notes if available
    release_notes = output_dir / "release_notes.md"
    if release_notes.exists():
        shutil.copy2(release_notes, export_dir / "release_notes.md")
        files_copied += 1
    
    # Reproducibility content
    console.print("[cyan]Copying reproducibility content...[/cyan]")
    repro_src = research_dir / "reproducibility"
    if repro_src.exists():
        repro_export = export_dir / "reproducibility"
        repro_export.mkdir(exist_ok=True)
        
        for f in repro_src.glob("*"):
            if f.is_file():
                shutil.copy2(f, repro_export / f.name)
                files_copied += 1
    
    # Also check reproducibility output directory
    root = research_dir.parent.parent
    repro_analysis = root / "reproducibility" / "output" / experiment_id / "analysis"
    if repro_analysis.exists():
        repro_export = export_dir / "reproducibility"
        repro_export.mkdir(exist_ok=True)
        
        for f in repro_analysis.glob("*"):
            if f.is_file():
                dst = repro_export / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    files_copied += 1
    
    # Create README for the export
    create_export_readme(export_dir, experiment_id, lite)
    files_copied += 1
    
    console.print(f"[bold green]Export created: {export_dir}[/bold green]")
    console.print(f"  Total files: {files_copied}")
    
    return export_dir


def create_export_readme(export_dir: Path, experiment_id: str, lite: bool) -> None:
    """Create README for the export."""
    readme_content = f"""# {experiment_id} - Research Data Export

This directory contains the research data export for experiment `{experiment_id}`.

## Contents

```
{experiment_id}-export/
├── data/
│   ├── merged.parquet      # Merged experiment data (efficient format)
{'│   ├── merged.jsonl        # Merged experiment data (JSON lines)' if not lite else ''}
│   └── summary.json        # Statistical summary
├── figures/
│   └── *.png, *.pdf        # Publication-ready figures
├── tables/
│   ├── *.tex               # LaTeX tables
│   └── *.md                # Markdown tables
├── report/
│   ├── report.tex          # LaTeX report
│   └── report.md           # Markdown report
├── metadata/
│   ├── manifest.json       # Complete bundle manifest
│   ├── provenance.json     # Experiment provenance
│   └── dataset_version.json
├── reproducibility/         # Reproducibility analysis (if available)
│   ├── variance_summary.json
│   ├── confidence_intervals.json
│   ├── stability_summary.json
│   ├── stability_matrix.png
│   └── reproducibility_report.md
└── README.md               # This file
```

## Usage

### Loading Data (Python)

```python
import pandas as pd

# Load from Parquet (recommended)
df = pd.read_parquet("data/merged.parquet")

# Load statistics
import json
with open("data/summary.json") as f:
    stats = json.load(f)
```

### Including in LaTeX

```latex
% Include a table
\\input{{tables/latency_quantiles.tex}}

% Include a figure
\\includegraphics[width=0.8\\textwidth]{{figures/latency_cdf.pdf}}
```

## Provenance

See `metadata/provenance.json` for complete experiment provenance including:
- Git commit hash
- Cluster configuration
- File checksums
- Worker synchronization data

## License

[License terms to be specified]

---
*Generated by Quantum-Resilient Packaging Tools*
"""
    
    with open(export_dir / "README.md", "w") as f:
        f.write(readme_content)


def create_lite_export(
    experiment_id: str,
    data_dir: Path,
    research_dir: Path,
    output_dir: Path,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Create lite export (no raw JSONL, smaller footprint)."""
    return create_export_structure(
        experiment_id=experiment_id,
        data_dir=data_dir,
        research_dir=research_dir,
        output_dir=output_dir,
        figures_dir=figures_dir,
        lite=True,
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Export publication-ready dataset")
    parser.add_argument("experiment_id", help="Experiment identifier")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--research-dir", required=True, help="Path to research output directory")
    parser.add_argument("--out", help="Output directory (default: packaging/output/<exp-id>)")
    parser.add_argument("--figures-dir", help="Additional figures directory")
    parser.add_argument("--lite", action="store_true", help="Create lite export (no JSONL)")
    parser.add_argument("--include-raw", action="store_true", help="Include raw worker files")
    
    args = parser.parse_args()
    
    output_dir = Path(args.out) if args.out else Path("packaging/output") / args.experiment_id
    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    
    export_dir = create_export_structure(
        experiment_id=args.experiment_id,
        data_dir=Path(args.data_dir),
        research_dir=Path(args.research_dir),
        output_dir=output_dir,
        figures_dir=figures_dir,
        lite=args.lite,
        include_raw=args.include_raw,
    )
    
    console.print(f"\n[bold]Export complete: {export_dir}[/bold]")


if __name__ == "__main__":
    main()

