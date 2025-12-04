#!/usr/bin/env python3
"""
Bundle figures for dissertation/publication.

Converts figures to multiple formats and creates:
- High-DPI PNG (300 DPI)
- PDF (vector)
- EPS (vector, for LaTeX)
- Manifest file with captions and labels
- ZIP tarball for dissertation appendix

Usage:
    python generate_figures_bundle.py --exp-id exp_001 --figures-dir analysis/figures/exp_001 --out research/output/exp_001/figures/
"""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Standard captions for common figure types
STANDARD_CAPTIONS = {
    "latency_cdf": "Cumulative distribution function (CDF) of cryptographic operation latency. Percentile markers indicate p50, p90, p95, and p99 values.",
    "latency_pdf": "Probability density function (PDF) of operation latency with kernel density estimation overlay.",
    "latency_tail": "Tail distribution (survival function) of latency on log scale, highlighting extreme values.",
    "latency_hist": "Histogram of cryptographic operation latency distribution.",
    "latency_algorithm_comparison": "Latency comparison across cryptographic algorithms showing CDF and box plot distributions.",
    "throughput_timeseries": "Throughput over time showing instantaneous operations per second with rolling average.",
    "throughput_distribution": "Distribution of per-second throughput measurements.",
    "throughput_curve": "Throughput curve showing system performance over the experiment duration.",
    "throughput_per_worker": "Per-worker throughput distribution and time series.",
    "queue_delay_distribution": "Queue delay distribution showing time spent waiting in the processing pipeline.",
    "queue_delay_vs_load": "Correlation between system load and queue delay.",
    "queue_delay_by_worker": "Queue delay distribution across different worker instances.",
    "queue_hist": "Histogram of queue delay distribution.",
}


def get_caption(filename: str) -> str:
    """Get standard caption for a figure based on filename."""
    stem = Path(filename).stem.lower()
    
    for key, caption in STANDARD_CAPTIONS.items():
        if key in stem:
            return caption
    
    # Generate default caption from filename
    return f"Figure: {stem.replace('_', ' ').title()}"


def get_label(filename: str, experiment_id: str) -> str:
    """Generate LaTeX label for a figure."""
    stem = Path(filename).stem.lower()
    exp_clean = experiment_id.replace("_", "-")
    return f"fig:{stem}-{exp_clean}"


def convert_to_pdf(input_path: Path, output_path: Path) -> bool:
    """Convert image to PDF using ImageMagick or PIL."""
    try:
        from PIL import Image
        
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(output_path, "PDF", resolution=300)
        return True
    except Exception as e:
        print(f"    Warning: Could not convert to PDF: {e}")
        return False


def convert_to_eps(input_path: Path, output_path: Path) -> bool:
    """Convert image to EPS using ImageMagick or PIL."""
    try:
        # Try using ImageMagick if available
        result = subprocess.run(
            ["convert", str(input_path), str(output_path)],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        from PIL import Image
        
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(output_path, "EPS")
        return True
    except Exception as e:
        print(f"    Warning: Could not convert to EPS: {e}")
        return False


def ensure_high_dpi_png(input_path: Path, output_path: Path, target_dpi: int = 300) -> bool:
    """Ensure PNG is at target DPI, resave if needed."""
    try:
        from PIL import Image
        
        with Image.open(input_path) as img:
            # Copy to output with explicit DPI
            img.save(output_path, "PNG", dpi=(target_dpi, target_dpi))
        return True
    except Exception as e:
        # Just copy if PIL not available
        shutil.copy2(input_path, output_path)
        return True


def generate_figures_bundle(
    experiment_id: str,
    figures_dir: Path,
    output_dir: Path,
    create_tarball: bool = True,
) -> dict:
    """Generate figure bundle with multiple formats."""
    print(f"Generating figures bundle for {experiment_id}")
    print(f"  Source: {figures_dir}")
    print(f"  Output: {output_dir}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "png").mkdir(exist_ok=True)
    (output_dir / "pdf").mkdir(exist_ok=True)
    (output_dir / "eps").mkdir(exist_ok=True)
    
    # Find all figures
    figure_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        figure_files.extend(figures_dir.glob(ext))
    
    # Also check for PDFs (copy as-is)
    figure_files.extend(figures_dir.glob("*.pdf"))
    
    if not figure_files:
        print("  No figures found!")
        return {"figures": [], "manifest_path": None}
    
    print(f"  Found {len(figure_files)} figures")
    
    # Process each figure
    manifest = {
        "experiment_id": experiment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figures": [],
    }
    
    for fig_path in sorted(figure_files):
        print(f"  Processing {fig_path.name}...")
        
        fig_info = {
            "name": fig_path.stem,
            "original": fig_path.name,
            "caption": get_caption(fig_path.name),
            "label": get_label(fig_path.name, experiment_id),
            "title": fig_path.stem.replace("_", " ").title(),
            "formats": [],
        }
        
        # High-DPI PNG
        png_output = output_dir / "png" / f"{fig_path.stem}.png"
        if fig_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            if ensure_high_dpi_png(fig_path, png_output):
                fig_info["formats"].append("png")
                fig_info["png_path"] = f"png/{fig_path.stem}.png"
        
        # PDF
        pdf_output = output_dir / "pdf" / f"{fig_path.stem}.pdf"
        if fig_path.suffix.lower() == ".pdf":
            shutil.copy2(fig_path, pdf_output)
            fig_info["formats"].append("pdf")
            fig_info["pdf_path"] = f"pdf/{fig_path.stem}.pdf"
        elif fig_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            if convert_to_pdf(fig_path, pdf_output):
                fig_info["formats"].append("pdf")
                fig_info["pdf_path"] = f"pdf/{fig_path.stem}.pdf"
        
        # EPS
        eps_output = output_dir / "eps" / f"{fig_path.stem}.eps"
        if fig_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            if convert_to_eps(fig_path, eps_output):
                fig_info["formats"].append("eps")
                fig_info["eps_path"] = f"eps/{fig_path.stem}.eps"
        
        # Default path for LaTeX includes
        fig_info["path"] = fig_info.get("pdf_path", fig_info.get("png_path", ""))
        
        manifest["figures"].append(fig_info)
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest to {manifest_path}")
    
    # Generate LaTeX include file
    latex_includes = []
    for fig in manifest["figures"]:
        latex_includes.append(f"""% {fig['title']}
\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.8\\textwidth]{{{fig['path']}}}
\\caption{{{fig['caption']}}}
\\label{{{fig['label']}}}
\\end{{figure}}
""")
    
    latex_path = output_dir / "figures_includes.tex"
    with open(latex_path, "w") as f:
        f.write("% Auto-generated LaTeX figure includes\n")
        f.write(f"% Experiment: {experiment_id}\n")
        f.write(f"% Generated: {manifest['generated_at']}\n\n")
        f.write("\n".join(latex_includes))
    print(f"  Saved LaTeX includes to {latex_path}")
    
    # Create tarball
    tarball_path = None
    if create_tarball:
        tarball_path = output_dir / f"figures_bundle_{experiment_id}.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(output_dir / "png", arcname="png")
            tar.add(output_dir / "pdf", arcname="pdf")
            tar.add(output_dir / "eps", arcname="eps")
            tar.add(manifest_path, arcname="manifest.json")
            tar.add(latex_path, arcname="figures_includes.tex")
        print(f"  Created tarball: {tarball_path}")
    
    return {
        "figures": manifest["figures"],
        "manifest_path": str(manifest_path),
        "tarball_path": str(tarball_path) if tarball_path else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate figure bundle for publication")
    parser.add_argument("--exp-id", required=True, help="Experiment identifier")
    parser.add_argument("--figures-dir", required=True, help="Source figures directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--no-tarball", action="store_true", help="Skip tarball creation")
    
    args = parser.parse_args()
    
    result = generate_figures_bundle(
        experiment_id=args.exp_id,
        figures_dir=Path(args.figures_dir),
        output_dir=Path(args.out),
        create_tarball=not args.no_tarball,
    )
    
    print(f"\nFigures bundle complete!")
    print(f"  Processed {len(result['figures'])} figures")
    if result.get("tarball_path"):
        print(f"  Tarball: {result['tarball_path']}")


if __name__ == "__main__":
    main()

