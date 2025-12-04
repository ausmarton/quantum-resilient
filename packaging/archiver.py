#!/usr/bin/env python3
"""
Create archival bundles (ZIP and TAR.GZ) for experiment results.

Creates secure, self-contained archives with:
- All data files
- Figures and tables
- Reports
- Manifest and provenance metadata

Usage:
    python -m packaging.archiver exp_001 --data-dir analysis/data/exp_001 --research-dir research/output/exp_001
"""

import json
import os
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def sanitize_path(path: str, base_name: str) -> str:
    """Sanitize archive path to prevent directory traversal attacks."""
    # Remove any leading slashes or parent directory references
    clean_path = path.lstrip("/").lstrip("\\")
    
    # Remove any .. components
    parts = []
    for part in clean_path.replace("\\", "/").split("/"):
        if part == "..":
            continue
        if part == ".":
            continue
        if part:
            parts.append(part)
    
    # Ensure path is under base_name
    return f"{base_name}/{'/'.join(parts)}"


def collect_bundle_files(
    experiment_id: str,
    data_dir: Path,
    research_dir: Path,
    figures_dir: Optional[Path] = None,
) -> list[tuple[Path, str]]:
    """Collect all files for the bundle with their archive paths."""
    files = []
    base_name = f"{experiment_id}-research-bundle"
    
    # Data files
    merged_dir = data_dir / "merged"
    if merged_dir.exists():
        for f in merged_dir.glob("*"):
            if f.is_file():
                archive_path = sanitize_path(f"data/{f.name}", base_name)
                files.append((f, archive_path))
    
    # Stats summary
    stats_file = data_dir / "stats" / "summary.json"
    if stats_file.exists():
        files.append((stats_file, sanitize_path("data/summary.json", base_name)))
    
    # Research outputs
    if research_dir.exists():
        # Metadata
        for meta_file in ["provenance.json", "dataset_version.json", "manifest.json"]:
            meta_path = research_dir / meta_file
            if meta_path.exists():
                files.append((meta_path, sanitize_path(f"metadata/{meta_file}", base_name)))
        
        # Tables
        tables_dir = research_dir / "tables"
        if tables_dir.exists():
            for f in tables_dir.glob("*"):
                if f.is_file():
                    files.append((f, sanitize_path(f"tables/{f.name}", base_name)))
        
        # Figures
        figs_dir = research_dir / "figures"
        if figs_dir.exists():
            for subdir in ["png", "pdf", "eps"]:
                subdir_path = figs_dir / subdir
                if subdir_path.exists():
                    for f in subdir_path.glob("*"):
                        if f.is_file():
                            files.append((f, sanitize_path(f"figures/{subdir}/{f.name}", base_name)))
            
            # Also check for files directly in figures dir
            for f in figs_dir.glob("*"):
                if f.is_file():
                    files.append((f, sanitize_path(f"figures/{f.name}", base_name)))
            
            # Figure manifest
            fig_manifest = figs_dir / "manifest.json"
            if fig_manifest.exists():
                files.append((fig_manifest, sanitize_path("figures/manifest.json", base_name)))
        
        # Reports
        for report in ["report.md", "report.tex"]:
            report_path = research_dir / report
            if report_path.exists():
                files.append((report_path, sanitize_path(f"report/{report}", base_name)))
    
    # External figures directory
    if figures_dir and figures_dir.exists():
        for f in figures_dir.rglob("*"):
            if f.is_file() and f.suffix in [".png", ".pdf", ".eps"]:
                rel_path = f.relative_to(figures_dir)
                files.append((f, sanitize_path(f"figures/{rel_path}", base_name)))
    
    return files


def create_zip_archive(
    files: list[tuple[Path, str]],
    output_path: Path,
    manifest: Optional[dict] = None,
) -> Path:
    """Create ZIP archive from file list."""
    console.print(f"[cyan]Creating ZIP archive: {output_path}[/cyan]")
    
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for source_path, archive_path in files:
            if source_path.exists():
                zf.write(source_path, archive_path)
        
        # Add manifest if provided and not already in files
        if manifest:
            manifest_in_files = any("manifest.json" in ap for _, ap in files)
            if not manifest_in_files:
                base_name = archive_path.split("/")[0]
                manifest_json = json.dumps(manifest, indent=2)
                zf.writestr(f"{base_name}/metadata/manifest.json", manifest_json)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"  [green]Created: {output_path} ({size_mb:.2f} MB)[/green]")
    
    return output_path


def create_tar_archive(
    files: list[tuple[Path, str]],
    output_path: Path,
    manifest: Optional[dict] = None,
) -> Path:
    """Create TAR.GZ archive from file list."""
    console.print(f"[cyan]Creating TAR.GZ archive: {output_path}[/cyan]")
    
    with tarfile.open(output_path, "w:gz") as tf:
        for source_path, archive_path in files:
            if source_path.exists():
                tf.add(source_path, arcname=archive_path)
        
        # Add manifest if provided and not already in files
        if manifest:
            manifest_in_files = any("manifest.json" in ap for _, ap in files)
            if not manifest_in_files:
                import io
                base_name = files[0][1].split("/")[0] if files else "bundle"
                manifest_json = json.dumps(manifest, indent=2).encode("utf-8")
                tarinfo = tarfile.TarInfo(name=f"{base_name}/metadata/manifest.json")
                tarinfo.size = len(manifest_json)
                tarinfo.mtime = int(datetime.now().timestamp())
                tf.addfile(tarinfo, io.BytesIO(manifest_json))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"  [green]Created: {output_path} ({size_mb:.2f} MB)[/green]")
    
    return output_path


def create_archives(
    experiment_id: str,
    data_dir: Path,
    research_dir: Path,
    output_dir: Path,
    figures_dir: Optional[Path] = None,
    manifest: Optional[dict] = None,
    formats: Optional[list[str]] = None,
) -> dict[str, Path]:
    """Create all archive formats."""
    console.print(f"[bold blue]Creating archives for {experiment_id}[/bold blue]")
    console.print(f"  Data directory: {data_dir}")
    console.print(f"  Research directory: {research_dir}")
    console.print(f"  Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if formats is None:
        formats = ["zip", "tar.gz"]
    
    # Collect files
    files = collect_bundle_files(experiment_id, data_dir, research_dir, figures_dir)
    console.print(f"  Collected {len(files)} files")
    
    # Load manifest from research dir if not provided
    if manifest is None:
        manifest_path = research_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
    
    archives = {}
    
    # Create ZIP
    if "zip" in formats:
        zip_path = output_dir / f"{experiment_id}-research-bundle.zip"
        create_zip_archive(files, zip_path, manifest)
        archives["zip"] = zip_path
    
    # Create TAR.GZ
    if "tar.gz" in formats:
        tar_path = output_dir / f"{experiment_id}-research-bundle.tar.gz"
        create_tar_archive(files, tar_path, manifest)
        archives["tar.gz"] = tar_path
    
    console.print(f"[bold green]Archives created successfully![/bold green]")
    
    return archives


def verify_archive(archive_path: Path, expected_files: Optional[list[str]] = None) -> bool:
    """Verify archive integrity and contents."""
    console.print(f"[cyan]Verifying archive: {archive_path}[/cyan]")
    
    if not archive_path.exists():
        console.print(f"  [red]Archive not found![/red]")
        return False
    
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                # Check for corruption
                bad_file = zf.testzip()
                if bad_file:
                    console.print(f"  [red]Corrupted file: {bad_file}[/red]")
                    return False
                
                file_list = zf.namelist()
        
        elif archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                file_list = tf.getnames()
        
        else:
            console.print(f"  [red]Unknown archive format[/red]")
            return False
        
        # Check for expected files
        if expected_files:
            missing = []
            for expected in expected_files:
                if not any(expected in f for f in file_list):
                    missing.append(expected)
            
            if missing:
                console.print(f"  [yellow]Missing files: {missing}[/yellow]")
        
        console.print(f"  [green]Archive verified: {len(file_list)} files[/green]")
        return True
    
    except Exception as e:
        console.print(f"  [red]Verification failed: {e}[/red]")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create experiment archives")
    parser.add_argument("experiment_id", help="Experiment identifier")
    parser.add_argument("--data-dir", required=True, help="Path to data directory")
    parser.add_argument("--research-dir", required=True, help="Path to research output directory")
    parser.add_argument("--out", help="Output directory (default: packaging/output/<exp-id>)")
    parser.add_argument("--figures-dir", help="Additional figures directory")
    parser.add_argument("--formats", nargs="+", default=["zip", "tar.gz"], help="Archive formats")
    parser.add_argument("--verify", action="store_true", help="Verify archives after creation")
    
    args = parser.parse_args()
    
    output_dir = Path(args.out) if args.out else Path("packaging/output") / args.experiment_id
    figures_dir = Path(args.figures_dir) if args.figures_dir else None
    
    archives = create_archives(
        experiment_id=args.experiment_id,
        data_dir=Path(args.data_dir),
        research_dir=Path(args.research_dir),
        output_dir=output_dir,
        figures_dir=figures_dir,
        formats=args.formats,
    )
    
    if args.verify:
        console.print("\n[bold]Verifying archives...[/bold]")
        for fmt, path in archives.items():
            verify_archive(path)


if __name__ == "__main__":
    main()

