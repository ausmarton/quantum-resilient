#!/usr/bin/env python3
"""
Packaging CLI using Typer.

Provides user-friendly commands for all packaging operations.

Usage:
    python -m packaging bundle exp_001
    python -m packaging export exp_001
    python -m packaging publish exp_001 --target gcs --uri gs://bucket/path
    python -m packaging manifest exp_001
    python -m packaging notes exp_001
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from . import archiver, exporter, manifest, publish, release_notes

app = typer.Typer(
    name="packaging",
    help="Quantum-Resilient experiment packaging and distribution tools.",
    add_completion=False,
)

console = Console()


def get_paths(
    experiment_id: str,
    data_dir: Optional[Path],
    research_dir: Optional[Path],
    figures_dir: Optional[Path],
    output_dir: Optional[Path],
) -> tuple[Path, Path, Optional[Path], Path]:
    """Resolve default paths."""
    root = Path(__file__).parent.parent
    
    data = data_dir or (root / "analysis" / "data" / experiment_id)
    research = research_dir or (root / "research" / "output" / experiment_id)
    figures = figures_dir or (root / "analysis" / "figures" / experiment_id)
    output = output_dir or (root / "packaging" / "output" / experiment_id)
    
    return data, research, figures if figures.exists() else None, output


@app.command()
def bundle(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
    research_dir: Optional[Path] = typer.Option(None, "--research-dir", "-r", help="Research output directory"),
    figures_dir: Optional[Path] = typer.Option(None, "--figures-dir", "-f", help="Figures directory"),
    output_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    formats: str = typer.Option("zip,tar.gz", "--formats", help="Archive formats (comma-separated)"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify archives after creation"),
):
    """Create research bundle (ZIP and TAR.GZ archives)."""
    console.print(Panel.fit(f"[bold]Creating bundle for {experiment_id}[/bold]"))
    
    data, research, figures, output = get_paths(
        experiment_id, data_dir, research_dir, figures_dir, output_dir
    )
    
    # First generate manifest
    console.print("\n[bold]Step 1: Generating manifest[/bold]")
    manifest.generate_manifest(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        figures_dir=figures,
    )
    
    # Then create archives
    console.print("\n[bold]Step 2: Creating archives[/bold]")
    format_list = [f.strip() for f in formats.split(",")]
    archives = archiver.create_archives(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        figures_dir=figures,
        formats=format_list,
    )
    
    if verify:
        console.print("\n[bold]Step 3: Verifying archives[/bold]")
        for fmt, path in archives.items():
            archiver.verify_archive(path)
    
    console.print(f"\n[bold green]Bundle complete![/bold green]")
    console.print(f"Output: {output}")


@app.command()
def export(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
    research_dir: Optional[Path] = typer.Option(None, "--research-dir", "-r", help="Research output directory"),
    figures_dir: Optional[Path] = typer.Option(None, "--figures-dir", "-f", help="Figures directory"),
    output_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    lite: bool = typer.Option(False, "--lite", help="Create lite export (no JSONL)"),
    include_raw: bool = typer.Option(False, "--include-raw", help="Include raw worker files"),
):
    """Create publication-ready export folder."""
    console.print(Panel.fit(f"[bold]Creating export for {experiment_id}[/bold]"))
    
    data, research, figures, output = get_paths(
        experiment_id, data_dir, research_dir, figures_dir, output_dir
    )
    
    export_dir = exporter.create_export_structure(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        figures_dir=figures,
        lite=lite,
        include_raw=include_raw,
    )
    
    console.print(f"\n[bold green]Export complete![/bold green]")
    console.print(f"Output: {export_dir}")


@app.command("manifest")
def generate_manifest(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
    research_dir: Optional[Path] = typer.Option(None, "--research-dir", "-r", help="Research output directory"),
    figures_dir: Optional[Path] = typer.Option(None, "--figures-dir", "-f", help="Figures directory"),
    output_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    storage_uri: Optional[str] = typer.Option(None, "--uri", help="Storage URI"),
):
    """Generate experiment manifest."""
    console.print(Panel.fit(f"[bold]Generating manifest for {experiment_id}[/bold]"))
    
    data, research, figures, output = get_paths(
        experiment_id, data_dir, research_dir, figures_dir, output_dir
    )
    
    manifest_data = manifest.generate_manifest(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        figures_dir=figures,
        storage_uri=storage_uri,
    )
    
    console.print(f"\n[bold green]Manifest generated![/bold green]")
    console.print(f"Files: {len(manifest_data['files'])}")


@app.command("notes")
def generate_notes(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
    research_dir: Optional[Path] = typer.Option(None, "--research-dir", "-r", help="Research output directory"),
    output_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    description: Optional[str] = typer.Option(None, "--description", help="Experiment description"),
    storage_uri: Optional[str] = typer.Option(None, "--uri", help="Storage URI"),
):
    """Generate release notes."""
    console.print(Panel.fit(f"[bold]Generating release notes for {experiment_id}[/bold]"))
    
    data, research, _, output = get_paths(
        experiment_id, data_dir, research_dir, None, output_dir
    )
    
    notes_path = release_notes.generate_release_notes(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        description=description,
        storage_uri=storage_uri,
    )
    
    console.print(f"\n[bold green]Release notes generated![/bold green]")
    console.print(f"Output: {notes_path}")


@app.command("publish")
def publish_bundle(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    bundle: Optional[Path] = typer.Option(None, "--bundle", "-b", help="Path to bundle file"),
    target: str = typer.Option(..., "--target", "-t", help="Target: gcs, s3, or github"),
    uri: str = typer.Option(..., "--uri", help="Target URI"),
    output_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    public: bool = typer.Option(False, "--public", help="Make files publicly readable"),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip verification"),
):
    """Publish bundle to cloud storage or GitHub."""
    console.print(Panel.fit(f"[bold]Publishing {experiment_id} to {target}[/bold]"))
    
    root = Path(__file__).parent.parent
    output = output_dir or (root / "packaging" / "output" / experiment_id)
    
    if bundle is None:
        bundle = output / f"{experiment_id}-research-bundle.zip"
    
    manifest_path = output / "manifest.json"
    notes_path = output / "release_notes.md"
    
    result = publish.publish_bundle(
        experiment_id=experiment_id,
        bundle_path=bundle,
        target=target,
        uri=uri,
        manifest_path=manifest_path if manifest_path.exists() else None,
        release_notes_path=notes_path if notes_path.exists() else None,
        public=public,
        verify=not no_verify,
    )
    
    console.print(f"\n[bold green]Publication complete![/bold green]")
    console.print(f"Files uploaded: {len(result['files_uploaded'])}")


@app.command("all")
def do_all(
    experiment_id: str = typer.Argument(..., help="Experiment identifier"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory"),
    research_dir: Optional[Path] = typer.Option(None, "--research-dir", "-r", help="Research output directory"),
    figures_dir: Optional[Path] = typer.Option(None, "--figures-dir", "-f", help="Figures directory"),
    output_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
    storage_uri: Optional[str] = typer.Option(None, "--uri", help="Storage URI for reproduction"),
):
    """Run complete packaging pipeline (manifest, notes, export, bundle)."""
    console.print(Panel.fit(f"[bold]Complete packaging for {experiment_id}[/bold]"))
    
    data, research, figures, output = get_paths(
        experiment_id, data_dir, research_dir, figures_dir, output_dir
    )
    
    # Step 1: Manifest
    console.print("\n[bold cyan]Step 1/4: Generating manifest[/bold cyan]")
    manifest.generate_manifest(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        figures_dir=figures,
        storage_uri=storage_uri,
    )
    
    # Step 2: Release notes
    console.print("\n[bold cyan]Step 2/4: Generating release notes[/bold cyan]")
    release_notes.generate_release_notes(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        storage_uri=storage_uri,
    )
    
    # Step 3: Export
    console.print("\n[bold cyan]Step 3/4: Creating export[/bold cyan]")
    exporter.create_export_structure(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        figures_dir=figures,
    )
    
    # Step 4: Archives
    console.print("\n[bold cyan]Step 4/4: Creating archives[/bold cyan]")
    archives = archiver.create_archives(
        experiment_id=experiment_id,
        data_dir=data,
        research_dir=research,
        output_dir=output,
        figures_dir=figures,
    )
    
    console.print(f"\n[bold green]Complete packaging finished![/bold green]")
    console.print(f"Output directory: {output}")
    console.print(f"Archives created: {list(archives.keys())}")


def main():
    app()


if __name__ == "__main__":
    main()

