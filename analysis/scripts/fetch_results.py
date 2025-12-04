#!/usr/bin/env python3
"""
Fetch experiment results from local filesystem, S3/MinIO, or GCS.

Usage:
    python fetch_results.py --experiment-id exp_001 --uri gs://bucket/exp_001 --out ./data/exp_001/
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import fastjsonschema
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

console = Console()

# JSONL event schema for validation
EVENT_SCHEMA = {
    "type": "object",
    "required": ["run_id", "event_id", "timestamp_utc_iso", "operation", "algorithm", "latency_us"],
    "properties": {
        "run_id": {"type": "string"},
        "event_id": {"type": "integer"},
        "timestamp_utc_iso": {"type": "string"},
        "timestamp_monotonic_ns": {"type": "integer"},
        "operation": {"type": "string"},
        "algorithm": {"type": "string"},
        "latency_us": {"type": "integer"},
        "queue_delay_us": {"type": "integer"},
        "worker_id": {"type": "integer"},
        "payload_size_bytes": {"type": "integer"},
        "ciphertext_size_bytes": {"type": ["integer", "null"]},
        "signature_size_bytes": {"type": ["integer", "null"]},
        "cpu_user_seconds": {"type": "number"},
        "memory_rss_bytes": {"type": "integer"},
        "error": {"type": ["string", "null"]},
    },
}

validate_event = fastjsonschema.compile(EVENT_SCHEMA)


@dataclass
class FetchMetadata:
    """Metadata about fetched results."""

    experiment_id: str
    uri: str
    files_found: int = 0
    files_downloaded: int = 0
    files_failed: int = 0
    total_events: int = 0
    validation_errors: int = 0
    workers: list = field(default_factory=list)


class StorageBackend:
    """Abstract storage backend interface."""

    def list_files(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        raise NotImplementedError


class LocalBackend(StorageBackend):
    """Local filesystem backend."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def list_files(self, prefix: str = "") -> list[str]:
        search_path = self.base_path / prefix if prefix else self.base_path
        if not search_path.exists():
            return []
        files = []
        for f in search_path.rglob("*.jsonl"):
            files.append(str(f.relative_to(self.base_path)))
        return files

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        src = self.base_path / remote_path
        if not src.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(src, local_path)
        return True


class GCSBackend(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(self, bucket_name: str):
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def list_files(self, prefix: str = "") -> list[str]:
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs if blob.name.endswith(".jsonl")]

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(str(local_path))
            return True
        except Exception as e:
            console.print(f"[red]Error downloading {remote_path}: {e}[/red]")
            return False


class S3Backend(StorageBackend):
    """S3/MinIO backend."""

    def __init__(self, bucket_name: str, endpoint_url: Optional[str] = None):
        import boto3

        self.s3 = boto3.client("s3", endpoint_url=endpoint_url)
        self.bucket_name = bucket_name

    def list_files(self, prefix: str = "") -> list[str]:
        files = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl"):
                    files.append(obj["Key"])
        return files

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket_name, remote_path, str(local_path))
            return True
        except Exception as e:
            console.print(f"[red]Error downloading {remote_path}: {e}[/red]")
            return False


def parse_uri(uri: str) -> tuple[StorageBackend, str]:
    """Parse URI and return appropriate backend and prefix."""
    parsed = urlparse(uri)

    if parsed.scheme == "gs":
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return GCSBackend(bucket), prefix
    elif parsed.scheme == "s3":
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        return S3Backend(bucket, endpoint), prefix
    elif parsed.scheme == "file" or not parsed.scheme:
        path = parsed.path if parsed.scheme == "file" else uri
        return LocalBackend(path), ""
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")


def validate_jsonl_file(filepath: Path) -> tuple[int, int]:
    """Validate JSONL file and return (valid_count, error_count)."""
    valid = 0
    errors = 0
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                validate_event(event)
                valid += 1
            except (json.JSONDecodeError, fastjsonschema.JsonSchemaException) as e:
                errors += 1
                if errors <= 5:  # Only show first 5 errors
                    console.print(f"[yellow]Validation error in {filepath}:{line_num}: {e}[/yellow]")
    return valid, errors


def fetch_results(
    experiment_id: str,
    uri: str,
    output_dir: Path,
    parallel: int = 4,
    validate: bool = True,
) -> FetchMetadata:
    """Fetch experiment results from storage."""
    console.print(f"[bold blue]Fetching results for experiment: {experiment_id}[/bold blue]")
    console.print(f"  URI: {uri}")
    console.print(f"  Output: {output_dir}")

    metadata = FetchMetadata(experiment_id=experiment_id, uri=uri)

    # Parse URI and get backend
    try:
        backend, prefix = parse_uri(uri)
    except Exception as e:
        console.print(f"[red]Failed to parse URI: {e}[/red]")
        return metadata

    # List files
    console.print("[cyan]Listing files...[/cyan]")
    files = backend.list_files(prefix)
    metadata.files_found = len(files)
    console.print(f"  Found {len(files)} JSONL files")

    if not files:
        console.print("[yellow]No files found![/yellow]")
        return metadata

    # Create output directory
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download files in parallel
    console.print(f"[cyan]Downloading files (parallel={parallel})...[/cyan]")

    def download_one(remote_path: str) -> tuple[str, bool]:
        filename = Path(remote_path).name
        local_path = raw_dir / filename
        success = backend.download_file(remote_path, local_path)
        return remote_path, success

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(download_one, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Downloading"):
            remote_path, success = future.result()
            if success:
                metadata.files_downloaded += 1
            else:
                metadata.files_failed += 1

    console.print(
        f"  Downloaded: {metadata.files_downloaded}, Failed: {metadata.files_failed}"
    )

    # Validate files if requested
    if validate:
        console.print("[cyan]Validating JSONL files...[/cyan]")
        for jsonl_file in tqdm(list(raw_dir.glob("*.jsonl")), desc="Validating"):
            valid, errors = validate_jsonl_file(jsonl_file)
            metadata.total_events += valid
            metadata.validation_errors += errors

            # Extract worker ID from filename if present
            if "worker_" in jsonl_file.name:
                try:
                    worker_id = int(jsonl_file.stem.split("_")[1])
                    metadata.workers.append(worker_id)
                except (IndexError, ValueError):
                    pass

        console.print(
            f"  Total events: {metadata.total_events}, Validation errors: {metadata.validation_errors}"
        )

    # Save metadata
    metadata_path = output_dir / "fetch_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "experiment_id": metadata.experiment_id,
                "uri": metadata.uri,
                "files_found": metadata.files_found,
                "files_downloaded": metadata.files_downloaded,
                "files_failed": metadata.files_failed,
                "total_events": metadata.total_events,
                "validation_errors": metadata.validation_errors,
                "workers": sorted(metadata.workers),
            },
            f,
            indent=2,
        )
    console.print(f"[green]Metadata saved to {metadata_path}[/green]")

    # Check for missing workers
    if metadata.workers:
        expected = set(range(max(metadata.workers) + 1))
        actual = set(metadata.workers)
        missing = expected - actual
        if missing:
            console.print(f"[yellow]Warning: Missing worker results: {sorted(missing)}[/yellow]")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Fetch experiment results from storage")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier")
    parser.add_argument(
        "--uri",
        required=True,
        help="Storage URI (gs://bucket/path, s3://bucket/path, or file:///path)",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel downloads")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation")

    args = parser.parse_args()

    metadata = fetch_results(
        experiment_id=args.experiment_id,
        uri=args.uri,
        output_dir=Path(args.out),
        parallel=args.parallel,
        validate=not args.no_validate,
    )

    if metadata.files_failed > 0:
        sys.exit(1)

    console.print("[bold green]Fetch complete![/bold green]")


if __name__ == "__main__":
    main()
