#!/usr/bin/env python3
"""
Publish experiment bundles to cloud storage or GitHub Releases.

Supported targets:
- GCS (Google Cloud Storage)
- S3 (AWS/MinIO)
- GitHub Releases

Usage:
    python -m packaging.publish exp_001 --bundle bundle.zip --target gcs --uri gs://bucket/path
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class GCSPublisher:
    """Publish to Google Cloud Storage."""
    
    def __init__(self, bucket: str, prefix: str = ""):
        from google.cloud import storage
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix.strip("/")
    
    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        public: bool = False,
    ) -> str:
        """Upload a file to GCS."""
        if self.prefix:
            blob_name = f"{self.prefix}/{remote_path}"
        else:
            blob_name = remote_path
        
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        
        if public:
            blob.make_public()
        
        return f"gs://{self.bucket.name}/{blob_name}"
    
    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download a file from GCS."""
        if self.prefix:
            blob_name = f"{self.prefix}/{remote_path}"
        else:
            blob_name = remote_path
        
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))
        return True
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in GCS."""
        if self.prefix:
            blob_name = f"{self.prefix}/{remote_path}"
        else:
            blob_name = remote_path
        
        blob = self.bucket.blob(blob_name)
        return blob.exists()


class S3Publisher:
    """Publish to S3 or S3-compatible storage (MinIO)."""
    
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: Optional[str] = None):
        import boto3
        
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
    
    def upload_file(
        self,
        local_path: Path,
        remote_path: str,
        public: bool = False,
    ) -> str:
        """Upload a file to S3."""
        if self.prefix:
            key = f"{self.prefix}/{remote_path}"
        else:
            key = remote_path
        
        extra_args = {}
        if public:
            extra_args["ACL"] = "public-read"
        
        self.s3.upload_file(str(local_path), self.bucket, key, ExtraArgs=extra_args or None)
        
        return f"s3://{self.bucket}/{key}"
    
    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download a file from S3."""
        if self.prefix:
            key = f"{self.prefix}/{remote_path}"
        else:
            key = remote_path
        
        self.s3.download_file(self.bucket, key, str(local_path))
        return True
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in S3."""
        if self.prefix:
            key = f"{self.prefix}/{remote_path}"
        else:
            key = remote_path
        
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


class GitHubPublisher:
    """Publish to GitHub Releases."""
    
    def __init__(self, repo: str, token: Optional[str] = None):
        import requests
        
        self.repo = repo
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN environment variable.")
        
        self.api_base = f"https://api.github.com/repos/{repo}"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.requests = requests
    
    def create_release(
        self,
        tag: str,
        name: str,
        body: str,
        draft: bool = False,
        prerelease: bool = False,
    ) -> dict:
        """Create a GitHub release."""
        url = f"{self.api_base}/releases"
        data = {
            "tag_name": tag,
            "name": name,
            "body": body,
            "draft": draft,
            "prerelease": prerelease,
        }
        
        response = self.requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def upload_asset(self, release_id: int, filepath: Path) -> str:
        """Upload an asset to a release."""
        upload_url = f"https://uploads.github.com/repos/{self.repo}/releases/{release_id}/assets"
        
        with open(filepath, "rb") as f:
            headers = {
                **self.headers,
                "Content-Type": "application/octet-stream",
            }
            params = {"name": filepath.name}
            
            response = self.requests.post(
                upload_url,
                headers=headers,
                params=params,
                data=f,
            )
            response.raise_for_status()
        
        return response.json().get("browser_download_url", "")
    
    def get_release_by_tag(self, tag: str) -> Optional[dict]:
        """Get release by tag name."""
        url = f"{self.api_base}/releases/tags/{tag}"
        response = self.requests.get(url, headers=self.headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def parse_uri(uri: str) -> tuple[str, str, str]:
    """Parse storage URI into (scheme, bucket, prefix)."""
    from urllib.parse import urlparse
    
    parsed = urlparse(uri)
    scheme = parsed.scheme
    bucket = parsed.netloc
    prefix = parsed.path.strip("/")
    
    return scheme, bucket, prefix


def publish_bundle(
    experiment_id: str,
    bundle_path: Path,
    target: str,
    uri: str,
    manifest_path: Optional[Path] = None,
    public: bool = False,
    verify: bool = True,
    release_notes_path: Optional[Path] = None,
) -> dict:
    """Publish bundle to specified target."""
    console.print(f"[bold blue]Publishing {experiment_id}[/bold blue]")
    console.print(f"  Bundle: {bundle_path}")
    console.print(f"  Target: {target}")
    console.print(f"  URI: {uri}")
    
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    
    # Compute bundle checksum
    console.print("[cyan]Computing bundle checksum...[/cyan]")
    bundle_checksum = compute_sha256(bundle_path)
    console.print(f"  SHA-256: {bundle_checksum[:16]}...")
    
    result = {
        "experiment_id": experiment_id,
        "bundle": str(bundle_path),
        "bundle_checksum": bundle_checksum,
        "target": target,
        "uri": uri,
        "files_uploaded": [],
    }
    
    if target == "gcs":
        scheme, bucket, prefix = parse_uri(uri)
        publisher = GCSPublisher(bucket, f"{prefix}/{experiment_id}" if prefix else experiment_id)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Upload bundle
            task = progress.add_task("Uploading bundle...", total=None)
            bundle_uri = publisher.upload_file(bundle_path, bundle_path.name, public=public)
            result["files_uploaded"].append(bundle_uri)
            progress.update(task, description=f"Uploaded: {bundle_uri}")
            
            # Upload manifest if provided
            if manifest_path and manifest_path.exists():
                task = progress.add_task("Uploading manifest...", total=None)
                manifest_uri = publisher.upload_file(manifest_path, "manifest.json", public=public)
                result["files_uploaded"].append(manifest_uri)
                progress.update(task, description=f"Uploaded: {manifest_uri}")
            
            # Upload release notes if provided
            if release_notes_path and release_notes_path.exists():
                task = progress.add_task("Uploading release notes...", total=None)
                notes_uri = publisher.upload_file(release_notes_path, "release_notes.md", public=public)
                result["files_uploaded"].append(notes_uri)
                progress.update(task, description=f"Uploaded: {notes_uri}")
            
            # Upload checksum file
            task = progress.add_task("Uploading checksum...", total=None)
            checksum_content = f"{bundle_checksum}  {bundle_path.name}\n"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sha256", delete=False) as f:
                f.write(checksum_content)
                checksum_file = Path(f.name)
            checksum_uri = publisher.upload_file(checksum_file, f"{bundle_path.name}.sha256", public=public)
            checksum_file.unlink()
            result["files_uploaded"].append(checksum_uri)
            progress.update(task, description=f"Uploaded: {checksum_uri}")
        
        # Verify upload
        if verify:
            console.print("[cyan]Verifying upload...[/cyan]")
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = Path(f.name)
            
            try:
                publisher.download_file(bundle_path.name, temp_path)
                downloaded_checksum = compute_sha256(temp_path)
                
                if downloaded_checksum == bundle_checksum:
                    console.print("  [green]✓ Checksum verified[/green]")
                    result["verified"] = True
                else:
                    console.print("  [red]✗ Checksum mismatch![/red]")
                    result["verified"] = False
            finally:
                temp_path.unlink(missing_ok=True)
    
    elif target == "s3":
        scheme, bucket, prefix = parse_uri(uri)
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        publisher = S3Publisher(bucket, f"{prefix}/{experiment_id}" if prefix else experiment_id, endpoint)
        
        # Similar upload logic as GCS
        console.print("[cyan]Uploading to S3...[/cyan]")
        bundle_uri = publisher.upload_file(bundle_path, bundle_path.name, public=public)
        result["files_uploaded"].append(bundle_uri)
        console.print(f"  Uploaded: {bundle_uri}")
        
        if manifest_path and manifest_path.exists():
            manifest_uri = publisher.upload_file(manifest_path, "manifest.json", public=public)
            result["files_uploaded"].append(manifest_uri)
        
        result["verified"] = True  # TODO: Implement S3 verification
    
    elif target == "github":
        # Parse repo from URI
        # Format: github://owner/repo or just owner/repo
        repo = uri.replace("github://", "").strip("/")
        
        console.print(f"[cyan]Publishing to GitHub: {repo}[/cyan]")
        publisher = GitHubPublisher(repo)
        
        tag = f"exp-{experiment_id}"
        release_name = f"Experiment: {experiment_id}"
        
        # Load release notes
        body = f"Research data release for experiment {experiment_id}"
        if release_notes_path and release_notes_path.exists():
            body = release_notes_path.read_text()
        
        # Create or get release
        existing = publisher.get_release_by_tag(tag)
        if existing:
            release_id = existing["id"]
            console.print(f"  Using existing release: {tag}")
        else:
            release = publisher.create_release(tag, release_name, body)
            release_id = release["id"]
            console.print(f"  Created release: {tag}")
        
        # Upload bundle
        console.print("[cyan]Uploading assets...[/cyan]")
        asset_url = publisher.upload_asset(release_id, bundle_path)
        result["files_uploaded"].append(asset_url)
        console.print(f"  Uploaded: {bundle_path.name}")
        
        result["release_url"] = f"https://github.com/{repo}/releases/tag/{tag}"
        result["verified"] = True
    
    else:
        raise ValueError(f"Unknown target: {target}")
    
    console.print(f"[bold green]Publication complete![/bold green]")
    console.print(f"  Files uploaded: {len(result['files_uploaded'])}")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Publish experiment bundles")
    parser.add_argument("experiment_id", help="Experiment identifier")
    parser.add_argument("--bundle", required=True, help="Path to bundle file")
    parser.add_argument("--target", required=True, choices=["gcs", "s3", "github"], help="Publication target")
    parser.add_argument("--uri", required=True, help="Target URI (gs://bucket, s3://bucket, github://owner/repo)")
    parser.add_argument("--manifest", help="Path to manifest.json")
    parser.add_argument("--release-notes", help="Path to release_notes.md")
    parser.add_argument("--public", action="store_true", help="Make files publicly readable")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    
    args = parser.parse_args()
    
    result = publish_bundle(
        experiment_id=args.experiment_id,
        bundle_path=Path(args.bundle),
        target=args.target,
        uri=args.uri,
        manifest_path=Path(args.manifest) if args.manifest else None,
        release_notes_path=Path(args.release_notes) if args.release_notes else None,
        public=args.public,
        verify=not args.no_verify,
    )
    
    console.print(f"\n[bold]Result:[/bold]")
    console.print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

