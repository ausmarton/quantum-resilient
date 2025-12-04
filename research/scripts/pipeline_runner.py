#!/usr/bin/env python3
"""
Complete research documentation pipeline.

Runs all steps for generating dissertation-ready artifacts:
1. Fetch results from storage
2. Merge JSONL files
3. Compute statistics
4. Compute effect sizes
5. Generate provenance metadata
6. Version dataset
7. Generate tables
8. Generate figures bundle
9. Generate final reports

Usage:
    python pipeline_runner.py --exp-id exp_001 --uri gs://qr-results/exp_001 --generate-all
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class PipelineRunner:
    """Orchestrates the complete research pipeline."""
    
    def __init__(
        self,
        experiment_id: str,
        storage_uri: Optional[str] = None,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        analysis_scripts_dir: Optional[Path] = None,
        research_scripts_dir: Optional[Path] = None,
    ):
        self.experiment_id = experiment_id
        self.storage_uri = storage_uri
        
        # Set up directories
        root = Path(__file__).parent.parent.parent
        
        self.data_dir = data_dir or (root / "analysis" / "data" / experiment_id)
        self.output_dir = output_dir or (root / "research" / "output" / experiment_id)
        self.figures_dir = root / "analysis" / "figures" / experiment_id
        
        self.analysis_scripts = analysis_scripts_dir or (root / "analysis" / "scripts")
        self.research_scripts = research_scripts_dir or (root / "research" / "scripts")
        self.packaging_dir = root / "packaging"
        self.packaging_output = root / "packaging" / "output" / experiment_id
        
        self.steps_completed = []
        self.errors = []
    
    def run_step(self, name: str, command: list[str]) -> bool:
        """Run a pipeline step."""
        print(f"\n{'='*60}")
        print(f"Step: {name}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,
            )
            self.steps_completed.append(name)
            print(f"✓ {name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.errors.append((name, str(e)))
            print(f"✗ {name} failed: {e}")
            return False
        except FileNotFoundError as e:
            self.errors.append((name, f"Command not found: {e}"))
            print(f"✗ {name} failed: Command not found")
            return False
    
    def step_fetch_results(self) -> bool:
        """Step 1: Fetch results from storage."""
        if not self.storage_uri:
            print("Skipping fetch (no URI provided, assuming data exists)")
            return True
        
        return self.run_step(
            "Fetch Results",
            [
                sys.executable,
                str(self.analysis_scripts / "fetch_results.py"),
                "--experiment-id", self.experiment_id,
                "--uri", self.storage_uri,
                "--out", str(self.data_dir),
                "--parallel", "8",
            ]
        )
    
    def step_merge_jsonl(self) -> bool:
        """Step 2: Merge JSONL files."""
        raw_dir = self.data_dir / "raw"
        if not raw_dir.exists():
            print(f"Warning: Raw directory not found: {raw_dir}")
            return False
        
        return self.run_step(
            "Merge JSONL",
            [
                sys.executable,
                str(self.analysis_scripts / "merge_jsonl.py"),
                "--input", str(raw_dir),
                "--output", str(self.data_dir / "merged"),
            ]
        )
    
    def step_compute_statistics(self) -> bool:
        """Step 3: Compute statistics."""
        merged_file = self.data_dir / "merged" / "merged.jsonl"
        if not merged_file.exists():
            merged_file = self.data_dir / "merged" / "merged.parquet"
        
        if not merged_file.exists():
            print(f"Warning: Merged file not found")
            return False
        
        return self.run_step(
            "Compute Statistics",
            [
                sys.executable,
                str(self.analysis_scripts / "compute_statistics.py"),
                "--input", str(merged_file),
                "--output", str(self.data_dir / "stats"),
                "--experiment-id", self.experiment_id,
            ]
        )
    
    def step_generate_plots(self) -> bool:
        """Step 3b: Generate plots."""
        merged_file = self.data_dir / "merged" / "merged.jsonl"
        if not merged_file.exists():
            merged_file = self.data_dir / "merged" / "merged.parquet"
        
        if not merged_file.exists():
            print("Warning: Merged file not found, skipping plots")
            return False
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Run each plot script
        plot_scripts = ["plot_latency.py", "plot_throughput.py", "plot_queue_delay.py"]
        
        for script in plot_scripts:
            script_path = self.analysis_scripts / script
            if script_path.exists():
                self.run_step(
                    f"Generate Plots ({script})",
                    [
                        sys.executable,
                        str(script_path),
                        "--input", str(merged_file),
                        "--output", str(self.figures_dir),
                        "--experiment-id", self.experiment_id,
                    ]
                )
        
        return True
    
    def step_generate_provenance(self) -> bool:
        """Step 5: Generate provenance metadata."""
        return self.run_step(
            "Generate Provenance",
            [
                sys.executable,
                str(self.research_scripts / "provenance.py"),
                "--exp-id", self.experiment_id,
                "--data-dir", str(self.data_dir),
                "--out", str(self.output_dir),
            ] + (["--storage-uri", self.storage_uri] if self.storage_uri else [])
        )
    
    def step_version_dataset(self, version: str = "1.0.0") -> bool:
        """Step 6: Version dataset."""
        return self.run_step(
            "Version Dataset",
            [
                sys.executable,
                str(self.research_scripts / "version_dataset.py"),
                "--exp-id", self.experiment_id,
                "--data-dir", str(self.data_dir),
                "--out", str(self.output_dir),
                "--version", version,
            ]
        )
    
    def step_generate_tables(self) -> bool:
        """Step 7: Generate tables."""
        stats_file = self.data_dir / "stats" / "summary.json"
        if not stats_file.exists():
            print(f"Warning: Stats file not found: {stats_file}")
            return False
        
        return self.run_step(
            "Generate Tables",
            [
                sys.executable,
                str(self.research_scripts / "generate_tables.py"),
                "--exp-id", self.experiment_id,
                "--stats-file", str(stats_file),
                "--out", str(self.output_dir / "tables"),
            ]
        )
    
    def step_generate_figures_bundle(self) -> bool:
        """Step 8: Generate figures bundle."""
        if not self.figures_dir.exists():
            print(f"Warning: Figures directory not found: {self.figures_dir}")
            return False
        
        return self.run_step(
            "Generate Figures Bundle",
            [
                sys.executable,
                str(self.research_scripts / "generate_figures_bundle.py"),
                "--exp-id", self.experiment_id,
                "--figures-dir", str(self.figures_dir),
                "--out", str(self.output_dir / "figures"),
            ]
        )
    
    def step_generate_reports(self) -> bool:
        """Step 9: Generate final reports."""
        success = True
        
        for fmt in ["tex", "md"]:
            result = self.run_step(
                f"Generate Report ({fmt.upper()})",
                [
                    sys.executable,
                    str(self.research_scripts / "generate_report.py"),
                    "--exp-id", self.experiment_id,
                    "--format", fmt,
                    "--out", str(self.output_dir),
                    "--data-dir", str(self.data_dir),
                ]
            )
            success = success and result
        
        return success
    
    def step_package_bundle(self) -> bool:
        """Step 10: Create packaging bundle."""
        return self.run_step(
            "Package Bundle",
            [
                sys.executable,
                "-m", "packaging",
                "all",
                self.experiment_id,
                "--data-dir", str(self.data_dir),
                "--research-dir", str(self.output_dir),
                "--figures-dir", str(self.figures_dir),
                "--out", str(self.packaging_output),
            ] + (["--uri", self.storage_uri] if self.storage_uri else [])
        )
    
    def step_publish(self, target: str, uri: str) -> bool:
        """Step 11: Publish bundle to storage."""
        bundle_path = self.packaging_output / f"{self.experiment_id}-research-bundle.zip"
        
        if not bundle_path.exists():
            print(f"Warning: Bundle not found: {bundle_path}")
            return False
        
        return self.run_step(
            f"Publish to {target}",
            [
                sys.executable,
                "-m", "packaging",
                "publish",
                self.experiment_id,
                "--bundle", str(bundle_path),
                "--target", target,
                "--uri", uri,
            ]
        )
    
    def run_all(self, version: str = "1.0.0", package: bool = False, publish_target: Optional[str] = None, publish_uri: Optional[str] = None) -> bool:
        """Run complete pipeline."""
        print(f"\n{'#'*60}")
        print(f"# Research Pipeline: {self.experiment_id}")
        print(f"# Started: {datetime.now(timezone.utc).isoformat()}")
        print(f"{'#'*60}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run steps in order
        steps = [
            ("fetch", self.step_fetch_results),
            ("merge", self.step_merge_jsonl),
            ("stats", self.step_compute_statistics),
            ("plots", self.step_generate_plots),
            ("provenance", self.step_generate_provenance),
            ("version", lambda: self.step_version_dataset(version)),
            ("tables", self.step_generate_tables),
            ("figures", self.step_generate_figures_bundle),
            ("reports", self.step_generate_reports),
        ]
        
        for name, step_func in steps:
            if not step_func():
                print(f"\nWarning: Step '{name}' failed, continuing...")
        
        # Optional: Package bundle
        if package:
            if not self.step_package_bundle():
                print("\nWarning: Packaging failed")
        
        # Optional: Publish
        if publish_target and publish_uri:
            if not self.step_publish(publish_target, publish_uri):
                print("\nWarning: Publishing failed")
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0
    
    def print_summary(self):
        """Print pipeline summary."""
        print(f"\n{'='*60}")
        print("Pipeline Summary")
        print(f"{'='*60}")
        print(f"Experiment: {self.experiment_id}")
        print(f"Output: {self.output_dir}")
        print(f"Completed steps: {len(self.steps_completed)}")
        print(f"Errors: {len(self.errors)}")
        
        if self.steps_completed:
            print(f"\n✓ Completed:")
            for step in self.steps_completed:
                print(f"    - {step}")
        
        if self.errors:
            print(f"\n✗ Errors:")
            for step, error in self.errors:
                print(f"    - {step}: {error}")
        
        print(f"\nGenerated artifacts:")
        for artifact in self.output_dir.rglob("*"):
            if artifact.is_file():
                rel_path = artifact.relative_to(self.output_dir)
                print(f"    - {rel_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete research documentation pipeline"
    )
    parser.add_argument("--exp-id", required=True, help="Experiment identifier")
    parser.add_argument("--uri", help="Storage URI (gs://, s3://, file://)")
    parser.add_argument("--data-dir", help="Override data directory")
    parser.add_argument("--out", help="Override output directory")
    parser.add_argument("--version", default="1.0.0", help="Dataset version")
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Run complete pipeline"
    )
    parser.add_argument(
        "--step",
        choices=[
            "fetch", "merge", "stats", "plots", "provenance",
            "version", "tables", "figures", "reports", "package"
        ],
        help="Run single step only"
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Also create packaging bundle after pipeline"
    )
    parser.add_argument(
        "--publish-target",
        choices=["gcs", "s3", "github"],
        help="Publish to storage after packaging"
    )
    parser.add_argument(
        "--publish-uri",
        help="URI for publishing (e.g., gs://bucket/path)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else None
    output_dir = Path(args.out) if args.out else None
    
    runner = PipelineRunner(
        experiment_id=args.exp_id,
        storage_uri=args.uri,
        data_dir=data_dir,
        output_dir=output_dir,
    )
    
    if args.generate_all:
        success = runner.run_all(
            version=args.version,
            package=args.package,
            publish_target=args.publish_target,
            publish_uri=args.publish_uri,
        )
        sys.exit(0 if success else 1)
    elif args.step:
        step_methods = {
            "fetch": runner.step_fetch_results,
            "merge": runner.step_merge_jsonl,
            "stats": runner.step_compute_statistics,
            "plots": runner.step_generate_plots,
            "provenance": runner.step_generate_provenance,
            "version": lambda: runner.step_version_dataset(args.version),
            "tables": runner.step_generate_tables,
            "figures": runner.step_generate_figures_bundle,
            "reports": runner.step_generate_reports,
            "package": runner.step_package_bundle,
        }
        success = step_methods[args.step]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        print("\nUse --generate-all to run complete pipeline")
        print("Use --step <name> to run a single step")
        print("Use --package to also create bundle")
        print("Use --publish-target and --publish-uri to publish")


if __name__ == "__main__":
    main()

