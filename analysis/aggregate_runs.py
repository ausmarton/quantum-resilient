#!/usr/bin/env python3
"""
Aggregate statistics across multiple experiment runs.

Computes:
- Mean, stddev, 95% CI for p50/p95/p99 latency and throughput
- Run-to-run coefficient of variation (CV)
- Stability metrics and flags

Usage:
    python analysis/aggregate_runs.py \
        --input results/native/scenario_id \
        --runs 5

Output:
    results/native/scenario_id/aggregated_stats.json
    results/native/scenario_id/stability_report.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


# Stability thresholds
CV_THRESHOLD_GOOD = 0.05      # < 5% CV is excellent
CV_THRESHOLD_ACCEPTABLE = 0.10  # < 10% CV is acceptable
CV_THRESHOLD_WARN = 0.15      # < 15% CV is warning
# > 15% CV is unstable


@dataclass
class RunStats:
    """Statistics from a single run."""
    run_index: int
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    mean_latency: float = 0.0
    std_latency: float = 0.0
    mean_throughput: float = 0.0
    max_throughput: float = 0.0
    total_events: int = 0
    duration_sec: float = 0.0


@dataclass
class AggregatedRunStats:
    """Aggregated statistics across multiple runs."""
    n_runs: int
    
    # p50 latency
    p50_mean: float = 0.0
    p50_std: float = 0.0
    p50_ci_low: float = 0.0
    p50_ci_high: float = 0.0
    p50_cv: float = 0.0
    
    # p95 latency
    p95_mean: float = 0.0
    p95_std: float = 0.0
    p95_ci_low: float = 0.0
    p95_ci_high: float = 0.0
    p95_cv: float = 0.0
    
    # p99 latency
    p99_mean: float = 0.0
    p99_std: float = 0.0
    p99_ci_low: float = 0.0
    p99_ci_high: float = 0.0
    p99_cv: float = 0.0
    
    # Mean latency
    mean_latency_mean: float = 0.0
    mean_latency_std: float = 0.0
    mean_latency_ci_low: float = 0.0
    mean_latency_ci_high: float = 0.0
    mean_latency_cv: float = 0.0
    
    # Throughput
    throughput_mean: float = 0.0
    throughput_std: float = 0.0
    throughput_ci_low: float = 0.0
    throughput_ci_high: float = 0.0
    throughput_cv: float = 0.0
    
    # Event counts
    total_events_mean: float = 0.0
    total_events_std: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'n_runs': self.n_runs,
            'latency': {
                'p50': {
                    'mean': round(self.p50_mean, 2),
                    'std': round(self.p50_std, 2),
                    'ci_95_low': round(self.p50_ci_low, 2),
                    'ci_95_high': round(self.p50_ci_high, 2),
                    'cv': round(self.p50_cv, 4),
                },
                'p95': {
                    'mean': round(self.p95_mean, 2),
                    'std': round(self.p95_std, 2),
                    'ci_95_low': round(self.p95_ci_low, 2),
                    'ci_95_high': round(self.p95_ci_high, 2),
                    'cv': round(self.p95_cv, 4),
                },
                'p99': {
                    'mean': round(self.p99_mean, 2),
                    'std': round(self.p99_std, 2),
                    'ci_95_low': round(self.p99_ci_low, 2),
                    'ci_95_high': round(self.p99_ci_high, 2),
                    'cv': round(self.p99_cv, 4),
                },
                'mean': {
                    'mean': round(self.mean_latency_mean, 2),
                    'std': round(self.mean_latency_std, 2),
                    'ci_95_low': round(self.mean_latency_ci_low, 2),
                    'ci_95_high': round(self.mean_latency_ci_high, 2),
                    'cv': round(self.mean_latency_cv, 4),
                },
            },
            'throughput': {
                'mean': round(self.throughput_mean, 2),
                'std': round(self.throughput_std, 2),
                'ci_95_low': round(self.throughput_ci_low, 2),
                'ci_95_high': round(self.throughput_ci_high, 2),
                'cv': round(self.throughput_cv, 4),
            },
            'events': {
                'mean': round(self.total_events_mean, 0),
                'std': round(self.total_events_std, 2),
            },
        }


@dataclass
class StabilityReport:
    """Report on experiment stability across runs."""
    n_runs: int
    overall_stable: bool = True
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'n_runs': self.n_runs,
            'overall_stable': self.overall_stable,
            'metrics': self.metrics,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
        }


def load_summary(path: Path) -> Optional[dict]:
    """Load summary.json from a run directory."""
    summary_paths = [
        path / 'stats' / 'summary.json',
        path / 'summary.json',
    ]
    
    for sp in summary_paths:
        if sp.exists():
            try:
                with open(sp) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    
    return None


def extract_run_stats(summary: dict, run_index: int) -> RunStats:
    """Extract statistics from a summary dict."""
    run = RunStats(run_index=run_index)
    
    if 'latency' in summary:
        lat = summary['latency']
        run.p50 = lat.get('p50', 0)
        run.p90 = lat.get('p90', 0)
        run.p95 = lat.get('p95', 0)
        run.p99 = lat.get('p99', 0)
        run.p999 = lat.get('p999', 0)
        run.mean_latency = lat.get('mean', 0)
        run.std_latency = lat.get('std', 0)
    
    if 'throughput' in summary:
        tput = summary['throughput']
        run.mean_throughput = tput.get('mean_msgs_per_sec', tput.get('mean', 0))
        run.max_throughput = tput.get('max_msgs_per_sec', tput.get('max', 0))
    
    run.total_events = summary.get('total_events', summary.get('count', 0))
    run.duration_sec = summary.get('duration_sec', 0)
    
    return run


def compute_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval using t-distribution."""
    if len(values) < 2:
        val = values[0] if values else 0
        return (val, val)
    
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)
    
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def compute_cv(values: list[float]) -> float:
    """Compute coefficient of variation."""
    if not values:
        return 0.0
    
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    
    std = np.std(values, ddof=1) if len(values) > 1 else 0
    return std / mean


def aggregate_runs(run_stats: list[RunStats]) -> AggregatedRunStats:
    """Aggregate statistics across runs."""
    if not run_stats:
        return AggregatedRunStats(n_runs=0)
    
    agg = AggregatedRunStats(n_runs=len(run_stats))
    
    # Extract values
    p50_vals = [r.p50 for r in run_stats if r.p50 > 0]
    p95_vals = [r.p95 for r in run_stats if r.p95 > 0]
    p99_vals = [r.p99 for r in run_stats if r.p99 > 0]
    mean_lat_vals = [r.mean_latency for r in run_stats if r.mean_latency > 0]
    throughput_vals = [r.mean_throughput for r in run_stats if r.mean_throughput > 0]
    event_vals = [r.total_events for r in run_stats if r.total_events > 0]
    
    # p50
    if p50_vals:
        agg.p50_mean = np.mean(p50_vals)
        agg.p50_std = np.std(p50_vals, ddof=1) if len(p50_vals) > 1 else 0
        agg.p50_ci_low, agg.p50_ci_high = compute_ci(p50_vals)
        agg.p50_cv = compute_cv(p50_vals)
    
    # p95
    if p95_vals:
        agg.p95_mean = np.mean(p95_vals)
        agg.p95_std = np.std(p95_vals, ddof=1) if len(p95_vals) > 1 else 0
        agg.p95_ci_low, agg.p95_ci_high = compute_ci(p95_vals)
        agg.p95_cv = compute_cv(p95_vals)
    
    # p99
    if p99_vals:
        agg.p99_mean = np.mean(p99_vals)
        agg.p99_std = np.std(p99_vals, ddof=1) if len(p99_vals) > 1 else 0
        agg.p99_ci_low, agg.p99_ci_high = compute_ci(p99_vals)
        agg.p99_cv = compute_cv(p99_vals)
    
    # Mean latency
    if mean_lat_vals:
        agg.mean_latency_mean = np.mean(mean_lat_vals)
        agg.mean_latency_std = np.std(mean_lat_vals, ddof=1) if len(mean_lat_vals) > 1 else 0
        agg.mean_latency_ci_low, agg.mean_latency_ci_high = compute_ci(mean_lat_vals)
        agg.mean_latency_cv = compute_cv(mean_lat_vals)
    
    # Throughput
    if throughput_vals:
        agg.throughput_mean = np.mean(throughput_vals)
        agg.throughput_std = np.std(throughput_vals, ddof=1) if len(throughput_vals) > 1 else 0
        agg.throughput_ci_low, agg.throughput_ci_high = compute_ci(throughput_vals)
        agg.throughput_cv = compute_cv(throughput_vals)
    
    # Events
    if event_vals:
        agg.total_events_mean = np.mean(event_vals)
        agg.total_events_std = np.std(event_vals, ddof=1) if len(event_vals) > 1 else 0
    
    return agg


def evaluate_stability(agg: AggregatedRunStats, run_stats: list[RunStats]) -> StabilityReport:
    """Evaluate experiment stability and generate report."""
    report = StabilityReport(n_runs=agg.n_runs)
    
    # Evaluate each metric
    metrics = {
        'p50_latency': {'cv': agg.p50_cv, 'mean': agg.p50_mean, 'std': agg.p50_std},
        'p95_latency': {'cv': agg.p95_cv, 'mean': agg.p95_mean, 'std': agg.p95_std},
        'p99_latency': {'cv': agg.p99_cv, 'mean': agg.p99_mean, 'std': agg.p99_std},
        'mean_latency': {'cv': agg.mean_latency_cv, 'mean': agg.mean_latency_mean, 'std': agg.mean_latency_std},
        'throughput': {'cv': agg.throughput_cv, 'mean': agg.throughput_mean, 'std': agg.throughput_std},
    }
    
    for metric_name, metric_data in metrics.items():
        cv = metric_data['cv']
        
        if cv < CV_THRESHOLD_GOOD:
            status = 'excellent'
            stable = True
        elif cv < CV_THRESHOLD_ACCEPTABLE:
            status = 'good'
            stable = True
        elif cv < CV_THRESHOLD_WARN:
            status = 'acceptable'
            stable = True
            report.warnings.append(f"{metric_name}: CV={cv:.1%} is borderline (threshold: {CV_THRESHOLD_ACCEPTABLE:.0%})")
        else:
            status = 'unstable'
            stable = False
            report.warnings.append(f"{metric_name}: CV={cv:.1%} exceeds threshold ({CV_THRESHOLD_WARN:.0%})")
            report.overall_stable = False
        
        report.metrics[metric_name] = {
            'cv': round(cv, 4),
            'cv_percent': f"{cv:.2%}",
            'status': status,
            'stable': stable,
            'mean': round(metric_data['mean'], 2),
            'std': round(metric_data['std'], 2),
        }
    
    # Check for outlier runs
    if run_stats:
        p95_vals = [r.p95 for r in run_stats if r.p95 > 0]
        if len(p95_vals) >= 3:
            q1, q3 = np.percentile(p95_vals, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [i for i, v in enumerate(p95_vals) if v < lower_bound or v > upper_bound]
            if outliers:
                report.warnings.append(f"Potential outlier runs detected: {outliers}")
    
    # Generate recommendations
    if not report.overall_stable:
        report.recommendations.append("Increase number of runs to improve statistical confidence")
        report.recommendations.append("Check for system interference (other processes, thermal throttling)")
        report.recommendations.append("Consider longer warm-up period before measurements")
    
    if agg.n_runs < 5:
        report.recommendations.append("Consider running at least 5 repeats for reliable statistics")
    
    if any(m['cv'] > CV_THRESHOLD_ACCEPTABLE for m in report.metrics.values()):
        report.recommendations.append("High variability detected - results may not be reproducible")
    
    return report


def discover_runs(base_dir: Path, expected_runs: Optional[int] = None) -> list[Path]:
    """Discover run directories."""
    run_dirs = []
    
    # Pattern 1: run-N directories
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith('run-'):
            run_dirs.append(d)
    
    # Pattern 2: run_N directories
    if not run_dirs:
        for d in sorted(base_dir.iterdir()):
            if d.is_dir() and d.name.startswith('run_'):
                run_dirs.append(d)
    
    # Pattern 3: numbered directories
    if not run_dirs:
        for d in sorted(base_dir.iterdir()):
            if d.is_dir() and d.name.isdigit():
                run_dirs.append(d)
    
    # If expected_runs specified, also check for indexed runs
    if expected_runs and not run_dirs:
        for i in range(1, expected_runs + 1):
            for pattern in [f'run-{i}', f'run_{i}', str(i)]:
                candidate = base_dir / pattern
                if candidate.exists():
                    run_dirs.append(candidate)
                    break
    
    return sorted(run_dirs, key=lambda p: p.name)


def main():
    parser = argparse.ArgumentParser(description="Aggregate statistics across multiple runs")
    parser.add_argument('--input', '-i', type=Path, required=True, 
                        help='Base directory containing run-N subdirectories')
    parser.add_argument('--runs', '-n', type=int, default=None,
                        help='Expected number of runs (optional)')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output directory (default: input directory)')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = args.output or args.input
    
    # Discover run directories
    run_dirs = discover_runs(args.input, args.runs)
    
    if not run_dirs:
        # Check if this is a single-run directory
        summary = load_summary(args.input)
        if summary:
            print(f"Single run detected in {args.input}")
            run_stats = [extract_run_stats(summary, 1)]
        else:
            print(f"Error: No run directories found in {args.input}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Found {len(run_dirs)} run directories")
        
        # Load statistics from each run
        run_stats: list[RunStats] = []
        for i, run_dir in enumerate(run_dirs, 1):
            summary = load_summary(run_dir)
            if summary:
                stats = extract_run_stats(summary, i)
                run_stats.append(stats)
                print(f"  Run {i}: p95={stats.p95:.0f}μs, throughput={stats.mean_throughput:.0f} ops/s")
            else:
                print(f"  Run {i}: No summary.json found")
    
    if not run_stats:
        print("Error: No valid run data found", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nLoaded {len(run_stats)} runs")
    
    # Aggregate statistics
    agg = aggregate_runs(run_stats)
    
    # Evaluate stability
    stability = evaluate_stability(agg, run_stats)
    
    # Print summary
    print(f"\n{'='*60}")
    print("AGGREGATED STATISTICS")
    print(f"{'='*60}")
    print(f"Runs: {agg.n_runs}")
    print(f"\nLatency (μs):")
    print(f"  p50:  {agg.p50_mean:.0f} ± {agg.p50_std:.0f} (CV: {agg.p50_cv:.1%})")
    print(f"  p95:  {agg.p95_mean:.0f} ± {agg.p95_std:.0f} (CV: {agg.p95_cv:.1%})")
    print(f"  p99:  {agg.p99_mean:.0f} ± {agg.p99_std:.0f} (CV: {agg.p99_cv:.1%})")
    print(f"  mean: {agg.mean_latency_mean:.0f} ± {agg.mean_latency_std:.0f} (CV: {agg.mean_latency_cv:.1%})")
    print(f"\nThroughput (ops/s):")
    print(f"  mean: {agg.throughput_mean:.0f} ± {agg.throughput_std:.0f} (CV: {agg.throughput_cv:.1%})")
    print(f"\n95% Confidence Intervals:")
    print(f"  p95 latency: [{agg.p95_ci_low:.0f}, {agg.p95_ci_high:.0f}] μs")
    print(f"  throughput:  [{agg.throughput_ci_low:.0f}, {agg.throughput_ci_high:.0f}] ops/s")
    
    print(f"\n{'='*60}")
    print("STABILITY REPORT")
    print(f"{'='*60}")
    print(f"Overall stable: {'✓ Yes' if stability.overall_stable else '✗ No'}")
    
    for metric_name, metric_data in stability.metrics.items():
        status_icon = '✓' if metric_data['stable'] else '✗'
        print(f"  {status_icon} {metric_name}: {metric_data['status']} (CV: {metric_data['cv_percent']})")
    
    if stability.warnings:
        print(f"\nWarnings:")
        for w in stability.warnings:
            print(f"  ⚠ {w}")
    
    if stability.recommendations:
        print(f"\nRecommendations:")
        for r in stability.recommendations:
            print(f"  → {r}")
    
    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregated stats
    agg_path = output_dir / 'aggregated_stats.json'
    agg_dict = agg.to_dict()
    agg_dict['per_run'] = [
        {
            'run': r.run_index,
            'p50': r.p50,
            'p95': r.p95,
            'p99': r.p99,
            'mean_latency': r.mean_latency,
            'throughput': r.mean_throughput,
            'events': r.total_events,
        }
        for r in run_stats
    ]
    
    with open(agg_path, 'w') as f:
        json.dump(agg_dict, f, indent=2)
    print(f"\nWritten: {agg_path}")
    
    # Stability report
    stability_path = output_dir / 'stability_report.json'
    with open(stability_path, 'w') as f:
        json.dump(stability.to_dict(), f, indent=2)
    print(f"Written: {stability_path}")
    
    # Exit code based on stability
    if not stability.overall_stable:
        print("\n⚠ Experiment shows high variability - results may not be reliable")
        sys.exit(0)  # Still exit 0, but warn
    else:
        print("\n✓ Experiment stable across runs")
        sys.exit(0)


if __name__ == "__main__":
    main()

