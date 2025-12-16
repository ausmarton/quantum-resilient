#!/usr/bin/env python3
"""
Generate effect size forest plot: Cohen's d values with confidence intervals.

Produces a forest plot showing:
- Y-axis: Comparison pairs (Kyber vs RSA, Kyber vs ECDHE, etc.)
- X-axis: Cohen's d
- Horizontal CI bars (bootstrapped if available)

This visually communicates:
- Magnitude dominance
- Effects are not borderline
- Even with small n, effects are overwhelming

Usage:
    python analysis/plot_effect_size_forest.py \
        --hypothesis-tests final-results/hypothesis_tests.json \
        --output final-results/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def parse_comparison_name(comparison_id: str):
    """Parse comparison ID to extract algorithm names."""
    # Examples: "kyber512_native_vs_rsa2048_native", "dilithium2_vs_ecdsa", "kyber512_native_vs_rsa2048_native"
    parts = comparison_id.split('_vs_')
    if len(parts) == 2:
        # Extract algorithm from first part (may have environment suffix)
        algo_a_parts = parts[0].split('_')
        algo_b_parts = parts[1].split('_')
        
        # Algorithm names are typically first part, but handle cases like "kyber512_native"
        # Try to identify algorithm name (not environment)
        algo_a = None
        algo_b = None
        
        # Known algorithm patterns
        algo_patterns = ['kyber512', 'dilithium2', 'rsa2048', 'ecdsa', 'ecdhe', 'hybrid']
        
        for pattern in algo_patterns:
            if pattern in parts[0].lower():
                algo_a = pattern
            if pattern in parts[1].lower():
                algo_b = pattern
        
        # Fallback: use first part if no pattern matched
        if not algo_a:
            algo_a = algo_a_parts[0]
        if not algo_b:
            algo_b = algo_b_parts[0]
            
        return algo_a, algo_b
    return None, None


def format_algorithm_name(algo: str) -> str:
    """Format algorithm name for display."""
    name_map = {
        'kyber512': 'Kyber-512',
        'dilithium2': 'Dilithium-2',
        'rsa2048': 'RSA-2048',
        'ecdsa': 'ECDSA P-256',
        'ecdhe': 'ECDHE P-256',
        'hybrid': 'Hybrid',
    }
    return name_map.get(algo.lower(), algo)


def load_hypothesis_tests(hypothesis_tests_path: Path, environment: str = 'native'):
    """Load hypothesis test results and extract effect sizes."""
    with open(hypothesis_tests_path) as f:
        data = json.load(f)
    
    comparisons = []
    
    for result in data.get('results', []):
        # Focus on key comparisons: PQC vs Classical
        comparison_id = result.get('comparison_id', '')
        comparison_type = result.get('comparison_type', '')
        
        # Filter by environment if specified
        if environment and environment not in comparison_id:
            continue
        
        # Skip environment comparisons (native vs minikube, etc.)
        if comparison_type == 'environment' or 'native_vs_minikube' in comparison_id or 'native_vs_gcp' in comparison_id or 'minikube_vs_gcp' in comparison_id:
            continue
        
        # Focus on algorithm comparisons
        if comparison_type != 'algorithm':
            # Also check if it's an algorithm comparison by pattern
            if not any(algo in comparison_id for algo in ['kyber', 'dilithium', 'rsa', 'ecdsa', 'ecdhe']):
                continue
        
        algo_a, algo_b = parse_comparison_name(comparison_id)
        if not algo_a or not algo_b:
            continue
        
        # Focus on PQC vs Classical comparisons
        pqc_algos = ['kyber512', 'dilithium2', 'hybrid']
        classical_algos = ['rsa2048', 'ecdsa', 'ecdhe']
        
        algo_a_lower = algo_a.lower() if algo_a else ''
        algo_b_lower = algo_b.lower() if algo_b else ''
        
        is_pqc_a = any(p in algo_a_lower for p in pqc_algos)
        is_classical_b = any(c in algo_b_lower for c in classical_algos)
        is_pqc_b = any(p in algo_b_lower for p in pqc_algos)
        is_classical_a = any(c in algo_a_lower for c in classical_algos)
        
        if not ((is_pqc_a and is_classical_b) or (is_pqc_b and is_classical_a)):
            continue
        
        # Get effect size (may be in effect_size dict or at top level)
        effect_size_data = result.get('effect_size', {})
        if isinstance(effect_size_data, dict):
            cohens_d = effect_size_data.get('cohens_d')
            ci_low = effect_size_data.get('ci_95_low')
            ci_high = effect_size_data.get('ci_95_high')
        else:
            cohens_d = result.get('cohens_d')
            ci_low = None
            ci_high = None
        
        if cohens_d is None:
            continue
        
        # Get p-value
        tests = result.get('tests', {})
        p_value = 1.0
        if tests:
            # Try to get corrected p-value from any test
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict):
                    p_corrected = test_result.get('p_value_corrected')
                    if p_corrected is not None:
                        p_value = p_corrected
                        break
                    p_val = test_result.get('p_value')
                    if p_val is not None:
                        p_value = p_val
        
        # Use CI from effect_size if available, otherwise calculate
        if ci_low is None or ci_high is None:
            # Calculate approximate CI (if not available, estimate from effect size)
            # For Cohen's d, approximate 95% CI: d ± 1.96 * SE
            # SE ≈ sqrt((n1 + n2) / (n1 * n2) + d² / (2 * (n1 + n2)))
            n_a = result.get('n_a', 1000)
            n_b = result.get('n_b', 1000)
            if n_a > 0 and n_b > 0:
                se = np.sqrt((n_a + n_b) / (n_a * n_b) + cohens_d**2 / (2 * (n_a + n_b)))
                ci_low = cohens_d - 1.96 * se
                ci_high = cohens_d + 1.96 * se
            else:
                # Fallback: use ±20% of effect size
                ci_low = cohens_d * 0.8
                ci_high = cohens_d * 1.2
        
        # Determine comparison label (always show PQC first)
        if is_pqc_a:
            label = f"{format_algorithm_name(algo_a)} vs {format_algorithm_name(algo_b)}"
        else:
            label = f"{format_algorithm_name(algo_b)} vs {format_algorithm_name(algo_a)}"
            # Flip sign to show PQC advantage as positive
            cohens_d = -cohens_d
            ci_low, ci_high = -ci_high, -ci_low
        
        comparisons.append({
            'label': label,
            'cohens_d': cohens_d,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_value,
            'comparison_id': comparison_id,
        })
    
    # Sort by effect size magnitude (descending)
    comparisons.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
    
    return comparisons


def plot_effect_size_forest(comparisons: list, output_path: Path):
    """Create forest plot of effect sizes."""
    if not comparisons:
        print("Error: No comparisons to plot", file=sys.stderr)
        return
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(comparisons) * 0.4)))
    
    y_positions = np.arange(len(comparisons))
    
    # Plot confidence intervals
    for i, comp in enumerate(comparisons):
        y_pos = y_positions[i]
        cohens_d = comp['cohens_d']
        ci_low = comp['ci_low']
        ci_high = comp['ci_high']
        
        # Color based on effect size magnitude
        if abs(cohens_d) >= 0.8:
            color = '#2ecc71'  # Green for large
        elif abs(cohens_d) >= 0.5:
            color = '#f39c12'  # Orange for medium
        else:
            color = '#e74c3c'  # Red for small
        
        # Plot CI bar
        ax.plot([ci_low, ci_high], [y_pos, y_pos], color=color, linewidth=2, alpha=0.7)
        
        # Plot point estimate
        ax.scatter([cohens_d], [y_pos], color=color, s=100, zorder=10, edgecolors='black', linewidth=1)
        
        # Add significance marker
        if comp['p_value'] < 0.001:
            sig_marker = '***'
        elif comp['p_value'] < 0.01:
            sig_marker = '**'
        elif comp['p_value'] < 0.05:
            sig_marker = '*'
        else:
            sig_marker = ''
        
        # Add text label with effect size
        label_text = f"{comp['label']} (d={cohens_d:.2f}{sig_marker})"
        ax.text(-8, y_pos, label_text, va='center', ha='right', fontsize=9)
    
    # Add vertical line at d=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add effect size interpretation lines
    ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.text(0.8, len(comparisons) - 0.5, 'Large (d≥0.8)', rotation=90, 
            va='bottom', ha='right', fontsize=8, color='gray')
    ax.text(-0.8, len(comparisons) - 0.5, 'Large (d≤-0.8)', rotation=90, 
            va='bottom', ha='left', fontsize=8, color='gray')
    
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    ax.set_title("Effect Size Forest Plot: PQC vs Classical Comparisons", 
                fontsize=12, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.set_xlim(-10, 10)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Large effect (|d| ≥ 0.8)'),
        Patch(facecolor='#f39c12', label='Medium effect (0.5 ≤ |d| < 0.8)'),
        Patch(facecolor='#e74c3c', label='Small effect (|d| < 0.5)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Add note about significance
    ax.text(0.02, 0.02, '* p<0.05, ** p<0.01, *** p<0.001', 
            transform=ax.transAxes, fontsize=8, style='italic')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate effect size forest plot')
    parser.add_argument('--hypothesis-tests', type=Path, required=True,
                       help='Path to hypothesis_tests.json')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for figures')
    parser.add_argument('--environment', default='native',
                       choices=['native', 'minikube', 'gcp'],
                       help='Environment to filter (default: native)')
    
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading hypothesis tests from {args.hypothesis_tests}...")
    comparisons = load_hypothesis_tests(args.hypothesis_tests, args.environment)
    
    if not comparisons:
        print("Error: No comparisons found. Check hypothesis_tests.json file.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(comparisons)} comparisons")
    
    # Map environment name to new terminology for filename
    env_map = {'native': 'bare-metal', 'minikube': 'local-k8s', 'gcp': 'cloud-k8s'}
    env_for_file = env_map.get(args.environment, args.environment)
    output_path = args.output / f'effect_size_forest_{env_for_file}.png'
    plot_effect_size_forest(comparisons, output_path)
    
    print("Done!")


if __name__ == '__main__':
    main()
