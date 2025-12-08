#!/usr/bin/env python3
"""
Check data sufficiency for research claims.

Analyzes what data exists and what analysis can be performed.
"""

import json
from pathlib import Path
from collections import defaultdict

def check_data_sufficiency():
    """Check what data exists and what analysis is possible."""
    
    results_base = Path("results")
    final_results = Path("final-results")
    
    print("=" * 70)
    print("DATA SUFFICIENCY CHECK")
    print("=" * 70)
    print()
    
    # Check raw data files
    print("=== Raw Data Files ===")
    data_status = {}
    
    for env in ['native', 'minikube', 'gcp']:
        env_dir = results_base / env
        if not env_dir.exists():
            data_status[env] = {'jsonl': 0, 'dirs': 0, 'has_stats': 0}
            print(f"{env.upper()}: No directory found")
            continue
        
        # Count directories
        dirs = [d for d in env_dir.iterdir() if d.is_dir() if not d.name.startswith('.')]
        
        # Count JSONL files
        jsonl_files = list(env_dir.rglob("raw/run.jsonl"))
        
        # Count stats files
        stats_files = list(env_dir.rglob("stats/summary.json"))
        
        data_status[env] = {
            'dirs': len(dirs),
            'jsonl': len(jsonl_files),
            'has_stats': len(stats_files),
        }
        
        print(f"{env.upper()}:")
        print(f"  Directories: {len(dirs)}")
        print(f"  JSONL files: {len(jsonl_files)}")
        print(f"  Stats files: {len(stats_files)}")
    
    print()
    
    # Check index
    print("=== Index Status ===")
    index_path = final_results / "index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        
        exps = [e for e in index.get('experiments', []) if e.get('status') in ['success', 'cached']]
        
        by_env = defaultdict(list)
        for exp in exps:
            env = exp.get('environment', 'unknown')
            by_env[env].append(exp)
        
        print(f"Total indexed experiments: {len(exps)}")
        for env, env_exps in sorted(by_env.items()):
            print(f"  {env}: {len(env_exps)}")
            
            # Check algorithms
            algorithms = set(e.get('algorithm') for e in env_exps)
            print(f"    Algorithms: {', '.join(sorted(algorithms))}")
            
            # Check replicas
            replicas = set(e.get('replicas', 1) for e in env_exps)
            print(f"    Replicas: {sorted(replicas)}")
    else:
        print("No index.json found")
        print("  Note: Data may exist but not be indexed")
    
    print()
    
    # Analysis capability check
    print("=== Analysis Capability ===")
    
    # 1. Native analysis
    native_jsonl = data_status.get('native', {}).get('jsonl', 0)
    if native_jsonl >= 468:
        print("✅ Native Analysis: FULL")
        print("   - Can do complete algorithm comparison")
        print("   - Can do statistical analysis")
        print("   - Can generate all native plots")
    elif native_jsonl > 0:
        print(f"⚠️  Native Analysis: PARTIAL ({native_jsonl}/468 experiments)")
        print("   - Can do limited algorithm comparison")
    else:
        print("❌ Native Analysis: NO DATA")
    
    # 2. Cross-environment comparison
    native_jsonl = data_status.get('native', {}).get('jsonl', 0)
    minikube_jsonl = data_status.get('minikube', {}).get('jsonl', 0)
    gcp_jsonl = data_status.get('gcp', {}).get('jsonl', 0)
    
    print()
    if native_jsonl > 0 and minikube_jsonl > 0:
        overlap = min(native_jsonl, minikube_jsonl)
        print(f"⚠️  Native vs Minikube: PARTIAL ({overlap} experiments)")
        print("   - Can compare overlapping experiments only")
        print("   - Limited statistical power")
    else:
        print("❌ Native vs Minikube: INSUFFICIENT DATA")
    
    if native_jsonl > 0 and gcp_jsonl > 0:
        overlap = min(native_jsonl, gcp_jsonl)
        if overlap >= 10:
            print(f"⚠️  Native vs GCP: PARTIAL ({overlap} experiments)")
        else:
            print(f"❌ Native vs GCP: INSUFFICIENT DATA ({overlap} experiments)")
    else:
        print("❌ Native vs GCP: INSUFFICIENT DATA")
    
    if minikube_jsonl > 0 and gcp_jsonl > 0:
        overlap = min(minikube_jsonl, gcp_jsonl)
        if overlap >= 10:
            print(f"⚠️  Minikube vs GCP: PARTIAL ({overlap} experiments)")
        else:
            print(f"❌ Minikube vs GCP: INSUFFICIENT DATA ({overlap} experiments)")
    else:
        print("❌ Minikube vs GCP: INSUFFICIENT DATA")
    
    # 3. Scaling analysis
    print()
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        scaling_exps = [e for e in index.get('experiments', []) 
                       if e.get('replicas', 1) > 1 and e.get('status') in ['success', 'cached']]
        
        if len(scaling_exps) > 0:
            print(f"⚠️  Scaling Analysis: PARTIAL ({len(scaling_exps)} scaling experiments)")
            by_env = defaultdict(int)
            for exp in scaling_exps:
                by_env[exp.get('environment')] += 1
            for env, count in sorted(by_env.items()):
                print(f"   - {env}: {count}")
        else:
            print("❌ Scaling Analysis: NO DATA")
            print("   - Need experiments with replicas 2, 4, 8")
    else:
        print("❌ Scaling Analysis: CANNOT CHECK (no index)")
    
    print()
    print("=== Recommendations ===")
    
    # Calculate gaps
    native_gap = max(0, 468 - data_status.get('native', {}).get('jsonl', 0))
    minikube_gap = max(0, 468 - data_status.get('minikube', {}).get('jsonl', 0))
    gcp_gap = max(0, 468 - data_status.get('gcp', {}).get('jsonl', 0))
    
    if native_gap == 0:
        print("✅ Native: Complete")
    else:
        print(f"❌ Native: Missing {native_gap} experiments")
    
    if minikube_gap == 0:
        print("✅ Minikube: Complete")
    else:
        print(f"⚠️  Minikube: Missing {minikube_gap} experiments ({minikube_gap/468*100:.1f}%)")
    
    if gcp_gap == 0:
        print("✅ GCP: Complete")
    else:
        print(f"❌ GCP: Missing {gcp_gap} experiments ({gcp_gap/468*100:.1f}%)")
    
    print()
    print("=== What You Can Do Now ===")
    
    if native_jsonl >= 468:
        print("✅ Run full native analysis")
        print("   python3 analysis/compute_statistics.py --input results/native/...")
        print("   python3 analysis/compare_native_vs_minikube.py --native ...")
    
    if native_jsonl > 0 and minikube_jsonl > 0:
        print("⚠️  Run partial cross-environment comparison")
        print("   (Limited to overlapping experiments)")
    
    if native_jsonl >= 468:
        print("✅ Generate native-only plots and figures")
        print("   (Algorithm comparison, latency distributions, etc.)")
    
    print()
    print("=== What You Cannot Do ===")
    
    if gcp_jsonl < 10:
        print("❌ GCP analysis (insufficient data)")
    
    if minikube_jsonl < 100:
        print("❌ Comprehensive minikube analysis")
    
    if not index_path.exists() or len(scaling_exps) == 0:
        print("❌ Scaling analysis (no scaling experiments)")
    
    if native_jsonl < 468 or minikube_jsonl < 468 or gcp_jsonl < 468:
        print("❌ Full 3-way cross-environment comparison")
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    check_data_sufficiency()

