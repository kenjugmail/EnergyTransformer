#!/usr/bin/env python3
"""Sequential reproduction of the v18 module study. No network or repository writes."""
from pathlib import Path
import sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
def run(*args):subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
def main():
    for kind in ['diagonal','euclidean','precision','delta','preconditioned_delta','softmax']:
        for seed in [11,22,33]:run('scripts/train_recall.py','--kind',kind,'--seed',str(seed))
    for seed in [11,22,33]:run('scripts/train_recall.py','--kind','precision','--objective','immediate','--seed',str(seed))
    run('scripts/evaluate_recall.py');run('scripts/convert_projected_memory.py');run('scripts/audit_projected_memory.py')
    run('scripts/fixed_recall_controls.py');run('-m','pytest','-q','tests/test_geometric_memory.py');run('scripts/benchmark_recall.py')
if __name__=='__main__':main()
