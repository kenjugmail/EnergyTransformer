#!/usr/bin/env python3
"""Export all three bounded parser/decoder checkpoints; no retraining."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_transformer.portable import export_portable

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('portable'))
    args = p.parse_args()
    for seed in [11, 22, 33]:
        config = export_portable(Path(f'checkpoints/v14/bridge_seed{seed}.pt'), args.output / f'seed{seed}')
        print(seed, config['weights_sha256'])
