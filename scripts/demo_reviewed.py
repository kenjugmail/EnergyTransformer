#!/usr/bin/env python3
"""Show a learned proposal followed by exact, authorized semantic approval."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from energy_transformer.review import Edit, ReviewedEnergySession

def main():
    torch.set_num_threads(1)
    checkpoint = ROOT / 'checkpoints/v14/bridge_seed11.pt'
    if not checkpoint.exists():
        raise SystemExit('Checkpoint absent. Use the complete bundle, or run scripts/train_bridge.py --seed 11.')
    s = ReviewedEnergySession(checkpoint)
    response = s.propose('Alice is in the garden.')
    print(response)
    print('Before review:', s.answer('alice'))
    if response['status'] != 'needs_review':
        raise SystemExit('No supported proposal; nothing was changed.')
    proposal = response['proposal']
    # The example explicitly supplies the intended meaning. Do NOT construct a
    # real review by copying the unreviewed model output automatically.
    s.approve(proposal['proposal_id'], reviewed_edit=Edit('set', 'alice', 'garden'),
              expected_revision=proposal['revision'], authorized=True)
    print('After explicit approval:', s.answer('alice'))
    s.propose('Maybe Alice went to the kitchen.')
    print('After an unapproved sentence:', s.answer('alice'))
    s.apply_edits([Edit('delete', 'alice')], expected_revision=s.revision, authorized=True)
    print('After explicit deletion:', s.answer('alice'))

if __name__ == '__main__':
    main()
