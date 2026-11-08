"""Validate the rotating source lineage before frozen joint prediction."""
import argparse
from pathlib import Path

import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal.source_diverse_views import diverse_epochs
from experiments.phase1_joint_particle import ground as joint_ground
from experiments.phase1_joint_particle.train import EPOCHS


def verify_schedule(root, source, *, preflight=False):
    folder = root/'runs'/(source+'-training-smoke' if preflight else source)
    summary = core.read(folder/('smoke.json' if preflight else 'source_summary.json'))
    state = torch.load(folder/('smoke.pt' if preflight else 'final.pt'),
                       map_location='cpu', weights_only=True)
    num_epochs = 1 if preflight else EPOCHS
    expected_schedule = [list(diverse_epochs(i)) for i in range(1, num_epochs+1)]
    actual_ids = 64 if preflight else 32768
    for obj in (summary, state):
        counts = obj.get('source_counts_by_epoch')
        if (obj.get('source_view_schedule') != expected_schedule or
                obj.get('actual_fit_ids') != actual_ids or not isinstance(counts, list) or
                len(counts) != num_epochs or
                any(not isinstance(row, dict) or sum(row.values()) != 3*actual_ids for row in counts)):
            raise ValueError('Incorrect rotating source-view lineage or exposure')
    return expected_schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('smoke', 'predict', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()
    if args.mode != 'score':
        verify_schedule(args.root, args.source, preflight=args.mode == 'smoke')
    joint_ground.main()


if __name__ == '__main__':
    main()
