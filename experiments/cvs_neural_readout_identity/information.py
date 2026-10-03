"""Public feature-space counterexample; no dataset, fitting, or model checkpoint.

This checks a local readout's expressivity, not information loss by the complete
network: upstream complex filters may already encode phase into power.
"""
import argparse
import json
from pathlib import Path

import torch
from experiments.cvs_equivariant_identity.model import InvariantReadout
from experiments.cvs_neural_readout_identity.model import LearnedInvariantReadout


def rotate_channels(z, phase):
    r, i = z.unbind(1)
    c, s = phase.cos()[None, :, None], phase.sin()[None, :, None]
    return torch.stack((r*c-i*s, r*s+i*c), 1)


@torch.no_grad()
def verify():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1907)
        z = torch.randn(3, 2, 32, 64, dtype=torch.float64)
        z[:, :, 1] = z[:, :, 0]
        phase = torch.zeros(32, dtype=z.dtype)
        phase[1] = torch.pi
        changed = rotate_channels(z, phase)
        old = InvariantReadout().double()
        plain = LearnedInvariantReadout(False).double()
        mixed = LearnedInvariantReadout(True).double()
        # Construct one legal parameter setting to exhibit capacity. These are
        # not learned weights or proposed initialization for the experiment.
        for block in (plain, mixed):
            for p in block.scores.parameters():
                p.zero_()
            block.project.weight.zero_()
            block.project.weight[0, 0] = 1.
        mixed.mixer.weight_real[0, 1, 0] = 1.
        common = rotate_channels(z, torch.full((32,), .37, dtype=z.dtype))
        error = lambda a, b: float((a-b).abs().max())
        result = dict(
            status='VERIFIED_PUBLIC_FEATURE_COUNTEREXAMPLE',
            feature_shape=list(z.shape),
            original_readout_independent_phase_error=error(old(z), old(changed)),
            attention_without_mixing_independent_phase_error=error(plain(z), plain(changed)),
            constructed_mixing_independent_phase_difference=error(mixed(z), mixed(changed)),
            constructed_mixing_common_phase_error=error(mixed(z), mixed(common)),
            formal_data_access=False, checkpoint_loaded=False, optimization_steps=0,
            experiment_initialization_changed=False,
            scope='Readout expressivity only; no whole-network information-loss, channel-invariance, TX-origin or performance claim',
        )
        assert result['original_readout_independent_phase_error'] < 1e-12
        assert result['attention_without_mixing_independent_phase_error'] < 1e-12
        assert result['constructed_mixing_independent_phase_difference'] > .1
        assert result['constructed_mixing_common_phase_error'] < 1e-12
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = verify()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))
