"""Isolated WiSig diagnostic using fixed source U and a generic known action."""
import json
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read
from experiments.pamf_repl_view_shuffle.observable_action import forward_action, power, phase
from experiments.pamf_repl_view_shuffle.observable_action_source import RUN, ARMS, load_arm, partition, received, source_u

rt = read(Path('runtime.json'))
assert rt['run_id'] == RUN
root = Path(rt['run_root'])
torch.set_num_threads(2)
sample = received(source_u(rt)[partition()[2][:128]], 'cpu')
models = {name: load_arm(root, name, 'cpu') for name in ARMS}
fitted = models['fitted']
rng = torch.Generator().manual_seed(2026112329)
size = len(sample)
t = fitted.index
lag = torch.randint(-37, 38, (size,), generator=rng)
lag = torch.where(lag.abs() < 8, lag + 13, lag)
gain = phase(torch.empty(size).uniform_(-2.5, 2.5, generator=rng))
h = sample.new_zeros(size, 7)
h[:, 2] = gain
h[:, 1] = .09j * gain
h[:, 5] = .15j * gain
h[:, 6] = .08 * gain
disturbed = forward_action(sample, h, lag, t, fitted.ramp)
noise = torch.randn(sample.shape, generator=rng, dtype=sample.dtype)
disturbed = disturbed + .3 * power(sample).mean(-1, keepdim=True).sqrt() * noise
zeros = torch.zeros_like(lag)

def summary(wave):
    nmse = power(wave-sample).sum(dim=1) / power(sample).sum(dim=1).clamp_min(1e-8)
    return nmse.quantile(torch.tensor([.1, .5, .9])).tolist()

outputs = {}
with torch.no_grad():
    for group, y, delay in (('clean', sample, zeros), ('phase_only', sample * gain[:, None], zeros),
                            ('shift_filter_and_noise', disturbed, lag)):
        rows = {'raw_nmse_q10_q50_q90': summary(y)}
        for arm, model in models.items():
            coarse, final, info = model(y)
            error = (info['action_delay'] - delay).abs()
            rows[arm] = dict(coarse_nmse_q10_q50_q90=summary(coarse), final_nmse_q10_q50_q90=summary(final),
                             median_absolute_delay_error=float(error.float().median()),
                             delay_within_two=int((error <= 2).sum()),
                             delay_zero=int((info['action_delay']==0).sum()),
                             median_held_residual=float(info['held_residual'].median()))
        outputs[group] = rows
print(json.dumps(dict(n=size, independent_generic_seed=2026112329, source_U_only=True,
                      source_truth_reads=0, recognizer_reads=0, target_reads=0, waves=outputs),
                 indent=2, allow_nan=False))
