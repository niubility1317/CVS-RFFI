"""Read-only held-source comparison of DURA's original and carrier-projected action."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from .antithetic_detail_source import load_arm
from .carrier_coupled import packet_frame
from .channel_blind import corruption_score
from .channel_blind_source import partition
from .data import SOURCE, VIEWS
from .dura_carrier_action import CARRIER_BINS, CarrierRestrictedAction
from .dura_carrier_action_source import CONTROL, REUSED, RUN
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_views import source_contract, source_role
from .joint_unroll_source import coordinates
from .offband_noise_source import replay
from .packet_physics import chirp_windows
from .relative_carrier import match_packet_power
from .runner import activate


def energy(wave):
    return wave.abs().square().mean((-2, -1))


def private_near(wave):
    isolated = wave - wave.mean(-2, keepdim=True)
    mask = torch.fft.fftfreq(wave.shape[-1]).abs() * wave.shape[-1]
    mask = (mask > max(CARRIER_BINS)) & (mask <= 8)
    return torch.fft.fft(isolated, norm='ortho')[...,mask]


def quantiles(values):
    return torch.quantile(values, values.new_tensor([.50, .90])).tolist()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rt = read(Path('runtime.json'))
    activate(rt)
    root, source = (Path(rt[key]) for key in ('run_root', 'source_root'))
    if root.name != RUN:
        raise ValueError('Expected the registered carrier projection experiment')
    contract = source_contract(source)
    raw, rx, ids, _ = source_role(source, contract, 'U_s')
    fit, cal, held = partition(rx)
    chosen = np.concatenate([group[:30] for group in np.split(held, len(SOURCE))])
    if np.intersect1d(chosen, np.concatenate((fit,cal))).size:
        raise ValueError('Audit requires held source U only')
    original, threshold = load_arm(root, CONTROL, contract, 'cpu', REUSED)
    tracker = PacketRestorer('consensus').eval()
    clean = complex_iq(tensor(np.stack([chirp_windows(raw[i]) for i in chosen])))
    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full', 'residual'):
            results[route] = {}
            for view in VIEWS[1:]:
                jobs = ((route, view, raw[i], rx[i], ids[i]) for i in chosen)
                received, noise = (tensor(np.stack(values)) for values in
                                   zip(*pool.map(replay, jobs, chunksize=6), strict=True))
                frame = coordinates(tracker, received)
                active = corruption_score(frame[0]) > threshold
                y, rotation, _, scale = packet_frame(frame)
                native = real_iq(y * rotation).flatten(0,1)
                _, action, _ = original.restorer.estimate_context(native)
                restricted = CarrierRestrictedAction(action, rotation.flatten(0,1))
                preserved = match_packet_power(clean, y * rotation)
                inputs = preserved.flatten(0,1)
                proposals = (inputs, action(inputs), restricted(inputs))

                # Exact replay components are used only as references below.
                expected = complex_iq(received-noise) / scale
                variance = energy(complex_iq(noise) / scale).clamp_min(1e-10)
                near_ref = private_near(expected * rotation.conj())
                rows = {}
                for mode, proposal in zip(('identity','original','projected_original'), proposals, strict=True):
                    wave = proposal.reshape_as(expected)
                    relative = energy(wave - expected) / variance
                    off_manifold = energy(private_near(wave * rotation.conj()) - near_ref) / variance
                    use = active if active.any() else torch.ones_like(active)
                    rows[mode] = dict(relative_action_error=quantiles(relative[use]),
                                      private_3to8khz_error=quantiles(off_manifold[use]))
                results[route][view] = dict(active=int(active.sum()), methods=rows)
                print('LORA_DURA_CARRIER_ACTION_U', route, view, int(active.sum()), flush=True)
    result = dict(status='VERIFIED', n=len(chosen), results=results, target_reads=0,
                  F_reads=0, TX_reads=0, V_reads=0, exact_replay_only_for_diagnostics=True,
                  frozen_original=True, optimization_steps=0)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
