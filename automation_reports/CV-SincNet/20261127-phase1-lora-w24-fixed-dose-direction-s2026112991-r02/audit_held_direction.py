"""Compare the two fixed-dose directions on the original source U holdout."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import tensor
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE
from experiments.pamf_lora_main.full_packet_views import source_contract, source_role
from experiments.pamf_lora_main.w24_fixed_dose_direction import (
    BATCH, FIT_PER_RX, RUN, VIEWS, W24_COMMIT, W24_RUN, fit_root, load_head, readouts, source_parent,
)
from experiments.pamf_lora_main.w24_fixed_dose_direction_model import ApertureDirection, CENTER
from experiments.pamf_lora_main.w24_practical_supervision import NAMESPACE, SEED


def energies(edit, wanted):
    energy = wanted.abs().square().flatten(1).sum(-1)
    inner = (edit.conj() * wanted).real.flatten(1).sum(-1)
    magnitude = edit.abs().square().flatten(1).sum(-1).sqrt()
    residual = (edit - wanted).abs().square().flatten(1).sum(-1)
    normalized_norm = magnitude / energy.sqrt().clamp_min(1e-12)
    along = inner / energy.clamp_min(1e-24)
    parallel = (along - 1.).square()
    orthogonal = (normalized_norm.square() - along.square()).clamp_min(0.)
    return torch.stack((inner / (magnitude * energy.sqrt()).clamp_min(1e-12),
                        normalized_norm, residual / energy.clamp_min(1e-24),
                        parallel, orthogonal), dim=-1)


@torch.no_grad()
def audit(project):
    torch.set_num_threads(2)
    root = project / 'runs' / RUN
    config = json.loads((root / 'fit-inputs.json').read_text(encoding='utf-8'))
    if fit_root(config) != root:
        raise ValueError('Expected the original closed source fit')
    source = Path(config['source_root'])
    contract = source_contract(source)
    iq, rx, _, _ = source_role(source, contract, 'U_s')
    _, _, held = partition(rx, fit=FIT_PER_RX)
    cache = Path(config['practical_cache'])
    proof = json.loads((cache / 'complete.json').read_text(encoding='utf-8'))
    if (len(held) != 500 or proof['status'] != 'SOURCE_TRACE_FROZEN'
            or proof['data_role'] != 'U_s' or proof['namespace'] != NAMESPACE
            or proof['seed'] != SEED or proof['count'] != len(iq)
            or any(proof[k] for k in ('F_reads', 'TX_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected the original, isolated source Practical holdout')

    scenes = np.empty(len(iq), dtype=np.int8)
    rng = np.random.default_rng(SEED)
    for receiver in SOURCE:
        rows = np.flatnonzero(rx == receiver)
        scenes[rows] = rng.permutation(np.arange(len(rows)) % 6)
    if dict(zip(VIEWS[1:], np.bincount(scenes, minlength=6).tolist(), strict=True)) != proof['scenes']:
        raise ValueError('Source scenario identity differs from the sealed bank')

    pairs = np.load(cache / 'pairs.npy', allow_pickle=False, mmap_mode='r')
    if pairs.shape != (len(iq), 2, 8192) or pairs.dtype != np.complex64:
        raise ValueError('Invalid source pair bank')
    parent, (head, _) = source_parent(config, 'cpu'), load_head(root)
    projection = ApertureDirection()
    values = []
    for start in range(0, len(held), BATCH):
        block = pairs[held[start:start+BATCH]].view(np.float32)
        y, s = tensor(block, 'cpu').view(torch.complex64).unbind(1)
        half = torch.lerp(y, parent.restore_native(y), .5) - y
        fixed, learned = projection.edits(half, head(y, half))
        wanted = projection.project_regions(s - y)
        values.append(torch.stack([
            torch.stack([energies(edit[..., part], wanted[..., part]) for edit in (fixed, learned)], dim=1)
            for part in (slice(None, CENTER), slice(CENTER, None))
        ], dim=1))

    result = {}
    metrics = np.from_dlpack(torch.cat(values))
    np.testing.assert_allclose(metrics[..., 2], metrics[..., 3] + metrics[..., 4], atol=1e-6)
    for name, selected in zip(('all', *VIEWS[1:]),
                              (np.ones(len(held), dtype=bool), *(scenes[held] == i for i in range(6))), strict=True):
        this = metrics[selected]
        result[name] = dict(n=len(this), median=np.median(this, axis=0).tolist(),
                            mean=this.mean(axis=0).tolist(),
                            improve_count=(this[:, :, 1, 2] < this[:, :, 0, 2]).sum(0).tolist())
    from experiments import pamf_lora_main

    release = project / 'releases' / f'{W24_RUN}-release'
    archive = json.loads((release / 'submit.json').read_text(encoding='utf-8'))
    if archive['code_commit'] != W24_COMMIT:
        raise ValueError('Expected the original W24 readout archive')
    pamf_lora_main.__path__.append(str(release / 'experiments' / 'pamf_lora_main'))
    single = pairs[held[:32], 0]
    for observed in single[:4]:
        readouts(parent, head, observed[None])
    elapsed = []
    for observed in single:
        start = perf_counter()
        readouts(parent, head, observed[None])
        elapsed.append((perf_counter() - start) * 1000.)

    return dict(result=result, regions=['center', 'edges'], readouts=['frozen_half', 'fixed_dose_direction'],
                metrics=['cosine', 'edit_to_noise_norm', 'residual_to_noise_power',
                         'parallel_error_to_noise_power', 'orthogonal_error_to_noise_power'],
                cpu_readouts_b1_ms=dict(n=len(elapsed), median=float(np.median(elapsed)),
                                       p95=float(np.quantile(elapsed, .95))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.project)))
