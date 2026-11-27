"""Build source-only Practical supervision and train native W24 recovery readouts."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .full_packet_views import source_contract, source_role
from .w24_native_center import NativeCenterProjection


NAMESPACE = 'w24_native_practical_supervision_source_u_20261127'
SEED = 2026112704
EDGE = 24


def source_pair(row):
    from .w24_factor_audit import trace_components

    y, s, _ = trace_components(*row, NAMESPACE)
    return np.stack((y, s)).astype(np.complex64, copy=False)


def cache_source(config):
    from .data import SOURCE, VIEWS
    from .runner import activate
    from .w24_native_full import native_full_view

    if set(config) != {'source_root', 'cache_root', 'frozen_sources', 'workers'}:
        raise ValueError('Practical trace construction must receive only source roles and simulator bindings')
    activate(config)
    src, out = Path(config['source_root']), Path(config['cache_root'])
    contract = source_contract(src)
    iq, receivers, ids, _ = source_role(src, contract, 'U_s')
    scenes = np.empty(len(iq), dtype=np.int8)
    rng = np.random.default_rng(SEED)
    for rx in SOURCE:
        rows = np.flatnonzero(receivers == rx)
        scenes[rows] = rng.permutation(np.arange(len(rows)) % 6)
    view = np.asarray(VIEWS[1:])[scenes]
    out.mkdir()
    arr = np.lib.format.open_memmap(out/'pairs.npy', mode='w+', dtype=np.complex64, shape=(len(iq), 2, 8192))
    row0 = (iq[0], str(view[0]), str(receivers[0]), str(ids[0]))
    if not np.array_equal(source_pair(row0)[0], native_full_view(*row0, NAMESPACE)):
        raise ValueError('Source trace changed the registered received waveform')
    jobs = zip(iq, map(str, view), map(str, receivers), map(str, ids), strict=True)
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        for i, pair in enumerate(pool.map(source_pair, jobs, chunksize=12)):
            if pair.shape != (2, 8192) or not np.isfinite(pair).all():
                raise ValueError('Invalid source trace pair')
            arr[i] = pair
            if (i+1) % 1000 == 0:
                print('W24_SOURCE_PRACTICAL_PAIRS', i+1, len(iq), flush=True)
    arr.flush()
    write(out/'complete.json', dict(status='SOURCE_TRACE_FROZEN', data_role='U_s', source_root=str(src),
          count=len(iq), namespace=NAMESPACE, seed=SEED, F_reads=0, TX_reads=0, V_reads=0, target_reads=0,
          source_trace_roles=['received', 'post_action_before_noise'],
          scenes=dict(zip(VIEWS[1:], np.bincount(scenes, minlength=6).tolist(), strict=True))))


def training_sets(roles, device, *, smoke, fit_per_rx, bank_per_rx):
    from .channel_blind_source import partition

    src, root = Path(roles['source_root']), Path(roles['practical_cache'])
    contract = source_contract(src)
    iq, rx, _, _ = source_role(src, contract, 'U_s')
    cache = read(root/'complete.json')
    if (cache['status'] != 'SOURCE_TRACE_FROZEN' or cache['source_root'] != str(src)
            or cache['count'] != len(iq) or cache['namespace'] != NAMESPACE
            or cache['data_role'] != 'U_s' or any(cache[k] for k in ('F_reads','TX_reads','V_reads','target_reads'))):
        raise ValueError('Invalid matched source-only Practical cache')
    pairs = np.load(root/'pairs.npy', allow_pickle=False, mmap_mode='r')
    if pairs.shape != (len(iq), 2, 8192) or pairs.dtype != np.complex64:
        raise ValueError('Expected complete source-only Practical pair bank')
    train, _, held = partition(rx, fit=5, cal=4, held=3) if smoke else partition(rx, fit=fit_per_rx)

    def load(indices):
        out = torch.empty((len(indices), 3, 8192), device=device, dtype=torch.complex64)
        for start in range(0, len(indices), 512):
            subset = indices[start:start+512]
            block = np.concatenate((iq[subset, None], pairs[subset]), axis=1)
            out[start:start+len(subset)] = tensor(block.view(np.float32), device).view(torch.complex64)
        return out

    return (load(train), torch.empty(0, device=device), load(held)), contract


def regional_power(value):
    return value.abs().square().mean((-2, -1), keepdim=True)


def matched_direction(ideal, learned):
    want = regional_power(ideal)
    scale = torch.where(want > 1e-16, (regional_power(learned) / want.clamp_min(1e-16)).sqrt(), 0.)
    return ideal * scale, scale


class PracticalLoss:
    def __init__(self, device, teacher=None):
        self.project = NativeCenterProjection().to(device)
        self.teacher = teacher

    def edges(self, x):
        chirps = x.unflatten(-1, (8, 1024))
        return torch.cat((chirps[..., :EDGE], chirps[..., -EDGE:]), dim=-1)

    def __call__(self, model, _tracker, raw, _rng, *, with_identity, recipe, step):
        clean, y, s = raw.unbind(1)
        if with_identity:
            y, s = (torch.cat((x[:-2], clean[-2:])) for x in (y, s))
        rms = y.abs().square().mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
        y, wanted = y/rms, (s-y)/rms
        with torch.no_grad():
            target_center, target_edges = self.project(wanted), self.edges(wanted)
            if self.teacher is not None:
                teacher_edit = .5 * (self.teacher.restore_native(y) - y)
                target_center, center_scale = matched_direction(target_center, self.project(teacher_edit))
                target_edges, edge_scale = matched_direction(target_edges, self.edges(teacher_edit))
                parts = wanted.unflatten(-1, (8, 1024))
                wanted = torch.cat((parts[..., :EDGE] * edge_scale, parts[..., EDGE:-EDGE] * center_scale,
                                    parts[..., -EDGE:] * edge_scale), dim=-1).flatten(-2)
        stages, _ = model(y)
        edits = tuple(.5 * (z-y) for z in stages)
        center = self.project(edits[-1])
        edge = self.edges(edits[-1])
        errors = [(edit-wanted).abs().square().mean() for edit in edits]
        center_error = (center-target_center).abs().square().mean()
        edge_error = (edge-target_edges).abs().square().mean()
        identity = edits[-1][-2:].abs().square().mean() if with_identity else center_error.new_zeros(())
        values = torch.stack((*errors, center_error, edge_error, identity))
        return values @ values.new_tensor((.12, .12, .60, 1., .35, 1.)), values
