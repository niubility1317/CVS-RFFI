"""Train an IQ-only WiSig restorer using isolated exact source Practical references."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import torch
from torch import nn

from experiments.cric_dura_comparison.io import read, tensor, write


ARMS = ('direction_only', 'exact_denoise')
REGIONS = (slice(24, 232), np.r_[:24, 232:256])
SEED = 2026112795
FIT = 20000
HELD = 512
STEPS = 24000
BATCH = 32


def complex_iq(x):
    return x[:, 0] + 1j*x[:, 1]


def split_iq(x):
    return np.stack((x.real, x.imag), axis=1).astype(np.float32)


def matched_target(observed, prior, noiseless):
    """Replace direction but preserve the legal, frozen source teacher's two edit norms."""
    out = prior.copy()
    for take in REGIONS:
        y, g, s = (a[:, take] for a in (observed, prior, noiseless))
        oracle = s-y
        power = np.square(np.abs(oracle)).sum(-1, keepdims=True)
        edit = np.square(np.abs(g-y)).sum(-1, keepdims=True)
        scale = np.sqrt(np.divide(edit, power, out=np.zeros_like(power), where=power>1e-15))
        out[:, take] = np.where(power>1e-15, y+scale*oracle, y)
    return out


class PacketDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        width = 32
        self.stem = nn.Conv1d(2, width, 7, padding=3)
        self.layers = nn.ModuleList(nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2*d, dilation=d), nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv1d(width, width, 1)) for d in (1, 2, 4, 8, 16))
        self.readout = nn.Conv1d(width, 2, 5, padding=2)
        nn.init.normal_(self.readout.weight, std=.002)
        nn.init.zeros_(self.readout.bias)

    def forward(self, received):
        level = received.square().sum(1, keepdim=True).mean(-1, keepdim=True).add(1e-10).sqrt()
        z = self.stem(received / level)
        for layer in self.layers:
            z = z + layer(z)
        return received + self.readout(z) * level


def objective(model, received, target, quiet):
    result = model(received)
    level = received.square().mean((1, 2)).clamp_min(1e-8)
    error = (result-target).square()
    center = error[..., 24:232].mean((1, 2)) / level
    edges = torch.cat((error[..., :24], error[..., 232:]), -1).mean((1, 2)) / level
    identity = (model(quiet)-quiet).square().mean() / quiet.square().mean().clamp_min(1e-8)
    return .75*center.mean() + .25*edges.mean() + .50*identity


def source_iq(source):
    source = Path(source)
    manifest = read(source/'complete.json')
    if (manifest['status'], manifest['train_count'], manifest['U_count'],
            manifest['TX_labels_read'], manifest['target_reads']) != (
            'RECEPTION_SOURCE_CACHE_COMPLETE', 63000, 56700, 0, 0):
        raise ValueError('WiSig source-only role contract mismatch')
    clean, roles, ids, rx, day = (np.load(source/'train'/f'{name}.npy', allow_pickle=False, mmap_mode='r')
                                 for name in ('clean', 'roles', 'physical_ids', 'rx', 'day'))
    if clean.shape != (63000, 2, 256) or clean.dtype != np.float32 or len(np.unique(ids)) != len(clean):
        raise ValueError('Unexpected native WiSig source array')
    perm = np.random.default_rng(SEED).permutation(np.flatnonzero(roles == 'U_s'))
    if len(perm) != 56700:
        raise ValueError('Expected only verified source U_s IDs')
    fit, held = perm[:FIT], perm[FIT:FIT+HELD]
    return clean, ids, rx, day, fit, held


def source_snapshot(item):
    from leo_practical import ChannelStream, Config, receiver_for_session, stable_seed

    y, ident, session, scene, draw = item
    config = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='full',
                    mode='post_sync', equalization_enabled=False, boundary='reflect')
    seed = stable_seed(SEED, 'wisig-known-noise/source-U', scene, str(ident), str(draw))
    receiver = receiver_for_session(config, SEED+1, session)
    observed, meta, trace = ChannelStream(config, seed, receiver).process(y, return_traces=True,
                                                                          return_operator=True)
    scale = 10**(meta['agc_gain_db']/20)
    signal, noise = (trace[name] * scale for name in ('pre_agc_signal', 'pre_agc_noise'))
    np.testing.assert_allclose(observed, signal+noise, rtol=2e-6, atol=2e-7)
    return observed.astype(np.complex64), signal.astype(np.complex64)


def make_pairs(clean, ids, rx, day, indices, views, draw, workers):
    rng = np.random.default_rng([SEED, draw])
    scenes = rng.permutation(np.arange(len(indices)) % len(views))
    samples = ((clean[i, 0]+1j*clean[i, 1], ids[i], f'rx{rx[i]}:day{day[i]}:eq1', views[k], draw)
               for i, k in zip(indices, scenes, strict=True))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = []
        for i, pair in enumerate(pool.map(source_snapshot, samples, chunksize=24), 1):
            rows.append(pair)
            if i % 1000 == 0:
                print('WISIG_KNOWN_NOISE_SOURCE_PAIR', draw, i, flush=True)
    return np.stack(rows).transpose(1, 0, 2)


@torch.no_grad()
def make_bank(rt):
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import core
    from experiments.pamf_unrolled_restore.wisig_native_fir_anchor_probe import reference
    from experiments.pamf_unrolled_restore.wisig_native_fir_source import build, runner

    source, root = (Path(rt[name]) for name in ('source_root', 'run_root'))
    root = root/'source-bank'
    root.mkdir()
    parent = Path(reference(dict(rt, reference=rt['teacher_release']))['run_root'])
    record = read(parent/'centered-complete.json')
    checkpoint = torch.load(parent/'centered.pt', map_location='cpu', weights_only=True)
    if checkpoint['evidence'] != record or any(record[k] != v for k, v in runner.proof('centered').items()):
        raise ValueError('Expected the frozen source-only Native FIR teacher')
    clean, ids, rx, day, fit, held = source_iq(source)
    views = core.baseline.VIEWS[1:]
    sets = [(name, take, make_pairs(clean, ids, rx, day, take, views, draw, rt['cpu_workers']))
            for name, take, draw in (('fit', fit, 0), ('held', held, 1))]
    teacher = build('centered').eval().requires_grad_(False)
    teacher.load_state_dict(checkpoint['weights'], strict=True)
    for name, take, (observed, noiseless) in sets:
        batches = []
        for batch, chunk in enumerate(np.array_split(observed, (len(observed)+63)//64), 1):
            received = tensor(split_iq(chunk))
            z = torch.complex(received[:, 0], received[:, 1])
            stages, *_ = teacher.restore(teacher.analyze(z))
            recovered = stages[-1]
            if recovered.shape != z.shape or not torch.isfinite(recovered).all():
                raise ValueError('Source-only teacher returned an invalid received-window estimate')
            batches.append(np.from_dlpack(recovered.contiguous()))
            if batch % 20 == 0:
                print('WISIG_KNOWN_NOISE_SOURCE_TEACHER', name, batch, flush=True)
        teacher_readout = np.concatenate(batches)
        target = matched_target(observed, teacher_readout, noiseless)
        with (root/f'{name}.npz').open('xb') as stream:
            np.savez(stream, ids=ids[take], observed=split_iq(observed), exact_denoise=split_iq(noiseless),
                     direction_only=split_iq(target), clean=clean[take])
        print('WISIG_KNOWN_NOISE_SOURCE_BANK_READY', name, len(take), flush=True)
    write(root/'complete.json', source_bank_proof())


def source_bank_proof():
    return dict(status='SOURCE_READY', fit=FIT, held=HELD, source_role='U_s',
                trace_scope='source_only', target_reads=0, TX_reads=0, F_reads=0)


def validate_bank(root):
    if read(root/'complete.json') != source_bank_proof():
        raise ValueError('Expected a completed isolated source bank')


def load_arrays(root, arm, device):
    if arm not in ARMS:
        raise ValueError('Unknown frozen supervision rule')
    validate_bank(root)
    arrays, ids = [], []
    for role, expected in (('fit', FIT), ('held', HELD)):
        with np.load(root/f'{role}.npz', allow_pickle=False) as src:
            ids.append(src['ids'])
            columns = [src[k] for k in ('observed', arm, 'clean')]
            if len(np.unique(ids[-1])) != expected or any(
                   x.shape != (expected, 2, 256) or x.dtype != np.float32 or not np.isfinite(x).all()
                   for x in columns):
                raise ValueError('Wrong source-bank rows')
        arrays.append(tuple(tensor(x, device) for x in columns))
    if np.intersect1d(*ids).size:
        raise ValueError('Held source physical IDs overlap fitting')
    return arrays


def fit(root, bank, arm):
    device = 'cpu'
    torch.set_num_threads(2)
    (received, target, clean), held = load_arrays(bank, arm, device)
    torch.manual_seed(SEED)
    model = PacketDenoiser().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=3e-5)
    rng = np.random.default_rng(SEED+2)
    start, cpu_start, trajectory = perf_counter(), process_time(), {}
    print('WISIG_KNOWN_NOISE_FIT_DEVICE', arm, device, torch.get_num_threads(), flush=True)
    for step in range(1, STEPS+1):
        rows = tensor(rng.choice(FIT, BATCH, replace=False).astype(np.int64), device)
        loss = objective(model, received[rows], target[rows], clean[rows][:8])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if step == 1 or step % 200 == 0:
            print('WISIG_KNOWN_NOISE_FIT', arm, step, float(loss.detach()), perf_counter()-start, flush=True)
        if step in (2000,6000,12000,18000,STEPS):
            with torch.no_grad():
                trajectory[str(step)] = float(objective(model, *held))
            print('WISIG_KNOWN_NOISE_HELD', arm, step, trajectory[str(step)], flush=True)
    record = dict(status='SOURCE_FROZEN', arm=arm, seed=SEED, batch=BATCH, steps=STEPS, fit=FIT, held=HELD,
                  parameters=sum(p.numel() for p in model.parameters()), seconds=perf_counter()-start,
                  device=device, cpu_threads=torch.get_num_threads(), cpu_seconds=process_time()-cpu_start,
                  peak_cuda=0, target_reads=0, TX_reads=0, F_reads=0,
                  trace_scope='source_only', trajectory=trajectory)
    with (root/f'{arm}.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), metadata=record), stream)
    write(root/f'{arm}-complete.json', record)


def load_models(root, device='cuda:0'):
    models = {}
    for arm in ARMS:
        info = read(root/f'{arm}-complete.json')
        saved = torch.load(root/f'{arm}.pt', map_location=device, weights_only=True)
        if (saved['metadata'] != info or info['status'] != 'SOURCE_FROZEN' or info['steps'] != STEPS
                or info['arm'] != arm or info['trace_scope'] != 'source_only'
                or any(info[k] for k in ('target_reads','TX_reads','F_reads'))):
            raise ValueError('Expected exact matched frozen source restoration weights')
        model = PacketDenoiser().to(device).eval().requires_grad_(False)
        model.load_state_dict(saved['weights'], strict=True)
        models[arm] = model
    return models
