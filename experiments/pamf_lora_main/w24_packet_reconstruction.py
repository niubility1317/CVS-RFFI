"""From-scratch time/spectrum packet reconstruction on lawful LoRa source IQ."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import importlib
from pathlib import Path, PurePosixPath
import sys
from time import monotonic
from types import ModuleType

import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.w24_edit_budget_audit import FULL, score
from experiments.pamf_lora_main.w24_edit_cross import check_controls
from experiments.pamf_lora_main.w24_factor_audit import load_classifier, panel
from experiments.pamf_lora_main.w24_native_full import SOURCE, VIEWS, classify_aperture
from experiments.pamf_lora_main.w24_native_residual_source import (
    AUG_RUN, BATCH, DATA_RUN, TRAIN_RUN, W24_RUN, dispatch,
)
from experiments.pamf_lora_main.w24_observed_dose import source_receive
from experiments.pamf_lora_main.w24_spectral_cov import CENTER, CHIRPS, PARENT, fit_examples
from experiments.pamf_lora_main.w24_wide_transfer import received_aperture, training_runtime


RUN = '20261126-phase1-lora-w24-packet-reconstruction-s2026112991-r01'
SEED, STEPS, FIT_PACKETS, HOLD_PACKETS, FIT_BATCH = 2026112991, 1200, 8192, 160, 8
EDGES, TOKENS, WIDTH = 48, 24, 96
MODES = ('direct', 'per_chirp_center', 'per_chirp_full', 'packet_center', 'packet_full')
CONTRASTS = (('packet_full', 'per_chirp_full'), ('packet_center', 'per_chirp_center'),
             ('packet_full', 'packet_center'), ('packet_full', 'direct'))


def complex_channels(z):
    return torch.stack((z.real, z.imag), dim=-2)


class Mixing(nn.Module):
    def __init__(self):
        super().__init__()
        self.normal = nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, 4, batch_first=True)
        self.feed = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 2*WIDTH), nn.GELU(),
                                  nn.Linear(2*WIDTH, WIDTH))

    def forward(self, x, mask):
        y = self.normal(x)
        x = x + self.attention(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.feed(x)


class PacketReconstructor(nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint
        self.front = nn.Sequential(nn.Conv1d(4, 48, 9, padding=4), nn.SiLU(),
                                   nn.Conv1d(48, 48, 5, padding=2), nn.SiLU())
        self.edges = nn.Sequential(nn.Conv1d(2, 16, 5, padding=2), nn.SiLU())
        self.down = nn.Conv1d(48, WIDTH, 16, stride=16)
        self.edge_token = nn.Linear(16, WIDTH)
        self.position = nn.Parameter(.02 * torch.randn(1, CHIRPS, TOKENS, WIDTH))
        self.mixer = nn.ModuleList(Mixing() for _ in range(2))
        self.decoder = nn.Sequential(nn.Conv1d(WIDTH + 48, 64, 7, padding=3), nn.SiLU(),
                                     nn.Conv1d(64, 32, 5, padding=2), nn.SiLU(),
                                     nn.Conv1d(32, 2, 3, padding=1))
        self.spectral = nn.Sequential(nn.Conv1d(2 + WIDTH, 64, 7, padding=3), nn.SiLU(),
                                      nn.Conv1d(64, 32, 5, padding=2), nn.SiLU(),
                                      nn.Conv1d(32, 2, 3, padding=1))
        self.boundary = nn.Sequential(nn.Conv1d(16 + WIDTH, 32, 5, padding=2), nn.SiLU(),
                                      nn.Conv1d(32, 2, 3, padding=1))
        self.mix = nn.Linear(WIDTH, 1)
        chirp = torch.arange(CHIRPS).repeat_interleave(TOKENS)
        self.register_buffer('mask', chirp[:, None] != chirp, persistent=False)
        for head in (self.decoder[-1], self.spectral[-1], self.boundary[-1]):
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)

    def forward(self, center, edges, quality):
        b = len(center)
        z = center.flatten(0, 1)
        edge = edges.flatten(0, 1)
        spectrum = torch.fft.fft(z, norm='ortho')
        side = spectrum[:, CENTER//4:3*CENTER//4].abs().square().median(-1).values.clamp_min(1e-9)
        state = torch.stack((quality.reshape(-1), side.log().clamp(-14., 3.)/7), dim=1)
        features = torch.cat((complex_channels(z), state[..., None].expand(-1, -1, CENTER)), dim=1)
        local = self.front(features)
        edge_parts = edge.reshape(b*CHIRPS*2, EDGES//2)
        edge_features = self.edges(complex_channels(edge_parts)).reshape(b*CHIRPS, 2, 16, EDGES//2)
        edge_summary = self.edge_token(edge_features.mean((1, 3))).unsqueeze(1)
        tokens = self.down(local).transpose(1, 2) + edge_summary
        tokens = (tokens.reshape(b, CHIRPS, TOKENS, WIDTH) + self.position).flatten(1, 2)
        mask = None if self.joint else self.mask
        for block in self.mixer:
            tokens = block(tokens, mask)
        tokens = tokens.reshape(b*CHIRPS, TOKENS, WIDTH)
        summary = tokens.mean(1)
        context = interpolate(tokens.transpose(1, 2), size=CENTER, mode='linear', align_corners=False)
        i, q = self.decoder(torch.cat((local, context), dim=1)).unbind(1)
        predicted = z + torch.complex(i, q)
        spec = torch.fft.fftshift(spectrum, dim=-1)
        a, c = self.spectral(torch.cat((complex_channels(spec),
                                      summary[..., None].expand(-1, -1, CENTER)), dim=1)).unbind(1)
        noise = torch.fft.ifft(torch.fft.ifftshift(torch.complex(a, c), dim=-1), norm='ortho')
        observed_projection = z - noise
        weight = self.mix(summary).sigmoid()
        final = weight * predicted + (1.-weight) * observed_projection
        edge_context = summary[:, None, :, None].expand(-1, 2, -1, EDGES//2)
        outer = torch.cat((edge_features, edge_context), dim=2).flatten(0, 1)
        i, q = self.boundary(outer).unbind(1)
        reconstructed_edge = edge + torch.complex(i, q).reshape_as(edge)
        return tuple(x.reshape(b, CHIRPS, -1) for x in (final, predicted, observed_projection, reconstructed_edge))


def source_tools(rt, device):
    activate(rt)
    archive, _ = training_runtime(rt)
    package = ModuleType('lora_packet_front_end')
    package.__path__ = [str(archive/'experiments'/'pamf_lora_main')]
    sys.modules[package.__name__] = package
    imports = importlib.import_module('experiments').__path__
    imports.insert(0, str(archive/'experiments'))
    try:
        helper = importlib.import_module(f'{package.__name__}.dura_w24_model')
    finally:
        imports.pop(0)
    return helper.WideTracker().to(device).eval(), helper.condition_waveforms


def receiver_frame(tracker, received, guard):
    center, phase, quality, rms = coordinates(tracker, real_iq(received))
    guard = guard / rms
    return (center, guard, quality), phase*rms, rms


def risks(output, target, guard):
    final, prior, consistency, edge = output
    residual = final - target
    e = torch.fft.fft(residual, norm='ortho')
    freq = torch.arange(CENTER, device=e.device)
    near = (freq >= 1) & (freq <= 32) | (freq >= CENTER-32)
    weight = torch.where(near, 4., 1.)
    # Preserve the narrow complex detail as well as the packet-average wave.
    detail = (e.abs().square() * weight).mean()
    differences = (e[:, 1:] - e[:, :-1]).abs().square().mean()
    value = (residual.abs().square().mean(), detail,
             (prior-target).abs().square().mean(), (consistency-target).abs().square().mean(),
             (edge-guard).abs().square().mean(), differences)
    return torch.stack(value)


def training_pair(raw, tracker, generic, rng, device):
    observed, _ = fit_examples(raw, tracker, generic, rng, device)
    count = len(raw)
    source = observed[count:]
    received = torch.cat((observed[:count-1], source[-1:]))
    x, rotation, rms = receiver_frame(tracker, received[..., :CENTER], received[..., CENTER:])
    return x, source[..., :CENTER]/rotation, source[..., CENTER:]/rms


def fit(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    tracker, generic = source_tools(rt, device)
    contract = source_contract(source)
    iq = native(source, 'U_s', contract['counts']['U_s'])
    sampling = np.random.default_rng(SEED)
    panel = sampling.choice(len(iq), FIT_PACKETS+HOLD_PACKETS, replace=False)
    torch.manual_seed(SEED)
    models = [PacketReconstructor(joint).to(device) for joint in (False, True)]
    models[1].load_state_dict(models[0].state_dict())
    opts = [torch.optim.AdamW(model.parameters(), lr=.0006) for model in models]
    rng = torch.Generator(device=device).manual_seed(SEED)
    with torch.no_grad():
        args, clean, guards = training_pair(iq[panel[:2]], tracker, generic, rng, device)
        if any(not torch.isfinite(risks(model(*args), clean, guards)).all() for model in models):
            raise ValueError('Source-only reconstruction smoke failed')
    write(root/'fit-smoke.json', dict(status='PASS', F_reads=0, target_reads=0, source_u_packets=2))
    print('W24_PACKET_RECONSTRUCTION_FIT_SMOKE PASS', flush=True)
    started = monotonic()
    torch.cuda.reset_peak_memory_stats()
    logs = [[], []]
    for step in range(STEPS):
        idx = sampling.choice(panel[:FIT_PACKETS], FIT_BATCH, replace=False)
        with torch.no_grad():
            args, ground, guard = training_pair(iq[idx], tracker, generic, rng, device)
        for log, model, opt in zip(logs, models, opts, strict=True):
            value = risks(model(*args), ground, guard)
            loss = value @ value.new_tensor((1., .65, .2, .2, .2, .15))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            if (step+1) % 100 == 0:
                row = value.detach().tolist()
                if not np.isfinite(row).all():
                    raise ValueError('Nonfinite source reconstruction risk')
                log.append(row)
        if (step+1) % 100 == 0:
            print('W24_PACKET_RECONSTRUCTION_STEP', step+1, [log[-1] for log in logs], flush=True)
    seconds = monotonic()-started
    peak = torch.cuda.max_memory_allocated()/1024**2
    heldout = [[], []]
    with torch.no_grad():
        for start in range(0, HOLD_PACKETS, FIT_BATCH):
            idx = panel[FIT_PACKETS+start:FIT_PACKETS+start+FIT_BATCH]
            args, clean, guards = training_pair(iq[idx], tracker, generic, rng, device)
            for log, model in zip(heldout, models, strict=True):
                log.append(risks(model(*args), clean, guards).tolist())
    torch.save([model.cpu().state_dict() for model in models], root/'packet-reconstruction.pt')
    write(root/'fit-complete.json', dict(status='SOURCE_FROZEN', initialization='scratch', seed=SEED, steps=STEPS,
          source_fit_packets=FIT_PACKETS, source_holdout_packets=HOLD_PACKETS, F_reads=0, target_reads=0,
          source_truth_reads=0, source_V_reads=0, peak_cuda_allocated_MiB=peak, seconds=seconds,
          heldout=[np.mean(log, axis=0).tolist() for log in heldout],
          parameters=sum(v.numel() for v in models[0].parameters())))


@torch.no_grad()
def render(f, tracker, models, raw, device):
    original = received_aperture(raw, device)
    center, edge = (complex_iq(x) for x in (original[..., :CENTER], original[..., CENTER:]))
    args, rotation, rms = receiver_frame(tracker, center, edge)
    labels = [classify_aperture(f, original)]
    for model in models:
        fit_center, _, _, fit_edge = model(*args)
        wave = original.clone()
        wave[..., :CENTER] = real_iq(fit_center * rotation)
        labels.append(classify_aperture(f, wave))
        wave[..., CENTER:] = real_iq(fit_edge * rms)
        labels.append(classify_aperture(f, wave))
    return np.stack(labels)


def predict(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    proof = read(root/'fit-complete.json')
    if (proof['status'], proof['initialization'], proof['steps']) != ('SOURCE_FROZEN', 'scratch', STEPS) or any(
            proof[k] for k in ('F_reads', 'target_reads', 'source_truth_reads', 'source_V_reads')):
        raise ValueError('Both scratch source models must be frozen before W24 loads')
    models = tuple(PacketReconstructor(joint).to(device).eval() for joint in (False, True))
    states = torch.load(root/'packet-reconstruction.pt', map_location=device, weights_only=True)
    for model, state in zip(models, states, strict=True):
        model.load_state_dict(state)
    predict_panel(rt, models, MODES)


def predict_panel(rt, models, modes):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    tracker, _ = source_tools(rt, device)
    contract = source_contract(source)
    if len(modes) != 1+2*len(models):
        raise ValueError('Each model needs center and full readouts')
    f = load_classifier(rt, contract, device)
    packet = native(source, 'U_s', contract['counts']['U_s'])[0]
    raw = np.stack([source_receive(view, (0, SOURCE[0], packet)) for view in ('clean', VIEWS[-1])])
    if render(f, tracker, models, raw, device).shape != (len(modes), 2):
        raise ValueError('Frozen W24 source smoke failed')
    write(root/'smoke.json', dict(status='PASS', source_u_packets=1, source_truth_reads=0, target_reads=0))
    print('W24_PACKET_RECONSTRUCTION_SMOKE PASS', flush=True)

    ix, rx, ids = panel(source, contract)
    raw = native(source, 'V', contract['counts']['V'])[ix]
    jobs = [(int(i), SOURCE[int(j)], packet) for i, j, packet in zip(ix, rx, raw, strict=True)]
    matrix = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for view in VIEWS:
            stream = pool.map(partial(source_receive, view), jobs, chunksize=6)
            results = []
            for off in range(0, len(jobs), BATCH):
                raw = np.stack([next(stream) for _ in range(min(BATCH, len(jobs)-off))])
                results.append(render(f, tracker, models, raw, device))
            matrix[view] = np.concatenate(results, axis=1)
            print('W24_PACKET_RECONSTRUCTION_PREDICTED', view, len(jobs), flush=True)
    archive = root.parent/PARENT/FULL
    check_controls(archive, ('direct',), [0], ids, matrix)
    with np.load(archive/'predictions.npz', allow_pickle=False) as old:
        k = read(archive/'source-predictions.json')['modes'].index('outer_objective_half')
        matrix = {v: np.concatenate((matrix[v], old[v][[k]])) for v in VIEWS}
    out = root/FULL
    out.mkdir()
    np.savez(out/'predictions.npz', **matrix)
    write(out/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', modes=[*modes, 'outer_half'],
          views=list(VIEWS), physical_ids=ids, target_reads=0, source_truth_reads=0,
          waveform=dict(route=FULL, scratch=True, archived_outer_control=True)))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    phases = ('fit-source', 'predict-source', 'score-source')
    for phase in ('dispatch', *phases):
        subs.add_parser(phase).add_argument('runtime', type=Path)
    pub = subs.add_parser('publish')
    for field in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__, phases=phases, gpu_phases=phases[:2])
    elif args.action == 'fit-source':
        fit(read(args.runtime))
    elif args.action == 'predict-source':
        predict(read(args.runtime))
    elif args.action == 'score-source':
        score(args.runtime, modes=(*MODES, 'outer_half'),
              contrasts=(*CONTRASTS, ('packet_full','outer_half'), ('packet_center','outer_half')))
    else:
        ctx = read(args.context)
        rt = {field:ctx[field] for field in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, releases = project/'runs', project/'releases'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'), classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[7, 3, 0, 6], cpu_workers=8)
        rt['external_releases'] = [rt[field] for field in ('source_root','classifier_root','w24_release_root',
                                                           'training_release_root')]
        rt['external_releases'].append(str(runs/PARENT))
        release(args.repo, args.output, rt, args.ssh_config)


if __name__ == '__main__':
    main()
