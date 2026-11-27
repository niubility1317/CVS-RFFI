"""Source fitting and deployment readout for the matched spectral block order."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .joint_unroll_source import coordinates
from .w24_aperture_noise_audit import CENTER, center_rms, model_iq, native_aperture
from .w24_component_directions import matched_component_edit
from .w24_fixed_dose_direction import BATCH, FIT_PER_RX, ISOLATION, SEED, dtype_policy, source_parent_and_tracker
from .w24_fixed_dose_direction_model import ApertureDirection
from .w24_post_peer_refinement import PostPeerRefiner, endpoint_risk
from .w24_practical_supervision import training_sets
from .w24_regional_dose_source import complex_tensor
from ..cric_dura_comparison.io import read, tensor, write


RUN = '20261128-phase1-lora-w24-post-peer-refinement-s2026112991-r01'
STEPS = 24000
ARMS = ('frequency_first', 'peer_first')
MODES = ('direct', 'frozen_half', *ARMS)
CONTRASTS = ((ARMS[1], ARMS[0]), (ARMS[0], 'frozen_half'), (ARMS[1], 'frozen_half'))


@torch.no_grad()
def prepare(aperture, parent, tracker, direction, wave):
    half = torch.lerp(wave, parent.restore_native(wave), .5)
    observed = aperture.project(wave)
    base = aperture.project(half) - observed
    _, phase, quality, _ = coordinates(tracker, aperture.normalize(observed))
    correction = direction(wave, half - wave)[1]
    proposal = aperture.project(wave + correction) - observed
    prior = matched_component_edit(base, proposal, phase)
    return (observed, prior, phase, quality), half


def frozen_refiner_inputs(rt, device):
    from .w24_peer_spectral_direction_source import COMPONENT_RUN, load_trained

    parent, tracker = source_parent_and_tracker(rt, device)
    direction = load_trained(Path(rt['project'])/'runs'/COMPONENT_RUN).to(device)
    return ApertureDirection().to(device), parent, tracker, direction


def evaluate(inputs, arms, batch, *, identity):
    aperture, *_ = inputs
    clean, received, source = batch.unbind(1)
    noisy = torch.ones(len(batch), dtype=torch.bool, device=batch.device)
    if identity:
        received = torch.cat((received[:-2], clean[-2:]))
        noisy[-2:] = False
    with torch.no_grad():
        frame, _ = prepare(*inputs, received)
        target = aperture.project(source)
    observed, prior, phase, _ = frame
    return [endpoint_risk(observed, target, prior, model(*frame), phase, noisy) for model in arms]


def fit(rt):
    from .w24_peer_spectral_direction_source import COMPONENT_RUN, fit_config

    fit_config(rt)
    root = Path(rt['run_root'])
    if root.name != RUN:
        raise ValueError('Expected an immutable post-peer fit root')
    device = 'cuda:0'
    dtype_policy()
    inputs = frozen_refiner_inputs(rt, device)
    torch.manual_seed(SEED)
    arms = (PostPeerRefiner(peer_first=False), PostPeerRefiner(peer_first=True))
    arms[1].load_state_dict(arms[0].state_dict(), strict=True)
    for model in arms:
        model.to(device)
    options = [torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4) for m in arms]
    schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(o, STEPS, eta_min=2e-5) for o in options]

    (probe, _, _), _ = training_sets(rt, device, smoke=True, fit_per_rx=FIT_PER_RX, bank_per_rx=0)
    for model, (loss, _) in zip(arms, evaluate(inputs, arms, probe[:BATCH], identity=True), strict=True):
        if not torch.isfinite(loss):
            raise ValueError('Real isolated source smoke produced a nonfinite endpoint loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        model.zero_grad(set_to_none=True)
    write(root/'fit-smoke.json', dict(status='PASS', source_role='U_s', model='post_peer_refinement', **ISOLATION))
    del probe, loss

    (training, _, held), _ = training_sets(rt, device, smoke=False, fit_per_rx=FIT_PER_RX, bank_per_rx=0)
    if (len(training), len(held)) != (20000, 500):
        raise ValueError('Unexpected fit and held physical source counts')
    rng = np.random.default_rng(SEED+1)
    started = perf_counter()
    history = {}
    for step in range(1, STEPS+1):
        indices = tensor(rng.choice(len(training), BATCH, replace=False), device)
        rows = evaluate(inputs, arms, training[indices], identity=True)
        for model, optimizer, schedule, (loss, _) in zip(arms, options, schedules, rows, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
        if step == 1 or step % 200 == 0:
            print('W24_POST_PEER_FIT', step, [{k:float(v) for k,v in report.items()} for _,report in rows], flush=True)
        if step in (2000, 6000, 12000, 18000, STEPS):
            for model in arms:
                model.eval()
            with torch.no_grad():
                reports = [(len(chunk), evaluate(inputs, arms, chunk, identity=False)) for chunk in held.split(BATCH)]
            history[str(step)] = [{k:sum(n*float(row[i][1][k]) for n,row in reports)/len(held)
                                   for k in reports[0][1][i][1]} for i in range(len(arms))]
            print('W24_POST_PEER_HELD', step, history[str(step)], flush=True)
            for model in arms:
                model.train()

    metadata = dict(status='SOURCE_FROZEN', base=COMPONENT_RUN, arms=list(ARMS), steps=STEPS,
                    model_seed=SEED, batch=BATCH, fit=len(training), held=len(held),
                    parameters=sum(p.numel() for p in arms[0].parameters()), seconds=perf_counter()-started,
                    peak_cuda=torch.cuda.max_memory_allocated(), held_trajectory=history, **ISOLATION)
    with (root/'refinement.pt').open('xb') as out:
        torch.save(dict(metadata=metadata, weights=[m.state_dict() for m in arms]), out)
    write(root/'fit-complete.json', metadata)


def verify_frozen(root):
    from .w24_peer_spectral_direction_source import COMPONENT_RUN

    metadata = read(root/'fit-complete.json')
    expected = dict(status='SOURCE_FROZEN', base=COMPONENT_RUN, arms=list(ARMS), steps=STEPS,
                    model_seed=SEED, batch=BATCH, fit=20000, held=500, **ISOLATION)
    if root.name != RUN or any(metadata.get(key) != value for key,value in expected.items()):
        raise ValueError('Unexpected source-only post-peer freeze')
    return metadata


def load_arms(root):
    evidence = verify_frozen(root)
    checkpoint = torch.load(root/'refinement.pt', map_location='cpu', weights_only=True)
    if checkpoint['metadata'] != evidence:
        raise ValueError('Checkpoint is not bound to the registered source freeze')
    models = tuple(PostPeerRefiner(peer_first=i == 1).eval().requires_grad_(False) for i in range(len(ARMS)))
    for model, weights in zip(models, checkpoint['weights'], strict=True):
        model.load_state_dict(weights, strict=True)
    return models


@torch.no_grad()
def receive(inputs, models, raw):
    wave = complex_tensor(raw)
    frame, half = prepare(*inputs, wave)
    original, retained = (native_aperture(np.from_dlpack(x.contiguous())) for x in (wave, half))
    out = [original, retained]
    for model in models:
        view = retained.copy()
        view[..., :CENTER] = np.from_dlpack((frame[0] + model(*frame)).contiguous())
        out.append(view)
    return tuple(tensor(model_iq(value, center_rms(value)), 'cpu') for value in out)
