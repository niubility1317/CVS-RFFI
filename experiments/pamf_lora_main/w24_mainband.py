"""Matched native-rate restoration with and without complex mainband fidelity."""
import argparse
from functools import partial
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.rf_noise_transport import colored_observation
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier
from experiments.pamf_lora_main.w24_native_full import DATA_RUN, W24_COMMIT, W24_RUN, classify
from experiments.pamf_lora_main.w24_native_restore_run import SPLIT_SEED, as_native, partition
from experiments.pamf_lora_main.w24_native_restorer import NativeRestorer, waveform_loss
from experiments.pamf_lora_main.w24_native_wiener import predict_pilot, raw_pilot_batch, receipt, score


RUN = '20261125-phase1-lora-w24-native-mainband-m2-r01'
ARMS = ('base', 'mainband')
MODES = ('direct', 'base_half', 'mainband_half')
PILOT_SEED = 2026112572
SEED = 2026112573
STEPS = 3600
BATCH = 6
LR = 2e-4
REFINED_WEIGHT = .15
COARSE_WEIGHT = .03


def private_mainband_risk(predicted, clean):
    """Relative complex chirp-private risk in the original 0–100 kHz band."""
    reference, _, scale = NativeRestorer.frame(clean)
    error = (predicted - clean) / scale
    freq = torch.fft.fftfreq(clean.shape[-1], device=clean.device)
    band = freq.abs() <= .1
    private = []
    for wave in (reference, error):
        variation = wave - wave.mean(1, keepdim=True)
        private.append(torch.fft.fft(variation, norm='ortho')[..., band])
    target, residual = private
    power = target.abs().square().mean(1, keepdim=True)
    floor = (.03 * power.mean(-1, keepdim=True)).clamp_min(1e-4)
    return (residual.abs().square() / (power + floor)).log1p().mean()


def calibration(rt):
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    contract = source_contract(source)
    packets, _, ids, _ = partition(source, contract)
    values = []
    with torch.no_grad():
        for ix in np.array_split(ids, 50):
            x = as_native(packets[ix], 'cpu')
            values.append(corruption_score(NativeRestorer.frame(x)[0]))
    threshold = float(torch.quantile(torch.cat(values), .999))
    if not np.isfinite(threshold) or threshold <= 0.:
        raise ValueError('Invalid source calibration')
    write(root / 'calibration.json', dict(status='SOURCE_FROZEN', protocol=contract['protocol'],
         n=len(ids), quantile=.999, threshold=threshold, split_seed=SPLIT_SEED,
         F_reads=0, V_reads=0, TX_reads=0, target_reads=0))


def fit(runtime, arm):
    rt = read(runtime)
    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    gate = read(root / 'calibration.json')
    if gate['status'] != 'SOURCE_FROZEN' or gate['protocol'] != contract['protocol']:
        raise ValueError('Matching source-calibrated gate required')
    packets, ids, _, held = partition(source, contract)
    out = root / arm
    out.mkdir()
    torch.manual_seed(SEED)
    rng = torch.Generator(device=device).manual_seed(SEED + 1)
    choices = np.random.default_rng(SEED)
    model = NativeRestorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=2e-5)
    t0 = time.perf_counter()
    with (out / 'steps.jsonl').open('x', encoding='utf-8') as log:
        for step in range(1, STEPS + 1):
            clean = as_native(packets[choices.choice(ids, BATCH, replace=False)], device)
            y = clean if step % 5 == 0 else colored_observation(clean, rng)
            optimizer.zero_grad(set_to_none=True)
            coarse, refined = model.stages(y)
            fit_loss = waveform_loss(coarse, refined, clean)
            extra = fit_loss.new_zeros(())
            if arm == 'mainband':
                extra = (REFINED_WEIGHT * private_mainband_risk(refined, clean)
                         + COARSE_WEIGHT * private_mainband_risk(coarse, clean))
            loss = fit_loss + extra
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native mainband objective')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            row = dict(step=step, waveform=float(fit_loss.detach()), mainband=float(extra.detach()),
                       total=float(loss.detach()), grad=float(gradient))
            log.write(json.dumps(row, allow_nan=False) + '\n')
            if step == 1 or step % 100 == 0:
                log.flush()
                print('W24_MAINBAND_FIT', arm, json.dumps(row), flush=True)
    torch.save(model.state_dict(), out / 'G.pt')
    model.eval()
    diagnostic, active = [], 0
    rng.manual_seed(SEED + 2)
    with torch.no_grad():
        for ix in np.array_split(held, 30):
            clean = as_native(packets[ix], device)
            received = colored_observation(clean, rng)
            corrected = model.restore(received, gate['threshold'], amount=.5)
            diagnostic.append((len(ix), float(private_mainband_risk(received, clean)),
                               float(private_mainband_risk(corrected, clean))))
            active += int((received != corrected).flatten(1).any(1).sum())
    values = np.asarray(diagnostic)
    write(out / 'frozen.json', dict(status='SOURCE_FROZEN', run=RUN, arm=arm, steps=STEPS, seed=SEED,
          source_contract=contract, parameters=sum(x.numel() for x in model.parameters()),
          threshold=gate['threshold'], n_fit=len(ids), n_held=len(held), held_activated=active,
          held_private_risk_received=float(np.average(values[:, 1], weights=values[:, 0])),
          held_private_risk_half=float(np.average(values[:, 2], weights=values[:, 0])),
          seconds=time.perf_counter()-t0, F_reads=0, TX_reads=0, V_reads=0, target_reads=0))


def render(wave, models, threshold):
    prepared = wave.reshape(len(wave), 8, 1024)
    aligned, quality, scale = NativeRestorer.frame(prepared)
    active = corruption_score(aligned) > threshold
    outputs = []
    if not active.any():
        return tuple(wave for _ in models), active
    for model in models:
        _, refined = model(aligned[active], quality[active])
        result = prepared.clone()
        result[active] = torch.lerp(prepared[active], refined * scale[active], .5)
        outputs.append(result.reshape_as(wave))
    return tuple(outputs), active


def evaluate(runtime):
    rt = read(runtime)
    activate(rt)
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract = source_contract(source)
    threshold = read(root / 'calibration.json')['threshold']
    models = []
    for arm in ARMS:
        path = root / arm
        c = read(path / 'frozen.json')
        if (c['status'] != 'SOURCE_FROZEN' or c['arm'] != arm or c['source_contract'] != contract
                or c['run'] != RUN or c['threshold'] != threshold or c['steps'] != STEPS or c['target_reads']
                or c['F_reads'] or c['TX_reads'] or c['V_reads']):
            raise ValueError('Native ablation checkpoint provenance differs')
        model = NativeRestorer().to(device).eval().requires_grad_(False)
        model.load_state_dict(torch.load(path / 'G.pt', map_location=device, weights_only=True), strict=True)
        models.append(model)
    classifier = load_classifier(rt, contract, device)
    packets, _, _, held = partition(source, contract)
    clean = as_native(packets[held[:4]], device)
    rng = torch.Generator(device=device).manual_seed(SEED + 5)
    noisy = colored_observation(clean, rng)
    with torch.no_grad():
        for q in (clean, noisy):
            candidates, mask = render(q.flatten(1), models, threshold)
            for x in candidates:
                if not torch.equal(x[~mask], q.flatten(1)[~mask]) or not torch.isfinite(x).all():
                    raise ValueError('Native ablation smoke violates finite or bypass requirement')
                classify(classifier, np.from_dlpack(x.cpu().contiguous()), device)
        if not mask.any():
            raise ValueError('Real source smoke did not exercise the native restoration path')
    write(root / 'smoke.json', dict(status='PASS', source_packets=4, target_reads=0))
    apply = partial(render, models=models, threshold=threshold)
    with torch.no_grad():
        predict_pilot(rt, partial(raw_pilot_batch, classifier, device, apply),
                      modes=MODES, seed=PILOT_SEED, log_prefix='W24_MAINBAND_PILOT')


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    if read(reference / 'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Exact W24 source release required')
    source_runtime = root / 'source-runtime.json'
    write(source_runtime, {k: rt[k] for k in ('run_root', 'source_root', 'frozen_sources')})
    write(root / 'state.json', dict(status='SOURCE_CALIBRATION'))
    source = read(source_runtime)
    activate(source)
    calibration(source)
    write(root / 'state.json', dict(status='WAIT_GPU'), exclusive=False)
    deadline = time.monotonic() + 1800
    allowed = rt['allowed_gpus']
    while not (cards := available(allowed, max_compute_processes=8)):
        if time.monotonic() >= deadline:
            write(root / 'state.json', dict(status='FAILED', error='GPU wait expired'), exclusive=False)
            return
        time.sleep(20)
    cards = [g for g in allowed if g in cards][:2]
    script = str(Path(__file__).resolve())

    def worker(gpu, phase, config, log_name, *extra):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                   PYTHONPATH=str(reference))
        with (root / log_name).open('x', encoding='utf-8') as log:
            return subprocess.Popen([rt['python'], '-u', script, phase, str(config), *extra],
                                    cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT)

    pending = list(ARMS)
    while pending:
        stage = {}
        for gpu in cards:
            if pending:
                arm = pending.pop(0)
                stage[arm] = (gpu, worker(gpu, 'fit', source_runtime, f'fit-{arm}.log', arm))
        write(root / 'state.json',
              dict(status='RUNNING', phase='fit', workers={a: {'pid': p.pid, 'gpu': g} for a, (g, p) in stage.items()}),
              exclusive=False)
        results = {name: proc.wait() for name, (_, proc) in stage.items()}
        if any(results.values()):
            write(root / 'state.json', dict(status='FAILED', error='fit', exit_codes=results), exclusive=False)
            return
    p = worker(cards[0], 'evaluate', runtime, 'evaluate.log')
    write(root / 'state.json', dict(status='RUNNING', phase='evaluate', pid=p.pid, gpu=cards[0]), exclusive=False)
    if p.wait():
        write(root / 'state.json', dict(status='FAILED', error='evaluate'), exclusive=False)
        return
    if read(root / 'predictions' / 'complete.json') != receipt(MODES, PILOT_SEED):
        raise ValueError('Frozen native ablation pilot matrix differs')
    write(root / 'state.json', dict(status='PREDICTIONS_FROZEN'), exclusive=False)


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'fit', 'evaluate'):
        cmd = sub.add_parser(phase)
        cmd.add_argument('runtime', type=Path)
        if phase == 'fit':
            cmd.add_argument('arm', choices=ARMS)
    pub = sub.add_parser('publish')
    for key in ('repo', 'context', 'ssh-config', 'output'):
        pub.add_argument('--' + key, type=Path, required=True)
    scoring = sub.add_parser('score')
    for key in ('predictions', 'prior', 'dataset', 'output'):
        scoring.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k: ctx[k] for k in ('project', 'python', 'frozen_sources')}
        project = PurePosixPath(rt['project'])
        data = project / 'releases' / f'{DATA_RUN}-release' / 'data'
        rt.update(run_id=RUN, code_commit=release_commit(args.repo), runner=__spec__.name,
                  release_root=str(project / 'releases' / f'{RUN}-release'),
                  run_root=str(project / 'runs' / RUN),
                  source_root=str(data / 'source'), native_target_root=str(data / 'target'),
                  classifier_root=str(project / 'runs' / AUG_RUN),
                  w24_release_root=str(project / 'releases' / f'{W24_RUN}-release'),
                  allowed_gpus=[2, 7, 4], cpu_workers=8)
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'score':
        score(args.predictions, args.prior, args.dataset, args.output, modes=MODES, seed=PILOT_SEED,
              contrast=('mainband_half', 'base_half'))
    elif args.action == 'fit':
        fit(args.runtime, args.arm)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    else:
        evaluate(args.runtime)


if __name__ == '__main__':
    main()
