"""Full-only W24 recovery with receive-estimated, packet-local readout strength."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from time import monotonic, sleep

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_repair import complex_iq, real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract, target_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_factor_audit import frozen_source_panel, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_formal import FULL, RUN_OUTER, append_blind_prediction, cases, receipt
from experiments.pamf_lora_main.w24_native_full import (
    BATCH as FORMAL_BATCH, LIMIT, SOURCE, VIEWS, W24_COMMIT, channel_job, classify_aperture, native_full_view,
)
from experiments.pamf_lora_main.w24_native_residual_source import BATCH as SOURCE_BATCH, WAVEFORM_NAMESPACE, ungated_latency
from experiments.pamf_lora_main.w24_outer_objective_source import RUN as PARENT, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import load_restorer, received_aperture, training_runtime


RUN = '20261126-phase1-lora-w24-observed-dose-s2026092702-r01'
OUTER_EVAL_COMMIT = '91fa158ab622f9029f8ec170db4740a9972bb8ab'
CANDIDATE = 'observed_dose'
MODES = ('direct', 'outer_half', CANDIDATE)
FULL_CASES = tuple(k for k, route in cases().items() if route == FULL)
SIX_GAIN, LOW_GAIN = 18, 10


def readout_gain(variance, rms):
    power = rms.square()
    noise = (variance * power).sum(dim=-2, keepdim=True) / power.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    return .5 + noise / (noise + .275**2)


def source_qualifies(scores):
    actual, baseline = scores[CANDIDATE], scores['outer_half']
    return (actual['six_correct'] >= baseline['six_correct'] + SIX_GAIN
            and actual['low_urban_correct'] >= baseline['low_urban_correct'] + LOW_GAIN
            and actual['by_scene']['clean'] >= scores['direct']['by_scene']['clean'])


@torch.no_grad()
def edits(model, tracker, observed):
    y, phase, quality, rms = coordinates(tracker, observed[..., :384])
    stages, _, _, variance, _ = model.propose(y, quality, rms, complex_iq(observed[..., 384:]))
    delta = real_iq((stages[-1] - y) * phase * rms)
    gain = readout_gain(variance, rms)
    return .5*delta, delta*gain.unsqueeze(-2), gain


@torch.no_grad()
def predict_batch(f, model, tracker, raw, device):
    observed = received_aperture(raw, device)
    half, adjusted, gain = edits(model, tracker, observed)
    rows = [classify_aperture(f, observed)]
    for delta in (half, adjusted):
        restored = observed.clone()
        restored[..., :384] += delta
        rows.append(classify_aperture(f, restored))
    return np.stack(rows), np.asarray(gain.flatten().tolist())


def load_models(rt, device):
    activate(rt)
    contract = source_contract(Path(rt['source_root']))
    f = load_classifier(rt, contract, device)
    _, tracker, _, _ = load_restorer(*training_runtime(rt), contract, device)
    parent = Path(rt['project'])/'runs'/PARENT
    g, summary = load_candidate(parent, contract, device)
    return f, g, tracker, parent, contract, summary


def source_receive(scene, item):
    row, rx, packet = item
    return packet if scene == 'clean' else native_full_view(packet, scene, rx, row, WAVEFORM_NAMESPACE)


def batches(f, g, tracker, waveforms, size, device, batch_size):
    decisions, gains = [], []
    for i in range(0, size, batch_size):
        raw = np.stack([next(waveforms) for _ in range(min(batch_size, size-i))])
        y, dose = predict_batch(f, g, tracker, raw, device)
        decisions.append(y)
        gains.append(dose)
    return np.concatenate(decisions, axis=1), np.concatenate(gains)


def predict_source(rt):
    device = 'cuda:0'
    root, source = Path(rt['run_root']), Path(rt['source_root'])
    f, g, tracker, parent, contract, info = load_models(rt, device)
    u = native(source, 'U_s', contract['counts']['U_s'])[0]
    clean, disturbed = (source_receive(v, (0, SOURCE[0], u)) for v in ('clean', VIEWS[-1]))
    raw = np.stack((clean, disturbed))
    labels, gain = predict_batch(f, g, tracker, raw, device)
    if labels.shape != (3,2) or not (np.isfinite(gain).all() and ((gain >= .5) & (gain <= 1.5)).all()):
        raise ValueError('Actual source-only W24 checkpoint smoke failed')
    observed = received_aperture(raw[:1], device)
    b1 = ungated_latency(g, tracker, observed, render=edits)
    write(root/'smoke.json', dict(status='PASS', source_packets=1, target_reads=0,
          source_truth_reads=0, parent=PARENT, steps=info['steps'], parameters=info['parameters'], b1_ms=b1))

    ix, rx, ids = panel(source, contract)
    prior = parent/FULL
    proof = read(prior/'source-predictions.json')
    expected = ('PREDICTIONS_FROZEN', ids, list(VIEWS), 0, 0)
    if tuple(proof[k] for k in ('status','physical_ids','views','source_truth_reads','target_reads')) != expected:
        raise ValueError('Prior Full source matrix physical IDs differ')
    comparison = [proof['modes'].index(m) for m in ('direct','outer_objective_half')]
    jobs = [(int(i), SOURCE[int(r)], w) for i,r,w in zip(ix,rx,native(source,'V',contract['counts']['V'])[ix],strict=True)]
    predictions, distribution = {}, {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool, np.load(prior/'predictions.npz',allow_pickle=False) as old:
        for v in VIEWS:
            stream = pool.map(partial(source_receive,v),jobs,chunksize=6)
            rows, gains = batches(f,g,tracker,stream,len(jobs),device,SOURCE_BATCH)
            if not np.array_equal(rows[:2],old[v][comparison]):
                raise ValueError(f'Archived frozen controls differ for {v}')
            predictions[v], distribution[v] = rows, np.quantile(gains,[.1,.5,.9]).tolist()
            print('W24_OBSERVED_DOSE_SOURCE',v,len(jobs),flush=True)
    out = root/FULL
    out.mkdir()
    np.savez(out/'predictions.npz',**predictions)
    write(out/'source-predictions.json',dict(status='PREDICTIONS_FROZEN',modes=list(MODES),views=list(VIEWS),
          physical_ids=ids,target_reads=0,source_truth_reads=0,waveform=dict(candidate=CANDIDATE,
          parent=PARENT,gain_q10_q50_q90=distribution,route=FULL)))


def score_source_phase(runtime, rt):
    root = Path(rt['run_root'])/FULL
    frozen_source_panel(Path(rt['source_root']),root,MODES)
    score_source(runtime,modes=MODES,output=root,contrasts=((CANDIDATE,'outer_half'),))


def prior_target(rt, case):
    p = Path(rt['project'])
    root = p/'runs'/RUN_OUTER
    release = read(p/'releases'/f'{RUN_OUTER}-release'/'runtime.json')
    state = read(root/'state.json')
    if (release['code_commit'] != OUTER_EVAL_COMMIT or Path(release['run_root']) != root
            or any(release[k] != rt[k] for k in ('source_root','native_target_root','w24_release_root'))
            or state != dict(status='PREDICTIONS_FROZEN',cases=list(cases()),target_truth_read=False)):
        raise ValueError('Expected the exact frozen formal W24 outer-objective reference')
    out = root/'predictions'/case
    if read(out/'complete.json') != receipt(case,True,True):
        raise ValueError('Malformed archived Full reference shard')
    return out


def formal_receipt(case):
    if case not in FULL_CASES:
        raise ValueError('Only Full practical is authorized')
    old = receipt(case,True,True)
    return dict(old,modes=[*old['modes'],CANDIDATE],reference_run=RUN_OUTER,
                candidate_run=RUN)


def predict_formal(rt, case):
    expected = formal_receipt(case)
    root = Path(rt['run_root'])
    frozen = read(root/FULL/'source-scores.json')
    if (frozen['status'] != 'ANALYZED' or not frozen['source_truth_last']
            or frozen['target_reads'] or not source_qualifies(frozen['scores'])
            or read(root/'smoke.json')['status'] != 'PASS'):
        raise ValueError('Registered Full source qualification is required before target')
    device = 'cuda:0'
    older = prior_target(rt,case)
    old_activity = read(older/'activity.json')
    f, g, tracker, *_ = load_models(rt,device)
    target = Path(rt['native_target_root'])
    target_contract(target)
    out = root/'predictions'/case
    out.mkdir(parents=True)
    activity = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers']) as pool:
        for rx in expected['receivers']:
            raw = native(target,rx,LIMIT)
            decisions = {}
            with np.load(older/f'{rx}.npz',allow_pickle=False) as old:
                if set(old.files) != set(VIEWS):
                    raise ValueError('Incomplete archived Full view set')
                for scene in VIEWS:
                    stream = (iter(raw) if scene == 'clean' else
                              pool.map(partial(channel_job,scene,rx),enumerate(raw),chunksize=16))
                    rows, _ = batches(f,g,tracker,stream,LIMIT,device,FORMAL_BATCH)
                    key = f'{rx}/{scene}'
                    if rows.shape != (3,LIMIT) or not np.array_equal(rows[1],old[scene][0,-1]):
                        raise ValueError(f'Archived outer candidate decisions differ for {key}')
                    decisions[scene], activity[key] = append_blind_prediction(
                        old[scene], rows[None,[0,2]], old_activity[key], LIMIT)
                    print('W24_OBSERVED_DOSE_FORMAL',rx,scene,flush=True)
            np.savez(out/f'{rx}.npz',**decisions)
    write(out/'activity.json',activity)
    write(out/'complete.json',expected)


def dispatch(runtime, rt):
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    state = root/'state.json'

    def status(**value):
        write(state,value,exclusive=False)

    def spawn(phase, name, gpu, *args):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   PYTHONPATH=str(reference), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        with (root/f'{name}.log').open('x',encoding='utf-8') as log:
            job = subprocess.Popen([rt['python'],'-u',str(Path(__file__).resolve()),phase,str(runtime),*args],
                                   cwd=reference,env=env,stdout=log,stderr=subprocess.STDOUT)
        print('W24_OBSERVED_DOSE_STARTED',phase,name,job.pid,gpu,flush=True)
        return job

    try:
        if read(reference/'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Expected exact W24 readout archive')
        status(status='WAIT_GPU')
        deadline = monotonic()+600
        while not (cards := available(rt['allowed_gpus'],rt['max_compute_processes'])):
            if monotonic() >= deadline:
                raise RuntimeError('Full source GPU wait expired')
            sleep(15)
        gpu = next(i for i in rt['allowed_gpus'] if i in cards)
        for phase, card in (('predict-source',gpu),('score-source',None)):
            job = spawn(phase,phase,card)
            status(status='RUNNING',phase=phase,pid=job.pid,gpu=card)
            if job.wait():
                raise RuntimeError(f'{phase} failed')
        promote = source_qualifies(read(root/FULL/'source-scores.json')['scores'])
        status(status='SOURCE_SCORED',promote=promote)
        if not promote:
            return
        status(status='TARGET_PREDICTING',route=FULL)
        schedule(rt,[('predict-formal',name) for name in FULL_CASES],
                 lambda _rt, phase, name, gpu: spawn(phase,name,gpu,name))
        for name in FULL_CASES:
            if read(root/'predictions'/name/'complete.json') != formal_receipt(name):
                raise ValueError(f'Missing formal Full shard {name}')
        status(status='PREDICTIONS_FROZEN',cases=list(FULL_CASES),target_truth_read=False)
    except Exception as error:
        status(status='FAILED',error=str(error))
        raise


def collect(args):
    from experiments.pamf_lora_main.w24_formal_score import score_observed_dose

    root = PurePosixPath(read(args.submit)['run_root'])
    if root.name != RUN or not re.fullmatch(r'/[A-Za-z0-9_./-]+',str(root)):
        raise ValueError('Unexpected immutable N607 root')
    connection = ['-F',str(args.ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10']
    end, last = monotonic()+12*60*60,None
    while monotonic() < end:
        state = None
        try:
            remote = subprocess.run(['ssh',*connection,'-T','N607','cat '+str(root/'state.json')],
                                    timeout=25,capture_output=True)
            if remote.returncode == 0:
                state = json.loads(remote.stdout)
        except (subprocess.TimeoutExpired,json.JSONDecodeError):
            pass
        if state is not None:
            if state['status'] != last:
                print(state,flush=True)
                last = state['status']
            if last == 'FAILED':
                raise RuntimeError(state['error'])
            if last == 'SOURCE_SCORED' and not state['promote']:
                return
            if last == 'PREDICTIONS_FROZEN':
                break
        sleep(40)
    else:
        raise TimeoutError('Full frozen matrix not available before the scorer deadline')
    args.output.mkdir(parents=True)
    subprocess.run(['scp',*connection,'-r','N607:'+str(root/'predictions'),str(args.output)],
                   check=True,timeout=180)
    score_observed_dose(args.output/'predictions',args.dataset,args.output/'scores')


def main():
    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_lora_main.w24_native_residual_source import AUG_RUN, DATA_RUN, TRAIN_RUN, W24_RUN

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    for name in ('dispatch','predict-source','score-source','predict-formal'):
        phase = sub.add_parser(name)
        phase.add_argument('runtime',type=Path)
        if name == 'predict-formal':
            phase.add_argument('case',choices=FULL_CASES)
    pub = sub.add_parser('publish')
    for name in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+name,type=Path,required=True)
    reader = sub.add_parser('collect')
    for name in ('submit','dataset','output','ssh-config'):
        reader.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k:ctx[k] for k in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs, release_dir = project/'runs',project/'releases'
        data = release_dir/f'{DATA_RUN}-release'/'data'
        rt.update(run_id=RUN,runner=__spec__.name,code_commit=release_commit(args.repo),
                  release_root=str(release_dir/f'{RUN}-release'),run_root=str(runs/RUN),
                  source_root=str(data/'source'),native_target_root=str(data/'target'),
                  classifier_root=str(runs/AUG_RUN),w24_release_root=str(release_dir/f'{W24_RUN}-release'),
                  training_release_root=str(release_dir/f'{TRAIN_RUN}-release'),
                  allowed_gpus=[3,0,5,6],cpu_workers=8,max_workers=2,max_compute_processes=8)
        rt['external_releases'] = [rt[k] for k in ('source_root','native_target_root','classifier_root',
                                                   'w24_release_root','training_release_root')]
        rt['external_releases'] += [str(runs/x) for x in (PARENT,RUN_OUTER)]
        release(args.repo,args.output,rt,args.ssh_config)
    elif args.action == 'collect':
        collect(args)
    else:
        rt = read(args.runtime)
        if args.action == 'dispatch':
            dispatch(args.runtime,rt)
        elif args.action == 'predict-source':
            predict_source(rt)
        elif args.action == 'score-source':
            score_source_phase(args.runtime,rt)
        else:
            predict_formal(rt,args.case)


if __name__ == '__main__':
    main()
