"""Run the fixed two-arm LoRa Full matrix after source weights freeze, independent of source scores."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import monotonic, sleep

import numpy as np
import torch

if __name__ == '__mp_main__':
    sys.path.insert(0, str(Path.cwd()))

if not __package__:
    from experiments import pamf_lora_main
    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import w24_joint_pamf as joint, w24_native_training as native


RUN = '20261127-phase1-lora-raw8192-paired-denoise-full-s2026113027-r01'
FIT_COMMIT = 'fc8c14c22b5de5c3198dcca00a3e45d3827a4555'
CONTROL, CANDIDATE = native.PAIR_ARMS


def formal_receipt(case):
    fit = joint.formal_receipt(case,native.PAIR_RUN)
    return dict(fit,modes=[*fit['modes'][:-1], CONTROL, CANDIDATE],candidate_run=RUN,
                fit_run=native.PAIR_RUN,fit_commit=FIT_COMMIT)


def fit_runtime(rt):
    project = Path(rt['project'])
    release = project/'releases'/f'{native.PAIR_RUN}-release'
    run = project/'runs'/native.PAIR_RUN
    fit = read(release/'runtime.json')
    if (fit['run_id'],fit['code_commit'],Path(fit['run_root']),Path(fit['release_root'])) != (
            native.PAIR_RUN,FIT_COMMIT,run,release):
        raise ValueError('Expected the registered two-arm source fit release')
    return fit,run,release


def await_original_dispatch(fit_root, fit_release):
    # Wait for the original dispatcher to finish writing, independent of the source decision values.
    pid = read(fit_release/'submit.json')['pid']
    signature = os.fsencode(fit_release/'runtime.json')
    deadline = monotonic()+12*60*60
    while monotonic()<deadline:
        try:
            live = signature in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            live = False
        if not live:
            if not all((fit_root/arm/'complete.json').is_file() for arm in native.PAIR_ARMS):
                raise ValueError('Registered training exited before both source weights froze')
            return
        sleep(30)
    raise TimeoutError('Original run still owns its output')


def baseline_for(fit, fit_root, case):
    # An already frozen blind candidate shard may be reused in full, regardless of any score.
    previous = fit_root/'predictions'/case
    if (previous/'complete.json').is_file():
        if read(previous/'complete.json') != joint.formal_receipt(case,native.PAIR_RUN):
            raise ValueError('Prior candidate shard differs from the frozen row')
        return previous, (CONTROL,)
    return joint.formal_reference(fit,case), native.PAIR_ARMS


def order_pair(rows, counts, *, reused):
    if reused:
        rows = rows.copy()
        rows[:, -2:] = rows[:, [-1, -2]]
        counts[-2:] = counts[-2:][::-1]
    return rows, counts


@torch.no_grad()
def smoke(rt):
    fit,_,_ = fit_runtime(rt)
    f, models, _, contract = native.load_readers(joint,fit,'cuda:0')
    source = Path(fit['source_root'])
    iq = joint.native(source,'U_s',contract['counts']['U_s'])[0]
    samples = np.stack([joint.source_receive(v,(0,joint.SOURCE[0],iq)) for v in ('clean',joint.VIEWS[-1])])
    for arm in native.PAIR_ARMS:
        if native.render(joint,f,models,arm,samples,'cuda:0').shape != (2,):
            raise ValueError('Real two-arm no-query smoke did not produce complete predictions')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',fit_commit=FIT_COMMIT,
          arms=list(native.PAIR_ARMS),source_u_count=1,target_reads=0,target_truth_reads=0))


@torch.no_grad()
def predict(rt,case):
    fit, fit_root, _ = fit_runtime(rt)
    expected = formal_receipt(case)
    baseline, arms = baseline_for(fit,fit_root,case)
    f, models, _, _ = native.load_readers(joint,fit,'cuda:0')
    target = Path(fit['native_target_root'])
    joint.target_contract(target)
    out = Path(rt['run_root'])/'predictions'/case
    out.mkdir(parents=True)
    counts = {}
    activity = read(baseline/'activity.json')
    with ProcessPoolExecutor(max_workers=rt['cpu_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for receiver in expected['receivers']:
            raw = joint.native(target,receiver,joint.LIMIT)
            predictions = {}
            with np.load(baseline/f'{receiver}.npz',allow_pickle=False) as old:
                for view in joint.VIEWS:
                    stream = iter(raw) if view=='clean' else pool.map(
                        partial(joint.channel_job,view,receiver),enumerate(raw),chunksize=16)
                    batch = 8
                    pieces = []
                    for i in range(0,joint.LIMIT,batch):
                        packets = np.stack([next(stream) for _ in range(min(batch,joint.LIMIT-i))])
                        direct = joint.classify_aperture(f,joint.received_aperture(packets,'cuda:0'))
                        labels = [native.render(joint,f,models,arm,packets,'cuda:0') for arm in arms]
                        pieces.append(np.stack((direct,*labels)))
                    key = f'{receiver}/{view}'
                    rows, used = joint.append_blind_prediction(
                        old[view],np.concatenate(pieces,axis=1)[None],activity[key],joint.LIMIT)
                    predictions[view],counts[key] = order_pair(rows,used,reused=arms==(CONTROL,))
                    print('PAIRED_NATIVE_FULL',receiver,view,'NEW_READOUTS',*arms,flush=True)
            np.savez(out/f'{receiver}.npz',**predictions)
    write(out/'activity.json',counts)
    write(out/'complete.json',expected)


def dispatch(path):
    rt = read(path)
    root = Path(rt['run_root'])
    root.mkdir()
    state = root/'state.json'

    def status(**fields):
        write(state,fields,exclusive=False)

    def launch(_, action, name, gpu):
        w24 = rt['w24_release_root']
        env = dict(os.environ,PYTHONPATH=w24,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        command = [rt['python'],'-u',str(Path(__file__).resolve()),action,str(path),name]
        with (root/f'{name}.log').open('x',encoding='utf-8') as out:
            process = subprocess.Popen(command,cwd=w24,env=env,stdout=out,stderr=subprocess.STDOUT)
        print('NATIVE_PAIRED_FULL_LAUNCHED',action,name,process.pid,gpu,flush=True)
        return process

    try:
        _,fit_root,fit_release = fit_runtime(rt)
        status(status='WAIT_FROZEN_PAIR',fit_run=native.PAIR_RUN)
        await_original_dispatch(fit_root,fit_release)
        status(status='SOURCE_SMOKE')
        joint.schedule(rt,[('smoke','pair-smoke')],launch)
        status(status='FORMAL_PREDICTING',route=joint.FULL)
        joint.schedule(rt,[('predict',case) for case in joint.FULL_CASES],launch)
        if any(read(root/'predictions'/case/'complete.json')!=formal_receipt(case) for case in joint.FULL_CASES):
            raise ValueError('Incomplete paired Full predictions')
        status(status='PREDICTIONS_FROZEN',cases=list(joint.FULL_CASES),target_truth_read=False)
    except Exception as exc:
        status(status='FAILED',error=str(exc))
        raise


def score(predictions,dataset,output):
    from experiments.pamf_lora_main.w24_formal_score import load_frozen, score_matrix, full_target_acceptance

    plans = {case:formal_receipt(case) for case in joint.FULL_CASES}
    matrices, activity = load_frozen(predictions,ungated=True,observed_plans=plans)
    plan = plans[joint.FULL_CASES[0]]
    score_matrix(matrices,activity,{joint.FULL:plan['modes']},dataset,output,tuple(plan['seeds']),
                 lambda results:{arm:full_target_acceptance(results[joint.FULL][arm]['pooled']) for arm in native.PAIR_ARMS},
                 additional_pairings=((CONTROL,'direct'),(CONTROL,'complex_response_half'),(CONTROL,native.TASK)))


def collect(args):
    root = PurePosixPath(read(args.submit)['run_root'])
    if root.name != RUN:
        raise ValueError('Expected the registered paired Full run')
    connection = ['-F',str(args.ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10']
    deadline, last = monotonic()+16*60*60, None
    while monotonic()<deadline:
        try:
            remote = subprocess.run(['ssh',*connection,'-T','N607','cat '+str(root/'state.json')],
                                    timeout=25,capture_output=True)
            status = json.loads(remote.stdout) if remote.returncode==0 else None
        except (subprocess.TimeoutExpired,ValueError):
            status = None
        if status and status['status']!=last:
            print(status,flush=True)
            last = status['status']
            if last=='FAILED':
                raise RuntimeError(status['error'])
            if last=='PREDICTIONS_FROZEN':
                break
        sleep(30)
    else:
        raise TimeoutError('Paired Full collector exceeded its observation window')
    args.output.mkdir(parents=True)
    subprocess.run(['scp',*connection,'-r','N607:'+str(root/'predictions'),str(args.output)],check=True,timeout=300)
    score(args.output/'predictions',args.dataset,args.output/'scores')


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    for action in ('dispatch','smoke','predict'):
        cmd = actions.add_parser(action)
        cmd.add_argument('runtime',type=Path)
        if action!='dispatch':
            cmd.add_argument('name')
    for action, fields in (('publish',('repo','context','ssh-config','output')),
                           ('collect',('submit','dataset','ssh-config','output'))):
        cmd = actions.add_parser(action)
        for field in fields:
            cmd.add_argument('--'+field,required=True,type=Path)
    args = parser.parse_args()
    if args.action=='dispatch':
        dispatch(args.runtime)
    elif args.action=='smoke':
        smoke(read(args.runtime))
    elif args.action=='predict':
        predict(read(args.runtime),args.name)
    elif args.action=='collect':
        collect(args)
    else:
        ctx = read(args.context)
        rt = {key:ctx[key] for key in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        runs,releases = project/'runs',project/'releases'
        w24 = releases/f'{joint.W24_RUN}-release'
        rt.update(run_id=RUN,runner=__spec__.name,code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{RUN}-release'),run_root=str(runs/RUN),
                  w24_release_root=str(w24),allowed_gpus=[6,7,2,0,3],max_workers=2,max_compute_processes=8,cpu_workers=6,
                  external_releases=[str(x) for x in (w24,releases/f'{native.PAIR_RUN}-release',runs/native.PAIR_RUN)])
        release(args.repo,args.output,rt,args.ssh_config)


if __name__=='__main__':
    main()
