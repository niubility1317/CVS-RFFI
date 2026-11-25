"""Select the paired estimator on source IQ, then independently audit it."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from . import wisig_independent_channel_audit as old
from . import wisig_paired_estimator as estimator

RUN = '20261125-diagnostic-wisig-paired-estimator-s2026112503-r01'
PANEL = '20261125-diagnostic-wisig-action-noise-s2026112501-r01'
SOURCE = '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
SEED = 2026112503
COUNT = 90
DEV = 360


def selected():
    rng = np.random.default_rng(SEED)
    previous = old.select()
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.arange(j,j+24),previous),
                                               6,replace=False)) for j in range(0,360,24)])


def prepare(root,panel):
    indices = selected()
    received = np.load(panel/'received.npy',mmap_mode='r',allow_pickle=False)
    if received.shape != (24,360,2,256):
        raise ValueError('Expected validated source V IQ')
    np.save(root/'challenge_received.npy',received[:,indices],allow_pickle=False)
    with np.load(panel/'scorer_reference.npz',allow_pickle=False) as reference:
        np.save(root/'challenge_clean.npy',reference['clean'][indices],allow_pickle=False)


def dev_pairs(source):
    contract = read(source/'complete.json')
    if (contract['status'],contract['U_count'],contract['TX_labels_read'],contract['target_reads']) != (
            'RECEPTION_SOURCE_CACHE_COMPLETE',56700,0,0):
        raise ValueError('Expected IQ-only source cache')
    rng = np.random.default_rng(SEED)
    roles = np.load(source/'train/roles.npy',allow_pickle=False)
    x = np.load(source/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    candidates = rng.permutation(np.flatnonzero(roles=='U_s'))[:2048]
    full, residual = (
        np.load(source/'epochs/01'/f'{r}.npy',mmap_mode='r',allow_pickle=False)[candidates]
        for r in ('full','residual'))
    valid = (np.sum(full.astype(np.float64)**2,axis=(1,2))>0) & (
        np.sum(residual.astype(np.float64)**2,axis=(1,2))>0)
    index = np.flatnonzero(valid)[:DEV]
    if index.shape != (DEV,):
        raise ValueError('Insufficient paired U_s IQ')
    y = np.where((np.arange(DEV)%2==0)[:,None,None],full[index],residual[index])
    return estimator.voltages(x[candidates[index]]),estimator.voltages(y)


def tune(source,output):
    clean,obs = dev_pairs(source)
    half = DEV//2
    config = estimator.tune(clean[:half],obs[:half],clean[half:],obs[half:])
    write(output,config)
    print('IQ_ONLY_SELECTION',config['selected'],flush=True)


def predict(clean_path,received_path,config_path,output):
    clean = np.load(clean_path,allow_pickle=False)
    received = np.load(received_path,mmap_mode='r',allow_pickle=False)
    config = read(config_path)
    if clean.shape != (COUNT,256) or received.shape != (24,COUNT,2,256):
        raise ValueError('Unexpected frozen evaluation sample set')
    rows = {key:[] for key in ('baseline','optimized')}
    for cell, iq in zip(old.CELLS,received):
        y = estimator.voltages(iq)
        chunks = [old.paired_fit(y[i:i+15],clean[i:i+15]) for i in range(0,COUNT,15)]
        d,h,_,v = map(np.concatenate,zip(*chunks))
        fixed = np.zeros((COUNT,len(estimator.OFFSETS)),complex)
        fixed[:,6+old.TAPS] = h
        rows['baseline'].append((d,fixed,v))
        rows['optimized'].append(estimator.estimate(clean,y,config))
        print('SOURCE_ESTIMATES',*cell,flush=True)
    fields = {}
    for key,values in rows.items():
        fields.update({f'{key}_{name}':np.stack(value)
                       for name,value in zip(('delay','coefficient','variance'),zip(*values))})
    if not all(np.isfinite(v).all() for v in fields.values()):
        raise ValueError('Nonfinite IQ-only estimator output')
    np.savez(output,**fields)


def metrics(clean,truth,reference,delay,coefficient,variance):
    predicted = estimator.apply(clean,delay,coefficient,old.TEST)
    signal = truth[:,old.TEST]
    nmse = np.mean(np.abs(predicted-signal)**2,axis=-1)/np.mean(np.abs(signal)**2,axis=-1).clip(1e-18)
    candidates = clean[:,old.reflect(old.FIT[None,:]-old.DELAYS[:,None])]
    inner = np.einsum('nlt,nt->nl',candidates.conj(),truth[:,old.FIT])
    best = old.DELAYS[(np.abs(inner)**2/np.sum(np.abs(candidates)**2,axis=-1).clip(1e-18)).argmax(1)]
    lag = delay+estimator.OFFSETS[np.abs(coefficient).argmax(1)]
    return np.stack((nmse,np.abs(lag-best),10*np.log10(variance.clip(1e-18)/reference)),axis=-1)


def summarize(values):
    error,lag,variance = values.reshape(-1,3).T
    quantile = lambda v: np.quantile(v,[.1,.5,.9]).tolist()
    return dict(count=len(error),nmse=quantile(error),lag_within_1=float(np.mean(lag<=1)),
                variance_db=quantile(variance),variance_abs_db=quantile(np.abs(variance)),
                variance_within_3=float(np.mean(np.abs(variance)<=3)))


def score(root,panel):
    methods = ('baseline','optimized')
    with np.load(root/'estimates.npz',allow_pickle=False) as record:
        predictions = {name: tuple(record[f'{name}_{field}'] for field in ('delay','coefficient','variance'))
                       for name in methods}
    shape = ((24,COUNT),(24,COUNT,len(estimator.OFFSETS)),(24,COUNT))
    if any(tuple(x.shape for x in row)!=shape for row in predictions.values()):
        raise ValueError('Freeze all IQ-only predictions before opening oracle')
    indices = selected()
    with np.load(panel/'scorer_reference.npz',allow_pickle=False) as reference:
        clean,signal,variance = reference['clean'][indices], reference['signal'][:,indices], reference['variance'][:,indices]
    outputs = {}
    for method,arrays in predictions.items():
        scored = np.stack([metrics(clean,signal[i],variance[i],*(x[i] for x in arrays)) for i in range(24)])
        outputs[method] = {f'{profile}/{route}': {view: summarize(np.stack(
            [scored[i] for i,case in enumerate(old.CELLS)
             if case[:2]==(profile,route) and (view=='combined' or view==case[2])]))
            for view in ('combined', *old.VIEWS[1:])} for profile in ('default','strong') for route in ('full','residual')}
    write(root/'scores.json',dict(status='SCORED',oracle_last=True,test_packets=COUNT,cases=24,
                                   target_reads=0,model_reads=0,results=outputs))


def dispatch(rt):
    root = Path(rt['run_root'])
    panel = Path(rt['panel_root'])
    root.mkdir()
    state = root/'state.json'
    cfg = root/'iq_only_config.json'
    tasks = (('tune',(rt['source_root'],cfg)),('prepare',(root,panel)),
             ('predict',(root/'challenge_clean.npy',root/'challenge_received.npy',cfg,root/'estimates.npz')),
             ('score',(root,panel)))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        for action,arguments in tasks:
            write(state,dict(status='RUNNING',stage=action),exclusive=False)
            command = [rt['python'],'-u','-m',rt['runner'],action,*(str(arg) for arg in arguments)]
            with (root/f'{action}.log').open('x') as log:
                p = subprocess.Popen(command,cwd=rt['release_root'],env=env,stdout=log,stderr=subprocess.STDOUT)
                print('PAIR_ESTIMATOR_WORKER',action,p.pid,flush=True)
                if p.wait():
                    raise RuntimeError(f'{action} failed')
        write(state,dict(status='ANALYZED'),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def publish(repo,output,context,ssh_config):
    from experiments.pamf_lora_main import publish as transport

    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    panel = str(base/'runs'/PANEL)
    rt = dict(project=ctx['project'],python=ctx['python'],frozen_sources={},
              run_id=RUN,runner=__spec__.name,code_commit=transport.release_commit(repo),
              release_root=str(base/'releases'/f'{RUN}-release'),run_root=str(base/'runs'/RUN),
              panel_root=panel,source_root=str(base/'runs'/SOURCE/'cache'),external_releases=[panel])
    transport.release(repo,output,rt,ssh_config)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    release = sub.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        release.add_argument('--'+key,type=Path,required=True)
    for stage,fields in (('dispatch',('runtime',)),('prepare',('root','panel')),
                         ('tune',('source','output')),('predict',('clean_path','received_path','config_path','output')),
                         ('score',('root','panel'))):
        command = sub.add_parser(stage)
        for field in fields:
            command.add_argument(field,type=Path)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action=='dispatch':
        dispatch(read(args['runtime']))
    else:
        {'publish':publish,'prepare':prepare,'tune':tune,'predict':predict,'score':score}[action](**args)


if __name__=='__main__':
    main()
