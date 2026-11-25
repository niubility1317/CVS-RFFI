"""Tune across source IQ from multiple simulators and score traces last."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from . import wisig_multichannel_estimator as solver
from . import wisig_multichannel_panel as data
from . import wisig_paired_estimator as paired
from . import wisig_paired_estimator_audit as reference

RUN = '20261125-diagnostic-wisig-multichannel-estimator-s2026112504-r01'


def tune(dev,output):
    with np.load(dev,allow_pickle=False) as data_file:
        clean,received = data_file['clean'],data_file['received']
    if clean.shape!=(len(data.DEVELOPMENT),data.NDEV,256) or received.shape!=clean.shape:
        raise ValueError('Expected complete IQ-only development strata')
    config = solver.tune(list(zip(clean,received)))
    write(output,config)
    print('IQ_ONLY_FROZEN',config['penalty'],config['color'],config['burst'],flush=True)


def predict(challenge,config,baseline,output):
    with np.load(challenge,allow_pickle=False) as data_file:
        clean,received = data_file['clean'],data_file['received']
    if received.shape!=(len(data.TEST_LABELS),data.NTEST,256) or clean.shape!=(data.NTEST,256):
        raise ValueError('Expected the frozen source holdout')
    options = read(config)
    old = read(baseline)
    rows = {'optimized':[], 'baseline':[]}
    for label,obs in zip(data.TEST_LABELS,received):
        delay,taps,variance = paired.estimate(clean,obs,old)
        legacy_coeff = np.stack([solver.embed(d,h) for d,h in zip(delay,taps)])
        rows['baseline'].append((legacy_coeff,np.zeros(len(obs)),variance,np.full(len(obs),-1)))
        rows['optimized'].append(solver.estimate(clean,obs,options))
        print('ESTIMATES_READY',label,flush=True)
    estimates = {name+'_'+field:np.stack(value) for name,row in rows.items()
                 for field,value in zip(('coeff','omega','variance','model'),zip(*row))}
    if not all(np.isfinite(v).all() for v in estimates.values()):
        raise ValueError('Nonfinite IQ-only estimates')
    np.savez(output,**estimates)


def metrics(clean,signal,variance,estimated):
    coeff,omega,noise,_ = estimated
    predicted = np.stack([solver.action(x,h,w,paired.TEST) for x,h,w in zip(clean,coeff,omega)])
    actual = signal[:,paired.TEST]
    error = np.mean(np.abs(actual-predicted)**2,axis=-1) / np.mean(np.abs(actual)**2,axis=-1).clip(1e-18)
    bank = clean[:,paired.reflect(paired.FIT[None,:]-solver.LAGS[:,None])]
    overlap = np.einsum('nlt,nt->nl',bank.conj(),signal[:,paired.FIT])
    best = solver.LAGS[(np.abs(overlap)**2/np.sum(np.abs(bank)**2,axis=-1).clip(1e-18)).argmax(1)]
    lag = solver.LAGS[np.abs(coeff[:,:len(solver.LAGS)]).argmax(1)]
    return np.stack((error,np.abs(lag-best),10*np.log10(noise.clip(1e-18)/variance)),axis=-1)


def score(root,methods=('baseline','optimized')):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('The separate oracle scorer must run on CPU')
    count = len(data.TEST_LABELS)
    with np.load(root/'estimates.npz',allow_pickle=False) as closed:
        estimates = {name:[closed[name+'_'+v] for v in ('coeff','omega','variance','model')]
                     for name in methods}
    expected = [(count,data.NTEST,2*len(solver.LAGS)), *[(count,data.NTEST)]*3]
    if any([v.shape for v in row]!=expected for row in estimates.values()):
        raise ValueError('Freeze all complete IQ-only estimate banks before reading the oracle')
    with np.load(root/'challenge.npz',allow_pickle=False) as data_file:
        clean = data_file['clean']
    with np.load(root/'oracle.npz',allow_pickle=False) as truth:
        signal,variance = truth['signal'],truth['variance']
    summary = {}
    for name,row in estimates.items():
        values = np.stack([metrics(clean,signal[i],variance[i],[x[i] for x in row])
                           for i in range(count)])
        families = {family:reference.summarize(values[[label.startswith(family+'/') for label in data.TEST_LABELS]])
                    for family,_ in data.DEVELOPMENT}
        cases = {label:reference.summarize(values[i]) for i,label in enumerate(data.TEST_LABELS)}
        histogram = np.bincount(row[-1].flatten()+1,minlength=1+len(solver.MODELS)).tolist()
        summary[name] = dict(families=families,cases=cases,model_counts=histogram)
    write(root/'scores.json',dict(status='SCORED',oracle_last=True,development_families=len(data.DEVELOPMENT),
                                  test_cases=count,test_packets=data.NTEST,target_reads=0,results=summary))


def dispatch(rt,stages=None):
    root = Path(rt['run_root'])
    root.mkdir()
    state = root/'state.json'
    if stages is None:
        stages = (('prepare',(root,rt['source_root'],rt['panel_root'])),
                  ('tune',(root/'development.npz',root/'configuration.json')),
                  ('predict',(root/'challenge.npz',root/'configuration.json',rt['baseline'],root/'estimates.npz')),
                  ('score',(root,)))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        for stage,arguments in stages:
            write(state,dict(status='RUNNING',stage=stage),exclusive=False)
            cmd = [rt['python'],'-u','-m',rt['runner'],stage,*(str(x) for x in arguments)]
            with (root/f'{stage}.log').open('x') as log:
                p = subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdout=log,stderr=subprocess.STDOUT)
                print('MULTICHANNEL_WORKER',stage,p.pid,flush=True)
                if p.wait():
                    raise RuntimeError(f'{stage} failed')
        write(state,dict(status='ANALYZED'),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def publish(repo,output,context,ssh_config):
    from experiments.pamf_lora_main import publish as transport

    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    runs = base/'runs'
    panel = runs/reference.PANEL
    config = runs/reference.RUN/'iq_only_config.json'
    rt = dict(project=ctx['project'],python=ctx['python'],frozen_sources={},
              run_id=RUN,runner=__spec__.name,code_commit=transport.release_commit(repo),
              release_root=str(base/'releases'/f'{RUN}-release'),run_root=str(runs/RUN),
              panel_root=str(panel),source_root=str(runs/reference.SOURCE/'cache'),
              baseline=str(config),external_releases=[str(panel),str(config)])
    transport.release(repo,output,rt,ssh_config)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    release = sub.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        release.add_argument('--'+field,type=Path,required=True)
    for stage,fields in (('dispatch',('runtime',)),('prepare',('root','source','panel')),
                         ('tune',('dev','output')),('predict',('challenge','config','baseline','output')),
                         ('score',('root',))):
        command = sub.add_parser(stage)
        for field in fields:
            command.add_argument(field,type=Path)
    args = vars(parser.parse_args())
    selected = args.pop('action')
    if selected=='dispatch':
        dispatch(read(args['runtime']))
    else:
        {'publish':publish,'prepare':data.prepare,'tune':tune,'predict':predict,'score':score}[selected](**args)


if __name__=='__main__':
    main()
