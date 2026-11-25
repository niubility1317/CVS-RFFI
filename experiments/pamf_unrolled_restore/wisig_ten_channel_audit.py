"""Run same-packet ten-family IQ estimation and independent oracle-last scoring."""

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from . import wisig_multichannel_audit as prior_runner
from . import wisig_multichannel_guard as guard
from . import wisig_multichannel_guard_audit as guarded_run
from . import wisig_ten_channel_estimator as estimator
from . import wisig_ten_channel_panel as panel
from .wisig_independent_channel_audit import FIT, N, TEST


RUN = '20261125-diagnostic-wisig-ten-channel-s2026112506-r01'
METHODS = ('guarded','adaptive','adaptive_matched_fit')


def prepare(root,source,source_panel,dev,seed):
    panel.development(root,source,dev)
    panel.challenge(root,source_panel,seed)


def tune(development,output):
    with np.load(development,allow_pickle=False) as data:
        x,y = data['clean'],data['received']
    if x.shape!=(len(panel.FAMILIES),panel.NDEV,N) or y.shape!=x.shape:
        raise ValueError('Expected ten anonymous IQ-only development groups')
    result = estimator.tune(x,y)
    write(output,result)
    print('IQ_ONLY_FROZEN',*[result[key] for key in ('complexity','shrink','blend')],flush=True)


def predict(challenge,config,previous,wide,baseline,output):
    with np.load(challenge,allow_pickle=False) as data:
        clean,received = data['clean'],data['received']
    if received.shape!=(len(panel.LABELS),panel.NTEST,N) or clean.shape!=(panel.NTEST,N):
        raise ValueError('Incomplete fresh source panel')
    fixed,old = read(config),read(previous)
    wide,base = guard.prepare(read(wide),read(baseline))
    waves = {name:[] for name in METHODS}
    power = {name:[] for name in METHODS}
    dofs = []
    for label,view in zip(panel.LABELS,received):
        for x,y in zip(clean,view):
            row = guard.packet(x,y,wide,base)
            _,fit = guard.action_choice(row,old['action_margin'])
            waves['guarded'].append(guard.solver.action(x,*fit[:2],TEST))
            power['guarded'].append(guard.variance(guard.noise_features(row,fit),old['color_threshold'],
                                                  old['spectrum_blend']))
            waveform,variance,dof = estimator.choose_and_apply(x,y,fixed)
            waves['adaptive'].append(waveform[TEST])
            power['adaptive'].append(variance)
            dofs.append(dof)
            matched,matched_variance,_ = estimator.choose_and_apply(x,y,fixed,train=FIT)
            waves['adaptive_matched_fit'].append(matched[TEST])
            power['adaptive_matched_fit'].append(matched_variance)
        print('ESTIMATES_READY',label,flush=True)
    shape = (len(panel.LABELS),panel.NTEST)
    arrays = {}
    for name in METHODS:
        arrays[f'{name}_action'] = np.asarray(waves[name]).reshape(*shape,len(TEST))
        arrays[f'{name}_variance'] = np.asarray(power[name]).reshape(shape)
    arrays['adaptive_dof'] = np.asarray(dofs).reshape(shape)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite estimate')
    np.savez(output,**arrays)


def summary(error,noise):
    return dict(count=error.size,action=np.quantile(error,[.5,.9]).tolist(),
                variance=np.quantile(np.abs(noise),[.5,.9]).tolist(),
                bias=float(np.median(noise)),within_3db=float(np.mean(np.abs(noise)<=3)))


def score(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Run the independent scorer on CPU')
    shape = (len(panel.LABELS),panel.NTEST)
    with np.load(root/'estimates.npz',allow_pickle=False) as prediction:
        closed = {name:(prediction[f'{name}_action'],prediction[f'{name}_variance']) for name in METHODS}
    for action,variance in closed.values():
        if action.shape!=(*shape,len(TEST)) or variance.shape!=shape or (variance<=0).any():
            raise ValueError('Freeze all complete estimates before opening the reference')
    with np.load(root/'oracle.npz',allow_pickle=False) as oracle:
        actual,true_v = oracle['signal'][:,:,TEST],oracle['variance']
    results = {}
    for name,(action,variance) in closed.items():
        nmse = np.mean(np.abs(action-actual)**2,axis=-1)/np.mean(np.abs(actual)**2,axis=-1).clip(1e-18)
        dB = 10*np.log10(variance/true_v)
        families = {}
        for family in panel.FAMILIES:
            include = np.asarray([label.startswith(family+'/') for label in panel.LABELS])
            families[family] = summary(nmse[include],dB[include])
        cases = {label:summary(nmse[i],dB[i]) for i,label in enumerate(panel.LABELS)}
        results[name] = dict(families=families,cases=cases)
    base = results['guarded']['families']
    macro = {}
    for row in METHODS[1:]:
        new = results[row]['families']
        ratios = {name:[new[family][name][0]/max(base[family][name][0],1e-18) for family in panel.FAMILIES]
                  for name in ('action','variance')}
        macro[row] = {name:float(np.exp(np.mean(np.log(np.maximum(value,1e-18))))) for name,value in ratios.items()}
    write(root/'scores.json',dict(status='SCORED',oracle_last=True,target_reads=0,families=list(panel.FAMILIES),
                                  packets=panel.NTEST,cases=len(panel.LABELS),results=results,
                                  macro_geometric_error_ratio=macro,
                                  goal_met=all(value<.5 for value in macro['adaptive'].values())))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    dev,config = root/'development.npz',root/'configuration.json'
    stages = (('prepare',(root,rt['source'],rt['source_panel'],rt['dev'],rt['seed'])),
              ('tune',(dev,config)),('predict',(root/'challenge.npz',config,rt['previous'],rt['wide'],
                                              rt['baseline'],root/'estimates.npz')),('score',(root,)))
    prior_runner.dispatch(rt,stages)


def publish(repo,output,context,ssh_config):
    from experiments.pamf_lora_main import publish as transport

    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    runs = base/'runs'
    ref = prior_runner.reference
    inputs = dict(source=str(runs/ref.SOURCE/'cache'),source_panel=str(runs/ref.PANEL),
                  dev=str(runs/prior_runner.RUN/'development.npz'),
                  wide=str(runs/prior_runner.RUN/'configuration.json'),
                  baseline=str(runs/ref.RUN/'iq_only_config.json'),
                  previous=str(runs/guarded_run.RUN/'configuration.json'))
    rt = dict(project=ctx['project'],python=ctx['python'],frozen_sources={},run_id=RUN,seed=panel.SEED,
              runner=__spec__.name,code_commit=transport.release_commit(repo),
              release_root=str(base/'releases'/f'{RUN}-release'),run_root=str(runs/RUN),
              external_releases=list(inputs.values()),**inputs)
    transport.release(repo,output,rt,ssh_config)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    release = sub.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        release.add_argument('--'+field,type=Path,required=True)
    commands = ((dispatch,('runtime',)),(prepare,('root','source','source_panel','dev','seed')),
                (tune,('development','output')),
                (predict,('challenge','config','previous','wide','baseline','output')),(score,('root',)))
    for command,fields in commands:
        entry = sub.add_parser(command.__name__)
        for field in fields:
            entry.add_argument(field,type=int if field=='seed' else Path)
    args = vars(parser.parse_args())
    selected = args.pop('action')
    {'publish':publish,**{command.__name__:command for command,_ in commands}}[selected](**args)


if __name__=='__main__':
    main()
