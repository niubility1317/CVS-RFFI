"""Validate strong Full Low Urban on the reserved source V physical packets."""

import argparse
import os
from pathlib import Path, PurePosixPath
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_dura_stress_data as stress
from . import wisig_low_urban_bayes as bayes
from . import wisig_ten_channel_audit as full_panel
from . import wisig_ten_channel_panel as panel
from .wisig_independent_channel_audit import N, TEST
from .wisig_lora_source_fit import source_iq
from .wisig_paired_estimator import voltages


RUN = '20261125-diagnostic-wisig-strong-low-urban-s2026112507-r01'
SEED = panel.SEED+1
NDEV = 128
LABELS = ('practical/strong/full/practical_low_urban',
          'practical/strong/full/practical_mid_urban','practical/default/full/practical_low_urban')
METHODS = ('guarded','adaptive','posterior')


def prepare(root,source,source_panel):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
    from leo_practical import Config

    iq,fit,_ = source_iq(source)
    rng = np.random.default_rng(SEED)
    take = np.sort(rng.choice(fit,NDEV,replace=False))
    ids,rx,day = [np.load(source/'train'/f'{column}.npy',allow_pickle=False)[take]
                  for column in ('physical_ids','rx','day')]
    sessions = [f'rx{r}:day{d}:eq1' for r,d in zip(rx,day)]
    cfg = stress.harder(Config(fs_hz=25e6,fc_hz=2462e6,scenario='practical_low_urban',
                               processing_route='full',mode='post_sync',equalization_enabled=False,
                               boundary='reflect'))
    observed = stress.channel(iq[take],cfg,ids,sessions,SEED,SEED+14,'strong_urban/paired_source_U',False)
    np.savez(root/'development.npz',clean=voltages(iq[take]),received=voltages(observed))
    slots = [panel.LABELS.index(case) for case in LABELS]
    new = panel.split(SEED)
    with np.load(source_panel/'scorer_reference.npz',allow_pickle=False) as oracle:
        clean = oracle['clean'][new]
        action,variance = (oracle[name][slots][:,new] for name in ('signal','variance'))
    raw = np.load(source_panel/'received.npy',mmap_mode='r',allow_pickle=False)
    frames = np.stack([raw[index,new] for index in slots])
    np.savez(root/'challenge.npz',clean=clean,received=voltages(frames))
    np.savez(root/'oracle.npz',action=action,variance=variance)
    print('SOURCE_PANELS_READY',NDEV,len(new),len(LABELS),flush=True)


def tune(development,current,output):
    with np.load(development,allow_pickle=False) as source:
        x,y = source['clean'],source['received']
    if x.shape!=(NDEV,N) or y.shape!=x.shape:
        raise ValueError('Incomplete anonymous source U pairs')
    fixed = bayes.tune(x,y,read(current))
    write(output,fixed)
    print('IQ_ONLY_FROZEN',*[fixed[key] for key in ('gate','mix','variance_mix')],flush=True)


def predict(challenge,settings,current,previous,wide,baseline,output):
    configured,full,old = (read(path) for path in (settings,current,previous))
    guard = full_panel.guard
    wide,base = guard.prepare(read(wide),read(baseline))
    prior = np.asarray(configured['prior'])
    with np.load(challenge,allow_pickle=False) as source:
        clean,received = source['clean'],source['received']
    if clean.shape!=(panel.NTEST,N) or received.shape!=(len(LABELS),panel.NTEST,N):
        raise ValueError('Unexpected reserved source panel')
    shape = (len(LABELS),panel.NTEST)
    signals = {method:[] for method in METHODS}
    powers = {method:[] for method in METHODS}
    for label,observations in zip(LABELS,received):
        for x,y in zip(clean,observations):
            pack = guard.packet(x,y,wide,base)
            _,fit = guard.action_choice(pack,old['action_margin'])
            signal = guard.solver.action(x,*fit[:2],TEST)
            variance = guard.variance(guard.noise_features(pack,fit),old['color_threshold'],old['spectrum_blend'])
            signals['guarded'].append(signal)
            powers['guarded'].append(variance)
            row = bayes.pair(x,y,bayes.adaptive.T,full,prior)
            signals['adaptive'].append(row[0][TEST])
            powers['adaptive'].append(row[1])
            signal,variance = bayes.select(row,configured['gate'],configured['mix'],configured['variance_mix'])
            signals['posterior'].append(signal[TEST])
            powers['posterior'].append(variance)
        print('ESTIMATES_READY',label,flush=True)
    arrays = {f'{name}_action':np.asarray(signals[name]).reshape(*shape,len(TEST)) for name in METHODS}
    arrays.update({f'{name}_variance':np.asarray(powers[name]).reshape(shape) for name in METHODS})
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Nonfinite IQ-only estimate')
    np.savez(output,**arrays)


def score(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Run an independent CPU scorer')
    shape = (len(LABELS),panel.NTEST)
    with np.load(root/'estimates.npz',allow_pickle=False) as frozen:
        predictions = {method:(frozen[f'{method}_action'],frozen[f'{method}_variance']) for method in METHODS}
    if any(wave.shape!=(*shape,len(TEST)) or power.shape!=shape or (power<=0).any()
           for wave,power in predictions.values()):
        raise ValueError('All three methods must be frozen before scoring')
    with np.load(root/'oracle.npz',allow_pickle=False) as oracle:
        wave,variance = oracle['action'][:,:,TEST],oracle['variance']
    results = {}
    for method,(proposed,power) in predictions.items():
        nmse = np.mean(np.abs(proposed-wave)**2,axis=-1)/np.mean(np.abs(wave)**2,axis=-1).clip(1e-18)
        noise = 10*np.log10(power/variance)
        results[method] = {case:full_panel.summary(nmse[i],noise[i]) for i,case in enumerate(LABELS)}
    write(root/'scores.json',dict(status='SCORED',oracle_last=True,target_reads=0,packets=panel.NTEST,
                                  seed=SEED,results=results))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    cfg,dev = root/'configuration.json',root/'development.npz'
    stages = (('prepare',(root,rt['source'],rt['source_panel'])),('tune',(dev,rt['current'],cfg)),
              ('predict',(root/'challenge.npz',cfg,rt['current'],rt['previous'],rt['wide'],rt['baseline'],
                          root/'estimates.npz')),('score',(root,)))
    full_panel.prior_runner.dispatch(rt,stages)


def publish(repo,output,context,ssh_config):
    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    runs = base/'runs'
    previous = full_panel.prior_runner
    ref = previous.reference
    inputs = dict(source=str(runs/ref.SOURCE/'cache'),source_panel=str(runs/ref.PANEL),
                  wide=str(runs/previous.RUN/'configuration.json'),baseline=str(runs/ref.RUN/'iq_only_config.json'),
                  previous=str(runs/full_panel.guarded_run.RUN/'configuration.json'),
                  current=str(runs/full_panel.RUN/'configuration.json'))
    rt = dict(project=ctx['project'],python=ctx['python'],frozen_sources={},run_id=RUN,
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
    specifications = ((dispatch,('runtime',)),(prepare,('root','source','source_panel')),
                      (tune,('development','current','output')),
                      (predict,('challenge','settings','current','previous','wide','baseline','output')),
                      (score,('root',)))
    for command,fields in specifications:
        entry = sub.add_parser(command.__name__)
        for field in fields:
            entry.add_argument(field,type=Path)
    args = vars(parser.parse_args())
    action = args.pop('action')
    {'publish':publish,**{command.__name__:command for command,_ in specifications}}[action](**args)


if __name__=='__main__':
    main()
