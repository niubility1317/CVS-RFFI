"""Compare three frozen IQ-only estimators against independent source references."""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from . import wisig_multichannel_audit as previous
from . import wisig_multichannel_estimator as solver
from . import wisig_multichannel_guard as guard
from . import wisig_multichannel_panel as data


RUN = '20261125-diagnostic-wisig-multichannel-guard-s2026112505-r01'
SEED = 2026112505
METHODS = ('baseline','optimized','guarded')


def prepare(root,panel):
    data.prepare_test(root,panel,data.fresh_holdout(SEED),SEED)


def tune(dev,wide,baseline,output):
    with np.load(dev,allow_pickle=False) as iq:
        clean,received = iq['clean'],iq['received']
    if clean.shape!=(len(data.DEVELOPMENT),data.NDEV,solver.N) or received.shape!=clean.shape:
        raise ValueError('Expected six complete IQ-only development groups')
    config = guard.tune(clean,received,read(wide),read(baseline))
    write(output,config)
    print('IQ_ONLY_FROZEN',config['action_margin'],config['color_threshold'],config['spectrum_blend'],flush=True)


def predict(challenge,config,wide,baseline,output):
    with np.load(challenge,allow_pickle=False) as iq:
        clean,received = iq['clean'],iq['received']
    if received.shape!=(len(data.TEST_LABELS),data.NTEST,solver.N) or clean.shape!=(data.NTEST,solver.N):
        raise ValueError('Expected complete fresh V IQ pairs')
    fixed = read(config)
    wide,old = guard.prepare(read(wide),read(baseline))
    results = {name:[] for name in METHODS}
    for label,obs in zip(data.TEST_LABELS,received):
        for x,y in zip(clean,obs):
            row = guard.packet(x,y,wide,old)
            coeff,omega,residual = row['wider']
            v_baseline = solver.noise(row['raw'],residual,0.,0.)
            v_wide = solver.noise(row['raw'],residual,wide['color'],wide['burst'])
            results['baseline'].append((*row['fallback'][:2],v_baseline,-1))
            results['optimized'].append((coeff,omega,v_wide,row['wide_index']))
            chosen,model = guard.action_choice(row,fixed['action_margin'])
            features = guard.noise_features(row,model)
            v_guard = guard.variance(features,fixed['color_threshold'],fixed['spectrum_blend'])
            results['guarded'].append((*model[:2],v_guard,chosen))
        print('ESTIMATES_READY',label,flush=True)
    shape = (len(data.TEST_LABELS),data.NTEST)
    estimates = {}
    for name,rows in results.items():
        for field,values in zip(('coeff','omega','variance','model'),zip(*rows)):
            array = np.asarray(values)
            estimates[name+'_'+field] = array.reshape(*shape,*array.shape[1:])
    if any(not np.isfinite(array).all() for array in estimates.values()):
        raise ValueError('Nonfinite IQ-only estimates')
    np.savez(output,**estimates)


def score(root):
    previous.score(root,METHODS)


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    wide,old = rt['wide'],rt['baseline']
    config = root/'configuration.json'
    stages = (('prepare',(root,rt['panel_root'])),('tune',(rt['dev'],wide,old,config)),
              ('predict',(root/'challenge.npz',config,wide,old,root/'estimates.npz')),
              ('score',(root,)))
    previous.dispatch(rt,stages)


def publish(repo,output,context,ssh_config):
    from experiments.pamf_lora_main import publish as transport

    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    runs = base/'runs'
    panel = runs/previous.reference.PANEL
    wide = runs/previous.RUN/'configuration.json'
    dev = runs/previous.RUN/'development.npz'
    config = runs/previous.reference.RUN/'iq_only_config.json'
    rt = dict(project=ctx['project'],python=ctx['python'],frozen_sources={},run_id=RUN,
              runner=__spec__.name,code_commit=transport.release_commit(repo),
              release_root=str(base/'releases'/f'{RUN}-release'),run_root=str(runs/RUN),
              panel_root=str(panel),wide=str(wide),dev=str(dev),baseline=str(config),
              external_releases=[str(path) for path in (panel,wide,dev,config)])
    transport.release(repo,output,rt,ssh_config)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    release = sub.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        release.add_argument('--'+field,type=Path,required=True)
    commands = ((dispatch,('runtime',)),(prepare,('root','panel')),
                (tune,('dev','wide','baseline','output')),
                (predict,('challenge','config','wide','baseline','output')),(score,('root',)))
    for command,fields in commands:
        subcommand = sub.add_parser(command.__name__)
        for field in fields:
            subcommand.add_argument(field,type=Path)
    args = vars(parser.parse_args())
    name = args.pop('action')
    {'publish':publish,**{command.__name__:command for command,_ in commands}}[name](**args)


if __name__=='__main__':
    main()
