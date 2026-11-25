"""Same-draw default and double-power evaluation for the frozen WiSig estimator."""

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main import publish as transport
from . import wisig_ten_channel_audit as baseline
from . import wisig_ten_channel_panel as panel
from .wisig_independent_channel_audit import N, TEST


RUN = '20261125-diagnostic-wisig-double-impairment-s2026112506-r01'
HARD = 'practical/strong/full/practical_low_urban'
LABELS = tuple(label for label in panel.LABELS if label.startswith('practical/default/') or
               label.split('/')[0] in ('weak_leo','legacy_sat') or
               label.endswith(('/nominal','_nominal')))+(HARD,)
SLOTS = tuple(panel.LABELS.index(label) for label in LABELS)
SHAPE = (len(LABELS),panel.NTEST)


def double_impairment(clean,signal,observed):
    energy = np.sum(np.abs(clean)**2,axis=-1,keepdims=True).clip(1e-18)
    gain = np.sum(clean.conj()*signal,axis=-1,keepdims=True)/energy
    aligned = gain*clean
    distortion = signal-aligned
    amplified = aligned+np.sqrt(2.)*distortion
    power = np.sum(np.abs(signal)**2,axis=-1,keepdims=True)
    new_power = np.sum(np.abs(amplified)**2,axis=-1,keepdims=True)
    stressed = amplified*np.sqrt(power/new_power.clip(1e-18))
    return stressed,stressed+np.sqrt(2.)*(observed-signal)


def prepare(root,parent):
    parent = Path(parent)
    if read(parent/'state.json')['status']!='ANALYZED' or not read(parent/'scores.json')['oracle_last']:
        raise ValueError('Expected the sealed, scored parent diagnostic')
    with np.load(parent/'challenge.npz',allow_pickle=False) as source:
        clean,observed = source['clean'],source['received'][list(SLOTS)]
    # This process constructs the two received views. The fitter receives the output IQ only.
    with np.load(parent/'oracle.npz',allow_pickle=False) as source:
        signal,variance = source['signal'][list(SLOTS)],source['variance'][list(SLOTS)]
    if clean.shape!=(panel.NTEST,N) or observed.shape!=(*SHAPE,N) or signal.shape!=observed.shape or variance.shape!=SHAPE:
        raise ValueError('Unexpected source evaluation bank')
    stronger,received = double_impairment(clean,signal,observed)
    np.savez(root/'challenge.npz',clean=clean,received=received)
    np.savez(root/'oracle.npz',default=signal[:,:,TEST],stronger=stronger[:,:,TEST],variance=variance)
    print('PAIRED_IQ_READY',*SHAPE,flush=True)


def predict(challenge,config,output):
    fixed = read(config)
    with np.load(challenge,allow_pickle=False) as staged:
        clean,received = staged['clean'],staged['received']
    if clean.shape!=(panel.NTEST,N) or received.shape!=(*SHAPE,N):
        raise ValueError('Expected complete paired input IQ')
    wave = np.empty((*SHAPE,len(TEST)),complex)
    power = np.empty(SHAPE)
    for i,(label,view) in enumerate(zip(LABELS,received)):
        for j,(x,y) in enumerate(zip(clean,view)):
            signal,power[i,j],_ = baseline.estimator.choose_and_apply(x,y,fixed)
            wave[i,j] = signal[TEST]
        print('DOUBLE_IMPAIRMENT_ESTIMATED',label,flush=True)
    if not (np.isfinite(wave).all() and np.isfinite(power).all() and (power>0).all()):
        raise ValueError('Invalid frozen estimates')
    np.savez(output,action=wave,variance=power)


def score(root,parent):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Use CPU for independent scoring')
    with np.load(root/'estimates.npz',allow_pickle=False) as frozen:
        strong = frozen['action'],frozen['variance']
    with np.load(parent/'estimates.npz',allow_pickle=False) as frozen:
        current = frozen['adaptive_action'][list(SLOTS)],frozen['adaptive_variance'][list(SLOTS)]
    if any(y.shape!=(*SHAPE,len(TEST)) or v.shape!=SHAPE or (v<=0).any() for y,v in (current,strong)):
        raise ValueError('Freeze both complete estimate arrays before loading the oracle')
    with np.load(root/'oracle.npz',allow_pickle=False) as source:
        default,stronger,power = source['default'],source['stronger'],source['variance']
    families,cases = {},{}
    for mode,((estimate,estimated_power),target,true_power) in (
            ('default',(current,default,power)),('double_power',(strong,stronger,2.*power))):
        rms = np.mean(np.abs(target)**2,axis=-1).clip(1e-18)
        nmse = np.mean(np.abs(estimate-target)**2,axis=-1)/rms
        db = 10.*np.log10(estimated_power/true_power)
        families[mode] = {}
        for family in panel.FAMILIES:
            select = np.asarray([label.startswith(family+'/') and label!=HARD for label in LABELS])
            families[mode][family] = baseline.summary(nmse[select],db[select])
        cases[mode] = {key:baseline.summary(nmse[i],db[i]) for i,key in enumerate(LABELS)}
    write(root/'scores.json',dict(status='SCORED',oracle_last=True,target_reads=0,packets=panel.NTEST,
                                  families=families,cases=cases,levels=[1.,2.],
                                  strength_units='distortion-to-aligned-direct power and injected noise power',
                                  paired_draws=True,parent=baseline.RUN))


def dispatch(runtime):
    rt = read(runtime)
    root = Path(rt['run_root'])
    parent,cfg = rt['panel_root'],rt['estimator']
    stages = (('prepare',(root,parent)),('predict',(root/'challenge.npz',cfg,root/'estimates.npz')),
              ('score',(root,parent)))
    baseline.prior_runner.dispatch(rt,stages)


def publish(repo,output,context,ssh_config):
    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    parent = base/'runs'/baseline.RUN
    config = parent/'configuration.json'
    rt = dict(project=ctx['project'],python=ctx['python'],frozen_sources={},run_id=RUN,
              runner=__spec__.name,code_commit=transport.release_commit(repo),
              release_root=str(base/'releases'/f'{RUN}-release'),run_root=str(base/'runs'/RUN),
              panel_root=str(parent),estimator=str(config),external_releases=[str(parent),str(config)])
    transport.release(repo,output,rt,ssh_config)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    release = sub.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        release.add_argument('--'+field,type=Path,required=True)
    specs = ((dispatch,('runtime',)),(prepare,('root','parent')),
             (predict,('challenge','config','output')),(score,('root','parent')))
    for command,fields in specs:
        stage = sub.add_parser(command.__name__)
        for field in fields:
            stage.add_argument(field,type=Path)
    args = vars(parser.parse_args())
    name = args.pop('action')
    {'publish':publish, **{command.__name__:command for command,_ in specs}}[name](**args)


if __name__=='__main__':
    main()
