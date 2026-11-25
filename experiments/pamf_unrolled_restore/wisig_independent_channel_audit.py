"""Compare two non-neural estimators to the recorded WiSig simulator reference."""

import argparse
from itertools import product
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_view_shuffle.plan import VIEWS

RUN = '20261125-diagnostic-wisig-analytic-action-noise-s2026112502-r01'
PARENT = '20261125-diagnostic-wisig-action-noise-s2026112501-r01'
SEED = 2026112502
PACKETS = 30
N = 256
DELAYS = np.arange(-40, 41)
TAPS = np.array([0, -1, 1])
CELLS = tuple(product(('default', 'strong'), ('full', 'residual'), VIEWS[1:]))
OBS = np.arange(96, N)
FIT = OBS[((OBS - 96)//8) % 2 == 0]
TEST = OBS[((OBS - 96)//8) % 2 == 1]


def select():
    """Two samples from each archived 24-packet receiver-day stratum."""
    rng = np.random.default_rng(SEED)
    return np.concatenate([np.sort(rng.choice(np.arange(j,j+24),2,replace=False))
                           for j in range(0,360,24)])


def prepare(root, panel):
    take = select()
    iq = np.load(panel/'received.npy', mmap_mode='r', allow_pickle=False)
    if iq.shape != (24,360,2,256) or iq.dtype != np.float32:
        raise ValueError('Expected the existing fixed source simulator panel')
    np.save(root/'received.npy', iq[:,take], allow_pickle=False)
    with np.load(panel/'scorer_reference.npz', allow_pickle=False) as values:
        clean = values['clean'][take]
    np.save(root/'paired_clean.npy', clean, allow_pickle=False)


def fit_bank(basis, observed):
    adjoint = basis.conj().swapaxes(-1,-2)
    gram = adjoint @ basis
    ridge = .005*np.trace(gram, axis1=-2, axis2=-1).real/len(TAPS)
    rhs = np.einsum('...ij,...j->...i', adjoint, observed)
    coeff = np.linalg.solve(gram+ridge[...,None,None]*np.eye(len(TAPS)), rhs[...,None])[...,0]
    residual = observed - np.einsum('...ij,...j->...i', basis, coeff)
    indices = (np.abs(residual)**2).mean(-1).argmin(-1)
    return DELAYS[indices], coeff[np.arange(len(indices)), indices]


def pilot_fit(y):
    from .wisig_joint_preamble import waveform
    from .wisig_timing import NOMINAL

    # Estimate CFO from disjoint-from-score receive samples in the repeated STF.
    cross = (y[:,68:96]*y[:,48:76].conj()).sum(-1)
    omega = np.angle(cross)/20
    sync = y * np.exp(-1j*omega[:,None]*np.arange(N))
    relative = FIT[None,:,None] - NOMINAL - DELAYS[:,None,None] - TAPS
    basis = waveform(relative)
    delta, coef = fit_bank(basis, sync[:,None,FIT])
    center = sync[np.arange(len(y))[:,None], 72+delta[:,None]+np.arange(80)]
    repeats = center.reshape(-1,4,20)
    variance = .5*np.mean(np.abs(repeats[:,1::2]-repeats[:,::2])**2, axis=(1,2))
    return delta, coef, omega, variance


def reflect(index):
    return np.where(index<0,-index,np.where(index>=N,2*(N-1)-index,index))


def applied(x, delta, coefficients, omega, indices):
    offsets = indices[None,:,None] - delta[:,None,None] - TAPS
    shifted = x[np.arange(len(x))[:,None,None], reflect(offsets)]
    return np.einsum('ntk,nk->nt', shifted, coefficients)*np.exp(1j*omega[:,None]*indices)


def paired_fit(y, clean):
    offsets = FIT[None,:,None] - DELAYS[:,None,None] - TAPS
    basis = clean[:,reflect(offsets)]
    delta, coef = fit_bank(basis, y[:,None,FIT])
    omega = np.zeros(len(y))
    error = y[:,TEST] - applied(clean,delta,coef,omega,TEST)
    return delta, coef, omega, np.mean(np.abs(error)**2,axis=-1)


def estimate(iq, output, clean=None):
    rows = np.load(iq, allow_pickle=False, mmap_mode='r')
    if rows.shape != (len(CELLS), PACKETS, 2, N):
        raise ValueError('Incorrect source diagnostic panel')
    reference = np.load(clean, allow_pickle=False) if clean is not None else None
    results = []
    for row in rows:
        z = row[:,0].astype(np.float64) + 1j*row[:,1]
        result = pilot_fit(z) if reference is None else paired_fit(z,reference)
        if not all(np.isfinite(v).all() for v in result):
            raise ValueError('Nonfinite estimator output')
        results.append(result)
    delta, coefficient, omega, variance = map(np.stack, zip(*results))
    np.savez(output, delta=delta, coefficient=coefficient, omega=omega, variance=variance)
    print('ESTIMATES_FROZEN', len(rows), PACKETS, flush=True)


def gain(reference, values):
    return (reference.conj()*values).sum(-1)/(np.abs(reference)**2).sum(-1).clip(1e-18)


def read_metrics(clean, truth, oracle_var, coeffs):
    delta, coefficients, omega, variance = coeffs
    main_lag = delta + TAPS[np.abs(coefficients).argmax(1)]
    cand = clean[:,reflect(FIT[None,:]-DELAYS[:,None])]
    corr = np.einsum('nlt,nt->nl',cand.conj(),truth[:,FIT])
    oracle_lag = DELAYS[(np.abs(corr)**2/(np.abs(cand)**2).sum(-1).clip(1e-18)).argmax(1)]
    fitted = applied(clean,delta,coefficients,omega,FIT)
    predicted = applied(clean,delta,coefficients,omega,TEST)
    expected = truth[:,TEST]
    scale = np.mean(np.abs(expected)**2,axis=-1).clip(1e-18)
    identity = clean[:,TEST]*gain(clean[:,FIT],truth[:,FIT])[:,None]
    return dict(
        lag_abs=np.abs(main_lag-oracle_lag),
        identity=np.mean(np.abs(identity-expected)**2,axis=-1)/scale,
        absolute=np.mean(np.abs(predicted-expected)**2,axis=-1)/scale,
        shape=np.mean(np.abs(predicted*gain(fitted,truth[:,FIT])[:,None]-expected)**2,axis=-1)/scale,
        noise_db=10*np.log10(variance.clip(1e-18)/oracle_var),
    )


def describe(rows):
    values = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
    out = {key: np.quantile(value,[.1,.5,.9]).tolist() for key,value in values.items()}
    return dict(records=len(values['lag_abs']), **out,
                lag_within_1=float(np.mean(values['lag_abs']<=1)),
                noise_med_abs_db=float(np.median(np.abs(values['noise_db']))),
                noise_within_3db=float(np.mean(np.abs(values['noise_db'])<=3)))


def score(root, panel):
    # Both complete coefficient archives must exist before opening the oracle.
    estimators = {}
    for method in ('preamble','paired'):
        with np.load(root/f'{method}.npz',allow_pickle=False) as record:
            estimates = tuple(record[key] for key in ('delta','coefficient','omega','variance'))
        if tuple(v.shape for v in estimates) != ((24,30),(24,30,3),(24,30),(24,30)):
            raise ValueError('Incomplete frozen estimator outputs')
        estimators[method] = estimates
    with np.load(panel/'scorer_reference.npz',allow_pickle=False) as reference:
        take = select()
        clean = reference['clean'][take]
        signal = reference['signal'][:,take]
        oracle_var = reference['variance'][:,take]
    score_rows = {}
    for method, estimates in estimators.items():
        measurements = {case: read_metrics(clean,signal[i],oracle_var[i],[v[i] for v in estimates])
                        for i,case in enumerate(CELLS)}
        score_rows[method] = {profile: {route: {view: describe(
            [row for case,row in measurements.items()
             if case[:2]==(profile,route) and (view=='combined' or view==case[2])])
            for view in ('combined', *VIEWS[1:])} for route in ('full','residual')}
            for profile in ('default','strong')}
    write(root/'scores.json',dict(status='SCORED',oracle_last=True,source_packets=PACKETS,
                                 target_reads=0,model_reads=0,results=score_rows))


def dispatch(rt):
    root = Path(rt['run_root'])
    panel = Path(rt['panel_root'])
    root.mkdir()
    state = root/'state.json'
    received = root/'received.npy'
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    tasks = (
        ('prepare', (root,panel)),
        ('estimate', (received,root/'preamble.npz')),
        ('estimate', (received,root/'paired.npz',root/'paired_clean.npy')),
        ('score', (root,panel)),
    )
    try:
        for number,(stage,arguments) in enumerate(tasks):
            write(state,dict(status='RUNNING',stage=stage,index=number),exclusive=False)
            with (root/f'{number}-{stage}.log').open('x') as log:
                p = subprocess.Popen([rt['python'],'-u','-m',rt['runner'],stage,
                                      *(str(arg) for arg in arguments)],
                                     cwd=rt['release_root'],env=env,stdout=log,stderr=subprocess.STDOUT)
                print('ANALYTIC_AUDIT_WORKER',number,stage,p.pid,flush=True)
                if p.wait():
                    raise RuntimeError(f'Stage {number} {stage} failed')
        write(state,dict(status='ANALYZED'),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main import publish as transport

    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    panel = str(base/'runs'/PARENT)
    rt = dict(project=ctx['project'],python=ctx['python'],frozen_sources={},
              run_id=RUN,runner=__spec__.name,code_commit=transport.release_commit(repo),
              release_root=str(base/'releases'/f'{RUN}-release'),run_root=str(base/'runs'/RUN),
              panel_root=panel,external_releases=[panel])
    transport.release(repo,output,rt,ssh_config)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action',required=True)
    release = sub.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        release.add_argument('--'+key,type=Path,required=True)
    for stage,fields in (('dispatch',('runtime',)),('prepare',('root','panel')),
                         ('estimate',('iq','output')),('score',('root','panel'))):
        command = sub.add_parser(stage)
        for field in fields:
            command.add_argument(field,type=Path)
        if stage=='estimate':
            command.add_argument('clean',type=Path,nargs='?')
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action=='dispatch':
        dispatch(read(args['runtime']))
    else:
        {'publish':publish,'prepare':prepare,'estimate':estimate,'score':score}[action](**args)


if __name__=='__main__':
    main()
