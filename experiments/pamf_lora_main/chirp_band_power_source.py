"""Test whether one received packet can identify its weak LoRa spectrum power."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from .antithetic_detail_source import RUN as PARENT, bind_models
from .channel_blind import corruption_score
from .channel_blind_source import partition, setup
from .chirp_band_power import BANDS, COMPONENTS, band_power, estimators, frequency_groups, packet_basis, spectrum
from .complex_vector_source import publish
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .full_packet_views import source_contract, source_role
from .gated_slope_source import restore
from .joint_unroll_source import coordinates
from .offband_noise_source import replay
from .refine_runner import start
from .rf_noise_transport import colored_observation
from .runner import activate


RUN = '20261124-phase1-lora-interleaved-band-power-source-cpu-r01'
PER_RX = 30
SEED = 2026115177
ISOLATION = dict(F_reads=0, V_reads=0, TX_reads=0, target_reads=0, fitted_weights=0)
VIEWS_HELD = (VIEWS[3], VIEWS[-1])


def prior_for(tracker, training):
    out = []
    for part in training.split(96):
        y = coordinates(tracker, part)[0]
        out.append(band_power(spectrum(y, packet_basis(y)), frequency_groups(y.shape[-1], y.device)))
    values = torch.cat(out)
    return values.mean(0), values.var(0, unbiased=False)


def visible(iq, tracker, model, threshold, prior):
    frame = coordinates(tracker, iq)
    y, rot, _, rms = frame
    estimates, basis, bands, raw, alias = estimators(y, *prior)
    active = corruption_score(y) > threshold
    if active.any():
        f = tuple(v[active] for v in frame)
        recovered, = restore(iq, active, model, f, (.5,))
    else:
        recovered = iq
    position = rot * rms
    estimates['frozen_recovery_power'] = band_power(spectrum(complex_iq(recovered)/position, basis), bands)
    return estimates, active, position, raw, alias


def correlation(prediction, reference):
    a, b = prediction - prediction.mean(), reference - reference.mean()
    denominator = a.norm() * b.norm()
    return float(a.dot(b)/denominator) if denominator > 1e-12 else None


def statistics(estimate, reference, bootstrap=None, prior_error=None):
    wanted = reference.mean()
    error = (estimate-reference).square()
    out = dict(power_mean=float(estimate.mean()), rmse_over_power_mean=float(error.mean().sqrt()/wanted),
               mse=float(error.mean()), signed_bias_over_mean=float((estimate-reference).mean()/wanted),
               negative_count=int((estimate<0).sum()), correlation=correlation(estimate,reference))
    if bootstrap is not None:
        differences = ((error-prior_error)[bootstrap].mean(-1)/prior_error.mean().clamp_min(1e-14))
        out['mse_change_relative_to_prior_ci'] = torch.quantile(differences, differences.new_tensor([.025, .975])).tolist()
    return out


def evaluate(iq, noise, tracker, model, threshold, prior, rx):
    positions = []
    measured = {}
    activation = []
    raw_part, alias_part = [], []
    for part in iq.split(14):
        samples, active, basis, raw, alias = visible(part, tracker, model, threshold, prior)
        for key, val in samples.items():
            measured.setdefault(key, []).append(val)
        positions.append(basis)
        activation.append(active)
        raw_part.append(raw)
        alias_part.append(alias)
    measured = {k:torch.cat(v) for k,v in measured.items()}
    pos = torch.cat(positions)
    observed_active = torch.cat(activation)
    projection = packet_basis(complex_iq(iq))
    groups = frequency_groups(iq.shape[-1], iq.device)
    aliased = tuple((g+iq.shape[-1]//2)%iq.shape[-1] for g in groups)

    # Only diagnostic code reconstructs the matched noiseless reference.
    physical = spectrum(complex_iq(iq-noise)/pos, projection)
    exact, exact_alias = band_power(physical,groups), band_power(physical,aliased)
    added = spectrum(complex_iq(noise)/pos, projection)
    variance_actual = band_power(added, groups)
    variance_alias = band_power(added, aliased)
    measured['isolated_exact_noise_power_control'] = torch.cat(raw_part)-variance_actual
    observed_alias = torch.cat(alias_part)
    if not all(v.isfinite().all() for v in (*measured.values(),exact,observed_alias)) or not (exact>0).all():
        raise ValueError('Nonfinite or zero-power LoRa diagnostic')

    rng = np.random.default_rng(SEED)
    indices = [np.flatnonzero(rx==name) for name in SOURCE]
    draws = tensor(np.concatenate([rng.choice(group, size=(1600,len(group))) for group in indices],axis=-1))
    indices = [tensor(ix) for ix in indices]
    cells={}
    for i,component in enumerate(COMPONENTS):
        cells[component]={}
        for j,band in enumerate(BANDS):
            proper=exact[:,i,j]
            prior_error=(measured['prior'][:,i,j]-proper).square()
            positive = observed_active
            cells[component][band] = dict(
                reference_mean=float(proper.mean()),
                true_alias_over_near=float(exact_alias[:,i,j].sum()/proper.sum()),
                actual_noise_over_reference=float(variance_actual[:,i,j].sum()/proper.sum()),
                alias_noise_over_near_noise=float(variance_alias[:,i,j].sum()/variance_actual[:,i,j].sum().clamp_min(1e-12)),
                predictions={k:statistics(v[:,i,j],proper,draws,prior_error) for k,v in measured.items()},
                active_predictions={k:statistics(v[positive,i,j],proper[positive]) for k,v in measured.items()}
                                   if positive.any() else {},
                by_rx={name:dict(n=len(ix),truth_mean=float(proper[ix].mean()),
                                predictions={k:statistics(v[ix,i,j],proper[ix]) for k,v in measured.items()})
                       for name,ix in zip(SOURCE,indices,strict=True)})
    return dict(n=len(iq),active=int(observed_active.sum()),cells=cells)


def model_binding(rt, contract, *, device='cpu'):
    root=Path(rt['run_root']).parent/PARENT
    if read(root/'state.json')['status']!='SOURCE_SCORED':
        raise ValueError('Need completed source-only parent recovery')
    _, arms, threshold=bind_models(root,contract,device)
    return arms[1],threshold


@torch.no_grad()
def smoke(rt):
    torch.set_num_threads(2)
    (clean,cal,_),contract=setup(rt,'cpu',smoke=True)
    tracker=PacketRestorer('consensus').eval()
    model,threshold=model_binding(rt,contract)
    prior=prior_for(tracker,clean)
    y=real_iq(colored_observation(complex_iq(cal),torch.Generator().manual_seed(SEED)))
    y=y[corruption_score(coordinates(tracker,y)[0])>threshold][:2]
    checked,active,_,_,_=visible(y,tracker,model,threshold,prior)
    if (len(y)!=2 or not active.all() or not all(v.shape==(2,2,2) and v.isfinite().all() for v in checked.values())
            or torch.equal(checked['median_raw'],checked['parity_raw'])):
        raise ValueError('Real checkpoint amplitude diagnostic smoke failed')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',source_U=len(y),**ISOLATION))
    print('LORA_BAND_POWER_REAL_CHECKPOINT_PASS',flush=True)


@torch.no_grad()
def probe(rt):
    torch.set_num_threads(2)
    start_time=perf_counter()
    source,root=Path(rt['source_root']),Path(rt['run_root'])
    raw,rx,ids,_=source_role(source,source_contract(source),'U_s')
    (training,_,held),contract=setup(rt,'cpu')
    indices=partition(rx)[2]
    held_ids=np.concatenate([group[:PER_RX] for group in np.split(indices,len(SOURCE))])
    clean=torch.cat([group[:PER_RX] for group in held.chunk(len(SOURCE))])
    tracker=PacketRestorer('consensus').eval()
    model,threshold=model_binding(rt,contract)
    prior=prior_for(tracker,training)
    reported={}
    # White positive control is generated in the clean tracked frame at fixed noise power.
    y,rot,_,rms=coordinates(tracker,clean)
    rng=torch.Generator().manual_seed(SEED)
    noise=.025*torch.randn(y.shape,dtype=y.dtype,generator=rng)*rot*rms
    added=real_iq(noise)
    reported['generic_white']=evaluate(clean+added,added,tracker,model,threshold,prior,rx[held_ids])
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            reported[route]={}
            for name in VIEWS_HELD:
                jobs=((route,name,raw[i],rx[i],ids[i]) for i in held_ids)
                iq,epsilon=(tensor(np.stack(p)) for p in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                reported[route][name]=evaluate(iq,epsilon,tracker,model,threshold,prior,rx[held_ids])
                print('LORA_BAND_POWER_SOURCE',route,name,len(iq),reported[route][name]['active'],flush=True)
    write(root/'source-u-band-power.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',n=len(clean),prior_source_U=len(training),
          worker_seconds=perf_counter()-start_time,prior_mean=prior[0].tolist(),prior_variance=prior[1].tolist(),
          reference_used_after_observed_estimators=True,rows=reported,**ISOLATION))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    try:
        for phase in ('smoke','probe'):
            if start(rt,phase,'source',module=rt['runner']).wait():
                raise RuntimeError(f'Weak-band received power {phase} failed')
        write(root/'state.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE'))
    except Exception as error:
        write(root/'state.json',dict(status='FAILED',error=str(error)))
        raise


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    actions=cli.add_subparsers(dest='action',required=True)
    pub=actions.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        pub.add_argument('--'+name,type=Path,required=True)
    for phase in ('dispatch','smoke','probe'):
        stage=actions.add_parser(phase)
        stage.add_argument('runtime',type=Path)
        if phase!='dispatch':
            stage.add_argument('row',choices=('source',))
    args=vars(cli.parse_args())
    action=args.pop('action')
    if action=='publish':
        return publish(**args,run=RUN,runner=__spec__.name)
    rt=read(args['runtime'])
    if action=='dispatch':
        return dispatch(rt)
    activate(rt)
    return smoke(rt) if action=='smoke' else probe(rt)


if __name__=='__main__':
    main()
