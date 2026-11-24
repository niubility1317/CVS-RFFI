"""Compare received-noise inverse regularizers in an isolated known-action source audit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read,tensor,write
from .chirp_spectrum_source import freeze,read_frozen
from .complex_vector_source import load_reader,publish
from .data import SOURCE,VIEWS
from .denoise_source import normalized,replay,summarize
from .discrepancy_inverse import inverse_row
from .full_packet_views import source_contract,source_role
from .joint_unroll_source import coordinates,source_control
from .learned_inverse_source import panel
from .packet_runner import schedule
from .physical_inverse_source import source_trace_implementation
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_signal_probe import observed_noise_rms


RUN = '20261124-phase1-lora-received-discrepancy-inverse-source-m2-r01'
PER_RX = 60
MODES = ('direct','reference','exact80_direct','raw_damped','weighted_damped',
         'raw_stop1','raw_stop075','weighted_stop1','weighted_stop075','exact80_damped')
STOP_MODES = MODES[5:9]
PAIRS = tuple((mode,'raw_damped') for mode in MODES[4:9]) + (
    ('weighted_stop1','raw_stop1'),('weighted_stop075','raw_stop075'),('exact80_damped','exact80_direct'))


def noise_and_reference(wave,baseline,tracker):
    control,active = baseline(wave)
    proxy = wave.new_zeros(wave.shape[:2])
    if active.any():
        observed,_,_,rms = coordinates(tracker,wave[active])
        proxy[active] = (observed_noise_rms(observed)*rms).squeeze(-1)
    return control,active,proxy


def solver_summary(steps):
    stats=np.asarray(steps,dtype=np.int32).reshape(-1,len(STOP_MODES))
    rows={}
    for name,iteration in zip(STOP_MODES,stats.T,strict=True):
        reached=iteration[iteration>0]
        rows[name]=dict(reached=int(len(reached)),bypassed=int(np.sum(iteration==0)),
                        no_crossing=int(np.sum(iteration<0)),
                        median_reached=float(np.median(reached)) if len(reached) else None)
    return rows


@torch.no_grad()
def evaluate(rt,route,*,smoke=False):
    started=perf_counter()
    root,device=Path(rt['run_root']),'cuda:0'
    source,cache,contract,baseline,tracker=source_control(rt,route,device)
    f=load_reader(root,contract,device)
    raw,rx,ids,_=source_role(source,contract,'U_s' if smoke else 'V')
    if smoke:
        chosen=np.concatenate([np.flatnonzero(rx==key)[:2] for key in SOURCE])
    else:
        chosen=panel(rx,PER_RX)
    jobs=list(zip(raw[chosen],rx[chosen],ids[chosen],strict=True))
    official=None if smoke or cache is None else np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    metrics,predictions={},{}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for view in ((VIEWS[-1],) if smoke else VIEWS):
            captures,isolated_noise=(np.stack(parts) for parts in zip(
                *pool.map(partial(replay,route,view),jobs,chunksize=6),strict=True))
            if official is not None and not np.array_equal(captures,official[VIEWS.index(view),chosen]):
                raise ValueError('Raw source replay differs from the registered full route')
            prepared=[noise_and_reference(value,baseline,tracker) for value in tensor(captures,device).split(30)]
            ref,active,noise_std=(torch.cat(values) for values in zip(*prepared,strict=True))
            picked=np.flatnonzero(active.tolist())
            raw_gpu=tensor(captures,device)
            amended=raw_gpu.unsqueeze(0).repeat(len(MODES)-3,1,1,1,1)
            phys_error=0.
            iterates=()
            if len(picked):
                if view=='clean':
                    raise ValueError('Historical source clean bypass changed')
                proxies=noise_std.cpu().tolist()
                locked=[(*jobs[i],captures[i],np.asarray(proxies[i],dtype=np.float64)) for i in picked]
                waves,iterates,errors=zip(*pool.map(partial(inverse_row,route,view),locked,chunksize=2),strict=True)
                if not np.isfinite(np.asarray(errors)).all():
                    raise FloatingPointError('Invalid projected action mismatch')
                phys_error=max(errors)
                amended[:,active]=tensor(np.stack(waves,axis=1),device)
            # Known-source noise enters only the explicitly labelled exact80 controls.
            noise=tensor(isolated_noise,device)
            known80=torch.where(active[:,None,None,None],normalized(raw_gpu-.8*noise,raw_gpu),raw_gpu)
            curves=torch.cat((torch.stack((raw_gpu,ref,known80)),amended),dim=0)
            if not torch.isfinite(curves).all():
                raise FloatingPointError('Nonfinite diagnostic inverse input')
            logits=[f(chunk.flatten(0,1))[0] for chunk in curves.split(18,dim=1)]
            if not all(torch.isfinite(values).all() for values in logits):
                raise FloatingPointError('Nonfinite diagnostic inverse prediction')
            predictions[view]=np.concatenate([values.argmax(1).reshape(len(MODES),-1).cpu().tolist()
                                              for values in logits],axis=1)
            metrics[view]=dict(active=len(picked),replay_relative_error=phys_error,
                               discrepancy_stops=solver_summary(iterates))
            print('LORA_DISCREPANCY_SOURCE',route,view,len(picked),phys_error,flush=True)
    if smoke:
        if not metrics[VIEWS[-1]]['active']:
            raise ValueError('Real source U operation smoke did not exercise the inverse')
        write(root/f'smoke-{route}.json',dict(status='PASS',source_u=len(chosen),target_reads=0,
              TX_reads=0,metrics=metrics))
    else:
        freeze(root,route,source,chosen,ids,predictions,metrics,MODES)
        write(root/f'{route}-inverse-cost.json',dict(worker_seconds=perf_counter()-started,
              cpu_readout_workers=8,trainable_parameters=0,actual_gpu=torch.cuda.get_device_name()))


def independent_score(rt):
    folder,source=Path(rt['run_root']),Path(rt['source_root'])
    frozen={r:read_frozen(folder,r,MODES,per_rx=PER_RX) for r in ROUTES}
    if frozen['full'][0]!=frozen['residual'][0]:
        raise ValueError('Two routes need identical frozen physical source V identifiers')
    ids=np.asarray(source_contract(source)['role_ids']['V'])
    lookup={key:i for i,key in enumerate(ids)}
    chosen=np.asarray([lookup[k] for k in frozen['full'][0]])
    with np.load(source/'V_metadata.npz',allow_pickle=False) as m:
        if not np.array_equal(chosen,panel(np.asarray(SOURCE)[m['rx_i']],PER_RX)):
            raise ValueError('Unexpected source V panel after prediction')
    for route,(_,rows,metrics) in frozen.items():
        summarize(folder,route,source,chosen,ids,rows,metrics,MODES,
                  predictions_written=True,comparisons=PAIRS,per_rx=PER_RX)
    print('LORA_DISCREPANCY_TRUTH_LAST_COMPLETE',len(chosen),flush=True)


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    launch=partial(start,module=rt['runner'])
    try:
        for action in ('smoke','evaluate'):
            schedule(rt,[(action,key) for key in ROUTES],launch)
        if launch(rt,'score','source').wait():
            raise RuntimeError('Independent LoRa inverse diagnostic scorer failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root/'state.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    entries=cli.add_subparsers(dest='action',required=True)
    release=entries.add_parser('publish')
    for key in ('repo','output','context','ssh-config','parent-submit'):
        release.add_argument('--'+key,type=Path,required=True)
    for phase in ('dispatch','smoke','evaluate','score'):
        sub=entries.add_parser(phase)
        sub.add_argument('runtime',type=Path)
        if phase!='dispatch':
            sub.add_argument('row',choices=('source',) if phase=='score' else ROUTES)
    options=vars(cli.parse_args())
    action=options.pop('action')
    if action=='publish':
        return publish(**options,run=RUN,runner=__spec__.name)
    rt=read(options['runtime'])
    if action=='dispatch':
        return dispatch(rt)
    if action=='score':
        return independent_score(rt)
    activate(rt)
    source_trace_implementation()
    evaluate(rt,options['row'],smoke=action=='smoke')


if __name__=='__main__':
    main()
