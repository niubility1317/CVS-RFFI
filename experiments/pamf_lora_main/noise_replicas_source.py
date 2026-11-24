"""Quantify independent noise opportunities on matched source LoRa packets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read,tensor,write
from .channel_blind import corruption_score,zero_carrier
from .chirp_spectrum_source import freeze,read_frozen
from .complex_vector_source import load_reader,publish
from .data import SOURCE,VIEWS
from .denoise_source import normalized,summarize
from .dura_repair import complex_iq,real_iq
from .full_packet_views import source_contract,source_role
from .joint_unroll_source import coordinates,source_control
from .learned_inverse_source import panel
from .noise_replicas import DRAWS,INVERSE_DRAWS,replicate
from .packet_physics import chirp_windows
from .packet_runner import schedule
from .physical_inverse_source import source_trace_implementation
from .refine_runner import start
from .runner import activate
from .spectral_unroll import ROUTES
from .weak_guard_source import parent_model


RUN = '20261124-phase1-lora-noise-replicas-source-m2-r01'
PER_RX = 60
MODES = ('direct','reference','rank_half','received_pool_half',*(f'average_{k}' for k in DRAWS),
         'average_25_common','average_25_private','exact80_same_noise',*(f'inverse_{k}' for k in INVERSE_DRAWS))
PAIRS = (('received_pool_half','rank_half'), *((f'average_{k}','direct') for k in DRAWS),
         ('average_25_common','direct'),('average_25_private','direct'),
         ('average_25','exact80_same_noise'),('inverse_8','average_8'),('inverse_25','average_25'))


def components(edit,basis):
    frame=complex_iq(edit)/basis
    common=frame.mean(1,keepdim=True)
    return real_iq(common*basis),real_iq((frame-common)*basis)


@torch.no_grad()
def readouts(iq,truth,averages,inverted,baseline,tracker,rank,threshold):
    reference,active=baseline(iq)
    y,rot,q,rms=coordinates(tracker,iq)
    basis=rot*rms
    rank_active=corruption_score(y)>threshold
    rank_iq=iq.clone()
    if rank_active.any():
        a,b,c=(v[rank_active] for v in (y,q,rot))
        proposal=rank(a,b,c)[-1]
        rank_iq[rank_active]+=real_iq((torch.lerp(a,proposal,.5)-a)*basis[rank_active])
    residual=zero_carrier(y)
    shared=residual.mean(1,keepdim=True)
    pool=iq+real_iq(.5*(shared-residual)*basis)
    common,private=components(averages[-1]-iq,basis)
    lines=(pool,*averages,iq+common,iq+private,torch.lerp(iq,truth,.8))
    selected=active[:,None,None,None]
    applied=[torch.where(selected,normalized(x,iq),iq) for x in lines]
    final=[torch.where(selected,x,iq) for x in inverted]
    return torch.stack((iq,reference,rank_iq,*applied,*final)),int(active.sum())


def noise_reduction(raw,truth,series):
    original=np.sum((raw.astype(np.float64)-truth)**2)
    if original<=0:
        raise ValueError('Replica matrix needs an actual positive added noise energy')
    return {str(k):float(np.sum((samples.astype(np.float64)-truth)**2)/original)
            for k,samples in zip(DRAWS,series,strict=True)}


@torch.no_grad()
def evaluate(rt,route,*,smoke=False):
    started=perf_counter()
    root,device=Path(rt['run_root']),'cuda:0'
    source,cache,contract,baseline,tracker=source_control(rt,route,device)
    rank,threshold=parent_model(root,contract,device)
    classifier=load_reader(root,contract,device)
    raw,rx,ids,_=source_role(source,contract,'U_s' if smoke else 'V')
    chosen=(np.concatenate([np.flatnonzero(rx==key)[:2] for key in SOURCE]) if smoke else panel(rx,PER_RX))
    inputs=list(zip(raw[chosen],rx[chosen],ids[chosen],strict=True))
    expected=None if smoke or cache is None else np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    scores,metrics={},{}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for view in ((VIEWS[-1],) if smoke else VIEWS):
            if view=='clean':
                observed=np.stack([chirp_windows(x[0]) for x in inputs])
                clear=observed
                average=[observed]*len(DRAWS)
                inverses=[observed]*len(INVERSE_DRAWS)
                ratios={}
                replay_error=0.
            else:
                a,b,c,d,e=zip(*pool.map(partial(replicate,route,view),inputs,chunksize=3),strict=True)
                observed,clear=np.stack(a),np.stack(b)
                average=[np.stack(v) for v in zip(*c,strict=True)]
                inverses=[np.stack(v) for v in zip(*d,strict=True)]
                replay_error=float(max(e))
                ratios=noise_reduction(observed,clear,average)
            if expected is not None and not np.array_equal(observed,expected[VIEWS.index(view),chosen]):
                raise ValueError('Fixed native replica diagnostic changed the original received route')
            stack=tensor(np.stack((observed,clear,*average,*inverses)),device)
            results=[]
            n_active=0
            for chunk in stack.split(16,dim=1):
                values,n=readouts(chunk[0],chunk[1],chunk[2:2+len(DRAWS)],chunk[2+len(DRAWS):],
                                  baseline,tracker,rank,threshold)
                logits=classifier(values.flatten(0,1))[0]
                if (values.shape[0]!=len(MODES) or not torch.isfinite(logits).all()
                        or (view=='clean' and (n or not torch.equal(values[2:],chunk[0].expand_as(values[2:]))))):
                    raise ValueError('Invalid finite readouts or source clean bypass')
                results.append(logits.argmax(1).reshape(len(MODES),-1).cpu().tolist())
                n_active+=n
            scores[view]=np.concatenate(results,axis=1)
            metrics[view]=dict(active=n_active,replay_relative_error=replay_error,noise_power_ratio=ratios)
            print('LORA_REPLICA_SOURCE',route,view,len(chosen),n_active,ratios,flush=True)
    if smoke:
        observed=metrics[VIEWS[-1]]
        if not observed['active'] or not .02<observed['noise_power_ratio']['25']<.065:
            raise ValueError('Real source smoke did not receive the expected independent noise replicas')
        write(root/f'smoke-{route}.json',dict(status='PASS',source_U=len(chosen),target_reads=0,TX_reads=0,**observed))
    else:
        freeze(root,route,source,chosen,ids,scores,metrics,MODES)
        write(root/f'{route}-inverse-cost.json',dict(worker_seconds=perf_counter()-started,
              cpu_workers=8,trainable_parameters=0,actual_gpu=torch.cuda.get_device_name()))


def independent_score(rt):
    root,source=Path(rt['run_root']),Path(rt['source_root'])
    frozen={r:read_frozen(root,r,MODES,per_rx=PER_RX) for r in ROUTES}
    if frozen['full'][0]!=frozen['residual'][0]:
        raise ValueError('Both routes must freeze identical source packets before any labels')
    ids=np.asarray(source_contract(source)['role_ids']['V'])
    index={key:i for i,key in enumerate(ids)}
    chosen=np.asarray([index[key] for key in frozen['full'][0]])
    with np.load(source/'V_metadata.npz',allow_pickle=False) as metadata:
        if not np.array_equal(chosen,panel(np.asarray(SOURCE)[metadata['rx_i']],PER_RX)):
            raise ValueError('Source V physical packet panel changed')
    for route,(_,predictions,metrics) in frozen.items():
        summarize(root,route,source,chosen,ids,predictions,metrics,MODES,
                  predictions_written=True,comparisons=PAIRS,per_rx=PER_RX)
    print('LORA_REPLICA_TRUTH_LAST_COMPLETE',len(chosen),flush=True)


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir()
    call=partial(start,module=rt['runner'])
    try:
        for phase in ('smoke','evaluate'):
            schedule(rt,[(phase,r) for r in ROUTES],call)
        if call(rt,'score','source').wait():
            raise RuntimeError('Independent noise replica scoring failed')
        write(root/'state.json',dict(status='SOURCE_SCORED'))
    except Exception as exc:
        write(root/'state.json',dict(status='FAILED',error=str(exc)))
        raise


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    actions=cli.add_subparsers(dest='action',required=True)
    release=actions.add_parser('publish')
    for name in ('repo','output','context','ssh-config','parent-submit'):
        release.add_argument('--'+name,type=Path,required=True)
    for action in ('dispatch','smoke','evaluate','score'):
        sub=actions.add_parser(action)
        sub.add_argument('runtime',type=Path)
        if action!='dispatch':
            sub.add_argument('row',choices=('source',) if action=='score' else ROUTES)
    args=vars(cli.parse_args())
    action=args.pop('action')
    if action=='publish':
        return publish(**args,run=RUN,runner=__spec__.name)
    rt=read(args['runtime'])
    if action=='dispatch':
        return dispatch(rt)
    if action=='score':
        return independent_score(rt)
    activate(rt)
    source_trace_implementation()
    evaluate(rt,args['row'],smoke=action=='smoke')


if __name__=='__main__':
    main()
