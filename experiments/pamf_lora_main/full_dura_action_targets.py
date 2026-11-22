"""Generate and test LoRa full-channel action supervision on source L/U packets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import SOURCE, VIEWS
from .full_dura_action_probe import FIT_PROBES, TOTAL_PROBES, apply_action, chirp_nmse, distribution, fit_action, source_probes
from .full_packet_data import COUNTS
from .full_packet_views import source_role
from .full_restore_draw import DRAW, RUN as DRAW_RUN, load_draw
from .full_supervised import handles
from .runner import activate


ACTION_BANK='source_full_action'


def full_teacher(raw, rx, identity, scene, received, namespace, realization, *, assess=True):
    probes,out,clean,response,noise=source_probes(
        raw,rx,identity,scene,received,namespace=namespace,realization=realization,
        probe_count=TOTAL_PROBES if assess else FIT_PROBES)
    fitted=fit_action(probes[:FIT_PROBES],out[:FIT_PROBES],mirror=False)
    clean_scale=np.sqrt(np.mean(np.abs(clean)**2))
    observed_scale=np.sqrt(np.mean(np.abs(response+noise)**2))
    main=fitted[...,0]*(clean_scale/observed_scale)
    coeff=np.stack((main.real,main.imag,np.zeros_like(main.real),np.zeros_like(main.imag)),axis=1)
    variance=.5*np.mean(np.abs(noise/observed_scale)**2,axis=-1)
    fir=np.zeros((len(received),2,64),dtype=np.float32)
    fir[:,0,0]=1.
    target=dict(received=received,kernel=fir,coefficients=coeff.astype(np.float32),
                variance=np.clip(variance,1e-8,1.).astype(np.float32))
    if (any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in target.values())
            or target['coefficients'].shape!=(len(received),4,256)):
        raise ValueError('Full teacher differs from the original DURA action array interface')
    if not assess:
        return target,None
    truth=out[FIT_PROBES:]
    held=chirp_nmse(apply_action(probes[FIT_PROBES:],fitted),truth)
    reconstructed=apply_action(clean,fitted)
    signal=chirp_nmse(reconstructed,response)
    mismatch=np.mean(np.abs(reconstructed-response)**2,axis=-1)
    noise_power=np.mean(np.abs(noise)**2,axis=-1)
    error_over_noise=mismatch/np.maximum(noise_power,1e-8)
    return target,dict(held_probe_nmse=held.reshape(-1),actual_packet_nmse=signal.reshape(-1),
                       mismatch_over_added_noise=error_over_noise,variance=variance,
                       coefficient_rms=np.sqrt(np.mean(np.abs(main)**2,axis=-1)))


def bank_row(item):
    target,_=full_teacher(*item,assess=False)
    return target['coefficients'][:,:2],target['variance']


def build_bank(rt):
    source,cache,c=handles(rt)
    root=Path(rt['run_root'])
    out=root/ACTION_BANK
    out.mkdir(exist_ok=False)
    count=COUNTS['L_s']+COUNTS['U_s']
    main=np.lib.format.open_memmap(out/'main.npy',mode='w+',dtype=np.float32,shape=(2,count,6,2,256))
    variance=np.lib.format.open_memmap(out/'variance.npy',mode='w+',dtype=np.float32,shape=(2,count,6))
    at=0
    with ProcessPoolExecutor(max_workers=12) as pool:
        for role,namespace in (('L_s','source'),('U_s','source-U')):
            raw,receivers,ids,slots=source_role(source,c,role)
            labeled=role=='L_s'
            primary=np.load(cache/f'{role}.npy' if labeled else cache/'U_s_practical.npy',
                            allow_pickle=False,mmap_mode='r')
            independent=load_draw(root.parent/DRAW_RUN,role)
            scene_ids=slots%6+1
            if labeled:
                np.save(out/'L_scene.npy',scene_ids.astype(np.int64),allow_pickle=False)
            for d in range(2):
                items=((raw[i],receivers[i],ids[i],VIEWS[view],
                        independent[i] if d else (primary[view,i] if labeled else primary[i]),
                        namespace,DRAW if d else None) for i,view in enumerate(scene_ids))
                for i,(coef,var) in enumerate(pool.map(bank_row,items,chunksize=6)):
                    main[d,at+i]=coef
                    variance[d,at+i]=var
                    if (i+1)%1000==0 or i+1==len(ids):
                        print('LORA_FULL_ACTION_READY',role,d,i+1,len(ids),flush=True)
            at+=len(ids)
    if at!=count:
        raise ValueError('Incomplete full action source roles')
    main.flush()
    variance.flush()
    write(out/'complete.json',dict(status='SOURCE_FULL_ACTION_READY',source_contract=c,n=count,
          draws=2,chirps=6,exact_noisy_view_replay=True,fit_probes=FIT_PROBES,
          roles=['L_s','U_s'],F_reads=0,TX_reads=0,V_reads=0,target_reads=0))


def pilot(rt):
    source,cache,c=handles(rt)
    root=Path(rt['run_root'])
    rng=np.random.default_rng(2026112272)
    rows={}
    for role,namespace in (('L_s','source'),('U_s','source-U')):
        raw,receivers,ids,slots=source_role(source,c,role)
        if role=='L_s':
            primary=np.load(cache/'L_s.npy',allow_pickle=False,mmap_mode='r')
        else:
            primary=np.load(cache/'U_s_practical.npy',allow_pickle=False,mmap_mode='r')
        independent=load_draw(root.parent/DRAW_RUN,role)
        for j,scene in enumerate(VIEWS[1:]):
            for rx in SOURCE:
                can_use=(receivers==rx)&(slots%6==j)
                i=int(rng.choice(np.flatnonzero(can_use)))
                for draw,view,realization in (('primary',primary[j+1,i] if role=='L_s' else primary[i],None),
                                               ('independent',independent[i],DRAW)):
                    _,metrics=full_teacher(raw[i],rx,ids[i],scene,view,namespace,realization)
                    group=rows.setdefault((scene,role,draw),{k:[] for k in metrics})
                    for name,x in metrics.items():
                        group[name].extend(x.tolist())
    statistics={f'{v}/{r}/{d}':{k:distribution(values) for k,values in row.items()}
                for (v,r,d),row in rows.items()}
    return dict(status='FULL_ACTION_TARGETS_PILOT_COMPLETE',source_packets=2*len(SOURCE)*6,
                exact_noisy_view_replay=True,roles=['L_s','U_s'],fit_probes=FIT_PROBES,
                TX_reads=0,F_reads=0,V_reads=0,target_reads=0,groups=statistics)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('config',type=Path)
    a=p.parse_args()
    r=read(a.config)
    if set(r)&{'truth','target_views','target_truth','F_checkpoint'}:
        raise ValueError('Expected the source-only runtime')
    activate(r)
    print(json.dumps(pilot(r),allow_nan=False),flush=True)
