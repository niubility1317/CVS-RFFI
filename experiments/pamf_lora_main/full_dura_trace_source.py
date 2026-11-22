"""Source-only physical signal components for LoRa's fixed practical draw."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import VIEWS
from .full_packet_views import source_role
from .full_restore_draw import DRAW, RUN as DRAW_RUN, load_draw
from .full_supervised import handles
from .packet_physics import CONTEXT, channel, chirp_samples, chirp_windows, hardware


TRACE_RUN='20261122-phase1-lora-dura-r69-physical-trace-m2-r01'
TRACE_ROLES=('L_s','U_s')


def source_trace(item):
    from leo_practical.channel import ChannelStream, normalize_input, stable_seed
    role,z,rx,pid,slot=item
    scene=VIEWS[1+int(slot)%6]
    namespace='source-U' if role=='U_s' else 'source'
    z,_=normalize_input(z)
    cfg=channel(scene)
    seed=stable_seed(2026112207,namespace,scene,rx,str(pid),'draw',DRAW)
    obs,meta,parts=ChannelStream(cfg,seed,hardware(scene,f'{namespace}/{rx}')).process(
        z[CONTEXT:],preceding_context=z[:CONTEXT],return_traces=True)
    delay=meta['numerical_filter_latency_samples']
    if (meta['channel_equalization_applied'] or meta['boundary']!='provided_context'
            or delay!=cfg.fractional_delay_half_length):
        raise ValueError('LoRa physical trace processing differs from the frozen source draw')
    shift=delay-CONTEXT
    received=chirp_windows(obs,offset=shift)
    signal=chirp_samples(parts['pre_agc_signal'],offset=shift)
    noise=chirp_samples(parts['pre_agc_noise'],offset=shift)
    scale=np.sqrt(np.mean(np.abs(signal+noise)**2))
    if not np.isfinite(scale) or scale<=1e-12:
        raise ValueError('Invalid same-coordinate physical trace')
    signal=np.stack((signal.real,signal.imag),axis=1)/scale
    return received,np.asarray(signal,dtype=np.float32)


def trace_indices(role,n):
    q=np.arange(n,dtype=np.int64)
    return q if role=='L_s' else q[q%5!=0][::3]


def build(rt):
    root=Path(rt['run_root'])
    src,_,contract=handles(rt)
    out=root/'source_trace'
    out.mkdir(exist_ok=False)
    counts={}
    with ProcessPoolExecutor(6) as pool:
        for role in TRACE_ROLES:
            x,rx,pid,slot=source_role(src,contract,role)
            source=load_draw(root.parent/DRAW_RUN,role)
            idx=trace_indices(role,len(x))
            path=out/f'{role}.npy'
            data=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(idx),6,2,256))
            jobs=((role,x[i],rx[i],pid[i],slot[i]) for i in idx)
            for step,(r,s) in enumerate(pool.map(source_trace,jobs,chunksize=20)):
                if not np.array_equal(r,source[idx[step]]) or s.shape!=(6,2,256) or not np.isfinite(s).all():
                    raise ValueError('LoRa trace is not physically identical to the registered source realization')
                data[step]=s
                if (step+1)%1000==0:
                    print('LORA_R69_TRACE_READY',role,step+1,len(idx),flush=True)
            data.flush()
            counts[role]=len(idx)
    write(out/'complete.json',dict(status='SOURCE_TRACE_READY',source_contract=contract,counts=counts,
                                   draw_run=DRAW_RUN,draw=DRAW,TX_reads=0,F_reads=0,target_reads=0))


def read_trace(root,c,role):
    out=root/'source_trace'
    meta=read(out/'complete.json')
    idx=trace_indices(role,c['counts'][role])
    if (meta['status']!='SOURCE_TRACE_READY' or meta['source_contract']!=c or meta['draw_run']!=DRAW_RUN
            or meta['draw']!=DRAW or meta['counts'][role]!=len(idx)
            or any(meta[k] for k in ('TX_reads','F_reads','target_reads'))):
        raise ValueError('LoRa physical traces do not match the same source role and draw')
    a=np.load(out/f'{role}.npy',allow_pickle=False,mmap_mode='r')
    if a.shape!=(len(idx),6,2,256) or a.dtype!=np.float32:
        raise ValueError('Incomplete source-only LoRa physical trace')
    return idx,a
