"""Make aligned seven-view LoRa caches from blind 1 MHz packet archives."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import ROLES, SOURCE, TARGET, VIEWS
from .full_packet_data import COUNTS, LIMIT, PROTOCOL
from .packet_data import CHIRPS, PACKET_LEN
from .packet_physics import NATIVE_RATE, aligned_view, chirp_windows, save_views
from .runner import activate

WORKERS=16


def source_contract(root):
    c=read(root/'contract.json')
    if (c['protocol']!=PROTOCOL or c['counts']!=COUNTS or not c['source_only']
            or c['source_receivers']!=list(SOURCE) or c['chirps']!=list(CHIRPS) or c['input_fs']!=NATIVE_RATE):
        raise ValueError('Registered full-rate source LoRa contract required')
    return c


def target_contract(root):
    c=read(root/'contract.json')
    if c!=dict(protocol=PROTOCOL,receivers=list(TARGET),per_receiver=LIMIT,TX_truth_read=False,
               input_fs=NATIVE_RATE):
        raise ValueError('Registered full-rate TX-blind target LoRa contract required')
    return c


def native(root, name, n):
    x=np.load(root/f'{name}_packets.npy',mmap_mode='r',allow_pickle=False)
    if x.shape!=(n,PACKET_LEN) or x.dtype!=np.complex64:
        raise ValueError(f'Incorrect physical native packet matrix {name}')
    return x


def source_role(root,c,role):
    x=native(root,role,c['counts'][role])
    with np.load(root/f'{role}_metadata.npz',allow_pickle=False) as m:
        if ('y' in m.files)!=(role!='U_s') or m['ids'].tolist()!=c['role_ids'][role]:
            raise ValueError('Full LoRa source role order or TX visibility differs')
        return x,np.asarray(SOURCE)[m['rx_i']],m['ids'],m['base_index']


def practical_u(row):
    iq,rx,pid,slot=row
    return aligned_view(iq,VIEWS[1+int(slot)%6],rx,pid,'source-U')


def build_source(data,out):
    c=source_contract(data)
    out.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for role in ROLES:
            x,rx,ids,base=source_role(data,c,role)
            views=VIEWS[:1] if role=='U_s' else VIEWS
            save_views(pool,out/f'{role}.npy',x,rx,ids,'source-V' if role=='V' else 'source',views)
            if role=='U_s':
                u=np.lib.format.open_memmap(out/'U_s_practical.npy',mode='w+',dtype=np.float32,
                        shape=(len(x),len(CHIRPS),2,256))
                for i,z in enumerate(pool.map(practical_u,zip(x,rx,ids,base,strict=True),chunksize=24)):
                    u[i]=z
                    if (i+1)%4000==0:
                        print('FULL_U_PRACTICAL',i+1,len(x),flush=True)
                u.flush()
    write(out/'complete.json',dict(status='SOURCE_READY',protocol=PROTOCOL,counts=COUNTS,input_fs=NATIVE_RATE,
          views=list(VIEWS),U_extra_practical=True,target_read=False))


def build_target(data,out):
    target_contract(data)
    out.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for rx in TARGET:
            x=native(data,rx,LIMIT)
            save_views(pool,out/f'{rx}.npy',x,[rx]*LIMIT,range(LIMIT),'target',VIEWS)
            print('FULL_RX_VIEWS_BLIND',rx,LIMIT,flush=True)
    write(out/'complete.json',dict(status='TARGET_READY',protocol=PROTOCOL,target_truth_read=False,
          receivers=list(TARGET),per_receiver=LIMIT,views=list(VIEWS)))


def smoke(data,root):
    c=source_contract(data)
    x,rx,ids,_=source_role(data,c,'L_s')
    clean=chirp_windows(x[0])
    sat,m=aligned_view(x[0],VIEWS[1],rx[0],ids[0],'source',trace=True)
    if clean.shape!=(6,2,256) or clean.dtype!=np.float32 or sat.dtype!=np.float32 or not np.isfinite(sat).all():
        raise ValueError('Aligned full LoRa source practical smoke failed')
    write(root/'smoke.json',dict(status='PASS',native_fs=NATIVE_RATE,latency=m['numerical_filter_latency_samples'],
          dtype=str(sat.dtype),target_reads=0))


def child(rt,role):
    root=Path(rt['run_root'])
    keys=('run_root','frozen_sources',('target_root' if role=='target' else 'source_root'))
    file=root/f'{role}_runtime.json'
    write(file,{k:rt[k] for k in keys})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (root/f'{role}.log').open('x',encoding='utf8') as log:
        proc=subprocess.Popen([sys.executable,'-u','-m',__spec__.name,role,str(file)],env=env,
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    print('FULL_VIEW_WORKER',role,proc.pid,flush=True)
    return proc


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state=root/'state.json'
    try:
        write(state,dict(status='SOURCE_SMOKE'))
        if child(rt,'smoke').wait():
            raise RuntimeError('Native LoRa full source smoke failed')
        write(state,dict(status='ALIGNING_BLIND_VIEWS'),exclusive=False)
        s,t=[child(rt,role) for role in ('source','target')]
        codes=(s.wait(),t.wait())
        if any(codes):
            raise RuntimeError(f'Full aligned workers failed: {codes}')
        write(state,dict(status='FULL_VIEWS_READY',target_truth_read=False),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=('dispatch','smoke','source','target'))
    p.add_argument('runtime',type=Path)
    a=p.parse_args()
    rt=read(a.runtime)
    root=Path(rt['run_root'])
    if a.kind=='dispatch':
        dispatch(rt)
    else:
        activate(rt)
        if a.kind=='smoke':
            smoke(Path(rt['source_root']),root)
        elif a.kind=='source':
            build_source(Path(rt['source_root']),root/'source_views')
        else:
            build_target(Path(rt['target_root']),root/'target_views')
