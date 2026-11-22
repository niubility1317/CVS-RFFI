"""Build one independent practical realization for every source training packet."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .data import VIEWS
from .full_packet_data import COUNTS, PROTOCOL
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_contract, source_role
from .packet_data import CHIRPS
from .packet_physics import aligned_view
from .publish import release, release_commit
from .runner import activate


RUN='20261122-phase1-lora-source-practical-draw-r01'
ROLES=('L_s','U_s')
DRAW=1
WORKERS=8


def alternate(args):
    iq,rx,pid,slot,role=args
    scene=VIEWS[1+int(slot)%6]
    namespace='source-U' if role=='U_s' else 'source'
    return aligned_view(iq,scene,rx,pid,namespace,realization=DRAW)


def handles(rt):
    parent=Path(rt['parent_root'])
    if parent.name!=DATA_RUN:
        raise ValueError('Registered native LoRa alignment is required')
    source=Path(rt['source_root'])
    original=parent/'source_views'
    proof=read(original/'complete.json')
    if (proof.get('status')!='SOURCE_READY' or proof.get('protocol')!=PROTOCOL
            or proof.get('counts')!=COUNTS or proof.get('target_read') is not False):
        raise ValueError('Missing completed LoRa source-only views')
    return source,original,source_contract(source)


def smoke(rt):
    src,original,c=handles(rt)
    x,rx,ids,base=source_role(src,c,'U_s')
    stored=np.load(original/'U_s_practical.npy',mmap_mode='r',allow_pickle=False)
    for i in (0,4200,8400,12600,16800,21000):
        args=(x[i],rx[i],ids[i],base[i],'U_s')
        scene=VIEWS[1+int(base[i])%6]
        historical=aligned_view(x[i],scene,rx[i],ids[i],'source-U')
        wave=alternate(args)
        if not np.array_equal(stored[i],historical) or np.array_equal(wave,stored[i]):
            raise ValueError('Source practical draws did not preserve original alignment or independence')
        if not np.array_equal(wave,alternate(args)):
            raise ValueError('Independent draw is not reproducible')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',paired_U=6,native_original_exact=True,
          alternate_repeat_exact=True,draw=DRAW,TX_reads=0,target_reads=0))


def dispatch(rt):
    root=Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state=root/'state.json'
    try:
        activate(rt)
        write(state,dict(status='SOURCE_SMOKE'))
        smoke(rt)
        write(state,dict(status='DRAWING_SOURCE'),exclusive=False)
        src,_,c=handles(rt)
        rows={}
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for role in ROLES:
                x,rx,ids,base=source_role(src,c,role)
                out=np.lib.format.open_memmap(root/f'{role}_practical.npy',mode='w+',dtype=np.float32,
                                              shape=(len(ids),len(CHIRPS),2,256))
                jobs=((iq,r,i,b,role) for iq,r,i,b in zip(x,rx,ids,base,strict=True))
                for i,wave in enumerate(pool.map(alternate,jobs,chunksize=24),start=1):
                    if wave.shape!=out.shape[1:] or not np.isfinite(wave).all():
                        raise ValueError('Alternate source practical IQ is invalid')
                    out[i-1]=wave
                    if i%2000==0:
                        print('SOURCE_PRACTICAL_DRAW',role,i,len(ids),flush=True)
                out.flush()
                rows[role]=dict(n=len(ids),views=np.bincount(base%6,minlength=6).tolist())
        write(root/'complete.json',dict(status='SOURCE_DRAW_READY',protocol=PROTOCOL,draw=DRAW,
             parent=DATA_RUN,views=list(VIEWS[1:]),rows=rows,TX_reads=0,target_reads=0))
        write(state,dict(status='SOURCE_DRAW_READY'),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


def publish(repo,output,context,ssh_config,data_submit):
    ctx=read(context)
    rt={k:ctx[k] for k in ('project','python','frozen_sources')}
    p=PurePosixPath(rt['project'])
    parent=p/'runs'/DATA_RUN
    data=p/'releases'/f'{DATA_RUN}-release'
    r=read(data_submit)
    if (r['archive_transfer']!='VERIFIED' or PurePosixPath(r['run_root'])!=parent
            or PurePosixPath(r['cwd'])!=data):
        raise ValueError('Expected original LoRa aligned data publication')
    rt.update(run_id=RUN,code_commit=release_commit(repo),runner=__spec__.name,
              release_root=str(p/'releases'/f'{RUN}-release'),run_root=str(p/'runs'/RUN),
              parent_root=str(parent),source_root=str(data/'data/source'))
    return release(repo,output,rt,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_subparsers(dest='kind',required=True)
    pub=modes.add_parser('publish')
    for k in ('repo','output','context','ssh-config','data-submit'):
        pub.add_argument('--'+k,type=Path,required=True)
    modes.add_parser('dispatch').add_argument('runtime',type=Path)
    args=vars(p.parse_args())
    if args.pop('kind')=='publish':
        publish(**args)
    else:
        dispatch(read(args['runtime']))
