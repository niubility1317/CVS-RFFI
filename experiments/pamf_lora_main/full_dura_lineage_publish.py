"""Release one intact WiSig DURA training chain on the registered LoRa split."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_adaptive import SET_RUN
from .full_dura_lineage import PHASE_RUN, RUN
from .full_dura_trace_source import TRACE_RUN
from .full_packet_publish import DATA_RUN
from .full_restore_draw import RUN as DRAW_RUN
from .publish import release, release_commit


def publish(repo,output,context,ssh_config,data_submit,draw_submit,trace_submit,tri_submit,reuse_actions=None):
    ctx=read(context)
    p=PurePosixPath(ctx['project'])
    handles=((data_submit,DATA_RUN),(draw_submit,DRAW_RUN),(trace_submit,TRACE_RUN),(tri_submit,SET_RUN))
    if reuse_actions is not None:
        handles=(*handles,(reuse_actions,RUN))
    for path,name in handles:
        r=read(path)
        if (r['archive_transfer']!='VERIFIED' or PurePosixPath(r['run_root'])!=p/'runs'/name
                or PurePosixPath(r['cwd'])!=p/'releases'/f'{name}-release'):
            raise ValueError('Wrong LoRa lineage dependency')
    data=p/'runs'/DATA_RUN
    runtime={k:ctx[k] for k in ('project','python','frozen_sources')}
    name=RUN if reuse_actions is None else PHASE_RUN
    gpu_order=list(range(8)) if name==RUN else [3,6,2,1,4,5,7,0]
    runtime.update(run_id=name,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.full_dura_lineage',
        release_root=str(p/'releases'/f'{name}-release'),run_root=str(p/'runs'/name),
        parent_root=str(data),source_root=str(p/'releases'/f'{DATA_RUN}-release/data/source'),
        target_views=str(data/'target_views'),allowed_gpus=gpu_order,max_compute_processes=4,max_workers=1)
    release(repo,output,runtime,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config','data-submit','draw-submit','trace-submit','tri-submit'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--reuse-actions',type=Path)
    publish(**vars(p.parse_args()))
