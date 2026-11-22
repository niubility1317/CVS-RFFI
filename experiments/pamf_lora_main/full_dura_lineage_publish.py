"""Release one intact WiSig DURA training chain on the registered LoRa split."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_adaptive import SET_RUN
from .full_dura_detail import DETAIL_CONFIRM_RUN
from .full_dura_lineage import DETAIL_PRIOR_RUN, FULL_ACTION_RUN, PHASE_RUN, REPLICA_RUNS, RUN
from .full_dura_trace_source import TRACE_RUN
from .full_packet_publish import DATA_RUN
from .full_restore_draw import RUN as DRAW_RUN
from .publish import release, release_commit


def publish(repo,output,context,ssh_config,data_submit,draw_submit,trace_submit,tri_submit,
            reuse_actions=None,full_action_submit=None,replica_seed=None,reuse_full_bank=None,
            confirm_submit=None,detail_prior_submit=None):
    ctx=read(context)
    p=PurePosixPath(ctx['project'])
    handles=((data_submit,DATA_RUN),(draw_submit,DRAW_RUN),(trace_submit,TRACE_RUN),(tri_submit,SET_RUN))
    if reuse_actions is not None:
        handles=(*handles,(reuse_actions,RUN))
    if full_action_submit is not None or detail_prior_submit is not None:
        if reuse_actions is None:
            raise ValueError('LoRa prefix reuse requires both existing source dependencies')
        if full_action_submit is not None and detail_prior_submit is not None:
            raise ValueError('Choose one LoRa matched continuation')
        handles=(*handles,(full_action_submit or detail_prior_submit,PHASE_RUN))
    if replica_seed is not None:
        if (replica_seed not in REPLICA_RUNS or reuse_actions is None
                or full_action_submit is not None or detail_prior_submit is not None
                or reuse_full_bank is None or confirm_submit is None):
            raise ValueError('LoRa replica requires the matched source banks and three-seed controls')
        handles=(*handles,(reuse_full_bank,FULL_ACTION_RUN),(confirm_submit,DETAIL_CONFIRM_RUN))
    elif reuse_full_bank is not None or confirm_submit is not None:
        raise ValueError('Replica source bank and comparison receipts require a replica seed')
    for path,name in handles:
        r=read(path)
        if (r['archive_transfer']!='VERIFIED' or PurePosixPath(r['run_root'])!=p/'runs'/name
                or PurePosixPath(r['cwd'])!=p/'releases'/f'{name}-release'):
            raise ValueError('Wrong LoRa lineage dependency')
    data=p/'runs'/DATA_RUN
    runtime={k:ctx[k] for k in ('project','python','frozen_sources')}
    name=FULL_ACTION_RUN if full_action_submit else PHASE_RUN if reuse_actions else RUN
    if detail_prior_submit is not None:
        name=DETAIL_PRIOR_RUN
    gpu_order={RUN:list(range(8)),PHASE_RUN:[3,6,2,1,4,5,7,0],
               FULL_ACTION_RUN:[6,2,1,4,5,7,0,3],
               DETAIL_PRIOR_RUN:[2,1,4,5,7,6,0,3]}[name]
    if replica_seed is not None:
        name=REPLICA_RUNS[replica_seed]
        gpu_order={2026092701:[1,4,5,2,7,6,0,3],2026092703:[2,5,1,4,7,6,0,3]}[replica_seed]
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
    p.add_argument('--full-action-submit',type=Path)
    p.add_argument('--detail-prior-submit',type=Path)
    p.add_argument('--replica-seed',type=int,choices=tuple(REPLICA_RUNS))
    p.add_argument('--reuse-full-bank',type=Path)
    p.add_argument('--confirm-submit',type=Path)
    publish(**vars(p.parse_args()))
