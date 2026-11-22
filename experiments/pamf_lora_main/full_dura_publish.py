"""Release the full LoRa recovery comparison using established source roles."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_adaptive import RUN as TRI, WARMUP_RUN
from .full_dura import COVERAGE_RUN, PROJECTED_RUN, RUN, SHAPE_RUN, SWEEP_RUN
from .full_packet_publish import DATA_RUN
from .publish import release, release_commit


def publish(repo,output,context,ssh_config,data_submit,tri_submit,dose_submit=None,shape=False,projected=False,
            coverage_submit=None):
    x=read(context)
    rt={k:x[k] for k in ('project','python','frozen_sources')}
    p=PurePosixPath(x['project'])
    new_run=shape or projected or coverage_submit is not None
    if new_run and dose_submit is None:
        raise ValueError('Shape protection requires the frozen full LoRa G')
    handles=[(data_submit,DATA_RUN),(tri_submit,WARMUP_RUN if new_run else TRI)]
    if dose_submit is not None:
        handles.append((dose_submit,RUN))
    if coverage_submit is not None:
        handles.append((coverage_submit,PROJECTED_RUN))
    for path,name in handles:
        o=read(path)
        if (o['archive_transfer']!='VERIFIED' or PurePosixPath(o['run_root'])!=p/'runs'/name
                or PurePosixPath(o['cwd'])!=p/'releases'/f'{name}-release'):
            raise ValueError('The full LoRa release receipts differ')
    data=p/'runs'/DATA_RUN
    if coverage_submit is not None:
        name=COVERAGE_RUN
    else:
        name=PROJECTED_RUN if projected else SHAPE_RUN if shape else SWEEP_RUN if dose_submit is not None else RUN
    rt.update(run_id=name,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.full_dura',
        release_root=str(p/'releases'/f'{name}-release'),run_root=str(p/'runs'/name),
        parent_root=str(data),source_root=str(p/'releases'/f'{DATA_RUN}-release/data/source'),
        target_views=str(data/'target_views'),allowed_gpus=[5,6,1,4,0,2,3] if new_run else list(range(7)),
        max_compute_processes=4)
    if coverage_submit is not None:
        rt['max_workers']=4
    return release(repo,output,rt,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config','data-submit','tri-submit'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--dose-submit',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--shape',action='store_true')
    mode.add_argument('--projected',action='store_true')
    mode.add_argument('--coverage-submit',type=Path)
    publish(**vars(p.parse_args()))
