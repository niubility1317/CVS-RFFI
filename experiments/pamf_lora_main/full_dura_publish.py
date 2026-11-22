"""Release the full LoRa recovery comparison using established source roles."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_adaptive import RUN as TRI
from .full_dura import RUN, SWEEP_RUN
from .full_packet_publish import DATA_RUN
from .publish import release, release_commit


def publish(repo,output,context,ssh_config,data_submit,tri_submit,dose_submit=None):
    x=read(context)
    rt={k:x[k] for k in ('project','python','frozen_sources')}
    p=PurePosixPath(x['project'])
    handles=[(data_submit,DATA_RUN),(tri_submit,TRI)]
    if dose_submit is not None:
        handles.append((dose_submit,RUN))
    for path,name in handles:
        o=read(path)
        if (o['archive_transfer']!='VERIFIED' or PurePosixPath(o['run_root'])!=p/'runs'/name
                or PurePosixPath(o['cwd'])!=p/'releases'/f'{name}-release'):
            raise ValueError('The full LoRa release receipts differ')
    data=p/'runs'/DATA_RUN
    name=SWEEP_RUN if dose_submit is not None else RUN
    rt.update(run_id=name,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.full_dura',
        release_root=str(p/'releases'/f'{name}-release'),run_root=str(p/'runs'/name),
        parent_root=str(data),source_root=str(p/'releases'/f'{DATA_RUN}-release/data/source'),
        target_views=str(data/'target_views'),allowed_gpus=list(range(7)),max_compute_processes=4)
    return release(repo,output,rt,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config','data-submit','tri-submit'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--dose-submit',type=Path)
    publish(**vars(p.parse_args()))
