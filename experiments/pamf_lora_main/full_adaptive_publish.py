"""Publish the full LoRa continuations and seven-view predictor on N607."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_adaptive import RUN, WARMUP_RUN
from .full_packet_publish import DATA_RUN
from .full_supervised import RUN as SUPERVISED
from .publish import release, release_commit


def publish(repo,output,context,ssh_config,data_submit,supervised_submit,prior_submit=None):
    ctx=read(context)
    rt={k:ctx[k] for k in ('project','python','frozen_sources')}
    p=PurePosixPath(rt['project'])
    dependencies=[(data_submit,DATA_RUN),(supervised_submit,SUPERVISED)]
    run=RUN
    if prior_submit is not None:
        dependencies.append((prior_submit,RUN))
        run=WARMUP_RUN
    for path,name in dependencies:
        r=read(path)
        if (r['archive_transfer']!='VERIFIED' or PurePosixPath(r['run_root'])!=p/'runs'/name
                or PurePosixPath(r['cwd'])!=p/'releases'/f'{name}-release'):
            raise ValueError('Expected exact complete release handle for the full LoRa dependency')
    data=p/'runs'/DATA_RUN
    rt.update(run_id=run,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.full_adaptive',
              release_root=str(p/'releases'/f'{run}-release'),run_root=str(p/'runs'/run),
              parent_root=str(data),source_root=str(p/'releases'/f'{DATA_RUN}-release/data/source'),
              target_views=str(data/'target_views'),allowed_gpus=list(range(7)),max_compute_processes=4)
    if run==WARMUP_RUN:
        rt['max_workers']=4
    return release(repo,output,rt,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config','data-submit','supervised-submit'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--prior-submit',type=Path)
    publish(**vars(p.parse_args()))
