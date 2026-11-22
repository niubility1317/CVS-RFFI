"""Submit the preregistered LoRa refinement using the existing immutable data."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .publish import release, release_commit


RUN='20261122-phase1-lora-packet-refine-m6-r01'
PARENT='20261122-phase1-lora-packet-aligned-pilot-m5-r02'
DATA='20261122-phase1-lora-packet-aligned-pilot-m5-r01'


def publish(repo,output,context,ssh_config,parent_submit,data_submit):
    cfg=read(context)
    rt={k:cfg[k] for k in ('project','python','frozen_sources')}
    p=PurePosixPath(rt['project'])
    parent=p/'runs'/PARENT
    data=p/'releases'/f'{DATA}-release'
    for submit,cwd,run in ((parent_submit,p/'releases'/f'{PARENT}-release',parent),
                           (data_submit,data,p/'runs'/DATA)):
        prior=read(submit)
        if (PurePosixPath(prior['cwd'])!=cwd or PurePosixPath(prior['run_root'])!=run
                or prior['archive_transfer']!='VERIFIED'):
            raise ValueError('Previously verified parent and data release required')
    rt.update(run_id=RUN,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.refine_runner',
              run_root=str(p/'runs'/RUN),release_root=str(p/'releases'/f'{RUN}-release'),
              parent_root=str(parent),source_root=str(data/'data/source'),target_views=str(parent/'target_views'),
              allowed_gpus=list(range(7)),max_compute_processes=4)
    return release(repo,output,rt,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config','parent-submit','data-submit'):
        p.add_argument('--'+k,type=Path,required=True)
    publish(**vars(p.parse_args()))
