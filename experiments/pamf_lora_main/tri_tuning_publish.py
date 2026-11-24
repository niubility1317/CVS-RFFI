"""Publish the LoRa PAMF-Tri shape/loss single-seed screen."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_packet_publish import DATA_RUN
from .publish import release, release_commit
from .tri_tuning import RUN, RUN2
from .tri_length import RUN as APERTURE_RUN


def publish(repo, output, context, ssh_config, budget=False, aperture=False):
    ctx = read(context)
    rt = {k:ctx[k] for k in ('project','python','frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project/'runs'/DATA_RUN
    run = APERTURE_RUN if aperture else RUN2 if budget else RUN
    if budget:
        rt['study'] = 'budget'
    rt.update(run_id=run, code_commit=release_commit(repo),
              runner='experiments.pamf_lora_main.tri_tuning',
              release_root=str(project/'releases'/f'{run}-release'),
              run_root=str(project/'runs'/run), parent_root=str(parent),
              source_root=str(project/'releases'/f'{DATA_RUN}-release/data/source'),
              target_views=str(parent/'target_views'),
              allowed_gpus=[3,5,6,4,0,7,2,1], max_workers=8, max_compute_processes=12)
    if budget:
        rt.update(allowed_gpus=[0,1,2,4,3,5,6,7],max_workers=6)
    if aperture:
        rt.update(study='aperture', native_target_root=str(project/'releases'/f'{DATA_RUN}-release/data/target'),
                  allowed_gpus=[3,5,6,7,0,1,2,4],max_workers=4,min_free_mib=14500)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--budget',action='store_true')
    mode.add_argument('--aperture',action='store_true')
    for k in ('repo','output','context','ssh-config'):
        parser.add_argument('--'+k,type=Path,required=True)
    publish(**vars(parser.parse_args()))
