"""Publish the LoRa PAMF-Tri shape/loss single-seed screen."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_packet_publish import DATA_RUN
from .publish import release, release_commit
from .tri_tuning import RUN


def publish(repo, output, context, ssh_config):
    ctx = read(context)
    rt = {k:ctx[k] for k in ('project','python','frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project/'runs'/DATA_RUN
    rt.update(run_id=RUN, code_commit=release_commit(repo),
              runner='experiments.pamf_lora_main.tri_tuning',
              release_root=str(project/'releases'/f'{RUN}-release'),
              run_root=str(project/'runs'/RUN), parent_root=str(parent),
              source_root=str(project/'releases'/f'{DATA_RUN}-release/data/source'),
              target_views=str(parent/'target_views'),
              allowed_gpus=[3,5,6,4,0,7,2,1], max_workers=8, max_compute_processes=12)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config'):
        parser.add_argument('--'+k,type=Path,required=True)
    publish(**vars(parser.parse_args()))
