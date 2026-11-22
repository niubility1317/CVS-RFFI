"""Launch the full LoRa supervised stage against immutable aligned inputs."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .full_packet_publish import DATA_RUN
from .full_supervised import RUN
from .publish import release, release_commit


def publish(repo,output,context,ssh_config,data_submit):
    cfg=read(context)
    rt={k:cfg[k] for k in ('project','python','frozen_sources')}
    p=PurePosixPath(rt['project'])
    base=p/'releases'/f'{DATA_RUN}-release'
    parent=p/'runs'/DATA_RUN
    d=read(data_submit)
    if (d['archive_transfer']!='VERIFIED' or PurePosixPath(d['cwd'])!=base
            or PurePosixPath(d['run_root'])!=parent):
        raise ValueError('Expected verified native full packet publication')
    rt.update(run_id=RUN,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.full_supervised',
              release_root=str(p/'releases'/f'{RUN}-release'),run_root=str(p/'runs'/RUN),parent_root=str(parent),
              source_root=str(base/'data/source'),allowed_gpus=list(range(7)),max_compute_processes=4)
    return release(repo,output,rt,ssh_config)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','context','ssh-config','data-submit'):
        p.add_argument('--'+k,type=Path,required=True)
    publish(**vars(p.parse_args()))
