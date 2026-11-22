"""Publish native full LoRa inputs and run the registered view builder on N607."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .data import ROLES, TARGET
from .full_packet_views import native, source_contract, source_role, target_contract
from .publish import release, release_commit

DATA_RUN='20261122-phase1-lora-full-aligned-data-r01'


def publish(repo,output,inputs,context,ssh_config):
    c=source_contract(inputs/'source')
    t=target_contract(inputs/'target')
    for r in ROLES:
        source_role(inputs/'source',c,r)
    for rx in TARGET:
        native(inputs/'target',rx,t['per_receiver'])
    ctx=read(context)
    rt={k:ctx[k] for k in ('project','python','frozen_sources')}
    project=PurePosixPath(rt['project'])
    folder=project/'releases'/f'{DATA_RUN}-release'
    rt.update(run_id=DATA_RUN,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.full_packet_views',
           run_root=str(project/'runs'/DATA_RUN),release_root=str(folder),source_root=str(folder/'data/source'),
           target_root=str(folder/'data/target'))
    return release(repo,output,rt,ssh_config,inputs,transfer_timeout=1200)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('repo','output','inputs','context','ssh-config'):
        p.add_argument('--'+k,type=Path,required=True)
    publish(**vars(p.parse_args()))
