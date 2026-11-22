"""Publish the single-seed input/recognition/restoration pilot to N607."""
import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from .packet_data import CHIRPS, ROLE_COUNTS, TARGET_LIMIT
from .publish import release, release_commit


RUN = '20261122-phase1-lora-packet-aligned-pilot-m5-r01'


def publish(repo, output, inputs, context, ssh_config):
    inputs = Path(inputs)
    a, b = [read(inputs/p/'contract.json') for p in ('source','target')]
    if (a['protocol'] != 'lora_packet_pilot_v1' or b['protocol'] != a['protocol']
            or a['counts'] != ROLE_COUNTS or a['chirps'] != list(CHIRPS)
            or b['per_receiver'] != TARGET_LIMIT or not a['source_only'] or b['TX_truth_read']
            or set(a['source_receivers']) & set(b['receivers'])):
        raise ValueError('LoRa packet development split differs')
    ctx = read(context)
    project = PurePosixPath(ctx['project'])
    folder = project/'releases'/f'{RUN}-release'
    runtime = {k:ctx[k] for k in ('project','python','frozen_sources')}
    runtime.update(run_id=RUN,code_commit=release_commit(repo),runner='experiments.pamf_lora_main.packet_runner',
        run_root=str(project/'runs'/RUN),release_root=str(folder),source_root=str(folder/'data/source'),
        target_root=str(folder/'data/target'),allowed_gpus=list(range(7)))
    return release(repo,output,runtime,ssh_config,inputs)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','output','inputs','context','ssh-config'):
        p.add_argument('--'+key,required=True)
    publish(**vars(p.parse_args()))
