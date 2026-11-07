"""Run the registered, unstarted projected-prior experiment on normal GPU5."""
import argparse
import os
from pathlib import Path

from . import packet_filter_resource_dispatch as dispatch

RUN='20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01'
ARM='TF_ProjectedPriorDetail'
SCIENCE_MODULE='experiments.pamf_phase3_restoration.projected_prior_detail_trial'
SCIENCE_COMMIT='8cf3669c4a5d98f6c4930117952203afd56f2f6e'


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or args.legacy_release!=args.root/'releases'/(RUN+'-release'):
        raise ValueError('CUDA-empty owner and exact original release required')
    if (args.legacy_release/'release_commit.txt').read_text().strip()!=SCIENCE_COMMIT:
        raise ValueError('Original scientific commit required')
    handoff=dispatch.read(args.root/'logs'/RUN/'resource_handover_GPU5.json')
    if handoff['status']!='VERIFIED_UNSTARTED_OWNER_RETIRED' or handoff['scientific_commit']!=SCIENCE_COMMIT:
        raise ValueError('Exact unstarted original queue handover required')
    proc=Path('/proc')/str(handoff['old_PID'])/'cmdline'
    try:command=proc.read_bytes().replace(bytes([0]),b' ').decode()
    except FileNotFoundError:command=''
    if SCIENCE_MODULE+' run ' in command and SCIENCE_COMMIT in command:
        raise ValueError('Old queue owner remains live; prevent duplicate')
    dispatch.RUN,dispatch.ARM=RUN,ARM
    dispatch.LEGACY_MODULE,dispatch.LEGACY_COMMIT=SCIENCE_MODULE,SCIENCE_COMMIT
    dispatch.controller(args,allowed_gpu=5,reuse_unstarted_root=True,
        resource_dependency='Normal GPU5 capacity; independent of other scientific scores or unfinished GPU3 work')


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','legacy-release','cache'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('Actual dispatcher commit required')
    controller(args)


if __name__=='__main__':main()
