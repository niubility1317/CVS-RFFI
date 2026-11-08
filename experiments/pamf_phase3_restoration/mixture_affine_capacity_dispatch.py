"""Adopt the exact unstarted affine-mixture run under GPU0 cap3."""
import argparse
from pathlib import Path
from . import mixture_capacity_dispatch as dispatch

RUN = '20261108-phase1-Restore-MixtureAffine-E30-s2026092701-r01'
ARM = 'TF_MixtureAffine'
SCIENCE_MODULE = 'experiments.pamf_phase3_restoration.mixture_affine_trial'
SCIENCE_COMMIT = 'b231eb29aae1ff0d85049960dba02a64b404b4a6'


def install():
    dispatch.RUN = RUN
    dispatch.ARM = ARM
    dispatch.SCIENCE_MODULE = SCIENCE_MODULE
    dispatch.SCIENCE_COMMIT = SCIENCE_COMMIT


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'legacy-release', 'cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Actual dispatcher code identity required')
    install()
    dispatch.controller(args, gpu=0)


if __name__ == '__main__':
    main()
