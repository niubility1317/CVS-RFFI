"""Adopt the exact unstarted feature-response run under GPU5 cap3."""
import argparse
from pathlib import Path
from . import mixture_capacity_dispatch as dispatch

RUN = '20261108-phase1-Restore-FeatureResponse-E30-s2026092701-r01'
ARM = 'TF_FeatureResponse'
SCIENCE_MODULE = 'experiments.pamf_phase3_restoration.feature_response_trial'
SCIENCE_COMMIT = '6a54c65a2b54886191f44b6a4d6d94530fcb217d'


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
    dispatch.controller(args, gpu=5)


if __name__ == '__main__':
    main()
