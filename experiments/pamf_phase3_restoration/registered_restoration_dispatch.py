"""Two fixed registered candidates, unchanged science and serial normal GPU5 use."""
import argparse
from pathlib import Path
import time

from . import packet_filter_resource_dispatch as dispatch

MODULE='experiments.pamf_phase3_restoration.registered_restoration_dispatch'
CASES={
    'modulation':dict(run='20261107-phase1-Restore-ModulationCovariance-E30-s2026092701-r01',arm='TF_ModulationCovariance',
        module='experiments.pamf_phase3_restoration.modulation_covariance_trial',commit='3fc9201628e7602ea58c6a5fb13d6ec934513e99'),
    'mismatched':dict(run='20261107-phase1-Restore-MismatchedPrior-E30-s2026092701-r01',arm='TF_MismatchedPrior',
        module='experiments.pamf_phase3_restoration.mismatched_prior_trial',commit='14e7315d6da5d6fe242e7568f20c97ade064e8d9'),
}


def predecessor_finished(root):
    previous=CASES['modulation'];folder=root/'runs'/previous['run']
    for stage in ('train','cross_rx','cross_both'):
        path=folder/('terminal-'+previous['arm']+'-'+stage+'.json')
        if path.is_file() and (dispatch.read(path)['exit_code']!=0 or stage=='cross_both'):
            return True
    return False


def run_candidate(args):
    case=CASES[args.candidate]
    if args.legacy_release!=args.root/'releases'/(case['run']+'-release'):
        raise ValueError('Exact original registered scientific release required')
    if (args.legacy_release/'release_commit.txt').read_text().strip()!=case['commit']:
        raise ValueError('Original registered scientific commit required')
    if args.candidate=='mismatched':
        print('WAITING_FOR_MODULATION_LAST_GPU_STAGE_NORMAL_GPU5',flush=True)
        while not predecessor_finished(args.root):time.sleep(5)
    dispatch.RUN,dispatch.ARM=case['run'],case['arm']
    dispatch.LEGACY_MODULE,dispatch.LEGACY_COMMIT=case['module'],case['commit']
    dispatch.controller(args,allowed_gpu=5,reuse_unstarted_root=True,
        resource_dependency='Serial GPU5: modulation last GPU stage before mismatched; no score-based dependency')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',choices=tuple(CASES),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--legacy-release',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:
        raise ValueError('Actual dispatcher Git execution identity required')
    run_candidate(args)


if __name__=='__main__':main()
