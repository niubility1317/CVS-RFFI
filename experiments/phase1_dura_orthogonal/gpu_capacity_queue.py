"""Wait for one registered N607 GPU slot, then launch one owned controller."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def capacity(gpu_csv, app_csv, gpu_index, minimum_mib=4096, max_apps=3):
    rows = [tuple(part.strip() for part in row.split(',')) for row in gpu_csv.splitlines() if row.strip()]
    matches = [row for row in rows if len(row) == 3 and row[0] == str(gpu_index)]
    if len(matches) != 1:
        raise ValueError('Exactly one selected GPU must be reported')
    _, uuid, free = matches[0]
    owners = [row for row in app_csv.splitlines() if row.split(',')[0].strip() == uuid]
    return int(free) >= minimum_mib and len(owners) < max_apps, int(free), len(owners)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--ground', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--objective', required=True, choices=('wide_temporal_prior_guard', 'wide_student_fir_guard', 'wide_frequency_regression', 'wide_reversible_guard', 'wide_balanced_spectral_guard', 'wide_multiview_dura_guard', 'wide_counterfactual_guard', 'wide_multiview_distill', 'wide_conditional_flow', 'wide_direct_flow', 'wide_conditional_flow_t0'))
    p.add_argument('--architecture', choices=('dura_multiscale', 'direct_student_fir', 'dura_frequency_regression', 'dura_reversible', 'direct_spectral_student', 'conditional_iq_flow', 'direct_iq_flow'), default='dura_multiscale')
    args = p.parse_args()
    if (args.objective, args.architecture) not in (('wide_temporal_prior_guard', 'dura_multiscale'), ('wide_student_fir_guard', 'direct_student_fir'), ('wide_frequency_regression', 'dura_frequency_regression'), ('wide_reversible_guard', 'dura_reversible'), ('wide_balanced_spectral_guard', 'dura_multiscale'), ('wide_multiview_dura_guard', 'dura_multiscale'), ('wide_counterfactual_guard', 'dura_multiscale'), ('wide_multiview_distill', 'direct_spectral_student'), ('wide_conditional_flow', 'conditional_iq_flow'), ('wide_direct_flow', 'direct_iq_flow'), ('wide_conditional_flow_t0', 'conditional_iq_flow')):
        raise ValueError('Registered objective and architecture pair required')
    runs = args.root/'runs'
    outputs = [runs/args.source, runs/(args.source+'-controller'), runs/(args.source+'-training-smoke'), runs/args.ground]
    if any(path.exists() for path in outputs) or not args.cache.is_file():
        raise ValueError('Unique output roots and fixed source cache required')
    print('GPU_QUEUE_WAITING', args.gpu, flush=True)
    while True:
        if any(path.exists() for path in outputs):
            raise FileExistsError('Another owner created an output while queued')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                       '--format=csv,noheader,nounits'], text=True)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
        ready, free, count = capacity(gpu, apps, args.gpu)
        if ready:
            print('GPU_QUEUE_SLOT', args.gpu, 'free_mib', free, 'other_apps', count, flush=True)
            break
        time.sleep(20)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    cmd = [sys.executable, '-X', 'utf8', '-u', '-m', 'experiments.phase1_dura_orthogonal.controller',
           '--root', str(args.root), '--source', args.source, '--ground', args.ground, '--commit', args.commit,
           '--device', 'cuda:0', '--prepare-batch', '32', '--objective', args.objective, '--deterministic',
           '--source-context-cache', str(args.cache), '--architecture', args.architecture]
    child = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdin=subprocess.DEVNULL)
    print('GPU_QUEUE_CONTROLLER_STARTED', child.pid, flush=True)
    code = child.wait()
    if code:
        raise RuntimeError('Owned controller failed: '+str(code))
    print('GPU_QUEUE_CONTROLLER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
