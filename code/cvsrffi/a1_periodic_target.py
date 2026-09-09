"""User-authorized exploratory target observation without training feedback."""
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch


def due(epoch, start, total):
    return int(start) > 0 and int(epoch) >= int(start) and (
        (int(epoch) - int(start)) % 10 == 0 or int(epoch) == int(total))


def evaluate_checkpoint(checkpoint, *, output, input_package, truth, run_id, row_id,
                        device, predictor=None, scorer=None, expected_records=672000):
    """Use a separate rebuilt model in this PID; truth only enters CPU scorer."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing existing evaluation output: {output}')
    release = Path(__file__).resolve().parents[2]
    if predictor is None:
        from scripts.predict_phase1_truth_last import main as predictor
    random_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = [torch.device(device).index or torch.cuda.current_device()] if torch.device(device).type == 'cuda' else []
    started = time.monotonic()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            predictor(['--checkpoint', str(checkpoint), '--output-root', str(output),
                '--input-package', str(input_package), '--run-id', run_id, '--row-id', row_id,
                '--mode', 'predict', '--device', str(device), '--num-workers', '0'])
            predictions = output / 'predictions.json'
            if not predictions.is_file():
                raise FileNotFoundError('Prediction must be complete before independent scoring')
            score_command = [sys.executable, str(release/'code/scripts/score_phase1_truth_last.py'),
                '--predictions', str(predictions), '--truth', str(truth), '--output', str(output/'score.json')]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            env['PYTHONPATH'] = os.pathsep.join([str(release/'code'), str(release), env.get('PYTHONPATH', '')])
            if scorer is None:
                with (output/'scorer.log').open('x', encoding='utf-8') as log:
                    subprocess.run(score_command, cwd=release, env=env,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                scorer(score_command, env)
            # Read coverage only. No accuracy, thresholds, or rankings enter training.
            count = json.loads((output/'score.json').read_text(encoding='utf-8'))['record_count']
            if count != expected_records:
                raise ValueError(f'Incomplete target prediction coverage: {count}')
            (output/'evaluation_scope.json').write_text(json.dumps({
                'scope': 'USER_AUTHORIZED_EXPLORATORY_TARGET_MONITOR_NOT_INDEPENDENT_CONFIRMATION',
                'feeds_training': False, 'checkpoint': str(checkpoint), 'record_count': count,
                'gpu_execution': 'same_training_pid_separate_rebuilt_model',
                'seconds': time.monotonic()-started}, indent=2)+'\n', encoding='utf-8')
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        gc.collect()
    return time.monotonic()-started


def run_epoch(args, epoch, output_dir, payload, save_fn, device):
    if not due(epoch, args.a1_periodic_target_start, args.epochs):
        return 0.0
    checkpoint = Path(output_dir)/f'epoch_{int(epoch):03d}_ssdg.pth'
    temporary = checkpoint.with_suffix('.pth.writing')
    if checkpoint.exists() or temporary.exists():
        raise FileExistsError(checkpoint)
    saved = dict(payload, checkpoint_role='fixed_epoch_exploratory_monitor_no_selection')
    save_fn(temporary, saved)
    temporary.rename(checkpoint)
    return evaluate_checkpoint(checkpoint, output=Path(output_dir)/'target_epochs'/f'E{int(epoch):03d}',
        input_package=args.a1_periodic_target_inputs, truth=args.a1_periodic_target_truth,
        run_id=args.run_id, row_id=f'{args.candidate_id}_E{int(epoch):03d}', device=device)
