"""Source-only learned-action falsification; one immutable run and one GPU worker."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from . import design as d
from .models import ActionRestorer
from .data import prepare_cache
from .runtime import load_coarse, save_checkpoint, load_checkpoint
from .evaluation import teacher_action, probe_error, predict, score
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.data import IQSource, write_json
from experiments.pamf_restoration.runner import array_tensor, tensor_array, code_commit
from experiments.pamf_restoration.dispatcher import capacity

ROOT = Path(__file__).resolve().parents[2]


def learning_rate(update):
    if update < 247:
        return 2e-4*(update+1)/247
    return 2e-4*(.1+.9*(1+math.cos(math.pi*(update-247)/(4940-248)))/2)


def source_smoke(args):
    source = IQSource(args.source_cache, 'L_s')
    x = array_tensor(source.iq[:2], 'cpu')
    coarse, parent = load_coarse(args.parent_restoration, 'cpu')
    model = ActionRestorer(coarse).eval()
    reader = load_reader(args.reader_folder, args.source_cache, 'cpu').requires_grad_(False).eval()
    with torch.no_grad():
        value = model(x)
        if not torch.isfinite(value).all() or not torch.isfinite(reader(value)).all():
            raise FloatingPointError('Actual-source/frozen-checkpoint smoke failed')
    optimizer = torch.optim.AdamW(model.estimator.parameters(), lr=2e-4)
    path = args.output/'source_smoke.pt'
    save_checkpoint(path, model, optimizer, 1, 0, parent)
    fresh, _ = load_checkpoint(path, 'cpu')
    with torch.no_grad():
        if not torch.allclose(fresh(x), value, atol=1e-6, rtol=1e-5):
            raise ValueError('Actual-source action checkpoint roundtrip failed')
    write_json(args.output/'smoke.json', dict(status='PASS', real_source_samples=2,
        TX_labels_read=0, target_reads=0, optimizer_updates=0, head_fit_updates=0, parent=parent))


def acquire(args):
    import fcntl
    while True:
        with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            card = capacity()[5]
            if card['free_mb'] >= 12000 and len(card['pids']) < 2:
                coarse, parent = load_coarse(args.parent_restoration, 'cuda:0')
                model = ActionRestorer(coarse).to('cuda:0')
                x = array_tensor(IQSource(args.source_cache, 'L_s').iq[:2], 'cuda:0')
                with torch.no_grad():
                    if not torch.isfinite(model.eval()(x)).all():
                        raise FloatingPointError('GPU actual-source smoke failed')
                torch.cuda.synchronize()
                return model, parent
        time.sleep(5)


def train(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '5':
        raise ValueError('Only physical GPU5 is registered')
    torch.manual_seed(d.MODEL_SEED); torch.cuda.manual_seed_all(d.MODEL_SEED)
    model, parent = acquire(args)
    row = args.output/'learned_operator'; row.mkdir()
    write_json(row/'gpu_ready.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv, physical_GPU=5))
    write_json(row/'resolved_config.json', d.configuration())
    optimizer = torch.optim.AdamW(model.estimator.parameters(), lr=2e-4, weight_decay=1e-4)
    roles = np.load(args.output/'cache/train/roles.npy', allow_pickle=False)
    permutation = np.random.default_rng(d.DATA_SEED)
    generator = torch.Generator(device='cuda:0').manual_seed(d.AUGMENTATION_SEED+7)
    updates = seen_l = seen_u = 0
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, d.EPOCHS+1):
            folder = args.output/'cache/epochs'/f'{epoch:02d}'
            arrays = {key: np.load(folder/(key+'.npy'), mmap_mode='r', allow_pickle=False)
                      for key in ('received', 'kernel', 'coefficients', 'variance')}
            with np.load(folder/'groups.npz', allow_pickle=False) as archive:
                groups = {key: archive[key] for key in archive.files}
            order = permutation.permutation(len(roles)); totals = dict(loss=0., action_error=0., log_variance_error=0.)
            strata = {}; clipped = 0; model.train()
            for offset in range(0, len(order), d.BATCH_SIZE):
                selected = order[offset:offset+d.BATCH_SIZE]
                received = array_tensor(arrays['received'][selected], 'cuda:0')
                target = teacher_action(arrays, selected, 'cuda:0')
                variance_target = array_tensor(arrays['variance'][selected], 'cuda:0')
                _, estimated, variance = model.estimate(received)
                probes = [torch.complex(*torch.randn((2, len(selected), 256), device='cuda:0', generator=generator))/2**.5 for _ in range(2)]
                action = probe_error(estimated, target, probes)
                noise = (variance.log()-variance_target.log()).square()
                loss = (action+.1*noise).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite action/noise training loss')
                lr = learning_rate(updates)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.estimator.parameters(), 1., error_if_nonfinite=True)
                clipped += int(norm > 1); optimizer.step(); updates += 1
                seen_l += int((roles[selected] == 'L_s').sum()); seen_u += int((roles[selected] == 'U_s').sum())
                totals['loss'] += float(loss)*len(selected)
                totals['action_error'] += float(action.sum()); totals['log_variance_error'] += float(noise.sum())
                errors = tensor_array(action)
                masks = {'L_s': roles[selected] == 'L_s', 'U_s': roles[selected] == 'U_s'}
                for key, values in groups.items():
                    masks.update({key+':'+str(v): values[selected] == v for v in np.unique(values)})
                for key, mask in masks.items():
                    item = strata.setdefault(key, dict(count=0, action_error_sum=0.))
                    item['count'] += int(mask.sum()); item['action_error_sum'] += float(errors[mask].sum())
                if updates == 1:
                    save_checkpoint(row/'initial_source_smoke.pt', model, optimizer, epoch, updates, parent)
                    fresh, _ = load_checkpoint(row/'initial_source_smoke.pt', 'cuda:0')
                    model.eval()
                    with torch.no_grad():
                        if not torch.allclose(model(received[:2]), fresh(received[:2]), atol=1e-6, rtol=1e-5):
                            raise ValueError('GPU trained action checkpoint roundtrip failed')
                    del fresh; model.train()
            record = dict(epoch=epoch, updates=updates, losses={key: value/len(roles) for key, value in totals.items()},
                learning_rate_last=lr, L_exposures=seen_l, U_exposures=seen_u,
                gradient_clip_fraction=clipped/247,
                strata={key: dict(count=value['count'], mean_action_error=value['action_error_sum']/max(1, value['count'])) for key, value in strata.items()},
                seconds=time.perf_counter()-started)
            log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
            print(json.dumps(dict(event='ACTION_EPOCH_COMPLETE', **record)), flush=True)
    if (updates, seen_l, seen_u) != (4940, 126000, 1134000):
        raise ValueError('Full registered training exposure required')
    save_checkpoint(row/'final.pt', model, optimizer, 20, updates, parent)
    write_json(row/'training_complete.json', dict(status='SOURCE_TRAINED', epochs=20, updates=updates,
        L_exposures=seen_l, U_exposures=seen_u, TX_labels_read=0, F_outputs_read=0, target_reads=0,
        coarse_updates=0, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.estimator.parameters()),
        seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    del model, optimizer; torch.cuda.empty_cache()
    predict(args)


def child_command(args, mode):
    command = [sys.executable, '-u', '-m', 'experiments.pamf_action_restore.runner', mode]
    for key in ('source_cache', 'parent_restoration', 'reader_folder', 'output', 'expected_commit'):
        command.extend(['--'+key.replace('_', '-'), str(getattr(args, key))])
    return command


def run(args):
    if args.output.exists():
        raise FileExistsError('Immutable action output exists; reconcile instead of relaunch')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU controller required')
    args.output.mkdir()
    write_json(args.output/'resolved_config.json', d.configuration())
    write_json(args.output/'launch.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv,
        commit=args.expected_commit, owner='codex/root/learned-action', started=time.time()))
    try:
        source_smoke(args)
        prepare_cache(args.parent_restoration/'cache', args.output/'cache')
        logs = args.output/'logs'; logs.mkdir()
        command = child_command(args, 'worker')
        with (logs/'worker.log').open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='5'), start_new_session=True)
        write_json(args.output/'worker_launch.json', dict(pid=process.pid, cwd=str(ROOT), argv=command,
                   GPU=5, log=str(logs/'worker.log')))
        if process.wait() != 0:
            raise RuntimeError('Source worker failed; preserve outputs,no automatic retry')
        subprocess.run(child_command(args, 'score'), cwd=ROOT, check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        write_json(args.output/'complete.json', dict(status='ANALYZED', source_only=True,
            target_reads=0, training_TX_labels_read=0, goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error),
            traceback=traceback.format_exc(), outputs_preserved=True, automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker', 'score'))
    for key in ('source-cache', 'parent-restoration', 'reader-folder', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    for key in ('source_cache', 'parent_restoration', 'reader_folder', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.name != d.RUN or code_commit() != args.expected_commit:
        raise ValueError('Registered run/code identity differs')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.mode == 'run':
        run(args)
    elif args.mode == 'worker':
        train(args)
    else:
        score(args)


if __name__ == '__main__':
    main()
