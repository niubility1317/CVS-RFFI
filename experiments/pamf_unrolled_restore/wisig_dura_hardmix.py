"""WiSig Main: balanced continuation, hard continuation, and hard-only retraining."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
import torch

from experiments.cric_dura_comparison import g_lineage, g_train
from experiments.cric_dura_comparison.evaluation import handle
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_action_restore import design as action_design
from experiments.pamf_lora_main import publish as transport
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_r65_four_route import design as r69_design, runtime as r69_runtime
from experiments.pamf_residual_restore import design as coarse_design
from experiments.pamf_stage2_inference.engine import InferenceEngine
from . import wisig_dura_stress as evaluation
from . import wisig_dura_hardmix_data as data
from .wisig_original_dura import restore, setup_torch
from .wisig_preamble_dura_target import historical


RUN = '20261126-phase1-wisig-dura-hardmix-s2026092701-m3-r01'
ARMS = {'balanced': 'balanced', 'hard': 'hard', 'scratch': 'hard'}
CASES = {split: f'default-residual-{split}' for split in evaluation.TASKS}
BASELINES = ('raw', 'R69')
MODEL_SEED = 2026092701
R69_COMMIT = 'b2a2ce937a64654e17e84c2aa817b9da4257a48f'
BATCH = 256
UPDATES = data.EPOCHS * 247
SOURCE_KEYS = ('run_root', 'action_iq', 'coarse_run', 'r69_run', 'r65_run', 'code_commit')


def historical_g(rt, device):
    path = Path(rt['r69_run']) / 'stage2/native_map_prior/final.pt'
    model, origin = r69_runtime.load_checkpoint(path, device)
    if (origin['arm'], origin['epoch'], origin['updates'], origin['code_commit']) != ('stage2', 20, 4940, R69_COMMIT):
        raise ValueError('Expected the exact frozen DURA v1.0 checkpoint')
    return model


def source_smoke(rt):
    setup_torch()
    root, action, coarse = map(Path, (rt['run_root'], rt['action_iq'], rt['coarse_run']))
    source = action / 'train'
    roles, ids, rx, day = (data.load(source / f'{field}.npy')
                          for field in ('roles', 'physical_ids', 'rx', 'day'))
    if (action.parent.name != action_design.RUN or coarse.name != coarse_design.PARENT_RUN
            or source.resolve() != (coarse / 'cache/train').resolve()
            or ids.shape != (data.COUNT,) or len(np.unique(ids)) != data.COUNT
            or (int((roles == 'L_s').sum()), int((roles == 'U_s').sum())) != (6300, 56700)
            or set(np.unique(rx)) != {1, 3, 4, 6, 8} or set(np.unique(day)) != {1, 2, 3}):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    r69_proof = read(Path(rt['r69_run']) / 'stage2/training_complete.json')
    expected = dict(status='SOURCE_TRAINED', epochs=20, updates=4940, L_exposures=126000,
                    U_exposures=1134000, TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                    target_reads=0, parent_run=r69_design.PARENT_RUN, parent_row=r69_design.PARENT_ROW)
    if any(r69_proof.get(k) != v for k, v in expected.items()):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    model = historical_g(rt, 'cpu').train()
    _, teacher = r69_runtime.initialize('stage2', Path(rt['r65_run']), 'cpu')
    clean = tensor(data.load(source / 'clean.npy')[:4])
    received = tensor(data.load(action / 'epochs/01/received.npy')[:4])
    loss = g_lineage.objective(model, teacher, 'r69', received, clean, 1, original_recipe=True).mean()
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if not torch.isfinite(loss) or not float(grad) > 0:
        raise ValueError('No-query real-weight R69 smoke failed')
    write(root / 'source_contract.json', dict(protocol='WiSig Main Phase1 source 392005', source=str(source),
          L_s=6300, U_s=56700, physical_ids=str(source / 'physical_ids.npy'), V_reads=0,
          TX_labels_read=0, F_reads=0, target_reads=0))
    write(root / 'source_smoke.json', dict(status='PASS', loss=float(loss.detach()), grad=float(grad),
          count=4, query_reads=0))
    print('HARDMIX_REAL_SOURCE_SMOKE_PASS', flush=True)


def fit(rt, arm):
    setup_torch()
    root = Path(rt['run_root'])
    view = data.SourceView(rt, ARMS[arm])
    if not view.ready() or torch.cuda.device_count() != 1:
        raise ValueError('One assigned GPU and the registered L_s/U_s waveform roles are required')
    contract = read(root / 'source_contract.json')
    output = root / 'models' / arm
    if arm == 'scratch':
        g_train.train(dict(output_root=str(output), model_seed=MODEL_SEED, code_commit=rt['code_commit'],
                           cache=dict(roles='LU', source_contract=str(root / 'source_contract.json'))),
                      view=view, original_recipe=True, final_epochs=data.EPOCHS, log_action_terms=True)
        return
    output.mkdir(parents=True)
    torch.manual_seed(MODEL_SEED)
    model = historical_g(rt, 'cuda:0').train()
    _, teacher = r69_runtime.initialize('stage2', Path(rt['r65_run']), 'cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(2026102001)
    updates = 0
    with (output / 'training.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, data.EPOCHS + 1):
            total = torch.zeros((), device='cuda:0', dtype=torch.float64)
            recipe_epoch = 1 + 19 * (epoch - 1) / (data.EPOCHS - 1)
            order = rng.permutation(len(view))
            for offset in range(0, len(view), BATCH):
                batch = view.batch(order[offset:offset + BATCH], epoch)
                received = tensor(batch['received'], 'cuda:0')
                clean = tensor(batch['clean'], 'cuda:0')
                optimizer.zero_grad(set_to_none=True)
                optimizer.param_groups[0]['lr'] = .5 * g_train.learning_rate(updates, data.EPOCHS)
                loss = g_lineage.objective(model, teacher, 'r69', received, clean, recipe_epoch,
                                           original_recipe=True)
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError('Nonfinite paired IQ training objective')
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                total += loss.detach().sum().double()
            row = dict(epoch=epoch, updates=updates, loss=float(total) / len(view),
                       L_exposures=6300 * epoch, U_exposures=56700 * epoch)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            print('HARDMIX_CONTINUATION_EPOCH', arm, row, flush=True)
    if updates != UPDATES or any(p.grad is not None for p in teacher.parameters()):
        raise ValueError('Incomplete continuation budget or unexpected teacher gradient')
    checkpoint = output / 'final.pt'
    with checkpoint.open('xb') as stream:
        torch.save(dict(model=model.state_dict(), code_commit=rt['code_commit'], arm=arm, updates=updates,
                        source_contract=contract, TX_labels_read=0, F_reads=0, target_reads=0), stream)
    write(output / 'complete.json', dict(status='SOURCE_FROZEN', epoch=data.EPOCHS, updates=updates,
          initialization=R69_COMMIT, source_contract=contract, TX_labels_read=0, F_reads=0, target_reads=0))


def frozen_arm(root, arm):
    root = Path(root).parent
    output = root / 'models' / arm
    contract = read(root / 'source_contract.json')
    done = read(output / 'complete.json')
    if done['status'] != 'SOURCE_FROZEN' or any(done[key] for key in ('TX_labels_read', 'F_reads', 'target_reads')):
        raise ValueError('Frozen source-only restoration checkpoint required')
    if arm == 'scratch':
        if (done['stages'] != list(g_lineage.STAGES)
                or done['updates'] != 247 * (140 + data.EPOCHS)):
            raise ValueError('The eight-stage scratch lineage must be complete')
        model, payload = g_lineage.load(output / 'r69/final.pt', contract, 'LU', 'cuda:0', epochs=data.EPOCHS)
        if (payload['stage'] != 'r69'
                or payload['ancestors'] != [str(output / stage / 'final.pt') for stage in g_lineage.STAGES[:-1]]):
            raise ValueError('Unexpected scratch checkpoint ancestry')
    else:
        payload = torch.load(output / 'final.pt', map_location='cpu', weights_only=True)
        if (payload['arm'], payload['updates'], payload['source_contract']) != (arm, UPDATES, contract):
            raise ValueError('Unexpected continuation checkpoint')
        model = r69_runtime.build('stage2')
        r69_runtime.validate_aliases(model, payload['model'])
        model.load_state_dict(payload['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 266951:
        raise ValueError('DURA v1.0 architecture changed')
    return InferenceEngine(model.to('cuda:0').eval().requires_grad_(False), reuse=True, graph=True, native=False), None


def predict(rt, name, *, route='residual'):
    setup_torch()
    arm, split = name.split(':')
    row = dict(rt, run_root=str(Path(rt['run_root']) / arm))
    training_row = Path(rt.get('training_root', rt['run_root'])) / arm
    case = f'default-{route}-{split}'
    evaluation.predict(row, case, methods=(*BASELINES, f'trained_{arm}'),
                       load=lambda _root, requested_arm: frozen_arm(training_row, requested_arm),
                       restore=restore, force_practical=True, clean_passthrough=False, measure_iq=True)


def score(rt, *, route='residual', arms=tuple(ARMS), output='scores.json'):
    from .wisig_dura_stress_score import cells, closed_predictions

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use the independent CPU scorer')
    root = Path(rt['run_root'])
    readers = Path(rt['recognizer_run'])
    original = historical(readers, 'predict', evaluation.SEEDS[0])
    ids_by_split, by_split, iq_by_split = {}, {}, {}
    for split in CASES:
        case = f'default-{route}-{split}'
        manifest, ids, _ = handle(evaluation.view_root(rt, original, route, split), original['classes'])
        if manifest['channel'] != f'{route}/post_sync/noeq' or len(ids) != evaluation.COUNTS[split]:
            raise ValueError('Unexpected WiSig Main physical split')
        ids_by_split[split] = ids
        predictions, iq = {}, {}
        for arm in arms:
            path = root / arm / 'predictions' / case
            methods = (*BASELINES, f'trained_{arm}')
            entries, _ = closed_predictions(path, ids, 'default', route, split, methods,
                                             force_practical=True, clean_passthrough=False)
            for key, values in entries.items():
                if key in predictions and not np.array_equal(values, predictions[key]):
                    raise ValueError('Historical baseline decisions differ between rows')
                predictions[key] = values
            metrics = read(path / 'complete.json')['mean_iq_nmse']
            for view in evaluation.VIEWS:
                iq.setdefault(view, {}).update(metrics[view])
        by_split[split] = predictions
        iq_by_split[split] = iq
    if np.intersect1d(*ids_by_split.values()).size:
        raise ValueError('Main split populations must be disjoint')
    truth = historical(readers, 'score', evaluation.SEEDS[0])
    if truth['classes'] != original['classes']:
        raise ValueError('Prediction and score TX orders differ')
    labels, receivers = {}, {}
    for split, ids in ids_by_split.items():
        rows = read(truth['tests'][split]['truth'])
        labels[split] = np.asarray([rows[str(ident)]['label'] for ident in ids], dtype=np.int64)
        receivers[split] = np.asarray([rows[str(ident)]['receiver'] for ident in ids])
    rng = np.random.default_rng(2026112602)
    results = {}
    for split in (*CASES, 'combined'):
        parts = tuple(CASES) if split == 'combined' else (split,)
        preds = {method: np.concatenate([by_split[part][method] for part in parts], axis=2)
                 for method in next(iter(by_split.values()))}
        results[split] = cells(preds, np.concatenate([labels[part] for part in parts]),
                               np.concatenate([receivers[part] for part in parts]), rng)
    write(root / output, dict(status='TARGET_SCORED', route=route, truth_last=True, results=results,
          mean_iq_nmse=iq_by_split, reader_seeds=list(evaluation.SEEDS), G_seed=MODEL_SEED))


def launch(rt, operation, name, gpu=None):
    root = Path(rt['run_root'])
    runtime = (root / 'source_runtime.json' if operation in ('smoke', 'bank', 'fit') else
               Path(rt['release_root']) / 'runtime.json')
    cmd = [rt['python'], '-u', '-m', rt['runner'], operation, str(runtime), name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (root / 'logs' / f'{operation}-{name.replace(":", "-")}.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=rt['release_root'], env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    print('HARDMIX_WORKER', operation, name, proc.pid, gpu, flush=True)
    return proc


def complete(proc, stage):
    if proc.wait():
        raise RuntimeError(f'{stage} failed. See the run-owned log.')


def arm_work(rt, arm):
    complete(launch(rt, 'fit', arm, os.environ['CUDA_VISIBLE_DEVICES']), f'{arm} source training')
    for split in CASES:
        complete(launch(rt, 'predict', f'{arm}:{split}', os.environ['CUDA_VISIBLE_DEVICES']),
                 f'{arm} {split} prediction')


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    state = root / 'state.json'
    write(state, dict(status='SOURCE_SMOKE'))
    try:
        write(root / 'source_runtime.json', {key: rt[key] for key in (
              *SOURCE_KEYS, 'python', 'release_root', 'runner')})
        complete(launch(rt, 'smoke', 'R69'), 'real-checkpoint no-query source smoke')
        write(state, dict(status='BANK_BUILDING'), exclusive=False)
        complete(launch(rt, 'bank', 'paired'), 'paired source bank')
        write(state, dict(status='SOURCE_FIT_THEN_ROW_PREDICT'), exclusive=False)
        schedule(rt, [('arm', arm) for arm in ARMS], launch)
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
        complete(launch(rt, 'score', 'cpu'), 'independent truth-last scoring')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    cfg = read(context)
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project / 'runs'
    paths = dict(action_iq=str(runs / action_design.RUN / 'cache'),
                 coarse_run=str(runs / coarse_design.PARENT_RUN),
                 r69_run=str(runs / r69_design.RUN), r65_run=str(runs / r69_design.PARENT_RUN),
                 parent_root=str(runs / '20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'),
                 baseline_root=str(runs / evaluation.BASELINE),
                 recognizer_run=str(runs / evaluation.F_RUN),
                 decision_replay=str(runs / '20261125-phase1-wisig-dura-force-main-m2-r01/predictions'))
    rt.update(paths, run_id=RUN, runner=__spec__.name, code_commit=transport.release_commit(repo),
              run_root=str(runs / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              external_releases=list(paths.values()), allowed_gpus=list(range(8)), max_workers=3,
              max_compute_processes=2)
    transport.release(repo, output, rt, ssh_config, archive_extras=(
        'tools/wisig_paired_main_uncertainty.py', 'tools/r17_activation_factorization.py'))


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='operation', required=True)
    release = actions.add_parser('publish')
    for arg in ('repo', 'output', 'context', 'ssh-config'):
        release.add_argument(f'--{arg}', type=Path, required=True)
    for operation in ('dispatch', 'smoke', 'bank', 'fit', 'arm', 'predict', 'score'):
        stage = actions.add_parser(operation)
        stage.add_argument('runtime', type=Path)
        stage.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.operation == 'publish':
        return publish(args.repo, args.output, args.context, args.ssh_config)
    rt = read(args.runtime)
    if args.operation == 'bank':
        root = Path(__file__).resolve().parents[2]
        os.environ['PYTHONPATH'] = os.pathsep.join((str(root / 'code'), str(root)))
        import sys
        sys.path.insert(0, str(root / 'code'))
    workers = dict(dispatch=dispatch, smoke=source_smoke, bank=data.prepare, score=score)
    if args.operation in workers:
        workers[args.operation](rt)
    else:
        dict(fit=fit, arm=arm_work, predict=predict)[args.operation](rt, args.name)


if __name__ == '__main__':
    main()
