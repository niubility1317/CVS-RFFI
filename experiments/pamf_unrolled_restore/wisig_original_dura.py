"""Replay original DURA training with estimated R07 action and variance targets."""

import argparse
from pathlib import Path, PurePosixPath
from shutil import copy2
from time import perf_counter, sleep

import numpy as np
import torch

from experiments.cric_dura_comparison import g_lineage, g_train
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_action_restore import design as action_design
from experiments.pamf_lora_main import publish as transport
from experiments.pamf_lora_main.packet_runner import available, schedule
from experiments.pamf_repl_view_shuffle.runner import activate
from experiments.pamf_residual_restore import design as coarse_design
from experiments.pamf_residual_restore.runtime import load_coarse
from experiments.pamf_stage2_inference.engine import InferenceEngine
from . import wisig_dura_estimated as workers
from . import wisig_dura_forced as baseline
from . import wisig_dura_stress as evaluation
from . import wisig_original_dura_targets as targets


RUN = '20261125-phase1-wisig-original-dura-adaptive-m1-r01'
OPERATOR_RUN = '20261126-phase1-wisig-original-dura-operator-m1-r01'
FAST_RUN = '20261126-phase1-wisig-original-dura-fast-operator-m1-r01'
MIXTURE_RUN = '20261126-diagnostic-wisig-practical-mixture-s2026112512-r01'
ARMS = dict(waveform='adaptive', operator='adaptive_operator', fast_mixture='fast_mixture_operator')
RUNS = dict(waveform=RUN, operator=OPERATOR_RUN, fast_mixture=FAST_RUN)
SEED = action_design.MODEL_SEED
SOURCE_FIELDS = ('run_root', 'coarse_run', 'action_iq', 'iq_estimator', 'code_commit', 'action_supervision')


def setup_torch():
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def source_smoke(rt):
    setup_torch()
    root, source, parent = map(Path, (rt['run_root'], rt['action_iq'], rt['coarse_run']))
    if source.parent.name != action_design.RUN or parent.name != coarse_design.PARENT_RUN:
        raise ValueError('Expected the original R07 observations and R02 coarse checkpoint')
    train = source / 'train'
    roles, ids = (targets.load(train, key) for key in ('roles', 'physical_ids'))
    if (train.resolve() != (parent / 'cache/train').resolve() or ids.shape != (targets.COUNT,)
            or len(np.unique(ids)) != targets.COUNT or roles.shape != ids.shape
            or (int((roles == 'L_s').sum()), int((roles == 'U_s').sum())) != (6300, 56700)):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    torch.manual_seed(SEED)
    coarse, provenance = load_coarse(parent, 'cpu')
    contract = dict(protocol='WiSig Main Phase1 source L_s/U_s', train_count=targets.COUNT,
                    physical_ids=str(train / 'physical_ids.npy'), roles=str(train / 'roles.npy'),
                    source_IQ=str(train / 'clean.npy'), historical_received_epochs=str(source / 'epochs'),
                    coarse_lineage=provenance, TX_labels_read=0, F_reads=0, target_reads=0)
    clean = targets.load(train, 'clean')
    received = targets.received(source, 1)
    index = np.flatnonzero(np.any(clean[:96] != received[:96], axis=(1, 2)))[:4]
    if len(index) != 4:
        raise ValueError('Expected original distorted source IQ pairs for smoke')
    mode = rt['action_supervision']
    fast = mode == 'fast_mixture'
    settings = targets.fast_settings(rt['mixture_root']) if fast else None
    variance_source = targets.variance_replay(rt) if fast else None
    cached_variance = targets.load(variance_source / '01', 'variance') if fast else None
    started = perf_counter()
    estimates = [targets.estimate_pair(clean[i], received[i], mode, settings=settings,
                 replayed_variance=cached_variance[i] if fast else None) for i in index]
    estimate_ms = (perf_counter() - started) * 1000 / len(index)
    if mode != 'waveform':
        if any(row[2] != targets.estimate_pair(clean[i], received[i], 'waveform')[2]
               for i, row in zip(index, estimates)):
            raise ValueError('Operator conversion changed Adaptive variance supervision')
    action = tuple(tensor(np.stack(values)) for values in tuple(zip(*estimates))[:3])
    model = g_lineage.construct('action', coarse).train()
    trace = dict(action_error=0., log_variance_error=0.)
    loss = g_lineage.objective(model, None, 'action', tensor(received[index]), tensor(clean[index]),
        1, action, torch.Generator().manual_seed(2026101919), action_trace=trace).mean()
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(g_lineage.trainable(model, 'action'), 1., error_if_nonfinite=True)
    if not torch.isfinite(loss) or not float(grad) > 0.:
        raise FloatingPointError('Original action-head smoke produced an invalid loss or gradient')
    prefix = root / 'inputs/coarse'
    prefix.mkdir(parents=True)
    torch.save(dict(schema='contract_matched_dura_stage_v1', stage='coarse', source_contract=contract,
        allowed_roles='LU', model_seed=SEED, model=coarse.state_dict(), ancestors=[], epoch=20, updates=4940,
        code_commit=provenance['commit'], TX_labels_read=0, F_reads=0, target_reads=0), prefix / 'final.pt')
    historical = parent / coarse_design.COARSE_NAME
    copy2(historical / 'training.jsonl', prefix / 'training.jsonl')
    duration = read(historical / 'training_complete.json')['seconds']
    write(prefix / 'complete.json', dict(status='SOURCE_FROZEN', stage='coarse', updates=4940, seconds=duration,
        TX_labels_read=0, F_reads=0, target_reads=0, original_checkpoint=str(historical / 'final.pt')))
    write(root / 'inputs/source_contract.json', contract)
    write(root / 'source_smoke.json', dict(status='PASS', count=len(index), loss=float(loss.detach()),
        action_supervision=mode, variance_matches_waveform_mode=True,
        estimate_ms_per_packet=estimate_ms,
        gradient_norm=float(grad), projection_relative_errors=[row[3] for row in estimates],
        real_variance=action[-1].tolist(), simulator_supervision_reads=0, TX_labels_read=0, F_reads=0, target_reads=0))
    print('ORIGINAL_DURA_SOURCE_SMOKE_PASS', flush=True)


class OriginalSourceView:
    """Open only original source clean/received IQ and this run's estimated target arrays."""

    def __init__(self, rt):
        self.source, self.root = map(Path, (rt['action_iq'], rt['run_root']))
        self.clean = targets.load(self.source / 'train', 'clean')
        self.is_L = targets.load(self.source / 'train', 'roles') == 'L_s'
        self.epoch = None
        self.mode = rt['action_supervision']

    def ready(self):
        return self.clean.shape == (targets.COUNT, 2, 256) and self.clean.dtype == np.float32

    def __len__(self):
        return len(self.clean)

    def batch(self, selected, epoch, coarse, action):
        if coarse:
            raise ValueError('Original coarse stage already has a matched frozen checkpoint')
        if self.epoch != (epoch, action):
            self.observed = targets.received(self.source, epoch)
            self.estimated = {}
            if action:
                part = self.root / 'estimates' / f'{epoch:02d}'
                marker = part / 'complete.json'
                while not marker.exists():
                    sleep(2.)
                completion = read(marker)
                if (completion['status'], completion['count'], completion['simulator_supervision_reads'],
                        completion['action_supervision']) != ('ESTIMATES_FROZEN', targets.COUNT, 0, self.mode):
                    raise ValueError('Original action stage requires complete IQ-estimated targets')
                self.estimated = {key: targets.load(part, key) for key in targets.SHAPES}
            self.epoch = epoch, action
        return dict(clean=self.clean[selected], received=self.observed[selected],
                    **{key: data[selected] for key, data in self.estimated.items()})


def train(rt):
    setup_torch()
    root = Path(rt['run_root'])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Use one capacity-checked visible GPU for source training')
    config = dict(output_root=str(root / 'G'), model_seed=SEED, code_commit=rt['code_commit'],
                  action_supervision=rt['action_supervision'],
                  cache=dict(roles='LU', source_contract=str(root / 'inputs/source_contract.json')))
    g_train.train(config, view=OriginalSourceView(rt), frozen_coarse=root / 'inputs/coarse', log_action_terms=True,
                  original_recipe=True)


def frozen_fit(root, arm):
    setup_torch()
    frozen, checkpoint = Path(root) / 'G', Path(root) / 'G/r69/final.pt'
    mode = read(frozen / 'config.json').get('action_supervision', 'waveform')
    if arm != ARMS[mode]:
        raise ValueError('Expected the source-frozen R69 estimate-supervised arm')
    closure = read(frozen / 'complete.json')
    if (closure['status'] != 'SOURCE_FROZEN' or closure['stages'] != list(g_lineage.STAGES)
            or Path(closure['checkpoint']) != checkpoint or closure['updates'] != 39520
            or any(closure[key] for key in ('TX_labels_read', 'F_reads', 'target_reads'))):
        raise ValueError('R69 estimate-supervised source lineage must be complete')
    model, payload = g_lineage.load(checkpoint, read(Path(root) / 'inputs/source_contract.json'), 'LU', 'cuda:0')
    expected = [str(frozen / stage / 'final.pt') for stage in g_lineage.STAGES[:-1]]
    if (payload['stage'] != 'r69' or payload['ancestors'] != expected
            or sum(p.numel() for p in model.parameters()) != 266951):
        raise ValueError('Expected the original native-map R69 architecture and estimated lineage')
    return InferenceEngine(model.requires_grad_(False), reuse=True, graph=True, native=False), None


def restore(engine, _gate, iq, _variance, *, force):
    if not force:
        raise ValueError('Full original R69 readout is required for every packet')
    return engine(iq)[0], torch.ones(len(iq), dtype=torch.bool, device=iq.device)


def evaluation_smoke(rt):
    activate(rt)
    root = Path(rt['run_root'])
    engine, _ = frozen_fit(root, ARMS[rt['action_supervision']])
    _, heads = evaluation.readers(rt)
    iq = tensor(targets.received(Path(rt['action_iq']), 1)[:4], 'cuda:0')
    recovered, active = restore(engine, None, iq, None, force=True)
    decisions = evaluation.classify(heads, recovered)
    if not active.all() or not (len(decisions) == 3 and all(len(row) == len(iq) for row in decisions)):
        raise ValueError('Frozen all-packet R69 to PAMF evaluation smoke failed')
    write(root / 'evaluation_smoke.json', dict(status='PASS', processed=len(iq), F_seeds=3, query_reads=0))


def launch(rt, operation, name, gpu=None):
    # Restrict the training-side runtime to source handles; prediction and scoring get the full manifest.
    worker_rt = dict(rt)
    if operation in ('source-smoke', 'estimate', 'train'):
        worker_rt['release_root'] = rt['release_root']
        # Reuse the existing subprocess launcher with its fixed source-manifest operation names.
        operation = {'source-smoke': 'fit-smoke', 'estimate': 'bank', 'train': 'fit'}[operation]
    return workers.launch(worker_rt, operation, name, gpu)


def source_workers(rt):
    root = Path(rt['run_root'])
    cpu = launch(rt, 'estimate', 'R07')
    learner = None
    try:
        marker = root / 'estimates/01/complete.json'
        while not marker.is_file():
            if cpu.poll() is not None:
                raise RuntimeError('R07 IQ estimate worker exited before first epoch closure')
            sleep(5.)
        excluded = set()
        for run in RUNS.values():
            sibling = root.parent / run / 'state.json'
            if run == root.name or not sibling.is_file():
                continue
            sibling_state = read(sibling)
            if sibling_state.get('status') == 'SOURCE_TRAINING':
                excluded.add(sibling_state['gpu'])
        allowed, samples = workers.gpu_selection(minimum=12288)
        allowed = [gpu for gpu in allowed if gpu not in excluded]
        while not (ready := available(allowed, max_compute_processes=None)):
            if cpu.poll() not in (None, 0):
                raise RuntimeError('R07 IQ estimate worker failed while waiting for GPU capacity')
            sleep(15.)
            allowed, samples = workers.gpu_selection(minimum=12288)
            allowed = [gpu for gpu in allowed if gpu not in excluded]
        gpu = next(gpu for gpu in allowed if gpu in ready)
        learner = launch(rt, 'train', 'DURA', gpu)
        write(root / 'state.json', dict(status='SOURCE_TRAINING', gpu=gpu,
              pid=learner.pid, estimates_pid=cpu.pid, gpu_samples=samples), exclusive=False)
        while learner.poll() is None or cpu.poll() is None:
            if learner.poll() not in (None, 0) or cpu.poll() not in (None, 0):
                raise RuntimeError('Source pipeline failed; inspect this run’s fit and estimate logs')
            sleep(5.)
        if learner.returncode or cpu.returncode:
            raise RuntimeError('Source pipeline failed; inspect this run’s fit and estimate logs')
    finally:
        for worker in (learner, cpu):
            if worker is not None and worker.poll() is None:
                worker.terminate()


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root / 'logs').mkdir()
    state = root / 'state.json'
    try:
        source = {key: rt[key] for key in SOURCE_FIELDS}
        if rt['action_supervision'] == 'fast_mixture':
            source.update({key: rt[key] for key in ('mixture_root', 'variance_replay')})
        write(root / 'source_runtime.json', source)
        write(state, dict(status='SOURCE_SMOKE'))
        workers.complete(launch(rt, 'source-smoke', 'original_R02'), 'original DURA source smoke')
        write(state, dict(status='ESTIMATING_SOURCE_SUPERVISION'), exclusive=False)
        source_workers(rt)
        allowed, _ = workers.gpu_selection(minimum=0)
        rt = dict(rt, allowed_gpus=allowed, max_workers=2, max_compute_processes=None)
        write(state, dict(status='FROZEN_CHECKPOINT_SMOKE'), exclusive=False)
        schedule(rt, [('smoke', 'readout')], launch)
        write(state, dict(status='TARGET_PREDICTING'), exclusive=False)
        schedule(rt, [('predict', case) for case in evaluation.CASES], launch)
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
        workers.complete(launch(rt, 'score', 'cpu'), 'independent truth-last score')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as exc:
        write(state, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, action_supervision):
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project / 'runs'
    paths = dict(coarse_run=str(runs / coarse_design.PARENT_RUN), action_iq=str(runs / action_design.RUN / 'cache'),
                 iq_estimator=str(runs / workers.calibration.RUN / 'configuration.json'),
                 baseline_root=str(runs / evaluation.BASELINE), recognizer_run=str(runs / evaluation.F_RUN),
                 strong_view_root=str(runs / evaluation.RUN / 'strong_target_iq'),
                 decision_replay=str(runs / baseline.RUN / 'predictions'))
    run = RUNS[action_supervision]
    if action_supervision == 'fast_mixture':
        paths['mixture_root'] = str(runs / MIXTURE_RUN)
        paths['variance_replay'] = str(runs / RUN)
    rt.update(run_id=run, runner=__spec__.name, code_commit=transport.release_commit(repo),
              action_supervision=action_supervision,
              run_root=str(runs / run), release_root=str(project / 'releases' / f'{run}-release'),
              external_releases=list(paths.values()), **paths)
    transport.release(repo, output, rt, ssh_config, archive_extras=(
        'tools/wisig_paired_main_uncertainty.py', 'tools/r17_activation_factorization.py'))


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='operation', required=True)
    release = actions.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        release.add_argument('--' + field, type=Path, required=True)
    release.add_argument('--action-supervision', choices=targets.SUPERVISION, default='waveform')
    for operation in ('dispatch', 'fit-smoke', 'bank', 'fit', 'smoke', 'predict', 'score'):
        stage = actions.add_parser(operation)
        stage.add_argument('runtime', type=Path)
        stage.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.operation == 'publish':
        return publish(args.repo, args.output, args.context, args.ssh_config, args.action_supervision)
    rt = read(args.runtime)
    methods = ('raw', 'R69', f"trained_{ARMS[rt['action_supervision']]}")
    operations = dict(dispatch=dispatch, bank=targets.prepare, fit=train, smoke=evaluation_smoke,
                      **{'fit-smoke':source_smoke})
    if args.operation == 'predict':
        return evaluation.predict(rt, args.name, methods=methods, load=frozen_fit, restore=restore,
                                  force_practical=True, clean_passthrough=False)
    if args.operation == 'score':
        from .wisig_dura_stress_score import score

        return score(rt, methods=methods, force_practical=True, clean_passthrough=False)
    operations[args.operation](rt)


if __name__ == '__main__':
    main()
