"""Continue the source-only direct Adaptive R17 to R69 student on Low Urban IQ."""

import argparse
from copy import deepcopy
from itertools import chain
import json
import math
from pathlib import Path, PurePosixPath
import sys
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison import g_lineage, g_train
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_repl_view_shuffle.runner import activate
from experiments.pamf_r44_pruned_refine.models import spectral_error
from experiments.pamf_stage2_inference.engine import InferenceEngine
from . import wisig_dura_estimated as workers
from . import wisig_dura_stress as evaluation
from . import wisig_dura_stress_data as data
from . import wisig_original_dura as replay
from . import wisig_r17_direct_distill as direct


RUN = '20261126-phase1-wisig-adaptive-direct-r69-low-urban-m1-r01'
ARM = 'low_urban'
METHODS = ('raw', 'R69', f'trained_{ARM}')
ROUTES = ('full',)
CASES = tuple(case for case, (_, route, _) in evaluation.CASES.items() if route in ROUTES)
SOURCE_KEYS = (*direct.SOURCE_KEYS, 'direct_root')
EPOCHS = 10
UPDATES = EPOCHS * 247
COUNT = 63000
AUGMENTATION_SEED = 2026112611
RECEIVER_SEED = 2026112612
SHUFFLE_SEED = 2026112613
SCENE = 'practical_low_urban'
SCHEMA = 'direct_r69_low_urban_v1'
CODE = Path(__file__).resolve().parents[2] / 'code'


def low_urban_configs():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    from leo_practical import Config

    full = Config(fs_hz=25e6, fc_hz=2462e6, scenario=SCENE, processing_route='full',
                  mode='post_sync', equalization_enabled=False, boundary='reflect')
    return full, data.harder(full)


def prepare(rt):
    source = Path(rt['action_iq']) / 'train'
    ids, rx, day = (replay.targets.load(source, key) for key in ('physical_ids', 'rx', 'day'))
    if source.parent.parent.name != replay.action_design.RUN or len(ids) != COUNT:
        raise ValueError('Expected the same Adaptive R07 source physical records')
    sessions = np.asarray([f'rx{r}:day{d}:eq1' for r, d in zip(rx, day, strict=True)])
    root = Path(rt['run_root']) / 'source_low_urban'
    root.mkdir()
    indices = np.arange(COUNT)
    with data.pool(source/'clean.npy', ids, CODE) as pool:
        for profile, config in zip(data.PROFILES, low_urban_configs(), strict=True):
            table = np.lib.format.open_memmap(root/f'{profile}.npy', mode='w+', dtype='float32',
                                              shape=(COUNT, 2, 256))
            jobs = data.batches(indices, config, sessions, AUGMENTATION_SEED, RECEIVER_SEED,
                                'wisig/r69-low-urban/source-only/v1', False)
            for selected, observed in pool.map(data.sample, jobs):
                table[selected] = observed
                if (int(selected[-1])+1) % 8192 == 0:
                    print('LOW_URBAN_SOURCE_IQ', profile, int(selected[-1])+1, COUNT, flush=True)
            table.flush()
            del table
            print('LOW_URBAN_SOURCE_IQ_COMPLETE', profile, COUNT, flush=True)
    write(root/'complete.json', dict(status='SOURCE_IQ_COMPLETE', profiles=list(data.PROFILES),
          scene=SCENE, route='full', count=COUNT, simulator_supervision_reads=0, target_reads=0))


def models(rt):
    replay.setup_torch()
    torch.manual_seed(replay.SEED)
    _, teacher, contract = direct.teacher_from(rt)
    root = Path(rt['direct_root'])
    folder = root/'student'
    proof = read(folder/'complete.json')
    path = folder/'final.pt'
    if (root.name != direct.RUN or proof['contract'] != contract or proof['teacher'] != str(teacher)
            or (proof['status'], proof['epoch'], proof['updates'], proof['parameters']) != (
                'SOURCE_FROZEN', 20, 4940, direct.PARAMETERS)):
        raise ValueError('Expected the direct Adaptive R17 to R69 checkpoint')
    anchor, payload = g_lineage.load(path, contract, 'LU')
    if (payload['stage'], payload['ancestors'], payload['model_seed']) != (
            'r69', [str(teacher)], replay.SEED):
        raise ValueError('Unexpected direct R69 ancestry')
    anchor.cuda().requires_grad_(False)
    model = deepcopy(anchor)
    model.prior.requires_grad_(True)
    if sum(p.numel() for p in model.parameters()) != direct.PARAMETERS or len(model.prior) != 2:
        raise ValueError('The continuation must preserve R69 topology')
    return model, anchor, path, contract


def batch(rt, view, selected, epoch, augmented):
    n = len(selected)
    a, b, c = n//4, n//2, 7*n//8
    wave = view.batch(selected, epoch, coarse=False, action=False)
    clean, historical = wave['clean'], wave['received']
    if augmented is None:  # A small real-IQ smoke probes both source profiles.
        from .wisig_dura_stress_data import channel

        source = Path(rt['action_iq'])/'train'
        ids, rx, day = (replay.targets.load(source, key)[selected[:b]]
                        for key in ('physical_ids', 'rx', 'day'))
        sessions = np.asarray([f'rx{r}:day{d}:eq1' for r, d in zip(rx, day, strict=True)])
        profiles = (channel(clean[sl], config, ids[sl], sessions[sl], AUGMENTATION_SEED,
                            RECEIVER_SEED, 'wisig/r69-low-urban/source-only/v1', False)
                    for config, sl in zip(low_urban_configs(), (slice(0, a), slice(a, b)), strict=True))
    else:
        profiles = (augmented[0][selected[:a]], augmented[1][selected[a:b]])
    received = np.concatenate((*profiles, historical[b:c], clean[c:]))
    return tensor(clean, 'cuda:0'), tensor(received, 'cuda:0'), (a, b, c)


def objective(student, anchor, received, clean, boundaries):
    _, b, c = boundaries
    context = student.estimate_context(received)
    recovered = student.stage_outputs(received, context)[0][-1]
    with torch.no_grad():
        baseline = anchor(received[b:c])
    energy = clean.square().mean((1, 2)).clamp_min(1e-4)
    paired = (recovered-clean).square().mean((1, 2)) / energy
    low = paired[:b] + .05 * spectral_error(recovered[:b], clean[:b])
    replay_loss = (recovered[b:c]-baseline).square().mean((1, 2)) / energy[b:c]
    return torch.cat((low, replay_loss, paired[c:]))


def smoke(rt):
    model, anchor, path, _ = models(rt)
    view = replay.OriginalSourceView(dict(rt, action_supervision='operator'))
    clean, received, cuts = batch(rt, view, np.arange(8), 1, None)
    value = objective(model, anchor, received, clean, cuts).mean()
    value.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.prior.parameters(), 1., error_if_nonfinite=True)
    if (not torch.isfinite(value) or not float(gradient) > 0.
            or any(p.grad is not None for p in chain(anchor.parameters(), model.frozen.parameters()))):
        raise FloatingPointError('Low Urban smoke has incorrect gradient routing')
    with torch.no_grad():
        engine = InferenceEngine(model.requires_grad_(False), reuse=True, graph=True, native=False)
        torch.testing.assert_close(engine(received)[0], anchor(received), rtol=5e-5, atol=5e-6)
    write(Path(rt['run_root'])/'source_smoke.json', dict(status='PASS', init=str(path), source_packets=8,
          loss=float(value.detach()), gradient_norm=float(gradient), simulator_supervision_reads=0,
          target_reads=0))


def learning_rate(step):
    if step < 100:
        return 2e-5 + 3e-5 * step / 100
    return 5e-6 + 4.5e-5 * (1 + math.cos(math.pi * (step-100)/(UPDATES-101))) / 2


def fit(rt):
    model, anchor, init, contract = models(rt)
    root = Path(rt['run_root'])
    cache = root/'source_low_urban'
    marker = read(cache/'complete.json')
    if (marker['status'], marker['profiles'], marker['scene'], marker['route'], marker['count'],
            marker['simulator_supervision_reads'], marker['target_reads']) != (
            'SOURCE_IQ_COMPLETE', list(data.PROFILES), SCENE, 'full', COUNT, 0, 0):
        raise ValueError('Expected frozen IQ-only Low Urban source views')
    augmented = tuple(np.load(cache/f'{p}.npy', mmap_mode='r', allow_pickle=False) for p in data.PROFILES)
    view = replay.OriginalSourceView(dict(rt, action_supervision='operator'))
    if (not view.ready() or int(view.is_L.sum()) != 6300
            or any(a.shape != view.clean.shape or a.dtype != np.float32 for a in augmented)):
        raise ValueError('Source physical record distribution or IQ layout differs')
    folder = root/'student'
    folder.mkdir()
    optimizer = torch.optim.AdamW(model.prior.parameters(), lr=learning_rate(0), weight_decay=1e-4)
    rng = np.random.default_rng(SHUFFLE_SEED)
    started, updates = perf_counter(), 0
    with (folder/'training.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, EPOCHS+1):
            model.prior.train()
            sums = np.zeros(4, dtype=np.float64)
            counts = np.zeros(4, dtype=np.int64)
            for selected in g_train.batches(len(view), rng):
                clean, received, cuts = batch(rt, view, selected, epoch, augmented)
                a, b, c = cuts
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(updates)
                optimizer.zero_grad(set_to_none=True)
                values = objective(model, anchor, received, clean, cuts)
                value = values.mean()
                if not torch.isfinite(value):
                    raise FloatingPointError('Nonfinite Low Urban continuation loss')
                value.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.prior.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                groups = values.detach().split((a, b-a, c-b, len(values)-c))
                sums += torch.stack([g.sum() for g in groups]).cpu().tolist()
                counts += [len(g) for g in groups]
                if updates == 1:
                    first = dict(updates=updates, loss=float(value.detach()), gradient_norm=float(gradient),
                                 trainable_parameters=sum(p.numel() for p in model.prior.parameters()))
                    write(folder/'first_update.json', first)
                    print('LOW_URBAN_FIRST_UPDATE', first, flush=True)
            row = dict(epoch=epoch, updates=updates, seconds=perf_counter()-started,
                       mix_counts=counts.tolist(), loss_by_group=(sums/counts).tolist(),
                       loss=float(sums.sum()/counts.sum()))
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
            write(root/'progress.json', row, exclusive=False)
            print('LOW_URBAN_EPOCH', row, flush=True)
    if updates != UPDATES or counts.sum() != COUNT:
        raise ValueError('Low Urban continuation did not reach the fixed budget')
    model.eval()
    if any(not torch.equal(a,b) for a,b in zip(model.frozen.state_dict().values(),
                                               anchor.frozen.state_dict().values(), strict=True)):
        raise ValueError('Frozen coarse/action context has changed')
    path = folder/'final.pt'
    torch.save(dict(schema=SCHEMA, stage='r69', source_contract=contract, allowed_roles='LU',
                    model_seed=replay.SEED, ancestors=[str(init)], epoch=EPOCHS, updates=updates,
                    model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    TX_labels_read=0, F_reads=0, simulator_supervision_reads=0, target_reads=0,
                    code_commit=rt['code_commit']), path)
    write(folder/'complete.json', dict(status='SOURCE_FROZEN', ancestors=[str(init)], epoch=EPOCHS,
          updates=updates, parameters=direct.PARAMETERS, seconds=perf_counter()-started, contract=contract))


def frozen_fit(root, arm):
    if arm != ARM:
        raise ValueError('Wrong Low Urban continuation arm')
    folder = Path(root)/'student'
    proof = read(folder/'complete.json')
    item = torch.load(folder/'final.pt', map_location='cpu', weights_only=True)
    if (proof['status'] != 'SOURCE_FROZEN' or item['schema'] != SCHEMA
            or (item['stage'], item['epoch'], item['updates'], item['allowed_roles'], item['model_seed'])
               != ('r69', EPOCHS, UPDATES, 'LU', replay.SEED)
            or item['source_contract'] != proof['contract'] or item['ancestors'] != proof['ancestors']
            or any(item[key] != 0 for key in ('TX_labels_read', 'F_reads', 'target_reads',
                                             'simulator_supervision_reads'))):
        raise ValueError('Expected the fixed source-final Low Urban student')
    model = g_lineage.skeleton('r69')
    model.load_state_dict(item['model'], strict=True)
    model.cuda().eval().requires_grad_(False)
    return InferenceEngine(model, reuse=True, graph=True, native=False), None


@torch.no_grad()
def readout_smoke(rt):
    activate(rt)
    replay.setup_torch()
    root = Path(rt['run_root'])
    engine, _ = frozen_fit(root, ARM)
    _, heads = evaluation.readers(rt)
    iq = tensor(replay.targets.received(Path(rt['action_iq']), 1)[:4], 'cuda:0')
    decisions = evaluation.classify(heads, engine(iq)[0])
    if len(decisions) != len(evaluation.SEEDS) or any(len(row) != len(iq) for row in decisions):
        raise ValueError('Frozen Low Urban student and PAMF readout differ')
    write(root/'evaluation_smoke.json', dict(status='PASS', source_packets=len(iq), target_reads=0))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    (root/'logs').mkdir()
    state = root/'state.json'
    try:
        write(root/'source_runtime.json', {k:rt[k] for k in SOURCE_KEYS})
        write(state, dict(status='SOURCE_SMOKE_AND_IQ_PREPARATION'))
        prep = evaluation.launch(rt, 'source-iq', 'cpu')
        allowed, _ = workers.gpu_selection(minimum=0)
        one = dict(rt, allowed_gpus=allowed, max_workers=1, max_compute_processes=None)
        schedule(one, [('fit-smoke', ARM)], evaluation.launch)
        workers.complete(prep, 'IQ-only Low Urban source augmentation')
        write(state, dict(status='SOURCE_TRAINING'), exclusive=False)
        schedule(one, [('fit', ARM)], evaluation.launch)
        write(state, dict(status='FROZEN_CHECKPOINT_SMOKE'), exclusive=False)
        schedule(one, [('smoke', 'readout')], evaluation.launch)
        write(state, dict(status='TARGET_PREDICTING'), exclusive=False)
        schedule(dict(one, max_workers=2), [('predict', case) for case in CASES], evaluation.launch)
        write(state, dict(status='PREDICTIONS_FROZEN'), exclusive=False)
        workers.complete(evaluation.launch(rt, 'score', 'cpu'), 'Full-only independent paired scoring')
        write(state, dict(status='ANALYZED'), exclusive=False)
    except Exception as e:
        write(state, dict(status='FAILED', error=str(e)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    settings = read(context)
    rt = {k:settings[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project/'runs'
    paths = dict(action_iq=str(runs/replay.action_design.RUN/'cache'),
                 teacher_root=str(runs/replay.OPERATOR_RUN), direct_root=str(runs/direct.RUN),
                 baseline_root=str(runs/evaluation.BASELINE), recognizer_run=str(runs/evaluation.F_RUN),
                 strong_view_root=str(runs/evaluation.RUN/'strong_target_iq'),
                 decision_replay=str(runs/replay.baseline.RUN/'predictions'))
    external = [*paths.values(), str(runs/direct.RUN/'student/final.pt'),
                str(runs/direct.RUN/'predictions'), str(runs/replay.OPERATOR_RUN/'G/r17/final.pt'),
                str(runs/replay.OPERATOR_RUN/'predictions')]
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo), run_root=str(runs/RUN),
              release_root=str(project/'releases'/f'{RUN}-release'), external_releases=external, **paths)
    release(repo, output, rt, ssh_config, archive_extras=(
        'tools/wisig_paired_main_uncertainty.py', 'tools/r17_activation_factorization.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    jobs = parser.add_subparsers(dest='action', required=True)
    pub = jobs.add_parser('publish')
    for key in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--'+key, type=Path, required=True)
    commands = {'dispatch':dispatch, 'source-iq':prepare, 'fit-smoke':smoke, 'fit':fit, 'smoke':readout_smoke}
    for key in (*commands, 'predict', 'score'):
        worker = jobs.add_parser(key)
        worker.add_argument('runtime', type=Path)
        worker.add_argument('name', nargs='?')
    args = parser.parse_args()
    if args.action == 'publish':
        return publish(args.repo, args.output, args.context, args.ssh_config)
    rt = read(args.runtime)
    if args.action == 'predict':
        if args.name not in CASES:
            raise ValueError('Only Full Practical is part of this experiment')
        return evaluation.predict(rt, args.name, methods=METHODS, load=frozen_fit, force_practical=True,
                                  restore=replay.restore, clean_passthrough=False)
    if args.action == 'score':
        from .wisig_dura_stress_score import score

        references = (
            ('adaptive_direct', Path(rt['direct_root'])/'predictions', 'trained_direct_distill'),
            ('adaptive_multistage', Path(rt['teacher_root'])/'predictions', 'trained_adaptive_operator'),
        )
        return score(rt, methods=METHODS, force_practical=True, clean_passthrough=False,
                     routes=ROUTES, references=references)
    return commands[args.action](rt)


if __name__ == '__main__':
    main()
