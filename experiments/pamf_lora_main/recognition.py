"""Run native PAMF continuation with the LoRa source contract and calibration."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .data import CLASSES, ROLES, SOURCE
from .physical import ChirpResponse, PracticalSource, rms_normalize


SEEDS = (2026092701, 2026092702, 2026092703)
METHODS = ('pamf_sup', 'pamf_pl', 'tri_original', 'pamf_adaptive', 'resnet18')
DOSES = (6., 4.5, 3.)
FOLDS = (0, 4)
STEPS = 222


def config(common, method, seed, folder, *, weight=6., held_rx=None):
    c = {key: deepcopy(value) for key, value in common['ssl'].items()}
    ssl = method in ('tri_original', 'pamf_adaptive')
    use_u = ssl or method == 'pamf_pl'
    if not ssl:
        c = {}
    c.update(run_id=common['run_id'], row_id=Path(folder).name, code_commit=common['code_commit'],
             output_root=str(folder), source_cache=common['source_cache'],
             source_contract=common['source_contract'], source_practical=common['source_practical'],
             source_rxs=list(range(len(SOURCE))), source_days=[0], num_classes=len(CLASSES),
             model_seed=seed, method_id='pamf_t1_e50', lora_method=method, held_rx=held_rx,
             ablation='F21' if use_u else 'F20', U_use=use_u, U_start_epoch=14,
             regime='matched_sat_plain_pl' if use_u else 'matched_sat_supervised',
             satellite_concat=False, satellite_CE_weight=0., satellite_consistency_weight=0.,
             paper_row=dict(structure='full', signature_views=3, permuted=ssl,
                            learning_regime='ssl' if ssl else 'pl' if use_u else 'supervised'),
             epochs=50, optimizer_updates=50*STEPS, default_head_joint=True, posthoc_head_fit=False)
    if ssl:
        c['hard_pl']['weight'] = c['hard_pl_gradient_guard']['weight'] = weight
        c['repl_row'] = dict(U=True, L_A=True, L_B=True,
                             blocks=dict(style=True, guard=True, joint=True, extrapolation=True))
    return c


def replace_reference(model, c):
    model.id_backbone.encoder.response = ChirpResponse()
    model._feature_learning_report.update(lora_reference=True, lora_dataset='lora_rx5_main_v1')
    return model


def positions(c):
    root = Path(c['source_cache'])
    result = {}
    for role in ROLES:
        with np.load(root / f'{role}_metadata.npz', allow_pickle=False) as m:
            if ('y' in m.files) != (role != 'U_s'):
                raise ValueError('Source label permissions differ')
            result[role] = np.flatnonzero(m['rx_i'] != c['held_rx'])
    return result


@contextmanager
def source_scope(c, holder, enabled):
    from experiments.cric_dura_comparison import f_data, f_train, t1_model, t1_practical
    from experiments.pamf_paper_completion.model import installed
    from experiments.pamf_repl_source_adapt.receiver_balance import BalancedREPL, positive_target
    from experiments.pamf_ssl_research import hard_pl_joint_objective as joint
    from experiments.pamf_repl_completion.objectives import objective

    old_context, old_alias = f_data.context, f_train.context
    old_build, old_practical, old_contrast = t1_model.install_backbone, t1_practical.Practical, joint.contrast

    def load(cfg):
        fold = positions(cfg) if cfg['held_rx'] is not None else None
        ctx = old_context(cfg, source_positions=fold)
        for name in ('train_loader', 'unlabeled_loader', 'val_loader'):
            if ctx[name] is not None:
                dataset = ctx[name].dataset
                dataset.x = rms_normalize(dataset.x)
        return ctx

    def build(net, cfg):
        return replace_reference(old_build(net, cfg), cfg)

    def contrast(maps, labels, receivers):
        target, statistics = positive_target(labels, receivers)
        holder['objective'].receiver_balance = statistics
        return old_contrast(maps, labels, receivers, positive_distribution=target)

    def validate(cfg):
        if cfg != c or any(cfg.get(k) for k in ('target_truth', 'truth', 'target_inputs', 'tests')):
            raise ValueError('LoRa source config must remain bound and target blind')
        if read(cfg['source_contract'])['protocol'] != 'lora_rx5_main_v1':
            raise ValueError('LoRa source contract mismatch')

    f_data.context = f_train.context = load
    t1_model.install_backbone = build
    t1_practical.Practical = PracticalSource
    balanced = enabled and c['lora_method'] == 'pamf_adaptive'
    if balanced:
        joint.contrast = contrast
    try:
        with installed(c, holder, enabled, validate=validate,
                       objective_factory=BalancedREPL if balanced else objective):
            yield
    finally:
        f_data.context, f_train.context = old_context, old_alias
        t1_model.install_backbone, t1_practical.Practical, joint.contrast = old_build, old_practical, old_contrast


def segment(c, name, offset, epochs, parent, *, smoke=False):
    from experiments.cric_dura_comparison.f_train import train
    folder = Path(c['output_root']) / name
    cfg = dict(c, output_root=str(folder), segment_epochs=epochs, epoch_offset=offset,
               step_offset=offset*STEPS, parent_source=str(parent) if parent else None)
    if parent:
        evidence = read(parent / 'complete.json')
        old = evidence['config']
        keys = ('source_contract', 'source_cache', 'model_seed', 'paper_row', 'held_rx', 'lora_method')
        if (evidence['status'] != 'SOURCE_FROZEN' or evidence['target_access'] or evidence['U_truth_read']
                or old['epoch_offset'] + old['segment_epochs'] != offset
                or any(cfg[key] != old[key] for key in keys)):
            raise ValueError('Incompatible source continuation')
    holder = {}
    enabled = bool(c.get('hard_pl')) and offset > 0
    with source_scope(cfg, holder, enabled):
        train(cfg, device='cpu' if smoke else 'cuda:0', smoke_steps=2 if smoke else None)
    evidence = read(folder / 'complete.json')
    if enabled:
        stats = holder['objective'].state_dict()
        if stats['calls'] != epochs*STEPS or stats['U_truth_read'] or stats['EMA_updates']:
            raise ValueError('LoRa REPL state does not match continuation')
        write(folder / 'repl_state.json', stats)
    return dict(folder=str(folder), **evidence)


@contextmanager
def reader_scope(c):
    from experiments.cric_dura_comparison import f_model
    from experiments.pamf_paper_completion.model import reader_scope as native_scope
    with native_scope(c['paper_row']):
        old = f_model.install
        f_model.install = lambda net, cfg: replace_reference(old(net, cfg), cfg)
        try:
            yield
        finally:
            f_model.install = old


@torch.no_grad()
def source_predict(path, c, rx):
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    root = Path(c['source_cache'])
    with np.load(root / 'V_metadata.npz', allow_pickle=False) as m:
        mask = m['rx_i'] == rx
        y = m['y'][mask]
    x = np.load(root / 'V_x.npy', mmap_mode='r', allow_pickle=False)[mask]
    payload = torch.load(path / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    with reader_scope(c):
        f = reader_from_payload(c, payload, 'cuda:0')
    prediction = [f(rms_normalize(batch.cuda())).argmax(1).cpu() for batch in tensor(x).split(256)]
    return np.asarray(torch.cat(prediction).tolist()) == y


def calibrate(common, rx, out):
    c = config(common, 'pamf_adaptive', SEEDS[0], out, held_rx=rx)
    out.mkdir(parents=True, exist_ok=False)
    prefix = Path(segment(c, 'segment20', 0, 20, None)['folder'])
    predictions = {}
    for weight in DOSES:
        candidate = deepcopy(c)
        candidate['hard_pl']['weight'] = candidate['hard_pl_gradient_guard']['weight'] = weight
        folder = Path(segment(candidate, f'weight{weight:g}_e30', 20, 10, prefix)['folder'])
        predictions[weight] = source_predict(folder, candidate, rx)
    base = predictions[DOSES[0]]
    evidence = {str(w): [len(base), int((predictions[w] & ~base).sum()), int((~predictions[w] & base).sum())]
                for w in DOSES[1:]}
    write(out / 'complete.json', dict(held_source_rx=rx, status='SOURCE_CALIBRATION_COMPLETE',
                                    reference_correct=int(base.sum()), same_row=evidence))


def choose_dose(calibrations):
    from experiments.pamf_repl_source_adapt.controller import choose_weight
    folds = {str(rx): ((rx, 0),) for rx in FOLDS}
    rows = {dose: {} for dose in DOSES[1:]}
    for path in calibrations:
        result = read(path)
        rx = result['held_source_rx']
        if result['status'] != 'SOURCE_CALIBRATION_COMPLETE' or str(rx) not in folds:
            raise ValueError('Unexpected source-only calibration result')
        for dose, cells in rows.items():
            cells[str(rx), (rx, 0)] = result['same_row'][str(dose)]
    return choose_weight(folds, rows, noise_aware=True)


def train_pamf(c):
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    segments = []
    parent = None
    for epochs, offset in ((20, 0), (10, 20), (20, 30)):
        result = segment(c, f'segment{offset + epochs}', offset, epochs, parent)
        segments.append(result)
        parent = Path(result['folder'])
    shutil.copyfile(parent / 'final_ssdg.pth', root / 'final_ssdg.pth')
    write(root / 'complete.json', dict(status='SOURCE_FROZEN', config=c, epoch=50, updates=11100,
          target_access=False, U_truth_read=False, segments=segments,
          initialization='same_contract_scratch20_own10_own20', source_contract=c['source_contract']))
