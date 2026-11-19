"""Paired E24 source trials and source-only dose selection."""
from copy import deepcopy
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_paper_completion import plan as paper
from .controller import ALTERNATIVES, NATIVE_WEIGHT, choose_weight
from .data import installed_training_fold, registered_fold
from .folds import fold_groups


RUN = '20261119-phase1-pamf-repl-sourceadapt-pilot-m8-r01'
SETTINGS = ('main_split', 'receiver_split_1', 'labeled_1pct', 'source_rx_2')
WEIGHTS = (NATIVE_WEIGHT, *ALTERNATIVES)
SEED = 2026092701
PILOT_EPOCHS = 4


def _weight_name(weight):
    return f'{weight:g}'.replace('.', 'p')


def prepare(rt):
    root = Path(rt['run_root'])
    ctx = read(rt['context'])
    rows = [row for row in paper.make_plan()['rows'] if row['family'] == 'factorial'
            and row['method'] == 'ssl' and row['model_seed'] == SEED and row['data_setting'] in SETTINGS]
    jobs = []
    for row in rows:
        original = paper.configuration(ctx, row, rt['project'], rt['code_commit'])
        groups = fold_groups(original['source_rxs'], original['source_days'])
        if len(groups) != 2:
            raise ValueError('Each first-wave development setting requires two held-out source folds')
        for fold in groups:
            job_id = f"train__{row['data_setting']}-{fold}"
            config = root / 'configs' / f'{job_id}.json'
            out = root / 'F' / row['data_setting'] / fold
            c = dict(original, run_id=RUN, row_id=job_id, output_root=str(out), source_fold=fold,
                     frozen_sources=rt['frozen_sources'])
            write(config, c)
            jobs.append(dict(id=job_id, kind='train', config=str(config), completion=str(out/'complete.json'),
                             successful_statuses=['SOURCE_PILOT_SCORED']))
    if len(jobs) != 2 * len(SETTINGS):
        raise ValueError('Incomplete first-wave source development matrix')
    write(root/'jobs.json', jobs)
    return jobs


def _segment(c, path, weight, prefix=None):
    cfg = deepcopy(c)
    offset = 0 if prefix is None else 20
    cfg.update(output_root=str(path), optimizer_updates=(20+PILOT_EPOCHS)*222,
               segment_epochs=20 if prefix is None else PILOT_EPOCHS,
               epoch_offset=offset, step_offset=offset*222, parent_source=str(prefix) if prefix else None)
    cfg['hard_pl']['weight'] = weight
    cfg['hard_pl_gradient_guard']['weight'] = weight
    return cfg


def _check_segment(folder, cfg):
    done = read(folder/'complete.json')
    if (done['status'] != 'SOURCE_FROZEN' or done['target_access'] or done['U_truth_read']
            or done['epoch'] != cfg['segment_epochs'] or done['updates'] != cfg['segment_epochs']*222
            or done['config'] != cfg or done['ancestors'] != ([cfg['parent_source']] if cfg['parent_source'] else [])):
        raise ValueError('Source-only fold segment or its ancestry differs')


def train_fold(c, device='cuda:0'):
    from experiments.cric_dura_comparison.f_train import train
    from experiments.pamf_paper_completion.model import installed

    _, partition = registered_fold(c)
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)

    def validate(candidate):
        if candidate != cfg or any(candidate.get(k) for k in ('tests', 'target_truth', 'views_root', 'checkpoint')):
            raise ValueError('Fold trainer must use only its prepared source config')

    prefix = out/'segment20'
    cfg = _segment(c, prefix, NATIVE_WEIGHT)
    with installed_training_fold(cfg, partition), installed(cfg, {}, False, validate=validate):
        train(cfg, device)
    _check_segment(prefix, cfg)
    for weight in WEIGHTS:
        folder = out/f'weight_{_weight_name(weight)}'
        cfg = _segment(c, folder, weight, prefix)
        holder = {}
        with installed_training_fold(cfg, partition), installed(cfg, holder, True, validate=validate):
            train(cfg, device)
        _check_segment(folder, cfg)
        state = holder['objective'].state_dict()
        if state['calls'] != PILOT_EPOCHS*222 or state['U_truth_read'] or state['student_views'] != 6*state['seen']:
            raise ValueError('Complete native U recipe required in each source arm')
        write(folder/'repl_state.json', state)
        del holder
    evaluate_fold(c, partition, device)


def evaluate_fold(c, partition, device):
    """Freeze all three models before using held V labels to compute paired net gains."""
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.pamf_paper_completion.model import reader_scope

    out = Path(c['output_root'])
    contract = read(c['source_contract'])
    positions = partition['held_v']
    data = SourceDataset(c['source_cache'], 'V', contract, positions=np.concatenate(list(positions.values())))
    predictions = {}
    for weight in WEIGHTS:
        folder = out/f'weight_{_weight_name(weight)}'
        cfg = _segment(c, folder, weight, out/'segment20')
        _check_segment(folder, cfg)
        payload = torch.load(folder/'final_ssdg.pth', map_location='cpu', weights_only=False)
        with reader_scope(cfg['paper_row']):
            model = reader_from_payload(cfg, payload, device)
        with torch.inference_mode():
            logits = (model.classify_features(model.features(x.to(device))) for x in data.x.split(256))
            predicted = torch.cat([batch.argmax(1).cpu() for batch in logits])
            predictions[weight] = np.asarray(predicted.tolist())
        del model, payload
    labels = data.metadata['y']
    result, start = [], 0
    for (rx, date), indices in positions.items():
        end = start + len(indices)
        base = predictions[NATIVE_WEIGHT][start:end] == labels[start:end]
        for weight in ALTERNATIVES:
            candidate = predictions[weight][start:end] == labels[start:end]
            result.append(dict(rx=rx, date=date, weight=weight, count=len(indices),
                               gain=int((candidate & ~base).sum()), damage=int((~candidate & base).sum())))
        start = end
    write(out/'complete.json', dict(status='SOURCE_PILOT_SCORED', setting=c['paper_row']['data_setting'],
         fold=c['source_fold'], source_evaluation='held_V_only', target_access=False, U_truth_read=False, rows=result))


def analyze(rt):
    root = Path(rt['run_root'])
    folds = {name: {} for name in SETTINGS}
    evidence = {name: {weight: {} for weight in ALTERNATIVES} for name in SETTINGS}
    for job in read(root/'jobs.json'):
        scores = read(job['completion'])
        if (scores['status'] != 'SOURCE_PILOT_SCORED' or scores['source_evaluation'] != 'held_V_only'
                or scores['target_access'] or scores['U_truth_read']):
            raise ValueError('Only untouched held source V may control dose selection')
        name, fold = scores['setting'], scores['fold']
        if fold in folds[name]:
            raise ValueError('Duplicate source fold')
        groups = folds[name][fold] = set()
        for row in scores['rows']:
            group = (row['rx'], row['date'])
            groups.add(group)
            arms = evidence[name][row['weight']]
            if (fold, group) in arms:
                raise ValueError('Duplicate fold scores')
            arms[fold, group] = (row['count'], row['gain'], row['damage'])
    decisions = {name: choose_weight(folds[name], evidence[name]) for name in SETTINGS}
    write(root/'analysis'/'source_decisions.json', dict(status='SOURCE_ADAPTATION_DECISIONS_FROZEN',
          selected_seed=SEED, decisions=decisions, target_access=False, U_truth_read=False))
