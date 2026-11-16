"""Independent post-freeze U correctness by receiver and update conflict.

Only aggregate diagnostics are emitted. This program must not be used by a
trainer or for selecting a model or hyperparameter.
"""
from collections import Counter
import gzip
from itertools import zip_longest
import json
from pathlib import Path


GROUPS = ('t2', 'cls_head', 'other')
STATUSES = ('conflict', 'nonconflict', 'undefined')


def _check_frozen(done):
    if (done.get('status') != 'SOURCE_FROZEN' or done.get('epoch') != 50
            or done.get('updates') != 11100 or done.get('target_access') is not False
            or done.get('U_truth_read') is not False):
        raise ValueError('Complete source-only E50 freeze required')


def _records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def analyze_segment(observations, guard_metrics, *, truth, receivers, step_offset,
                    expected_steps, source_completion):
    _check_frozen(source_completion)
    if not truth or set(truth) != set(receivers) or expected_steps < 1:
        raise ValueError('Exact nonempty U truth and receiver mappings required')
    totals, epochs = Counter(), {}
    group_totals = {g: {s: Counter() for s in STATUSES} for g in GROUPS}
    by_receiver = {}
    n = 0
    for n, (pseudo, guard) in enumerate(zip_longest(_records(observations), _records(guard_metrics)), 1):
        if (pseudo is None or guard is None or pseudo['step'] != step_offset+n
                or guard['step'] != n):
            raise ValueError('Complete same-step observations and guard metrics required')
        arrays = [pseudo[key] for key in ('physical_ids', 'pseudo', 'accepted')]
        if not arrays[0] or not all(len(value) == len(arrays[0]) for value in arrays):
            raise ValueError('Aligned nonempty U batch required')
        statuses = {}
        for group in GROUPS:
            cosine = guard['groups'][group]['anchor_cosines_before']['L_full']
            statuses[group] = ('undefined' if cosine is None else
                               'conflict' if cosine < -1e-6 else 'nonconflict')
            group_totals[group][statuses[group]]['steps'] += 1
        epoch = str((pseudo['step']-1)//222+1)
        epoch_total = epochs.setdefault(epoch, Counter())
        for sid, label, selected in zip(*arrays):
            if sid not in truth or not isinstance(label, int) or not isinstance(selected, bool):
                raise ValueError('Known U ID, pseudo class and boolean mask required')
            key = str(receivers[sid])
            if key not in by_receiver:
                by_receiver[key] = dict(counts=Counter(), seen=set(), accepted=set(),
                    groups={g: {s: Counter() for s in STATUSES} for g in GROUPS})
            entry = by_receiver[key]
            correct = int(selected and label == truth[sid])
            for counter in (totals, epoch_total, entry['counts']):
                counter['seen'] += 1
                counter['accepted'] += int(selected)
                counter['accepted_correct'] += correct
            entry['seen'].add(sid)
            if selected:
                entry['accepted'].add(sid)
            for group, status in statuses.items():
                for counter in (group_totals[group][status], entry['groups'][group][status]):
                    counter['seen'] += 1
                    counter['accepted'] += int(selected)
                    counter['accepted_correct'] += correct
    if n != expected_steps:
        raise ValueError('Complete expected source segment required')
    def values(counter):
        return {key: int(counter[key]) for key in ('seen', 'accepted', 'accepted_correct')}
    def grouped(groups, *, with_steps=False):
        return {g: {s: values(groups[g][s]) | ({'steps': int(groups[g][s]['steps'])}
                 if with_steps else {}) for s in STATUSES} for g in GROUPS}
    result_receivers = {}
    receiver_sizes = Counter(str(value) for value in receivers.values())
    for key, entry in by_receiver.items():
        result_receivers[key] = dict(**values(entry['counts']), total_U=receiver_sizes[key],
            unique_seen=len(entry['seen']), unique_accepted=len(entry['accepted']),
            groups=grouped(entry['groups']))
    return dict(steps=n, overall=values(totals), receivers=result_receivers,
        epochs={key: values(value) for key, value in epochs.items()}, groups=grouped(group_totals, with_steps=True),
        truth_used_only_after_freeze=True, training_or_selection_use=False)


def analyze_frozen_run(source):
    import numpy as np
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_repl_step_guard.u_scoring import load_truth

    root = Path(source).resolve()
    if root.name != 'source' or root.parent.parent.name != 'F':
        raise ValueError('A registered source trajectory is required')
    done = read(root/'complete.json')
    _check_frozen(done)
    cfg = done['config']
    if cfg['row_id'] != root.parent.name:
        raise ValueError('Frozen row identity differs')
    score_root = root.parents[2]/'readouts'/cfg['row_id']/'U_diagnostic'
    score_done, score = read(score_root/'complete.json'), read(score_root/'scores.json')
    if (score_done.get('status') != 'SCORED_U_DIAGNOSTIC'
            or score_done.get('truth_last') is not True or score_done.get('source_epoch') != 50
            or score.get('training_or_selection_use') is not False):
        raise ValueError('Independent frozen U score must already be complete')
    ids, truth = load_truth(cfg, source_completion=done)
    with np.load(Path(cfg['source_cache'])/'U_s_metadata.npz', allow_pickle=False) as meta:
        if 'y' in meta.files or [str(x) for x in meta['ids'].tolist()] != ids:
            raise ValueError('Original blind U metadata required')
        receivers = dict(zip(ids, [int(x) for x in meta['rx_i']]))
    segments = {}
    for end, offset, count in ((30, 4440, 2220), (50, 6660, 4440)):
        folder = root/f'segment{end}'
        segments[str(end)] = analyze_segment(folder/'U_pseudo_observations.jsonl.gz',
            folder/'step_guard_metrics.jsonl.gz', truth=truth, receivers=receivers,
            step_offset=offset, expected_steps=count, source_completion=done)
    combined = Counter()
    for segment in segments.values():
        combined.update(segment['overall'])
    if dict(combined) != score['overall']:
        raise ValueError('Independent U scoring counts do not match')
    return dict(status='ANALYZED_U_RECEIVER_CONFLICT', row_id=cfg['row_id'],
        segments=segments, overall=dict(combined), truth_last=True, training_or_selection_use=False,
        interpretation='Repeated U exposures; associations are descriptive, not causal.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_frozen_run(args.source), ensure_ascii=True, allow_nan=False))
