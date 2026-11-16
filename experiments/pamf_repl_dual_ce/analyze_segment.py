"""Analyze a closed source segment without reading U truth or test inputs."""
import argparse
from collections import Counter, defaultdict
import gzip
from itertools import zip_longest
import json
import math
from pathlib import Path

GROUPS = ('t2', 'cls_head', 'other')


def records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def guard_counts(value):
    if not value['actual_parameter_check'] or not value['norm_nonexpansion']:
        raise ValueError('Actual AdamW parameter projection check failed')
    result = {'fallback': int(value['quantization_fallback'])}
    for anchor, label in (('L_full', 'full'), ('L_clean', 'ce')):
        for stage in ('before', 'after'):
            cosine = value['anchor_cosines_'+stage][anchor]
            if cosine is not None and not math.isfinite(cosine):
                raise ValueError('Finite anchor cosine required')
            result[f'{label}_{stage}_conflict'] = int(cosine is not None and cosine < -1e-6)
            result[f'{label}_{stage}_undefined'] = int(cosine is None)
            if stage == 'after' and result[f'{label}_{stage}_conflict']:
                raise ValueError('Protected step has a material anchor conflict')
    return result


def analyze(folder, *, offset, expected_steps):
    folder = Path(folder)
    done = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    state = json.loads((folder/'repl_state.json').read_text(encoding='utf-8'))
    guard_state = state['U_parameter_step_guard']
    if (done['status'] != 'SOURCE_FROZEN' or done['updates'] != expected_steps
            or done['target_access'] or done['U_truth_read'] or state['U_truth_read']
            or state['calls'] != expected_steps or guard_state['calls'] != expected_steps
            or guard_state['extra_autograd_calls'] != expected_steps
            or guard_state['second_anchor'] != 'ordinary_labeled_CE_recorded_as_L_clean'):
        raise ValueError('Closed source-only dual-CE segment required')
    epoch_data = [json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(epoch_data)*222 != expected_steps:
        raise ValueError('Complete source epoch records required')
    paths = [folder/'step_metrics.jsonl.gz', folder/'step_guard_metrics.jsonl.gz', folder/'U_pseudo_observations.jsonl.gz']
    groups = {g: Counter() for g in GROUPS}
    epoch = defaultdict(Counter)
    seen_ids, accepted_ids = set(), set()
    seen = accepted = 0
    n = 0
    for n, triple in enumerate(zip_longest(*(records(path) for path in paths)), 1):
        step, guard, pseudo = triple
        if None in triple or step['step'] != n or guard['step'] != n or pseudo['step'] != offset*222+n:
            raise ValueError('Step, protection and blind U records must align')
        arrays = [pseudo[k] for k in ('physical_ids', 'pseudo', 'confidence', 'accepted')]
        if not all(len(a) == len(arrays[0]) for a in arrays):
            raise ValueError('U fields must align')
        seen += len(arrays[0]); accepted += sum(arrays[3])
        seen_ids.update(arrays[0]); accepted_ids.update(i for i,a in zip(arrays[0],arrays[3]) if a)
        metrics = step['metrics']
        if int(metrics['HPL_seen']) != seen or int(metrics['HPL_accepted']) != accepted:
            raise ValueError('Online U counters differ from blind observations')
        e = offset+(n-1)//222+1
        ce = float(metrics['train/loss_tx_labeled'])
        if not math.isfinite(ce):
            raise ValueError('Finite labeled CE required')
        epoch[e]['CE_sum'] += ce; epoch[e]['steps'] += 1
        for group in GROUPS:
            item = guard['groups'][group]
            groups[group].update(guard_counts(item))
            for metric in ('L_norm','U_norm','applied_U_norm','conflict','capped'):
                value = float(metrics[f'HPL_GP_{group}_{metric}'])
                if not math.isfinite(value):
                    raise ValueError('Finite original gradient telemetry required')
                groups[group][metric+'_sum'] += value
            for field in ('raw_increment_norm','applied_increment_norm','retention'):
                groups[group][field+'_sum'] += float(item[field])
    if n != expected_steps or seen != state['seen'] or accepted != state['accepted']:
        raise ValueError('Every segment step and U decision required')
    return dict(status='ANALYZED_CLOSED_SOURCE_SEGMENT', row_id=done['config']['row_id'],
        offset=offset, steps=n, U_seen=seen, U_accepted=accepted, U_acceptance=accepted/seen,
        unique_U_seen=len(seen_ids), unique_U_accepted=len(accepted_ids),
        epochs=[dict(epoch=offset+i+1, labeled_CE=epoch[offset+i+1]['CE_sum']/222,
                     source_val_accuracy=row['val_tx_acc']) for i,row in enumerate(epoch_data)],
        groups={g: dict(values) for g,values in groups.items()},
        target_inputs_read=False, U_truth_read=False,
        interpretation='Counts are repeated training steps or U exposures; CE is the ordinary labeled CE anchor.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--offset', required=True, type=int)
    parser.add_argument('--steps', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(analyze(args.folder, offset=args.offset, expected_steps=args.steps), allow_nan=False))
