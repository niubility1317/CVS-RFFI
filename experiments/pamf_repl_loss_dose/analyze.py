"""Compare frozen clean predictions for the matched CE × auxiliary design."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.pamf_repl_weight.analyze_paired import compare, prediction, read
from experiments.pamf_repl_loss_dose.plan import RECIPES, candidate_rows


TESTS = ('cross_rx', 'cross_both')


def contrasts():
    """Coefficients for matched effects on the percentage-point scale."""
    result = {}
    for ce in (0., .25, .5, 1.):
        result[f'auxiliary_at_ce_{ce}'] = {(1., ce): 1, (0., ce): -1}
    for ce in (0., .25, .5):
        result[f'ce_{ce}_vs_1_with_auxiliary'] = {(1., ce): 1, (1., 1.): -1}
        result[f'ce_{ce}_vs_1_without_auxiliary'] = {(0., ce): 1, (0., 1.): -1}
        result[f'interaction_ce_{ce}_vs_1'] = {
            (1., ce): 1, (1., 1.): -1, (0., ce): -1, (0., 1.): 1,
        }
    return result


def effect_counts(correctness, weights):
    """Count the signed per-ID effect, including the interaction's ±2."""
    effects = sum(weight * correctness[cell].astype(np.int8) for cell, weight in weights.items())
    unique, counts = np.unique(effects, return_counts=True)
    return dict(zip(map(int, unique), map(int, counts)))


def score_complete(entry):
    folder = Path(entry['scores'])
    if (read(folder / 'complete.json')['status'] != 'SCORED_COMPLETE'
            or not read(folder / 'scores.json')['truth_last']):
        raise ValueError('Independent clean score required')


def analyze(root, context):
    state = read(root / 'pipeline_state.json')
    if state['status'] != 'ANALYZED' or state['counts'] != {'COMPLETE': 64}:
        raise ValueError('Wait for all 16 registered rows and independent scores')
    sources, controls = read(root / 'sources.json'), read(root / 'controls.json')
    registered = candidate_rows()
    if set(sources) != {row['row_id'] for row in registered}:
        raise ValueError('Scored matrix and registered rows differ')
    for row in registered:
        row_id = row['row_id']
        entry = sources[row_id]
        score_complete(entry)
        frozen = read(Path(entry['source_output']) / 'complete.json')
        if (frozen['status'] != 'SOURCE_FROZEN' or frozen['epoch'] != 50
                or frozen['target_access'] or frozen['U_truth_read']
                or frozen['config']['tuning_recipe'] != RECIPES[row['recipe']]):
            raise ValueError(f'Expected source-only fixed E50 for {row_id}')
    result = {}
    for setting in ('ten_class', 'source_rx_6'):
        rows = [row for row in registered if row['setting'] == setting]
        candidates = {}
        for row in rows:
            recipe = RECIPES[row['recipe']]
            cell = recipe['auxiliary_dose'], recipe['ce_dose']
            candidates[cell] = sources[row['row_id']]
        current_controls = controls[setting]
        for name in ('pl', 'tri'):
            score_complete(current_controls[name])
        paired = {cell: {} for cell in candidates}
        effects = {name: Counter() for name in contrasts()}
        seen = set()
        for task in TESTS:
            ids, pl = prediction(current_controls['pl']['predictions'], task)
            tri_ids, tri = prediction(current_controls['tri']['predictions'], task)
            physical_ids = set(map(str, ids))
            if len(physical_ids) != len(ids) or physical_ids & seen or not np.array_equal(tri_ids, ids):
                raise ValueError(f'Invalid or unmatched {setting} physical IDs for {task}')
            seen.update(physical_ids)
            truth = read(context['handles'][rows[0]['protocol_id']]['tests'][task]['truth'])
            labels = np.asarray([truth[str(i)]['label'] for i in ids], dtype=np.int64)
            correctness = {}
            for cell, entry in candidates.items():
                candidate_ids, pred = prediction(entry['predictions'], task)
                if not np.array_equal(ids, candidate_ids):
                    raise ValueError(f'Unmatched candidate IDs for {setting}, {cell}, {task}')
                correctness[cell] = pred == labels
                paired[cell][task] = dict(pl=compare(ids, pred, pl, truth),
                                          tri=compare(ids, pred, tri, truth))
            for name, weights in contrasts().items():
                effects[name].update(effect_counts(correctness, weights))
        summaries = []
        for cell, tests in paired.items():
            pooled = {name: Counter() for name in ('pl', 'tri')}
            for comparisons in tests.values():
                for name in pooled:
                    pooled[name].update(comparisons[name]['overall'])
            if any(counts['count'] != len(seen) for counts in pooled.values()):
                raise ValueError('Paired denominators differ')
            summaries.append(dict(auxiliary=cell[0], ce=cell[1], pooled=pooled, tests=tests))
        result[setting] = dict(n=len(seen), candidates=summaries, matched_effects={
            name: dict(count_by_signed_effect=counts,
                       delta_pp=100. * sum(sign * n for sign, n in counts.items()) / len(seen))
            for name, counts in effects.items()
        })
    return dict(sets=result, protocol='independently scored E50, paired cross_rx + cross_both',
                scope='single selected seed and previously exposed test, descriptive effects')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True, type=Path)
    args = parser.parse_args()
    release = args.run_root.parents[1] / 'releases' / (args.run_root.name + '-release')
    context = read(read(release / 'runtime.json')['context'])
    print(json.dumps(analyze(args.run_root, context), ensure_ascii=False, allow_nan=False))
