"""Recompute paired uncertainty for the archived WiSig Main split R69 run."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


REPLICATES = 1600


def paired(raw, restored, rng, n_resamples=REPLICATES, *, labels=('raw', 'r69')):
    """Bootstrap whole physical packets with all seeds and requested views attached."""
    if raw.shape != restored.shape or raw.ndim != 3 or raw.shape[-1] == 0:
        raise ValueError('Expected matching seed, view and nonempty physical packet axes')
    repaired = restored & ~raw
    harmed = raw & ~restored
    changes = repaired.sum(axis=(0, 1)) - harmed.sum(axis=(0, 1))
    support, frequency = np.unique(changes, return_counts=True)
    # Equal-valued physical packets are interchangeable for summing a bootstrap draw.
    bootstrap = rng.multinomial(len(changes), frequency / len(changes), size=n_resamples) @ support
    row = dict(physical_packets=raw.shape[-1], judgments=raw.size, repairs=int(repaired.sum()),
               harms=int(harmed.sum()), net=int(changes.sum()), gain_pp=100 * changes.sum().item() / raw.size,
               **{f'{labels[0]}_correct': int(raw.sum()), f'{labels[1]}_correct': int(restored.sum())})
    return row, bootstrap


def summarize(raw, restored, receivers, views, rng, *, labels=('raw', 'r69')):
    grouped = {name: np.flatnonzero(receivers == name) for name in np.unique(receivers)}
    report = {}
    selections = [(view, slice(index, index + 1)) for index, view in enumerate(views)]
    for view, select in (*selections, ('six_practical', slice(1, None))):
        a, b = raw[:, select], restored[:, select]
        cells, draws = {}, []
        for name, indices in grouped.items():
            row, bootstrap = paired(a[..., indices], b[..., indices], rng, labels=labels)
            row['ci95_pp'] = (100 * np.quantile(bootstrap, (.025, .975)) / row['judgments']).tolist()
            cells[name] = row
            draws.append(bootstrap)
        fields = ('physical_packets', 'judgments', *(f'{label}_correct' for label in labels), 'repairs', 'harms', 'net')
        overall = {field: sum(row[field] for row in cells.values()) for field in fields}
        total = overall['judgments']
        overall['gain_pp'] = 100 * overall['net'] / total
        overall['ci95_pp'] = (100 * np.quantile(np.sum(draws, axis=0), (.025, .975)) / total).tolist()
        report[view] = dict(overall=overall, by_receiver=cells)
    return report


def audit(rt):
    from experiments.cric_dura_comparison import evaluation
    from experiments.cric_dura_comparison.io import read
    from experiments.pamf_repl_view_shuffle import dura_baseline as baseline

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use an independent CPU process with no visible GPUs')
    root = Path(rt['run_root'])
    if rt['run_id'] != baseline.RUN or read(root / 'state.json')['status'] != 'ANALYZED':
        raise ValueError('Require the complete archived pairwise Main split analysis')
    original = read(root / 'summary.json')
    if original['truth_last'] is not True:
        raise ValueError('Archived analysis must be truth-last')
    evaluation.tasks = lambda _: list(baseline.TASKS)
    configs = {(route, seed): baseline.score_config(rt, route, seed)
               for route in baseline.ROUTES for seed in baseline.SEEDS}
    tables = {key: evaluation.preflight(cfg) for key, cfg in configs.items()}
    ordered, reference_cfg = next(iter(tables.values())), next(iter(configs.values()))
    ids = {split: ordered[split][0] for split in baseline.TASKS}
    if (any(len(ids[split]) != baseline.COUNTS[split] for split in baseline.TASKS)
            or len(set(np.concatenate(list(ids.values())))) != sum(baseline.COUNTS.values())):
        raise ValueError('Main split union must be disjoint and complete')
    for key, prepared in tables.items():
        for split in baseline.TASKS:
            if (not np.array_equal(prepared[split][0], ids[split])
                    or configs[key]['tests'][split]['truth'] != reference_cfg['tests'][split]['truth']):
                raise ValueError('Main split frozen packet order or truth reference differs')

    truth, receivers = {}, {}
    for split in baseline.TASKS:
        rows = read(reference_cfg['tests'][split]['truth'])
        truth[split] = np.asarray([rows[str(ident)]['label'] for ident in ids[split]], dtype=np.int64)
        receivers[split] = np.asarray([rows[str(ident)]['receiver'] for ident in ids[split]])
        if truth[split].min() < 0 or truth[split].max() >= len(reference_cfg['classes']):
            raise ValueError('Archived Main split truth class order changed')

    recorded = {(row['route'], row['seed'], row['split']): row['by_view'] for row in original['rows']}
    routes, rng = {}, np.random.default_rng(20261124)
    for route in baseline.ROUTES:
        per_split = {}
        routes[route] = {}
        for split in baseline.TASKS:
            hit = {method: np.stack([[tables[route, seed][split][1][f'{method}__{view}'] == truth[split]
                                      for view in baseline.VIEWS] for seed in baseline.SEEDS])
                   for method in baseline.READOUTS}
            for index, seed in enumerate(baseline.SEEDS):
                for column, view in enumerate(baseline.VIEWS):
                    a, b = hit['raw'][index, column], hit['R69'][index, column]
                    row = dict(correct={'raw': int(a.sum()), 'R69': int(b.sum())},
                               repairs=int((~a & b).sum()), harms=int((a & ~b).sum()))
                    if row != recorded[route, seed, split][view]:
                        raise ValueError('Independent paired totals differ from the original score')
            per_split[split] = hit
            routes[route][split] = summarize(hit['raw'], hit['R69'], receivers[split], baseline.VIEWS, rng)
        merged = [np.concatenate([per_split[split][name] for split in baseline.TASKS], axis=2)
                  for name in baseline.READOUTS]
        routes[route]['combined'] = summarize(*merged, np.concatenate(list(receivers.values())), baseline.VIEWS, rng)
    return dict(run=baseline.RUN, truth_last=True, seeds=list(baseline.SEEDS), bootstrap_replicates=REPLICATES,
                physical_packet_cluster='all_seeds_all_views', strata='receiver', routes=routes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    runtime = json.loads(parser.parse_args().runtime.read_bytes())
    print(json.dumps(audit(runtime), allow_nan=False))
