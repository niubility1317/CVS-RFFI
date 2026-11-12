"""Compare reduced weights with matched PL, Hybrid, and full-weight confidence Hybrid."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import hybrid_confidence_report as base, hybrid_weight_plan as plan
from .target_tuning_report import load_score


def refresh_full_weight(rt):
    root = Path(rt['run_root'])
    prior = Path(rt['project']) / 'runs/20261113-phase1-pamf-hybrid-confidence-m30-r01'
    prior_sources = read(prior / 'sources.json')
    results = read(root / 'analysis/results.json')['rows']
    pairs, pending = [], []
    for row in results:
        key = f"{row['setting']}-hybrid-confidence-s{row['model_seed']}"
        folder = Path(prior_sources[key]['scores'])
        if not (folder / 'complete.json').is_file():
            pending.append(key)
            continue
        full = load_score(folder)
        if full['counts'] != row['current']['counts']:
            raise ValueError('Full-weight and reduced-weight test populations differ')
        pairs.append(dict(row_id=row['row_id'], model_seed=row['model_seed'], auxiliary_weight=row['auxiliary_weight'],
                          full_weight=full, clean_delta=row['current']['clean'] - full['clean'],
                          practical_delta=row['current']['practical_mean'] - full['practical_mean']))
    write(root / 'analysis/full_weight_comparison.json', dict(pairs=pairs, pending=sorted(set(pending)),
          interpretation='same-setting same-seed, previously exposed test; exploratory'), exclusive=False)


def analyze(rt):
    old = base.plan
    base.plan = plan
    try:
        base.analyze(rt)
    finally:
        base.plan = old
    refresh_full_weight(rt)
