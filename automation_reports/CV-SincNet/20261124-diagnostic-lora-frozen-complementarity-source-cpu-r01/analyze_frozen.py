"""Truth-side-only overlap analysis of already frozen LoRa source predictions."""

import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np


ROUTES = ('full', 'residual')
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
RX = ('b200_1', 'b200_mini_1', 'b210_1', 'pluto_1', 'rtl_1')
BASELINE = ('direct', 'reference')
ANCHOR = '20261124-phase1-lora-empirical-iq-prior-source-m2-r01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prediction(root, run, route, selected):
    frozen = read(root / run / f'{route}-predictions.json')
    ids, modes = frozen['ids'], frozen['modes']
    if len(ids) != 900 or len(set(ids)) != 900 or set(frozen['rows']) != set(VIEWS):
        raise ValueError('physical coverage differs from the fixed 900-packet panel')
    if not set(selected).issubset(modes) or len(set(modes)) != len(modes):
        raise ValueError('a specified frozen mode is missing or ambiguous')
    raw = np.asarray([frozen['rows'][v] for v in VIEWS])
    if (raw.dtype.kind not in 'iu' or raw.shape != (len(VIEWS), len(modes), len(ids))
            or raw.min() < 0 or raw.max() >= 10):
        raise ValueError('invalid frozen prediction array')
    return tuple(ids), {name: raw[:, modes.index(name)].astype(np.int8) for name in selected}


def collect(root, specs):
    anchors = {r: prediction(root, ANCHOR, r, BASELINE) for r in ROUTES}
    panel = anchors['full'][0]
    if anchors['residual'][0] != panel:
        raise ValueError('anchor routes differ in physical IDs')
    labels = ['direct', 'reference']
    families = ['baseline', 'baseline']
    primary = [True, True]
    frozen = {r: [anchors[r][1][k] for k in BASELINE] for r in ROUTES}
    included, omitted = [], []
    for s in specs:
        run = s['run']
        selected = [*BASELINE, *s['main'], *s['extra']]
        if len(set(selected)) != len(selected):
            raise ValueError(f'Nonunique fixed modes in {run}')
        try:
            rows = {r: prediction(root, run, r, selected) for r in ROUTES}
            for r, (ids, decisions) in rows.items():
                if ids != panel:
                    raise ValueError(f'{r} ordering or physical sample set differs')
                for k in BASELINE:
                    mismatch = int((anchors[r][1][k] != decisions[k]).sum())
                    if mismatch:
                        raise ValueError(f'{r} {k} differs from anchor in {mismatch} decisions')
        except (OSError, ValueError, KeyError) as exc:
            omitted.append(dict(run=run, reason=str(exc)))
            continue
        included.append(dict(run=run, primary=len(s['main']), extra=len(s['extra'])))
        for group, active in (('main', True), ('extra', False)):
            for key in s[group]:
                labels.append(f'{run}/{key}')
                families.append(s['family'])
                primary.append(active)
                for r in ROUTES:
                    frozen[r].append(rows[r][1][key])
    if not included:
        raise ValueError('No complete paired historical run')
    return panel, labels, families, np.asarray(primary), {r: np.stack(v) for r, v in frozen.items()}, included, omitted


def counts(hit):
    return dict(clean=int(hit[0].sum()), six=int(hit[1:].sum()), low=int(hit[-1].sum()),
                views={view: int(row.sum()) for view, row in zip(VIEWS, hit, strict=True)})


def oracles(hit, selected, rx):
    choices = hit[selected]
    cover = choices.any(0)
    baseline = hit[1]
    added = cover & ~baseline
    packed = np.packbits(choices.reshape(len(choices), -1), axis=1)
    return dict(counts=counts(cover), potential_repairs=counts(added),
                remaining_wrong=counts(~cover), distinct_correctness_patterns=len(np.unique(packed, axis=0)),
                by_rx={name: dict(n=int((rx == i).sum()), cover=counts(cover[:, rx == i]),
                                  additional=counts(added[:, rx == i]))
                       for i, name in enumerate(RX)})


def shared_pair(hits, labels, is_main, target):
    # The diagnostic chooses common arm identities, with ideal per-observation
    # selection within each route. Reference and direct are also always available.
    baseline = {r: hits[r][:2].any(0) for r in ROUTES}
    arms = np.flatnonzero(is_main)[2:]
    rank = []
    for a, b in combinations(arms, 2):
        summary = {r: counts(baseline[r] | hits[r][a] | hits[r][b]) for r in ROUTES}
        full, residual = (summary[r] for r in ROUTES)
        key = (min(full['six'], residual['six']),
               min(full['low'], residual['low']), full['six'] + residual['six'])
        rank.append((key, a, b, summary))
    if not rank:
        raise ValueError('Need at least two fixed primary arms')
    _, a, b, summary = max(rank, key=lambda x: x[0])
    meets = sum(x[3]['full']['six'] >= target['six'] and x[3]['full']['low'] >= target['low'] for x in rank)
    best_full = max(rank, key=lambda x: (x[3]['full']['six'], x[3]['full']['low']))
    return dict(pair=[labels[a], labels[b]], counts=summary, studied_pairs=len(rank),
                pairs_meeting_numeric_full_targets=meets,
                max_full_six_best_pair=best_full[3]['full']['six'])


def sensitivity(hits, frozen, labels, rx, target):
    """Post-hoc checks for aggressive modes that inflate a truth-side union."""
    checks = {
        'clean_same': np.logical_and.reduce([(f[:, 0] == f[0, 0]).all(1) for f in frozen.values()]),
        'clean_same_raw_floor': np.logical_and.reduce([
            (f[:, 0] == f[0, 0]).all(1) & (hits[r][:, 1:].sum((1, 2)) >= hits[r][0, 1:].sum())
            for r, f in frozen.items()]),
        'clean_same_dura_floor': np.logical_and.reduce([
            (f[:, 0] == f[0, 0]).all(1) & (hits[r][:, 1:].sum((1, 2)) >= hits[r][1, 1:].sum())
            for r, f in frozen.items()]),
    }
    output = {}
    for key, valid in checks.items():
        arm_count = int(valid[2:].sum())
        valid[:2] = True
        output[key] = dict(eligible_arms=arm_count,
                           routes={r: oracles(hits[r], valid, rx) for r in ROUTES})

    # Descriptive greedy coverage with Full as primary and Residual as tie
    # breaker. Chosen on source labels, these counts cannot select a method.
    cover = {r: hits[r][:2].any(0) for r in ROUTES}
    remaining = np.arange(2, len(labels))
    steps = []
    while len(remaining) and (cover['full'][1:].sum() < target['six']
                              or cover['full'][-1].sum() < target['low']):
        gain = [(hits[r][remaining, 1:] & ~cover[r][1:]).sum((1, 2)) for r in ROUTES]
        idx = int(np.lexsort((gain[1], gain[0]))[-1])
        pick = remaining[idx]
        if gain[0][idx] == 0:
            break
        cover = {r: value | hits[r][pick] for r, value in cover.items()}
        steps.append(dict(arm=labels[pick], by_route={r: counts(cover[r]) for r in ROUTES}))
        remaining = np.delete(remaining, idx)
    return dict(screened_unions=output, unrestricted_greedy=steps)


def analyze(root, source, specs):
    panel, names, family, is_main, frozen, included, omitted = collect(root, specs)
    ids = np.asarray(read(source / 'contract.json')['role_ids']['V'])
    lookup = {key: i for i, key in enumerate(ids)}
    chosen = np.asarray([lookup[key] for key in panel])
    # Identity is opened only here. Existing full and residual predictions are
    # fixed, validated and aligned before any label is read.
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as held:
        if not np.array_equal(held['ids'][chosen], ids[chosen]):
            raise ValueError('Score-side data ordering mismatch')
        y, rx = held['y'][chosen], held['rx_i'][chosen]
    if (len(y) != 900 or not np.array_equal(np.bincount(rx), [180] * len(RX))
            or np.any((y < 0) | (y >= 10))):
        raise ValueError('Expected a balanced legal source V panel')
    hits = {r: frozen[r] == y for r in ROUTES}
    target = dict(six=3618, low=477)
    indexes = np.arange(len(names))
    family = np.asarray(family)
    summary = {}
    for r, hit in hits.items():
        controls = {names[i]: counts(hit[i]) for i in range(2)}
        base = hit[1]
        per_arm = {k: dict(counts=counts(h), repair=counts(h & ~base), harm=counts(~h & base))
                   for k, h in zip(names, hit, strict=True)}
        summary[r] = dict(controls=controls, choices={
            'raw_or_DURA': oracles(hit, indexes < 2, rx),
            'primary': oracles(hit, is_main, rx),
            'all': oracles(hit, indexes >= 0, rx)},
            family={str(k): oracles(hit, (family == k) | (indexes < 2), rx)
                    for k in np.unique(family) if k != 'baseline'},
            arms=per_arm,
            best_primary_arm=max((k for k, main in zip(names, is_main, strict=True) if main),
                                 key=lambda k: per_arm[k]['counts']['six']))
    return dict(status='ANALYZED', n=900, targets_for_scale=target, target_reads=0,
                truth_after_validated_predictions=True, new_predictions=0, fitted_parameters=0,
                included=included, omitted=omitted, n_primary=int(is_main.sum()), n_all=len(names),
                common_pair=shared_pair(hits, names, is_main, target),
                posthoc=sensitivity(hits, frozen, names, rx, target), routes=summary)


def main(spec):
    runtime = read(Path('runtime.json'))
    if runtime['run_id'] != ANCHOR:
        raise ValueError('Use the existing immutable anchor release CWD')
    parent = Path(runtime['run_root']).parent
    print(json.dumps(analyze(parent, Path(runtime['source_root']), spec)), flush=True)


if __name__ == '__main__':
    main(read(Path(sys.argv[1])))
