"""Source-only post-hoc analysis of already saved backbone-gradient matrices.

No model, IQ, optimizer, prediction, or truth loader is imported.  Standard output
is the only output; the caller retains the result outside the original run root.
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time

import torch

RX = (1, 3, 4, 6, 8)
EPOCHS = (1, 17, 33, 50)
ARMS = ('SIG_TIME', 'BONE_RAW_FISHR', 'BONE_COND_FISHR')
FIRST = '20261014-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s01-m4-r01'
REMAINING = '20261015-phase1-backbone-fishr-oldgate-cos-e50-clean-manysig-s0203-m8-r01'


def decompose(v):
    """Exact pairwise radial/angular decomposition of the *normalized* loss."""
    v = v.double()
    if v.ndim != 2 or v.shape[0] != 5 or not torch.isfinite(v).all() or (v < 0).any():
        raise ValueError('Five finite nonnegative RX variance vectors required')
    avg = v.mean(0)
    numerator = (v - avg).square().mean()
    denominator = avg.square().mean()
    norms = v.norm(dim=1)
    radial = []
    angular = []
    cosines = []
    for i, j in itertools.combinations(range(5), 2):
        # The squared-norm identity avoids unstable 1-cos subtraction.
        distance2 = (v[i] - v[j]).square().sum()
        radial2 = (norms[i] - norms[j]).square()
        angular2 = distance2 - radial2
        if angular2 < -1e-12 * max(float(distance2), float(radial2), 1e-30):
            raise ValueError('Negative angular energy beyond roundoff')
        radial.append(radial2 / v.shape[1])
        angular.append(angular2.clamp_min(0) / v.shape[1])
        if norms[i] > 0 and norms[j] > 0:
            cosines.append(float((v[i] @ v[j]) / (norms[i] * norms[j])))
    factor = 4 / 10  # (R-1)/(2R), R=5
    radial = factor * torch.stack(radial).mean()
    angular = factor * torch.stack(angular).mean()
    if not torch.allclose(radial + angular, numerator, rtol=1e-9, atol=1e-35):
        raise ValueError('Radial/angular decomposition does not recover numerator')
    denom = denominator.clamp_min(1e-30)
    return dict(
        normalized_loss=float(numerator / denom),
        radial_component=float(radial / denom),
        angular_component=float(angular / denom),
        numerator=float(numerator), denominator=float(denominator),
        denominator_floor_active=bool(denominator < 1e-30),
        radial_fraction=float(radial / numerator) if numerator > 0 else None,
        angular_fraction=float(angular / numerator) if numerator > 0 else None,
        variance_norm_mean=float(norms.mean()),
        variance_norm_population_CV=float(norms.std(unbiased=False) / norms.mean())
        if norms.mean() > 0 else None,
        variance_shape_mean_cosine=statistics.mean(cosines) if cosines else None,
        variance_shape_valid_pairs=len(cosines),
    )


def checks():
    base = torch.tensor([1., 2., 4., 3.], dtype=torch.float64)
    same = base.repeat(5, 1)
    assert decompose(same)['normalized_loss'] == 0
    radial = decompose(torch.arange(1., 6.).double()[:, None] * same)
    assert abs(radial['radial_fraction'] - 1) < 1e-12
    angular = decompose(torch.eye(5, dtype=torch.float64))
    assert abs(angular['angular_fraction'] - 1) < 1e-12
    general = torch.tensor([[1., 3., 7.], [5., 2., 1.], [3., 8., 2.],
                            [7., 4., 5.], [2., 1., 6.]], dtype=torch.float64)
    original = decompose(general)
    for scale in (.01, .5, 2., 100.):
        scaled = decompose(general * scale)
        assert abs(scaled['normalized_loss'] - original['normalized_loss']) < 1e-12
        assert abs(scaled['radial_component'] - original['radial_component']) < 1e-12
        assert abs(scaled['angular_component'] - original['angular_component']) < 1e-12
    assert decompose(general * 1e-20)['denominator_floor_active']
    for invalid in (general[:4], general * float('nan'), -general):
        try:
            decompose(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid variance vectors accepted')
    return dict(status='PASS', exact_decomposition=True, pure_radial_and_angular=True,
                scale_invariance_above_floor=True, numerical_floor_explicit=True,
                invalid_inputs_rejected=True, model_or_query_read=False)


def run(project):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    started = time.time()
    records = []
    matching = {}
    matrices = 0
    for seed in (2026092701, 2026092702, 2026092703):
        parent = FIRST if seed == 2026092701 else REMAINING
        for arm in ARMS:
            row = f'{arm}-s{seed}'
            source = Path(project) / 'runs' / parent / row / 'source'
            reference = json.loads((source / 'backbone_fishr_gradient_records.json').read_text())
            if reference['epochs'] != list(EPOCHS) or reference['U_truth_read']:
                raise ValueError('Saved source-gradient scope differs')
            for epoch in EPOCHS:
                values = torch.load(source / f'backbone_fishr_gradients_e{epoch}.pt',
                                    map_location='cpu', weights_only=False)
                expected = [r for r in reference['batch_records'] if r['epoch'] == epoch]
                if len(values) != 13 or len(expected) != 13:
                    raise ValueError('Missing saved source cohorts')
                for batch, (entry, ref) in enumerate(zip(values, expected), 1):
                    y, rx, day = (entry[k].cpu() for k in ('y', 'rx', 'day'))
                    ids = entry['source_physical_ids']
                    if len(ids) != 180 or len(set(ids)) != 180 or entry['step'] != ref['step']:
                        raise ValueError('Saved source cohort identity differs')
                    layout = (tuple(y.tolist()), tuple(rx.tolist()), tuple(day.tolist()), tuple(ids))
                    key = (seed, epoch, batch)
                    if arm == 'SIG_TIME':
                        matching[key] = layout
                    elif matching[key] != layout:
                        raise ValueError('Paired saved source packet roles differ')
                    for tx, dd, receiver in itertools.product(range(6), (1, 2, 3), RX):
                        if int(((y == tx) & (day == dd) & (rx == receiver)).sum()) != 2:
                            raise ValueError('Expected true-L TX/day/RX layout differs')
                    terms = {'RAW': [], 'TX_DAY_CENTERED': []}
                    for layer, g in enumerate(entry['parameter_gradient_matrices']):
                        if g.shape != (180, (3456, 6912)[layer]) or not torch.isfinite(g).all():
                            raise ValueError('Saved actual parameter-gradient object differs')
                        g = g.double()
                        group = y * 3 + day - 1
                        means = torch.stack([g[group == i].mean(0) for i in range(18)])
                        residual = g - means[group]
                        matrices += 1
                        for view, vectors in (('RAW', g), ('TX_DAY_CENTERED', residual)):
                            vv = torch.stack([vectors[rx == receiver].var(0, unbiased=True)
                                              for receiver in RX])
                            result = decompose(vv)
                            terms[view].append(result['normalized_loss'])
                            records.append(dict(seed=seed, arm=arm, epoch=epoch,
                                                cohort=batch, layer=('t1', 't2')[layer],
                                                view=view, **result,
                                                mean_packet_gradient_norm=float(vectors.norm(dim=1).mean())))
                    for view in terms:
                        if abs(statistics.mean(terms[view]) - ref['raw_losses'][view]) > 1e-8:
                            raise ValueError('Independent loss decomposition disagrees with saved objective')
                del values
    grouped = {}
    for r in records:
        key = (r['seed'], r['arm'], r['epoch'], r['layer'], r['view'])
        grouped.setdefault(key, []).append(r)
    summaries = []
    keys = ('normalized_loss', 'radial_component', 'angular_component', 'radial_fraction',
            'angular_fraction', 'variance_norm_mean', 'variance_norm_population_CV',
            'variance_shape_mean_cosine', 'mean_packet_gradient_norm')
    for key, rows in grouped.items():
        if len(rows) != 13:
            raise ValueError('Incomplete source aggregate')
        summaries.append(dict(zip(('seed', 'arm', 'epoch', 'layer', 'view'), key),
                              cohorts=13, **{k: statistics.mean(r[k] for r in rows) for k in keys}))
    if matrices != 936 or len(records) != 1872 or len(summaries) != 144:
        raise ValueError('Incomplete fixed nine-model saved-gradient analysis')
    return dict(status='VERIFIED', analysis='posthoc_saved_source_variance_decomposition',
                parents=[FIRST, REMAINING], model_count=9, actual_saved_matrices=matrices,
                variance_views=len(records), grouped_source_records=len(summaries),
                saved_source_cohorts_matched=True,
                floor_active_views=sum(r['denominator_floor_active'] for r in records),
                source_only=True, new_model_forwards=0, new_gradient_evaluations=0,
                model_updates=0, head_fit_updates=0, optimizer_steps=0,
                query_IQ_read=False, truth_or_predictions_read=False,
                original_run_writes=0, summaries=summaries, records=records,
                scope='Four diagnostic epochs, 13 correlated source cohorts each; not 936 model seeds. '
                      'No target-feedback rule, new model selection, or causal explanation of target errors.',
                seconds=time.time() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path)
    parser.add_argument('--checks', action='store_true')
    args = parser.parse_args()
    if args.checks:
        result = checks()
    elif args.project:
        result = run(args.project)
    else:
        parser.error('--project or --checks is required')
    print(json.dumps(result, allow_nan=False))
