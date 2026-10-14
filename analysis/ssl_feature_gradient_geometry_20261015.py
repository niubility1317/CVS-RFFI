"""Read existing source-training vectors only; no IQ, model, truth or updates.

Recorded components are L CE, weighted U CE+entropy and feature consistency.
Their sum is a local component diagnostic, not an AdamW/EMA trajectory or a
claim about U correctness, target performance or causal transfer failure.
"""
import argparse
import json
from pathlib import Path
import statistics as st
import time
import torch

ROLE = 'saved_source_training_gradients'
EPS = 1e-12
PARAMETERS = (
    'id_backbone.encoder.core.id_backbone.t1.pw.weight',
    'id_backbone.encoder.core.id_backbone.t2.pw.weight',
    'id_backbone.encoder.sequence_pool.views.0.local_projection.weight',
)


def geometry(L, U, F, raw, *, role=ROLE):
    if role != ROLE:
        raise ValueError('Only saved legal source-training derivatives allowed')
    values = [v.detach().double().flatten() for v in (L, U, F, raw)]
    if len({v.shape for v in values}) != 1 or not values[0].numel():
        raise ValueError('Aligned actual derivative coordinates required')
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError('Nonfinite saved source gradient')
    L, U, F, raw = values
    nl, nu, nf, nr = [float(v.norm()) for v in values]
    cosine = lambda a, b: float(a @ b / (a.norm() * b.norm())) if a.norm() > EPS and b.norm() > EPS else None
    result = dict(CE_norm=nl, U_norm=nu, feature_norm=nf, raw_feature_norm=nr,
        CE_identifiable=nl > EPS, feature_CE_cosine=cosine(L, F),
        raw_feature_CE_cosine=cosine(L, raw), feature_native_components_cosine=cosine(L + U, F))
    if nl <= EPS:
        result.update(feature_over_CE=None, U_over_CE=None, raw_feature_over_CE=None,
            feature_signed_CE=None, native_components_CE_direction=None, all_components_CE_direction=None)
    else:
        uf, ff = float(L @ U) / nl ** 2, float(L @ F) / nl ** 2
        result.update(feature_over_CE=nf / nl, U_over_CE=nu / nl, raw_feature_over_CE=nr / nl,
            feature_signed_CE=ff, native_components_CE_direction=1 + uf,
            all_components_CE_direction=1 + uf + ff)
    return result


def summarize(values):
    output = dict(records=len(values), CE_identifiable=sum(v['CE_identifiable'] for v in values),
        effective_feature_nonzero=sum(v['feature_norm'] > EPS for v in values),
        raw_feature_nonzero=sum(v['raw_feature_norm'] > EPS for v in values))
    for field in ('feature_over_CE', 'U_over_CE', 'raw_feature_over_CE', 'feature_signed_CE',
                  'native_components_CE_direction', 'all_components_CE_direction',
                  'feature_CE_cosine', 'raw_feature_CE_cosine', 'feature_native_components_cosine'):
        available = [v[field] for v in values if v[field] is not None]
        output[field] = dict(count=len(available), mean=st.mean(available) if available else None,
                            median=st.median(available) if available else None,
                            minimum=min(available) if available else None,
                            maximum=max(available) if available else None)
    valid = [v for v in values if v['CE_identifiable']]
    output.update(effective_feature_CE_conflicts=sum(v['feature_CE_cosine'] is not None and v['feature_CE_cosine'] < 0 for v in valid),
        raw_feature_CE_conflicts=sum(v['raw_feature_CE_cosine'] is not None and v['raw_feature_CE_cosine'] < 0 for v in valid),
        feature_norm_exceeds_CE=sum(v['feature_over_CE'] > 1 for v in valid),
        native_components_reverse_CE=sum(v['native_components_CE_direction'] < 0 for v in valid),
        all_components_reverse_CE=sum(v['all_components_CE_direction'] < 0 for v in valid),
        feature_addition_reverses_CE=sum(v['native_components_CE_direction'] >= 0 and v['all_components_CE_direction'] < 0 for v in valid),
        feature_addition_restores_CE=sum(v['native_components_CE_direction'] < 0 and v['all_components_CE_direction'] >= 0 for v in valid))
    return output


def analyze(inputs):
    torch.set_num_threads(2); started = time.time(); groups = []
    batches = vectors = 0; max_reference_difference = 0.
    if len(inputs) != 12 or len({(r['run_id'], r['row_id']) for r in inputs}) != 12:
        raise ValueError('All twelve fixed source models required')
    for row in inputs:
        root = Path(row['source_root'])
        observation = json.loads((root / 'feature_mean_teacher_training_observation.json').read_text())
        if (observation['status'] != 'VERIFIED' or observation['head_fit_updates']
                or observation['U_truth_read'] or observation['steps'] != 11100
                or observation['gradient_epochs'] != [33, 34, 50]):
            raise ValueError('Existing source observation differs')
        for epoch in (33, 34, 50):
            feature = torch.load(root / f'feature_mean_teacher_gradients_e{epoch}.pt', map_location='cpu', weights_only=False)
            native = torch.load(root / f'ssl_gradients_e{epoch}.pt', map_location='cpu', weights_only=False)
            if len(feature) != 222 or len(native) != 222:
                raise ValueError('Complete saved source epoch required')
            layers = [[], [], []]
            for batch, (a, b) in enumerate(zip(feature, native), 1):
                expected = (epoch - 1) * 222 + batch
                if a['step'] != expected or b['step'] != expected or a['epoch'] != epoch or b['epoch'] != epoch:
                    raise ValueError('Saved component steps differ')
                for k in range(3):
                    L = a['vectors']['main_L_CE'][k]
                    other = b['vectors']['main_L_CE'][k]
                    if L.shape != other.shape:
                        raise ValueError('Paired source derivative object differs')
                    error = float((L.double() - other.double()).norm()) / max(float(L.double().norm()), EPS)
                    max_reference_difference = max(max_reference_difference, error)
                    if error > 1e-5:
                        raise ValueError('Source CE references are not the same numerical gradient')
                    values = geometry(L, b['vectors']['weighted_U_CE_entropy'][k],
                                      a['vectors']['effective'][k], a['vectors']['feature'][k])
                    if (epoch == 33 or row['arm'] == 'SIG_TIME') and values['feature_norm'] != 0:
                        raise ValueError('Pre-U or zero-control feature gradient differs')
                    layers[k].append(values)
                batches += 1
                vectors += 18  # Both saved files contain three components x three objects.
            for name, values in zip(PARAMETERS, layers):
                groups.append(dict(method=row['method'], arm=row['arm'], model_seed=row['model_seed'],
                    run_id=row['run_id'], row_id=row['row_id'], epoch=epoch, parameter=name,
                    statistics=summarize(values)))
            del feature, native
    return dict(status='VERIFIED', models=12, diagnostic_epochs=[33, 34, 50],
        paired_gradient_batches=batches, layer_batch_pairs=batches * 3,
        saved_vectors_read=vectors, max_relative_CE_reference_difference=max_reference_difference,
        groups=groups, new_model_forwards=0, new_autograd_calls=0, optimizer_updates=0,
        head_fit_updates=0, IQ_or_checkpoint_or_truth_read=False, source_only=True,
        limitation='Three measured parameter objects and three recorded loss components only; local infinitesimal CE geometry, not full AdamW/EMA updates, U correctness or DG causality',
        seconds=time.time() - started)


def check():
    torch.set_num_threads(2)
    L = torch.tensor([1., 0.]); U = torch.tensor([0., 1.]); F = torch.tensor([-2., 0.])
    value = geometry(L, U, F, F)
    assert value['feature_signed_CE'] == -2 and value['native_components_CE_direction'] == 1
    assert value['all_components_CE_direction'] == -1 and value['feature_over_CE'] == 2
    s = summarize([value])
    assert s['feature_addition_reverses_CE'] == 1 and s['feature_norm_exceeds_CE'] == 1
    orthogonal = geometry(L, U, torch.tensor([0., 2.]), F)
    assert orthogonal['all_components_CE_direction'] == 1
    zero = geometry(L * 0, U, F, F)
    assert not zero['CE_identifiable'] and zero['all_components_CE_direction'] is None
    failures = 0
    for role in ('query', 'target', 'U_truth'):
        try:
            geometry(L, U, F, F, role=role)
        except ValueError:
            failures += 1
        else:
            raise AssertionError('Forbidden gradient role accepted')
    for bad in (torch.tensor([float('nan'), 0.]), torch.zeros(3)):
        try:
            geometry(L, U, bad, F)
        except ValueError:
            failures += 1
        else:
            raise AssertionError('Invalid derivative accepted')
    return dict(status='PASS', known_parallel_and_orthogonal_components=True,
                known_CE_reversal=True, zero_reference_identifiable=False,
                forbidden_or_invalid_cases=failures, real_source_artifacts_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.check:
        parser.error('Use --check locally; analyze(inputs) only reads the fixed saved source vectors')
    print(json.dumps(check()))
