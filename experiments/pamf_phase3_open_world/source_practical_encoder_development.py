"""Frozen E52 source encoders on isolated development Q; independent scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import development_transfer as transfer, source_practical_development as parent
from . import source_practical_encoder as native, source_open_readout as core
from . import source_consensus as consensus, scores
from .source_open_export import DIMENSIONS, _tensor
from .source_risk_calibration import METHODS


RUN = native.RUN
ARMS = native.ARMS
STATES = ('parent_E51', 'increment')
PRIMARY = 'mlp16_oe'
SCHEMA = 'pamf_phase3_source_practical_encoder_development_v1'
PREDICTED = 'SOURCE_PRACTICAL_ENCODER_DEVELOPMENT_PREDICTIONS_COMPLETE'
SCORED = 'SCORED_SOURCE_PRACTICAL_ENCODER_DEVELOPMENT_COMPLETE'
FIELDS = {'run_id', 'code_commit', 'arm', 'condition', 'view_family', 'Q_public_root', 'Q_capsule_id',
          'Q_split_id', 'Q_expected_count', 'train_root', 'source_root', 'parent_prediction_root',
          'output_root', 'batch_size', 'dura_checkpoint', 'dura_release'}
PREDICTION_FIELDS = parent.PREDICTION_FIELDS
ZERO_FIELDS = ('target_U_reads', 'formal_target_reads', 'source_IQ_reads', 'new_encoder_updates',
               'new_identity_head_updates', 'new_readout_updates', 'new_threshold_calibrations')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_config(c):
    if not isinstance(c, dict) or set(c) != FIELDS:
        raise ValueError('Only frozen own-source states and public Q; no U, truth or unknown composition')
    if any(not isinstance(c[k], str) or not c[k].strip() for k in FIELDS -
           {'Q_expected_count', 'batch_size', 'dura_checkpoint', 'dura_release'}):
        raise ValueError('Nonempty frozen state identities and public Q paths required')
    if (c['run_id'] != RUN or c['arm'] not in ARMS or not re.fullmatch('[0-9a-f]{40}', c['code_commit'])
            or type(c['Q_expected_count']) is not int or c['Q_expected_count'] != 864
            or type(c['batch_size']) is not int or c['batch_size'] != 128):
        raise ValueError('Fixed source encoder run/arm, actual commit, Q864 and batch128 required')
    family = {'C0': 'clean_cross_receiver', 'P1': 'practical_cross_receiver'}
    if c['condition'] not in family or c['view_family'] != family[c['condition']]:
        raise ValueError('Only independent clean C0 and practical P1 conditions are allowed')
    public, train, source = (Path(c[k]) for k in ('Q_public_root', 'train_root', 'source_root'))
    if public.name != 'public' or public.parent.name != 'evaluation':
        raise ValueError('Only development evaluation public Q is allowed')
    if source.name != 'source' or source.parent != train.parent or train.parent.name != c['arm']:
        raise ValueError('Source aggregates and checkpoint must belong to the same declared encoder arm')
    if any(Path(c['output_root']).resolve().is_relative_to(Path(c[k]).resolve()) for k in
           ('Q_public_root', 'train_root', 'source_root', 'parent_prediction_root')):
        raise ValueError('Prediction output cannot modify public or frozen source/reference inputs')
    if c['condition'] == 'C0':
        if c['dura_checkpoint'] is not None or c['dura_release'] is not None:
            raise ValueError('Independent clean C0 has no G or practical reference')
    elif any(not isinstance(c[k], str) or not c[k].strip() for k in ('dura_checkpoint', 'dura_release')):
        raise ValueError('Practical P1 requires the same frozen source DURA; no fallback')
    return c


def _load_bundle(root):
    from .source_practical_encoder_readout import load_bundle
    return load_bundle(root)


def _dura_identity(g, c):
    expected = dict(deployment_id='dura_v1', arm='stage2', epoch=20, updates=4940,
        training_commit='b2a2ce937a64654e17e84c2aa817b9da4257a48f', parameters=266951,
        reuse=True, graph=True, native=False, precision='FP32', TF32=False)
    if not isinstance(g, dict) or any(g.get(k) != v for k, v in expected.items()):
        raise ValueError('The unchanged frozen DURA v1 R69 source identity is required')
    if c['condition'] == 'P1' and (g.get('checkpoint') != c['dura_checkpoint'] or g.get('release_root') != c['dura_release']):
        raise ValueError('Source views and development P1 must use the identical frozen DURA')


def _sources(c):
    """Only completed own-row aggregates and old source metadata; no IQ rows."""
    bundle = _load_bundle(c['source_root'])
    training = _read(Path(c['train_root']) / 'complete.json')
    tc, identity = training['config'], bundle['encoder_identity']
    if (training.get('status') != native.STATUS or training.get('schema') != native.SCHEMA
            or training.get('encoder_identity') != identity or training.get('updates') != 222
            or training.get('cumulative_updates') != 11544
            or any(tc.get(k) != c[k] for k in ('run_id', 'arm', 'code_commit'))
            or Path(tc.get('output_root', '')).resolve() != Path(c['train_root']).resolve()
            or any(identity.get(k) != c[k] for k in ('run_id', 'arm', 'code_commit'))
            or identity.get('updates') != 222 or identity.get('cumulative_updates') != 11544
            or Path(identity.get('checkpoint', '')).resolve() != (Path(c['train_root']) / 'final.pt').resolve()):
        raise ValueError('Own frozen source encoder must be this arm/code/root E51 plus 222-update endpoint')
    source_marker = bundle.get('source_marker', {})
    expected_schema = 'pamf_phase3_source_practical_encoder_evaluation_v1'
    for marker, status in ((bundle['fit_marker'], 'SOURCE_PRACTICAL_ENCODER_READOUTS_FROZEN'),
            (bundle['calibration_marker'], 'SOURCE_PRACTICAL_ENCODER_THRESHOLDS_FROZEN'),
            (source_marker, 'SOURCE_PRACTICAL_ENCODER_SOURCE_COMPLETE')):
        if (marker.get('status') != status or marker.get('schema') != expected_schema
                or marker.get('encoder_identity') != identity or marker.get('source_only') is not True
                or any(marker.get(k) != c[k] for k in ('run_id', 'arm', 'code_commit'))):
            raise ValueError('Own geometry/readouts and known-only thresholds must be frozen before Q')
    sc = source_marker.get('config', {})
    if (Path(sc.get('train_root', '')).resolve() != Path(c['train_root']).resolve()
            or Path(sc.get('output_root', '')).resolve() != Path(c['source_root']).resolve()
            or source_marker.get('frozen_parameters_buffers_unchanged') is not True
            or source_marker.get('frozen_own_source_arrays_unchanged') is not True):
        raise ValueError('Completed own source aggregates must belong to the declared immutable row')
    thresholds = bundle['thresholds']
    if thresholds.shape != (7,) or thresholds.dtype != np.float64 or not np.isfinite(thresholds).all():
        raise ValueError('Seven original own-source frozen threshold scalars required')
    core._geometry(bundle['geometry']); core._readout_state(bundle['state'])
    _dura_identity(bundle['dura_identity'], c)
    if training.get('dura_identity') != bundle['dura_identity']:
        raise ValueError('Encoder cache and source readout DURA identities differ')
    old = transfer._source_bundle(tc)
    if identity.get('parent_encoder_identity') != old['marker']['encoder_state_identity']:
        raise ValueError('Encoder/source ancestry must be the same fixed uniform-OE parent E51')
    return old, bundle


def _base(c, source):
    return dict(run_id=c['run_id'], code_commit=c['code_commit'], public_root=c['Q_public_root'],
        capsule_id=c['Q_capsule_id'], split_id=c['Q_split_id'], expected_count=c['Q_expected_count'],
        source_row_root=source['source_row_root'], risk_prediction_root=source['risk_prediction_root'],
        output_root=c['output_root'], input_paths=[c['condition']], batch_size=c['batch_size'],
        view_family=c['view_family'], dura_checkpoint=c['dura_checkpoint'], dura_release=c['dura_release'])


def _backend(c, source, bundle, device):
    model = native.load_trained(c['train_root'], device)
    if getattr(model, '_phase3_encoder_identity', None) != bundle['encoder_identity']:
        raise ValueError('Loaded full native encoder and own-source geometry/readouts differ')
    if c['condition'] == 'C0':
        return model, None
    # The old frozen reader loader establishes its already-verified train_config
    # for the G source contract. No source IQ or source feature bank is read.
    previous, baseline = transfer._load_readers(source, device)
    del previous, baseline
    return model, transfer._load_dura(source, _base(c, source), device)


def _compact(model, x, bundle):
    import torch
    with torch.no_grad():
        data = transfer._encode_new(model.id_backbone.encoder, x)
    branches = {name: data['features__' + name] for name in DIMENSIONS}
    desc = consensus.descriptor(branches, data['logits'], bundle['geometry'])
    static = scores.knownness_scores(branches['fused160'], data['logits'], bundle['geometry']['branches']['fused160']['prototypes'])
    # Compact row-local API: no legacy scorer requiring source IDs or fitting.
    values = np.concatenate((static, core.score_readouts(desc, bundle['state'])), axis=1).astype(np.float64)
    if values.shape != (len(x), 7) or not np.isfinite(values).all():
        raise ValueError('Seven finite own-geometry/readout scores required')
    return values, data['logits'].argmax(1).astype(np.int64)


def _smoke(model, g, bundle, c, device):
    import torch
    t = torch.arange(256, dtype=torch.float32, device=device)
    x = torch.stack((torch.stack((torch.cos(.11*t), torch.sin(.11*t))),
                     torch.stack((torch.cos(.19*t+.3), torch.sin(.19*t+.3)))))
    if g is not None:
        with torch.no_grad(): x = g[1](x)[0].detach().clone()
    values, closed = _compact(model, x, bundle)
    if (x.dtype != torch.float32 or tuple(x.shape) != (2, 2, 256) or not bool(torch.isfinite(x).all())
            or values.shape != (2, 7) or closed.shape != (2,)):
        raise ValueError('Actual-checkpoint two-row no-query smoke failed')
    result = dict(status='PASS', real_checkpoint=True, synthetic_rows=2, source_IQ_reads=0,
        target_reads=0, optimizer_updates=0, default_head_classes=6, descriptor_dimension=100)
    print('SOURCE_ENCODER_DEVELOPMENT_SYNTHETIC_SMOKE_PASS', json.dumps(result), flush=True)
    return result


def _check(models, snapshots, source, bundle, arrays):
    for model, snapshot in zip(models, snapshots): transfer._check_model(model, snapshot)
    after = transfer._array_snapshot(dict(source=source, bundle=bundle))
    if set(after) != set(arrays) or any(not np.array_equal(arrays[k], after[k]) for k in arrays):
        raise ValueError('Frozen own-source aggregate/readout/threshold state changed')


def _reference(c, source):
    done, p = parent._predictions(c['parent_prediction_root'])
    fit, calibration = Path(done.get('fit_root', '')), Path(done.get('calibration_root', ''))
    if (done.get('run_id') != parent.RUN or done.get('condition') != c['condition']
            or done.get('view_family') != c['view_family'] or done.get('count') != c['Q_expected_count']
            or done.get('public_capsule_id') != c['Q_capsule_id'] or done.get('public_split_id') != c['Q_split_id']
            or done.get('encoder_state_identity') != source['marker']['encoder_state_identity']
            or Path(done.get('source_row_root', '')).resolve() != Path(source['source_row_root']).resolve()
            or fit.name != 'fit' or fit.parent.name != 'source' or fit.parent.parent.name != parent.RUN
            or calibration.name != 'calibration' or calibration.parent != fit.parent
            or not np.array_equal(p['baseline_closed_prediction'], np.repeat(p['baseline_closed_prediction'][2:3], 3, axis=0))):
        raise ValueError('Parent must be the same-input previous source-practical augmented E51 readout row')
    selected = {k: p[k][2].copy() for k in ('knownness', 'closed_prediction', 'baseline_closed_prediction',
                                          'thresholds', 'accepted', 'prediction')}
    selected['ids'] = p['ids'].copy()
    return done, selected


def predict(c, device='cuda:0'):
    validate_config(c); output = Path(c['output_root'])
    if output.exists(): raise FileExistsError('Immutable encoder development prediction root already exists')
    source, bundle = _sources(c)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, g = _backend(c, source, bundle, device)
    models = [model] + ([] if g is None else [g[0]])
    snapshots = [transfer._snapshot_model(m) for m in models]
    arrays = transfer._array_snapshot(dict(source=source, bundle=bundle))
    smoke = _smoke(model, g, bundle, c, device)
    _check(models, snapshots, source, bundle, arrays)  # Before either reference or Q is opened.
    old_marker, previous = _reference(c, source)
    _, ids, iq = transfer._public(_base(c, source))
    try:
        if not np.array_equal(ids, previous['ids']):
            raise ValueError('Parent and new Q opaque IDs/order must align exactly')
        n = len(ids); values = np.empty((2, n, 7), np.float64); closed = np.empty((2, n), np.int64)
        values[0], closed[0] = previous['knownness'], previous['closed_prediction']
        for start in range(0, n, c['batch_size']):
            stop = min(start + c['batch_size'], n); x = _tensor(iq[start:stop], device)
            if g is not None:
                with torch.no_grad(): x = g[1](x)[0].detach().clone()
            values[1, start:stop], closed[1, start:stop] = _compact(model, x, bundle)
            if start == 0 or stop == n: print('SOURCE_ENCODER_DEVELOPMENT_PROGRESS', c['arm'], c['condition'], stop, n, flush=True)
        _check(models, snapshots, source, bundle, arrays)
        thresholds = np.stack((previous['thresholds'], bundle['thresholds']))
        accepted = values >= thresholds[:, None, :]
        prediction = np.where(accepted, closed[:, :, None], -1).astype(np.int64)
        baseline = np.repeat(previous['baseline_closed_prediction'][None], 2, axis=0)
        output.mkdir(parents=True, exist_ok=False)
        with (output / 'predictions.npz').open('xb') as stream:
            np.savez(stream, ids=ids, state_names=np.asarray(STATES), method_names=np.asarray(METHODS),
                closed_prediction=closed, baseline_closed_prediction=baseline, knownness=values,
                accepted=accepted, prediction=prediction, thresholds=thresholds)
        result = dict(status=PREDICTED, schema=SCHEMA, config=dict(c), run_id=RUN, arm=c['arm'], code_commit=c['code_commit'],
            condition=c['condition'], view_family=c['view_family'], count=n, state_names=list(STATES), method_names=list(METHODS),
            primary_state='increment', primary_method=PRIMARY, public_capsule_id=c['Q_capsule_id'], public_split_id=c['Q_split_id'],
            encoder_identity=bundle['encoder_identity'], parent_encoder_identity=source['marker']['encoder_state_identity'],
            dura_identity=bundle['dura_identity'], source_root=c['source_root'], train_root=c['train_root'],
            parent_prediction_root=c['parent_prediction_root'], parent_run_id=old_marker['run_id'], parent_state='augmented_readout',
            source_state_frozen_before_Q=True, truth_read=False, query_fit=False,
            frozen_parameters_buffers_unchanged=True, frozen_source_aggregates_unchanged=True,
            same_input_original_F6_baseline_reused=True, increment_closed_diff_from_parent_count=int(np.sum(closed[1] != closed[0])),
            real_checkpoint_synthetic_smoke_passed=True, smoke=smoke, anonymous_registration_performed=False, final_phase3_complete=False)
        result.update({k: 0 for k in ZERO_FIELDS}); _write(output / 'complete.json', result)
        return result
    finally:
        if hasattr(iq, '_mmap'): iq._mmap.close()


def _predictions(root):
    root = Path(root)
    try: done = _read(root / 'complete.json')
    except (OSError, ValueError) as exc: raise ValueError('Truth closed: complete encoder development predictions required') from exc
    validate_config(done.get('config'))
    if (done.get('status') != PREDICTED or done.get('schema') != SCHEMA or done.get('run_id') != RUN
            or done.get('state_names') != list(STATES) or done.get('method_names') != list(METHODS)
            or done.get('primary_state') != 'increment' or done.get('primary_method') != PRIMARY
            or done.get('source_state_frozen_before_Q') is not True or done.get('query_fit') is not False
            or done.get('truth_read') is not False or done.get('frozen_parameters_buffers_unchanged') is not True
            or done.get('frozen_source_aggregates_unchanged') is not True or done.get('count') != 864
            or any(done.get(k) != done['config'][k] for k in ('run_id', 'arm', 'code_commit', 'condition', 'view_family'))
            or any(done.get(k) != 0 for k in ZERO_FIELDS)):
        raise ValueError('Truth closed: frozen source states or completed two-state prediction boundary differs')
    with np.load(root / 'predictions.npz', allow_pickle=False) as stored:
        if set(stored.files) != PREDICTION_FIELDS: raise ValueError('Truth closed: prediction fields differ')
        p = {k: stored[k].copy() for k in stored.files}
    n = done['count']
    if (p['ids'].shape != (n,) or p['ids'].dtype.kind != 'U' or len(set(p['ids'].tolist())) != n or np.any(p['ids'] == '')
            or p['state_names'].tolist() != list(STATES) or p['method_names'].tolist() != list(METHODS)
            or p['knownness'].shape != (2, n, 7) or p['knownness'].dtype != np.float64 or not np.isfinite(p['knownness']).all()
            or p['thresholds'].shape != (2, 7) or p['thresholds'].dtype != np.float64 or not np.isfinite(p['thresholds']).all()
            or p['accepted'].shape != (2, n, 7) or p['accepted'].dtype.kind != 'b'
            or p['prediction'].shape != (2, n, 7) or p['prediction'].dtype != np.int64):
        raise ValueError('Truth closed: aligned finite two-state prediction arrays required')
    for key in ('closed_prediction', 'baseline_closed_prediction'):
        if p[key].shape != (2, n) or p[key].dtype != np.int64 or np.any(p[key] < 0) or np.any(p[key] >= 6):
            raise ValueError('Truth closed: complete old-six default-head/baseline predictions required')
    if (not np.array_equal(p['baseline_closed_prediction'][0], p['baseline_closed_prediction'][1])
            or not np.array_equal(p['accepted'], p['knownness'] >= p['thresholds'][:, None, :])
            or not np.array_equal(p['prediction'], np.where(p['accepted'], p['closed_prediction'][:, :, None], -1))):
        raise ValueError('Truth closed: same-input baseline, frozen thresholds or decisions differ')
    return done, p


def score(predictions_root, truth, output, targets):
    output = Path(output)
    if output.exists(): raise FileExistsError('Immutable encoder development score output already exists')
    done, p = _predictions(predictions_root)
    spec, minimum = transfer._targets(targets, done['view_family'])
    actual = transfer._truth(truth, p['ids'])
    y, rx, scenes = actual['y'], actual['receiver'], actual['scenario']
    if not bool((rx == 26).all()) or np.any(y >= 9):
        raise ValueError('Only RX24-1 index26 and the fixed development TX mapping may be scored')
    tx = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20', '19-15', '19-16', '7-1']
    rows = {}
    for i, state in enumerate(STATES):
        methods = {}
        for j, method in enumerate(METHODS):
            def metrics(mask):
                return transfer._metrics(y[mask], p['closed_prediction'][i, mask], p['baseline_closed_prediction'][i, mask],
                    p['knownness'][i, mask, j], p['accepted'][i, mask, j])
            methods[method] = dict(overall=metrics(np.ones(len(y), bool)),
                by_scenario={s: metrics(scenes == s) for s in sorted(set(scenes.tolist()))},
                by_transmitter={tx[int(label)]: metrics(y == label) for label in np.unique(y)},
                planned_discovery_unknown=metrics(np.isin(y, [6, 7])), planned_later_unknown=metrics(y == 8),
                planned_cohorts_diagnostic_only=True, diagnostic_only=method != PRIMARY)
        main = methods[PRIMARY]; later = main['planned_later_unknown']['unknown_macro_rejection']
        later_scenes = {s: None if not np.any((scenes == s) & (y == 8)) else
            float(np.mean(~p['accepted'][i, (scenes == s) & (y == 8), METHODS.index(PRIMARY)])) for s in set(scenes.tolist())}
        rows[state] = dict(methods=methods,
            primary_goals=transfer._goals(main['overall'], main['by_scenario'], minimum, done['view_family']),
            later_unknown_risk=dict(diagnostic_only=True, target_U_exposure=0, rejection=later, by_scenario=later_scenes,
                at_condition_target=later is not None and later >= minimum and all(v is not None and v >= minimum for v in later_scenes.values())))
    keys = ('unknown_macro_rejection', 'known_macro_acceptance', 'known_macro_open_accuracy', 'known_macro_frr',
            'original_closed_minus_new_open', 'encoder_closed_loss', 'auroc_known', 'oscr')
    def difference(a, b): return {k: None if a[k] is None or b[k] is None else a[k] - b[k] for k in keys}
    deltas = {method: difference(rows['increment']['methods'][method]['overall'], rows['parent_E51']['methods'][method]['overall']) for method in METHODS}
    passed = rows['increment']['primary_goals']['pass']
    result = dict(status=SCORED, schema=SCHEMA, run_id=RUN, arm=done['arm'], condition=done['condition'],
        view_family=done['view_family'], truth_last=True, development_only=True, count=len(y), states=rows,
        deltas=dict(increment_minus_parent_E51=deltas), primary_state='increment', primary_method=PRIMARY, primary_pass=passed,
        condition_unknown_rejection_min=minimum, targets_schema=spec['schema'], target_U_reads=0, query_fit=False,
        formal_target_reads=0, anonymous_registration_performed=False, final_phase3_complete=False,
        primary_pass_rule='Current overall and each main-scene unknown/known-acceptance/original-F0 loss targets',
        planned_cohort_note='Scorer-only discovery/later cohorts; no target U exposure, adaptation or registration',
        later_unknown_risk_affects_primary_pass=False, state_selection='Increment MLP preregistered; no target-driven state/method selection',
        next_step='SOURCE_ENCODER_PASSED_DEVELOPMENT_ONLY' if passed else 'NO_PROMOTION')
    output.mkdir(parents=True, exist_ok=False); _write(output / 'scores.json', result)
    _write(output / 'complete.json', {k: result[k] for k in ('status', 'schema', 'truth_last', 'arm', 'condition',
        'primary_state', 'primary_method', 'primary_pass', 'next_step', 'final_phase3_complete')})
    return result


def main(argv=None):
    print(json.dumps(dict(event='SOURCE_PRACTICAL_ENCODER_DEVELOPMENT_PROCESS', pid=os.getpid(), cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv))), flush=True)
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('predict'); p.add_argument('--config', required=True); p.add_argument('--device', default='cuda:0')
    s = sub.add_parser('score'); s.add_argument('--predictions-root', required=True); s.add_argument('--truth', required=True)
    s.add_argument('--targets', required=True); s.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = predict(_read(args.config), args.device) if args.action == 'predict' else score(args.predictions_root, args.truth, args.output, args.targets)
    print(json.dumps(dict(status=result['status'])), flush=True); return 0


if __name__ == '__main__': raise SystemExit(main())
