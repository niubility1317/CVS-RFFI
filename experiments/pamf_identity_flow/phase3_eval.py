"""Freeze a new source-trained G and evaluate it with the existing Phase3 F0/PCD.

Prediction consumes public received IQ and inherited frozen model states only.
The separate score entrypoint opens novel/query truth after every prediction is
complete. No target U adaptation, threshold refit, or registration is done.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase3_open_world import fresh_dura_run as reference
from experiments.pamf_phase3_open_world import fresh_dura_score as reference_score
from experiments.pamf_phase3_open_world import fresh_world_score as world_score
from . import source_train as candidate

RUN = '20261101-phase3-identity-flow-source-E30-s2026092701-r01'
STATUS = 'IDENTITY_FLOW_PHASE3_PREDICTIONS_COMPLETE'
FIELDS = {'run_id', 'code_commit', 'candidate_code_commit', 'reference_runtime', 'candidate_checkpoint', 'output_root'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def validate_config(c):
    if not isinstance(c, dict) or set(c) != FIELDS or c['run_id'] != RUN:
        raise ValueError('Exact contextual Phase3 evaluation fields required')
    if any(not isinstance(c[k], str) or not c[k] for k in FIELDS):
        raise ValueError('Nonempty immutable paths and identifiers required')
    if any(len(c[k]) != 40 or any(v not in '0123456789abcdef' for v in c[k]) for k in ('code_commit','candidate_code_commit')):
        raise ValueError('Actual release commit required')
    runtime = _read(c['reference_runtime'])
    old = reference.validate_config(runtime['eval_config'])
    output = Path(c['output_root']).resolve()
    inputs = [Path(c['reference_runtime']).resolve(), Path(c['candidate_checkpoint']).resolve(),
              Path(old['output_root']).resolve(), Path(old['reference_config']['output_root']).resolve()]
    inputs.extend(Path(q['public_root']).resolve() for q in old['Q_conditions'].values())
    if any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError('New output must not overlap input roots')
    return runtime, old


@torch.no_grad()
def _current(backend, models, restorer, values, device):
    a = np.asarray(values)
    if a.dtype != np.float32 or a.shape != (len(a), 2, 256) or not len(a) or not np.isfinite(a).all():
        raise ValueError('Finite fixed public received IQ required')
    x = torch.tensor(a.tolist(), dtype=torch.float32, device=device)
    restored = restorer(x)
    if restored.shape != x.shape or not torch.isfinite(restored).all():
        raise ValueError('New frozen restorer returned invalid IQ')
    f = backend['reader'].features(restored)
    logits = backend['reader'].classify_features(f)
    vectors = [models[name](restored) for name in reference.MODEL_NAMES]
    f = np.asarray(f.detach().cpu().tolist(), np.float32)
    logits = np.asarray(logits.detach().cpu().tolist(), np.float32)
    z = np.stack([np.asarray(v.detach().cpu().tolist(), np.float32) for v in vectors], axis=1)
    if (f.shape != (len(a), 160) or logits.shape != (len(a), 6) or z.shape != (len(a), 2, 160)
            or not all(np.isfinite(v).all() for v in (f, logits, z))
            or not np.allclose(np.linalg.norm(z.astype(np.float64), axis=2), 1., rtol=0, atol=2e-6)):
        raise ValueError('Exact frozen F0 and E0/PCD feature contract required')
    closed = logits.argmax(1).astype(np.int64)
    return dict(default_logits=logits, closed=closed, baseline_closed=closed.copy(),
                raw_signal_features=f, signal_features=z)


def predict(c, device='cuda:0'):
    runtime, old = validate_config(c)
    out = Path(c['output_root'])
    if out.exists():
        raise FileExistsError('New immutable prediction output required')
    completed = _read(Path(c['candidate_checkpoint']).parent/'training_complete.json')
    if (completed.get('status'), completed.get('epoch'), completed.get('updates'),
        completed.get('F0_frozen'), completed.get('target_IQ_reads')) != ('SOURCE_TRAINED', candidate.EPOCHS, candidate.STEPS, True, 0):
        raise ValueError('Complete fixed source-only E30 conditional-flow candidate required')
    # Validate the original six-path artifact and its exact inherited DURA
    # values before using any control or reading the new received records.
    old_done, _, old_rows = reference_score._prediction(old['output_root'])
    parent, states, tau, _ = reference._reference(old)
    if old_done['config'] != old or old_done['parent_F0_identity'] != parent['parent_F0_identity']:
        raise ValueError('Verified DURA control must share the original F0')
    if completed.get('F0') != parent['parent_F0_identity'] or any(completed.get(k) != 0 for k in ('F0_updates','TX_labels_read','V_IQ_reads','OE_IQ_reads','target_IQ_reads')):
        raise ValueError('Candidate must use the exact frozen source F0 without held-out or target training inputs')
    backend = reference._backend(old, device)
    pcd, pcd_identity = reference._load_model(old, device)
    models = dict(E0=backend['model'], PCD=pcd)
    if (backend['model_identity'] != parent['model_identities']['E0']
            or pcd_identity != parent['model_identities']['TARGET_PCD']
            or backend['reader_identity'] != parent['parent_F0_identity']
            or backend['dura_identity'] != parent['loaded_dura_identity']):
        raise ValueError('Original frozen F0/E0/PCD/DURA ancestry differs')
    restorer = candidate.load(Path(c['candidate_checkpoint']), device, c['candidate_code_commit'])
    modules = [backend['reader'], backend['dura'][0], *models.values(), restorer]
    if any(p.requires_grad or p.dtype != torch.float32 for m in modules for p in m.parameters()):
        raise ValueError('All networks must remain frozen FP32')
    snapshots = [(m, reference.fresh.best._snapshot_model(m)) for m in modules]
    fixed_states, fixed_tau = deepcopy(states), tau.copy()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from experiments.pamf_phase3_open_world.star_open_feature_export import _synthetic
    smoke = np.asarray(_synthetic(device).detach().cpu().tolist(), np.float32)
    _current(backend, models, restorer, smoke, device)
    out.mkdir(parents=True)
    _write(out/'config.json', c)
    _write(out/'smoke.json', dict(status='PASS', candidate_F0_E0_PCD_checkpoints=True,
                                 real_data_reads=0, optimizer_updates=0))
    source_ids = set()
    counts = {}
    def finalise(a):
        a['raw_scores'] = reference._raw(states, a['signal_features'], a['closed'])
        a['scores'] = a['raw_scores'] - tau[:, a['closed']].T
        return a
    for role, cache_role in (('known', 'V'), ('novel', 'novel')):
        previous = old_rows['source_DURA'][role]
        iq = reference._source_role(old, cache_role)
        try:
            if (not np.array_equal(iq['ids'], previous['ids'])
                    or iq['manifest']['parent_F0_identity'] != parent['parent_F0_identity']
                    or iq['manifest']['dura_identity'] != parent['dura_identity']
                    or any(not np.array_equal(iq['metadata'][k], previous[k]) for k in
                           ('rx_i', 'day_i') + (('y', 'scenario') if role == 'known' else ()))):
                raise ValueError('Exact source physical order and metadata required')
            ids = set(iq['ids'].tolist())
            if ids & source_ids:
                raise ValueError('Source physical roles must be disjoint')
            source_ids.update(ids)
            blocks = [_current(backend, models, restorer, iq['received'][j:j+128], device)
                      for j in range(0, len(iq['ids']), 128)]
            current = {k: np.concatenate([v[k] for v in blocks]) for k in blocks[0]}
            a = {k: v.copy() for k, v in previous.items()}
            a.update({k: v for k, v in current.items() if k in a})
            reference._validate(finalise(a), role)
            reference._save(out/'source'/(role+'.npz'), a)
            counts[role] = len(a['ids'])
            print('IDENTITY_FLOW_SOURCE_COMPLETE', role, counts[role], flush=True)
        finally:
            reference.received.close_role(iq)
    paired = None
    for condition in reference.QUERY_CONDITIONS:
        previous = old_rows[condition]
        manifest, ids, iq = reference._public(old, condition)
        try:
            if not np.array_equal(ids, previous['ids']) or set(ids.tolist()) & source_ids:
                raise ValueError('Exact source-disjoint paired target IDs required')
            if paired is not None and not np.array_equal(ids, paired):
                raise ValueError('Clean and star IDs must be in the same order')
            paired = ids.copy()
            blocks = [_current(backend, models, restorer, iq[j:j+128], device)
                      for j in range(0, len(ids), 128)]
            current = {k: np.concatenate([v[k] for v in blocks]) for k in blocks[0]}
            a = dict(ids=ids.copy(), **{k:v for k,v in current.items() if k in reference.TARGET_FIELDS})
            reference._validate(finalise(a), 'Q')
            name = condition.replace('_DURA', '')
            reference._save(out/'targets'/(name+'.npz'), a)
            counts[name] = len(ids)
            print('IDENTITY_FLOW_TARGET_COMPLETE', name, len(ids), flush=True)
        finally:
            if hasattr(iq, '_mmap'):
                iq._mmap.close()
    for model, snapshot in snapshots:
        reference.fresh.best._check_model(model, snapshot)
    if not reference._same(states, fixed_states) or not np.array_equal(tau, fixed_tau):
        raise ValueError('Frozen source density or threshold changed')
    if counts != dict(known=27000, novel=3150, clean=10238, star=10238):
        raise ValueError('Every fixed source and target prediction is required')
    _write(out/'complete.json', dict(status=STATUS, run_id=RUN, code_commit=c['code_commit'],
        config=c, candidate_code_commit=c['candidate_code_commit'], counts=counts, parent_F0_identity=parent['parent_F0_identity'],
        model_identities=dict(E0=parent['model_identities']['E0'], PCD=parent['model_identities']['TARGET_PCD']),
        all_predictions_before_truth=True, all_networks_and_thresholds_frozen=True,
        private_truth_reads=0, target_U_reads=0, U_ID_reads=0, new_updates=0,
        density_fit_updates=0, threshold_fit_updates=0, historical_PCD_target_U_ancestor=True,
        candidate_training_F_access=True, final_phase3_complete=False))


def _arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def _prediction(root):
    root = Path(root)
    done = _read(root/'complete.json')
    if (done.get('status') != STATUS or done.get('run_id') != RUN
            or done.get('counts') != dict(known=27000, novel=3150, clean=10238, star=10238)
            or not done.get('all_predictions_before_truth') or not done.get('all_networks_and_thresholds_frozen')
            or any(done.get(k) != 0 for k in ('private_truth_reads','target_U_reads','U_ID_reads',
                                               'new_updates','density_fit_updates','threshold_fit_updates'))):
        raise ValueError('Complete frozen predictions required before private truth')
    runtime, old = validate_config(done['config'])
    old_done, _, verified_controls = reference_score._prediction(old['output_root'])
    parent, states, tau, calibration = reference._reference(old)
    if (done['candidate_code_commit'] != done['config']['candidate_code_commit'] or done['parent_F0_identity'] != parent['parent_F0_identity']
            or old_done['config'] != old or old_done['parent_F0_identity'] != parent['parent_F0_identity']):
        raise ValueError('Candidate and F0 identities differ')
    rows = {'source': {role:_arrays(root/'source'/(role+'.npz')) for role in ('known','novel')},
            'clean':_arrays(root/'targets/clean.npz'), 'star':_arrays(root/'targets/star.npz')}
    controls = {'source':verified_controls['source_DURA'],
                'clean':verified_controls['clean_DURA'],
                'star':verified_controls['star_DURA'],
                'raw_clean':verified_controls['clean_raw']}
    for role in ('known','novel'):
        reference._validate(rows['source'][role], role)
        keys = ('ids','rx_i','day_i') + (('y','scenario') if role == 'known' else ())
        reference_score._align(rows['source'][role], controls['source'][role], keys)
    for condition in ('clean','star'):
        reference._validate(rows[condition], 'Q')
        reference_score._align(rows[condition], controls[condition], ('ids',))
        reference_score._align(rows[condition], controls['raw_clean'], ('ids',))
    source_ids = [set(rows['source'][role]['ids'].tolist()) for role in ('known','novel')]
    if source_ids[0] & source_ids[1] or (source_ids[0] | source_ids[1]) & set(rows['star']['ids'].tolist()):
        raise ValueError('Source roles and fixed target records must be disjoint')
    reference_score._align(rows['clean'], rows['star'], ('ids',))
    for a in [*rows['source'].values(), rows['clean'], rows['star']]:
        raw = reference._raw(states, a['signal_features'], a['closed'])
        if not np.allclose(a['raw_scores'], raw, rtol=1e-12, atol=1e-9) or not np.array_equal(a['scores'], a['raw_scores']-tau[:,a['closed']].T):
            raise ValueError('Frozen density and predicted-class thresholds must reconstruct')
    return done, runtime, calibration, rows, controls


def paired_clean_gaps(clean_closed, star_closed, truth, scenes, clean_accepted=None, star_accepted=None):
    """Each scene compares the exact same known physical records and TX roster."""
    if (clean_accepted is None) != (star_accepted is None):
        raise ValueError('Acceptance masks must be supplied as a pair')
    gaps = {}
    y = truth['y']
    for scene in scenes:
        mask = truth['scenario'] == scene
        rates = []
        for tx in range(6):
            selected = mask & (y == tx)
            if not selected.any():
                rates = []
                break
            clean_correct = clean_closed[selected] == tx
            star_correct = star_closed[selected] == tx
            if clean_accepted is not None:
                clean_correct &= clean_accepted[selected]
                star_correct &= star_accepted[selected]
            rates.append((float(np.mean(clean_correct)), float(np.mean(star_correct))))
        gaps[scene] = None if not rates else 100 * (np.mean([x[0] for x in rates])-np.mean([x[1] for x in rates]))
    return gaps


def score(prediction_root, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU-only scorer required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('New immutable independent score output required')
    done, runtime, cal, rows, controls = _prediction(prediction_root)
    # These are the first private truth reads, after all four predictions and
    # all frozen controls/states have been validated above.
    novel = reference_score._source_truth(runtime['source_novel_truth'], rows['source']['novel'])
    truth = {k:reference_score._future_truth(runtime[k+'_truth'], rows[k]['ids']) for k in ('clean','star')}
    for kind, values in truth.items():
        allowed = ('clean',) if kind == 'clean' else world_score.common.SCENES
        if not np.isin(values['scenario'], allowed).all():
            raise ValueError('Clean/star truth scene roster differs')
    if any(not np.array_equal(truth['clean'][k], truth['star'][k]) for k in ('ids','y','tx','receiver','day')):
        raise ValueError('Clean/star must be the same labelled physical records')
    index = reference.PRIMARY_INDEX
    results = {}
    for name, data in (('candidate', rows), ('DURA', controls)):
        results[name] = dict(source=reference_score._source(data['source'], novel, index),
            clean=reference_score._target(data['clean'], truth['clean'], 'clean_DURA', index),
            star=reference_score._target(data['star'], truth['star'], 'star_DURA', index))
    # Reuse the established paired-count math with candidate as the new path
    # and independently verified DURA as the old path. The temporary keys are
    # the helper's interface only; result metadata names the actual models.
    pair_rows, pair_paths = {}, {}
    for condition in ('source','clean','star'):
        pair_rows[condition+'_DURA'] = rows[condition]
        pair_rows[condition+'_raw'] = controls[condition]
        pair_paths[condition+'_DURA'] = results['candidate'][condition]
        pair_paths[condition+'_raw'] = results['DURA'][condition]
    paired = reference_score._pairs(pair_rows, truth, novel, index, pair_paths)
    gaps = paired_clean_gaps(controls['raw_clean']['closed'], rows['star']['closed'], truth['star'], world_score.common.SCENES)
    open_gaps = paired_clean_gaps(controls['raw_clean']['closed'], rows['star']['closed'], truth['star'],
        world_score.common.SCENES, controls['raw_clean']['scores'][:,index] >= 0, rows['star']['scores'][:,index] >= 0)
    known = truth['clean']['y'] < 6
    clean_masks = [known & (truth['clean']['y']==c) for c in range(6)]
    clean_loss = None
    if all(mask.any() for mask in clean_masks):
        clean_raw = np.mean([np.mean(controls['raw_clean']['closed'][m] == c) for c,m in enumerate(clean_masks)])
        clean_new = np.mean([np.mean(rows['clean']['closed'][m] == c) for c,m in enumerate(clean_masks)])
        clean_loss = 100*float(clean_raw-clean_new)
    result = dict(status='IDENTITY_FLOW_PHASE3_SCORES_COMPLETE', run_id=RUN, code_commit=done['code_commit'],
        candidate_code_commit=done['candidate_code_commit'], results=results, raw_clean_minus_candidate_star_F0_macro_pp=gaps,
        paired_candidate_minus_DURA=paired,
        raw_clean_minus_candidate_star_AK_macro_pp=open_gaps,
        raw_clean_minus_candidate_clean_F0_macro_pp=clean_loss,
        clean_damage_at_most_1pp=bool(clean_loss is not None and clean_loss <= 1.),
        six_scene_gap_3pp_passed=all(v is not None and v <= 3. for v in gaps.values()),
        gap_scope='Known TX macro F0 closed accuracy; raw clean and candidate star use the same physical IDs within each fixed star scene',
        primary_method='PCD_predicted_ratio', original_F0_unchanged=True,
        reference='DURA v1 on the same IDs and inherited PCD weights/density/thresholds',
        truth_last=True, target_U_reads=0, new_neural_updates=0, threshold_fit_updates=0,
        evaluation_neural_updates=0, candidate_training_F_access=True,
        candidate_source_guidance='Frozen original Phase3 F0 source feature anchor and path agreement',
        candidate_source_training_epochs=candidate.EPOCHS, candidate_source_training_updates=candidate.STEPS,
        historical_PCD_target_U_ancestor=True, target_REG=False, final_phase3_complete=False,
        scope='Exposed development RX3-2 and source V/novel diagnostics; no independent blind claim or registration claim')
    output.mkdir(parents=True)
    _write(output/'results.json', result)
    print('IDENTITY_FLOW_PHASE3_SCORES_COMPLETE', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('predict','score'))
    parser.add_argument('--config')
    parser.add_argument('--prediction-root')
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.mode == 'predict':
        if not args.config:
            parser.error('--config required')
        predict(_read(args.config), args.device)
    else:
        if not args.prediction_root or not args.output:
            parser.error('--prediction-root and --output required')
        score(args.prediction_root, args.output)


if __name__ == '__main__':
    main()
