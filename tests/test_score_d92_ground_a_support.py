"""Synthetic packet/cache and real candidate trace -> separate Ground A join."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import score_d92_ground_a_support as scorer
import export_d92_ground_classifier_a_packet as packets
from cvsrffi.d92_ground_classifier_a import GroundClassifierA, WEIGHT_KEY


def _packet(root):
    native = ['g3', 'g0', 'g5', 'g1', 'g4', 'g2']; old = sorted(native)
    weight = torch.zeros((6, 160), dtype=torch.float32)
    for j, name in enumerate(native): weight[j, int(name[1:])] = 1.
    checkpoint = root / 'synthetic.pt'; torch.save(dict(model={WEIGHT_KEY: weight}), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source = dict(checkpoint_sha256=digest, verdict='MATCHED_SOURCE_ONLY_SCRATCH', classes=old,
                  target_access_before_freeze=False, checkpoint_inheritance=[], model_seed=17,
                  checkpoint_epoch=200, source_role_comparison='EXACT_MATCH')
    binding = dict(checkpoint_sha256=digest, checkpoint_weight_key=WEIGHT_KEY, ordered_classes=native,
        scale=9.5, norm_eps=1e-4, feature_contract=dict(checkpoint_sha256=digest, cache_key='z_id', source_tensor='feat_joint',
        representation='raw', dtype='float32', feature_dim=160), logit_corrections='none',
        class_row_order_source='synthetic explicit original native rows', factory_scale_source='synthetic factory scale 9.5',
        corrections_source='synthetic none')
    packet = root / 'packet'; packets.export_packet(checkpoint=checkpoint, binding=binding, provenance=source, output=packet)
    return packet, digest, source, old, native


def _case(root, *, method='conditional', k=2, new=1):
    root.mkdir(); packet, digest, source, old, native = _packet(root)
    classes = old + (['new'] if new else []); y = np.repeat(np.arange(len(classes)), k).astype(np.int64)
    ids = ['physical_' + str(i).zfill(3) for i in range(len(y))]
    raw = dict(z_id=np.zeros((len(y), 160), dtype=np.float32), fft=np.zeros((len(y), 96), dtype=np.float32),
        t_emb=np.zeros((len(y), 160), dtype=np.float32), f_emb=np.zeros((len(y), 160), dtype=np.float32), pa_local=np.zeros((len(y), 160), dtype=np.float32))
    for i, label in enumerate(y):
        if classes[int(label)] != 'g0': raw['z_id'][i, int(label) if label < 6 else 0] = 2. + i / 16
    cache = root / 'cache'; cache.mkdir()
    split = dict(split_id='synthetic_split', receiver='synthetic_receiver', scenario='practical_high', k=k, support_seed=11,
        registered_classes=classes, support_ids=ids, support_indices=list(range(len(ids))), support_labels=y.tolist())
    np.savez(cache / scorer.CACHE_NAME, **raw, ids=np.asarray(ids), indices=np.arange(len(ids), dtype=np.int64),
             labels=np.asarray([classes[int(label)] for label in y]), checkpoint_sha256=np.asarray(digest),
             capsule_id=np.asarray('synthetic_capsule'), feature_contract_json=np.asarray(json.dumps(scorer.FEATURE_CONTRACT, sort_keys=True)))
    common = dict(schema=scorer.CACHE_SCHEMA, checkpoint_sha256=digest, capsule_id='synthetic_capsule', model_seed=17,
        feature_contract=scorer.FEATURE_CONTRACT, query_iq_access=False, source_data_access=False, truth_read=False,
        adapted_state_inherited=False, query_rows_read=0, native_eval=True, new_source_payload_bytes=0, new_ground_statistics_bytes=0)
    marker = dict(common, status='BRANCH_SUPPORT_FEATURES_COMPLETE', dtype='float32', view_count_per_observation=1,
        query_used_for_fitting=False, encoder_updated=False, native_parameters_unchanged=True, native_buffers_unchanged=True,
        classes=old, count=len(ids), shapes={name: list(value.shape) for name, value in raw.items()}, feature_file_bytes=(cache / scorer.CACHE_NAME).stat().st_size)
    scorer.write(cache / 'features_complete.json', marker)
    scorer.write(cache / 'startup.json', dict(common, provenance=source, query_fit_access=False))
    scorer.write(cache / 'checkpoint_provenance.json', source)
    scorer.write(cache / 'support_splits.json', dict(schema=scorer.CACHE_SCHEMA, capsule_id='synthetic_capsule', checkpoint_sha256=digest, splits=[split]))
    if method == 'conditional':
        import evaluate_d92_conditional_joint_probe as entry
        probe = entry.probe_conditional_joint
    else:
        import evaluate_d92_affine_joint_probe as entry
        probe = entry.probe_affine_joint
    lane = root / 'lane'; lane.mkdir(); archive = entry.StateArchive(lane)
    binding = dict(run_id='synthetic_run', row_id='synthetic_row', checkpoint_sha256=digest, capsule_id='synthetic_capsule', model_seed=17)
    record = probe(**raw, support_labels=y, support_ids=ids, classes=classes, old_classes=old,
        context=dict(run_id=binding['run_id'], row_id=binding['row_id'], split_id=split['split_id']), state_callback=archive)
    archive.finalize('COMPLETE')
    identity = entry.split_identity(split, old)
    record = entry.json_native(dict(record, **identity, scope=entry.SCOPE, query_rows_used=0, source_rows_used=0))
    with (lane / 'fit_trace.jsonl').open('x', encoding='utf-8') as stream: stream.write(json.dumps(record) + '\n')
    selected = dict(splits=[identity])
    source_context = dict(binding, schema=entry.SCHEMA, method=entry.METHOD, scope=entry.SCOPE,
        query_rows_used=0, source_rows_used=0, truth_read=False, support_features=str(cache.resolve()), provenance=source, episodes=1)
    scorer.write(lane / 'startup.json', dict(source_context, config=dict(selection=selected)))
    scorer.write(lane / 'probe_complete.json', dict(source_context, status=entry.STATUS, selection=selected))
    return dict(root=root, packet=packet, cache=cache, lane=lane, digest=digest, native=native, old=old, record=record, split=split, binding=binding)


@pytest.fixture(scope='module', params=['conditional', 'affine'])
def production(tmp_path_factory, request):
    return _case(tmp_path_factory.mktemp('ground_a_support') / request.param, method=request.param)


def _args(case, output):
    return dict(packet=case['packet'], support_features=case['cache'], output=output,
        selection=scorer.read(case['lane'] / 'startup.json')['config']['selection'],
        run_id=case['binding']['run_id'], row_id=case['binding']['row_id'], expected_checkpoint_sha256=case['digest'],
        expected_capsule_id='synthetic_capsule', expected_model_seed=17)


def _fixed(case, root):
    scorer.predict_support(**_args(case, root)); return root


def _copied_trace(case, root, mutate):
    root.mkdir()
    for name in ('startup.json', 'probe_complete.json'): shutil.copyfile(case['lane'] / name, root / name)
    record = deepcopy(case['record']); mutate(record)
    with (root / 'fit_trace.jsonl').open('x', encoding='utf-8') as stream: stream.write(json.dumps(record) + '\n')
    return root / 'fit_trace.jsonl'


def test_real_candidate_trace_pairing_original_ground_columns_and_no_old_summary(production, tmp_path):
    old_report = production['lane'] / 'synthetic_existing_report.md'
    if not old_report.exists(): old_report.write_text('original unchanged', encoding='utf-8')
    out = tmp_path / 'supplement'
    result = scorer.score_support(**_args(production, out), fit_trace=production['lane'] / 'fit_trace.jsonl')
    assert result['status'] == scorer.STATUS and result['query_rows_used'] == 0
    assert result['ordered_ground_classes'] == production['native']
    assert result['original_summary_modified'] is result['training_performed'] is result['calibration_performed'] is False
    parent = result['parents'][0]; metrics = parent['oof']['metrics']
    assert parent['oof']['old_held_physical_ids'] == production['split']['support_ids'][:12]
    assert metrics['A_old_accuracy'] == pytest.approx(5 / 6)
    assert metrics['adaptation_gain_B_minus_A'] == pytest.approx(metrics['B_old_accuracy'] - 5 / 6)
    assert all(example['A_prediction'] == 'g3' for path in parent['paired_paths'] for example in path['examples'] if example['truth'] == 'g0')
    assert result['resources']['score_call_count'] == len(production['split']['support_ids'])
    assert result['resources']['incremental_transmission_bytes'] is None and result['resources']['deployment_peak_memory_bytes'] is None
    assert old_report.read_text(encoding='utf-8') == 'original unchanged'
    assert (out / 'pairing.json').exists() and (out / 'complete.json').exists()


def test_scores_are_persisted_before_any_truth_or_trace_access(production, tmp_path, monkeypatch):
    out = tmp_path / 'guarded'
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.name == 'fit_trace.jsonl':
            assert (out / 'prediction_complete.json').is_file(), 'Truth was read before scores were persisted'
        return original_open(path, *args, **kwargs)
    original_read = scorer.read
    class MetadataOnlySplit(dict):
        def __getitem__(self, key):
            if key in ('support_labels', 'support_indices'):
                assert (out / 'prediction_complete.json').is_file(), 'Manifest truth was indexed while selecting model inputs'
            return super().__getitem__(key)
    def guarded_read(path):
        value = original_read(path)
        if Path(path).name == 'support_splits.json' and not (out / 'prediction_complete.json').exists():
            value['splits'] = [MetadataOnlySplit(split) for split in value['splits']]
        return value
    with np.load(production['cache'] / scorer.CACHE_NAME, allow_pickle=False) as archive: archive_class = type(archive)
    original_getitem = archive_class.__getitem__
    def guarded_getitem(archive, key):
        if key in ('labels', 'indices'):
            assert (out / 'prediction_complete.json').is_file(), 'Cache truth was read during model inference'
        return original_getitem(archive, key)
    original_score = GroundClassifierA.score; calls = []
    def single_record(head, *, z_id, feature_contract):
        assert tuple(z_id.shape) == (1, 160) and z_id.dtype == torch.float32
        calls.append(z_id.tolist()); return original_score(head, z_id=z_id, feature_contract=feature_contract)
    monkeypatch.setattr(Path, 'open', guarded_open); monkeypatch.setattr(archive_class, '__getitem__', guarded_getitem)
    monkeypatch.setattr(scorer, 'read', guarded_read)
    monkeypatch.setattr(GroundClassifierA, 'score', single_record)
    scorer.score_support(**_args(production, out), fit_trace=production['lane'] / 'fit_trace.jsonl')
    assert len(calls) == len(production['split']['support_ids'])


def test_truth_join_is_independent_of_packet_and_model(production, tmp_path, monkeypatch):
    fixed = _fixed(production, tmp_path / 'fixed')
    def forbidden(*args, **kwargs): raise AssertionError('Model/packet cannot run during truth join')
    monkeypatch.setattr(scorer, 'load_packet', forbidden); monkeypatch.setattr(GroundClassifierA, 'score', forbidden)
    result = scorer.score_fixed_support(predictions=fixed, fit_trace=production['lane'] / 'fit_trace.jsonl', support_splits=production['cache'] / 'support_splits.json')
    assert result['model_called_during_truth_join'] is False


@pytest.mark.parametrize('tamper', ['other_run', 'missing_old_id', 'state_other_run', 'missing_candidate', 'wrong_class_columns', 'nonfinite', 'missing_outer_archive', 'bad_prediction_status'])
def test_actual_b_c_pairing_rejects_tampered_trace(production, tmp_path, tamper):
    fixed = _fixed(production, tmp_path / 'fixed')
    schema = production['record']['schema']; candidate = scorer.METHODS[schema][3]
    def mutate(record):
        entry = record['folds'][0]
        if tamper == 'other_run': record['inheritance_binding']['run_id'] = 'another_run'
        elif tamper == 'missing_old_id': entry['b_ids'].pop()
        elif tamper == 'state_other_run':
            ref = entry['candidate_stages'][0]['final_state_ref']; namespace = json.loads(ref['namespace'])
            namespace['run_id'] = 'another_run'; ref['namespace'] = json.dumps(namespace)
        elif tamper == 'missing_candidate': entry['paths'].pop(candidate)
        elif tamper == 'wrong_class_columns': entry['b_classes'].reverse()
        elif tamper == 'missing_outer_archive': entry['outer_features_state_ref'] = None
        elif tamper == 'bad_prediction_status': entry['prediction_status'] = 'NOT_FIXED'
        else: entry['paths'][candidate]['c_scores'][0][0] = float('nan')
    trace = _copied_trace(production, tmp_path / 'trace', mutate)
    with pytest.raises(ValueError): scorer.score_fixed_support(predictions=fixed, fit_trace=trace, support_splits=production['cache'] / 'support_splits.json')


@pytest.mark.parametrize('method', ['conditional', 'affine'])
@pytest.mark.parametrize('new', [0, 1])
def test_true_k1_has_no_held_a_or_adaptation_gain(tmp_path, method, new):
    case = _case(tmp_path / 'case', method=method, k=1, new=new)
    result = scorer.score_support(**_args(case, tmp_path / 'supplement'), fit_trace=case['lane'] / 'fit_trace.jsonl')
    parent = result['parents'][0]
    assert result['k1_parent_count'] == 1 and parent['oof']['metrics'] == dict.fromkeys(scorer.METRICS)
    assert parent['proxy'] is None and parent['paired_paths'] == []
    assert all(row['measured_parent_count'] == 0 and row['mean'] is None for row in result['statistics']['by_k_new_count'])


def test_output_is_exclusive_before_packet_read(production, tmp_path, monkeypatch):
    out = tmp_path / 'owned'; out.mkdir()
    monkeypatch.setattr(scorer, 'load_packet', lambda *args: (_ for _ in ()).throw(AssertionError('Do not load packet')))
    with pytest.raises(FileExistsError): scorer.predict_support(**_args(production, out))


def test_packet_checkpoint_binding_cannot_be_substituted(production, tmp_path):
    args = _args(production, tmp_path / 'bad'); args['expected_checkpoint_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='binding'): scorer.predict_support(**args)
    assert not args['output'].exists()


@pytest.mark.parametrize('tamper', ['dtype', 'normalized_contract'])
def test_float64_or_normalized_contract_cache_is_rejected(production, tmp_path, tamper):
    cache = tmp_path / 'cache'; shutil.copytree(production['cache'], cache)
    with np.load(cache / scorer.CACHE_NAME, allow_pickle=False) as original: values = {name: original[name] for name in original.files}
    if tamper == 'dtype': values['z_id'] = values['z_id'].astype(np.float64)
    np.savez(cache / scorer.CACHE_NAME, **values)
    marker = scorer.read(cache / 'features_complete.json'); marker['feature_file_bytes'] = (cache / scorer.CACHE_NAME).stat().st_size
    if tamper == 'normalized_contract':
        marker['feature_contract'] = dict(marker['feature_contract'], normalization='unit_normalized')
    (cache / 'features_complete.json').write_text(json.dumps(marker), encoding='utf-8')
    args = _args(production, tmp_path / 'bad'); args['support_features'] = cache
    with pytest.raises(ValueError, match='float32|contract'): scorer.predict_support(**args)


def test_truth_mismatch_preserves_fixed_predictions_and_does_not_overwrite(production, tmp_path, monkeypatch):
    cache = tmp_path / 'cache'; shutil.copytree(production['cache'], cache)
    trace = _copied_trace(production, tmp_path / 'trace', lambda record: None)
    for name in ('startup.json', 'probe_complete.json'):
        metadata = scorer.read(trace.parent / name); metadata['support_features'] = str(cache.resolve())
        (trace.parent / name).write_text(json.dumps(metadata), encoding='utf-8')
    args = _args(production, tmp_path / 'supplement'); args['support_features'] = cache
    original_join = scorer.score_fixed_support
    def mismatched_after_prediction(**kwargs):
        assert (args['output'] / 'prediction_complete.json').is_file()
        plan = scorer.read(cache / 'support_splits.json'); plan['splits'][0]['support_labels'][0] = 1
        (cache / 'support_splits.json').write_text(json.dumps(plan), encoding='utf-8')
        return original_join(**kwargs)
    monkeypatch.setattr(scorer, 'score_fixed_support', mismatched_after_prediction)
    with pytest.raises(ValueError): scorer.score_support(**args, fit_trace=trace)
    assert (args['output'] / 'prediction_complete.json').is_file() and (args['output'] / 'scoring_failed.json').is_file()
    assert not (args['output'] / 'complete.json').exists()
    scorer.load_fixed_predictions(args['output'])
    with pytest.raises(FileExistsError): scorer.score_support(**args, fit_trace=trace)


def test_missing_selected_physical_id_is_rejected_before_prediction(production, tmp_path, monkeypatch):
    cache = tmp_path / 'cache'; shutil.copytree(production['cache'], cache)
    plan = scorer.read(cache / 'support_splits.json'); plan['splits'][0]['support_ids'][0] = 'missing_physical'
    (cache / 'support_splits.json').write_text(json.dumps(plan), encoding='utf-8')
    args = _args(production, tmp_path / 'no_prediction'); args['support_features'] = cache
    monkeypatch.setattr(GroundClassifierA, 'score', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Missing ID reached the model')))
    with pytest.raises(ValueError, match='missing'): scorer.predict_support(**args)
    assert not args['output'].exists()


def test_partial_trace_and_missing_prediction_ids_are_rejected(production, tmp_path):
    fixed = _fixed(production, tmp_path / 'fixed')
    trace = _copied_trace(production, tmp_path / 'trace', lambda record: None)
    trace.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='Partial'): scorer.score_fixed_support(predictions=fixed, fit_trace=trace, support_splits=production['cache'] / 'support_splits.json')
    with np.load(fixed / scorer.PREDICTION_FILE, allow_pickle=False) as saved: values = {name: saved[name] for name in saved.files}
    values['physical_ids'][0] = 'wrong_physical'; np.savez(fixed / scorer.PREDICTION_FILE, **values)
    marker = scorer.read(fixed / 'prediction_complete.json'); marker['prediction_file_bytes'] = (fixed / scorer.PREDICTION_FILE).stat().st_size
    (fixed / 'prediction_complete.json').write_text(json.dumps(marker), encoding='utf-8')
    with pytest.raises(ValueError): scorer.load_fixed_predictions(fixed)


def test_predictor_has_no_truth_role_quota_or_trace_arguments(production, tmp_path):
    for name in ('truth', 'role', 'quota', 'fit_trace'):
        with pytest.raises(TypeError): scorer.predict_support(**_args(production, tmp_path / name), **{name: []})


def test_selection_is_mandatory_and_output_cannot_change_original_inputs(production):
    args = _args(production, production['cache'] / 'supplement'); args.pop('selection')
    with pytest.raises(TypeError): scorer.predict_support(**args)
    args['selection'] = scorer.read(production['lane'] / 'startup.json')['config']['selection']
    with pytest.raises(ValueError, match='outside'): scorer.predict_support(**args)
    args['output'] = production['lane'] / 'supplement'
    with pytest.raises(ValueError, match='outside'): scorer.score_support(**args, fit_trace=production['lane'] / 'fit_trace.jsonl')


def test_extra_cache_records_and_unselected_splits_never_enter_model_or_statistics(production, tmp_path, monkeypatch):
    cache = tmp_path / 'cache'; shutil.copytree(production['cache'], cache)
    extra_ids = ['zz_extra_' + str(i) for i in range(6)]
    with np.load(cache / scorer.CACHE_NAME, allow_pickle=False) as original: values = {name: original[name] for name in original.files}
    n = len(values['ids'])
    for name in ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local'):
        values[name] = np.concatenate((values[name], np.full((6, values[name].shape[1]), 999., dtype=np.float32)))
    values['ids'] = np.asarray(values['ids'].astype(str).tolist() + extra_ids)
    values['indices'] = np.arange(n + 6, dtype=np.int64)
    values['labels'] = np.asarray(values['labels'].astype(str).tolist() + production['old'])
    np.savez(cache / scorer.CACHE_NAME, **values)
    marker = scorer.read(cache / 'features_complete.json'); marker.update(count=n + 6,
        feature_file_bytes=(cache / scorer.CACHE_NAME).stat().st_size,
        shapes={name: list(values[name].shape) for name in ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')})
    (cache / 'features_complete.json').write_text(json.dumps(marker), encoding='utf-8')
    plan = scorer.read(cache / 'support_splits.json')
    plan['splits'].append(dict(split_id='not_selected', receiver='other', scenario='other', k=1, support_seed=91,
        registered_classes=production['old'], support_ids=extra_ids, support_indices=list(range(n, n + 6)), support_labels=list(range(6))))
    (cache / 'support_splits.json').write_text(json.dumps(plan), encoding='utf-8')
    lane = tmp_path / 'lane'; lane.mkdir()
    shutil.copyfile(production['lane'] / 'fit_trace.jsonl', lane / 'fit_trace.jsonl')
    for name in ('startup.json', 'probe_complete.json'):
        metadata = scorer.read(production['lane'] / name); metadata['support_features'] = str(cache.resolve()); scorer.write(lane / name, metadata)
    original_score = GroundClassifierA.score; calls = []
    def selected_only(head, *, z_id, feature_contract):
        assert not bool((z_id == 999.).any()), 'Unauthorized extra cache feature reached the model'
        calls.append(z_id.tolist()); return original_score(head, z_id=z_id, feature_contract=feature_contract)
    monkeypatch.setattr(GroundClassifierA, 'score', selected_only)
    args = _args(production, tmp_path / 'supplement'); args['support_features'] = cache
    result = scorer.score_support(**args, fit_trace=lane / 'fit_trace.jsonl')
    assert len(calls) == n and result['ground_prediction_record_count'] == n and result['parent_count'] == 1
    fixed = scorer.load_fixed_predictions(args['output'])
    assert not set(extra_ids) & set(fixed['physical_ids']) and fixed['physical_ids'] == production['split']['support_ids']


def test_parent_h_gap_and_gain_are_averaged_after_parent_calculation():
    metrics = []
    for old, new in ((.2, 1.), (1., .2)):
        metrics.append(dict.fromkeys(scorer.METRICS) | dict(A_old_accuracy=.1, B_old_accuracy=.8, C_old_accuracy=old,
            C_new_accuracy=new, C_h=2 * old * new / (old + new), adaptation_gain_B_minus_A=.7,
            total_old_accuracy_drop=.8 - old, C_abs_new_old_gap=abs(old - new)))
    parents = [dict(path='actual', k=2, new_count=1, receiver='rx', scenario='scene', oof=dict(metrics=value), proxy=None) for value in metrics]
    rows = scorer._statistics(parents, dict(model_seed=17, row_id='row'))['by_k_new_count']
    lookup = {(row['diagnostic'], row['metric']): row for row in rows}
    assert lookup['oof', 'C_h']['mean'] == pytest.approx(1 / 3)
    assert lookup['oof', 'C_h']['mean'] != pytest.approx(.6)
    assert lookup['oof', 'C_abs_new_old_gap']['mean'] == pytest.approx(.8)
    assert lookup['oof', 'adaptation_gain_B_minus_A']['mean'] == pytest.approx(.7)
