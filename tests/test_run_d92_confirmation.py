import json
from pathlib import Path
import sys
import threading
import shutil

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_d92_confirmation as runner


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def specification(tmp_path):
    root, data = tmp_path / 'run', tmp_path / 'data'
    data_cfg = tmp_path / 'data.json'
    save(data_cfg, dict(output_root=str(data)))
    spec = dict(execution=dict(remote_run_root=str(root), launch_owner='synthetic-owner'),
        confirmation=dict(capsule=str(data / 'capsule'), truth=str(data / 'score_only' / 'truth.json'),
            data_config=str(data_cfg), candidate_config=str(tmp_path / 'candidate.json'),
            native_code=str(tmp_path / 'native'), source_contract=str(tmp_path / 'source_contract.json')),
        rows=[dict(row_id=f'seed-{seed}', output_root=str(root / f'seed-{seed}'), seeds=dict(model=seed),
            source_root=str(tmp_path / f'source-{seed}'), ground_source=str(tmp_path / f'ground-{seed}'))
            for seed in runner.MODEL_SEEDS])
    path = tmp_path / 'spec.json'
    save(path, spec)
    return path, spec


def install_workers(monkeypatch, spec, fail=None):
    calls, lock = [], threading.Lock()
    root = Path(spec['execution']['remote_run_root'])
    candidate = runner.candidate_definition(spec['confirmation'])
    splits = spec['confirmation'].get('expected_split_count', 900)
    def mock_invoke(tool, arguments, log, *, gpu=False):
        args = list(map(str, arguments))
        def arg(name):
            return args[args.index('--' + name) + 1]
        with lock:
            calls.append((tool, args, gpu))
        if fail and tool == fail[0] and fail[1] in str(log):
            raise RuntimeError('synthetic failure')
        if tool == 'build_d92_confirmation_data.py':
            save(Path(spec['confirmation']['capsule']) / 'manifest.json', dict(protocol_schema='p2_min_v1',
                phase2_data_status='VALIDATED_ONCE', capsule_id='synthetic', split_count=splits))
        elif tool == 'cvs_native_artifacts.py':
            assert gpu is True
            save(Path(arg('output')) / 'features_complete.json', dict(status='FROZEN_FEATURES_COMPLETE',
                capsule_id='synthetic', query_used_for_fitting=False))
        elif tool in ('cvs_d92_matched.py', candidate['candidate_predictor']):
            assert gpu is False
            assert len([c for c in calls if c[0] == 'cvs_native_artifacts.py']) == 4
            output = Path(arg('output'))
            if tool == candidate['candidate_predictor']:
                assert output.name == candidate['candidate_folder']
                assert (output.parent / 'predictions_complete.json').exists()
            save(output / 'predictions_complete.json', dict(status='PREDICTIONS_COMPLETE',
                capsule_id='synthetic', split_count=splits, truth_read=False))
        elif tool == 'score_d92_confirmation.py':
            assert all(s['status'] == 'PREDICTIONS_COMPLETE' for s in runner.read(root / 'state.json').values())
            assert len([c for c in calls if c[0] == candidate['candidate_predictor']]) == 4
            # Deliberately no truth file exists: orchestration must never open it.
            assert not Path(spec['confirmation']['truth']).exists()
            save(arg('output'), dict(status='SCORED', results=[{}], selection_feedback_forbidden=True))
        else:
            raise AssertionError(tool)
    monkeypatch.setattr(runner, 'invoke', mock_invoke)
    monkeypatch.setattr(runner, 'prepare_ground', lambda row, confirmation: 'synthetic-checkpoint-sha')
    return calls


def test_all_models_freeze_before_independent_score(tmp_path, monkeypatch):
    path, spec = specification(tmp_path)
    calls = install_workers(monkeypatch, spec)
    runner.run(path, 'synthetic-commit')
    assert calls[0][0] == 'build_d92_confirmation_data.py'
    assert [c[0] for c in calls[1:5]] == ['cvs_native_artifacts.py'] * 4
    assert calls[-1][0] == 'score_d92_confirmation.py'
    root = Path(spec['execution']['remote_run_root'])
    assert runner.read(root / 'complete.json')['status'] == 'SCORED'
    startup = runner.read(root / 'startup.json')
    assert startup['spec'] == spec and startup['commit'] == 'synthetic-commit'
    assert startup['worker_policy']['max_feature_workers'] == 1


@pytest.mark.parametrize('reuse', [False, True])
@pytest.mark.parametrize('candidate_index', [1, 2])
def test_trained_candidate_with_300_splits_completes_every_row_before_score(tmp_path, monkeypatch, reuse, candidate_index):
    path, spec = specification(tmp_path)
    candidate=runner.CANDIDATES[candidate_index]
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS, candidate)))
    spec['confirmation']['expected_split_count'] = 300
    if reuse:
        spec['confirmation']['reuse_validated_capsule_id'] = 'synthetic'
        save(Path(spec['confirmation']['capsule']) / 'manifest.json', dict(protocol_schema='p2_min_v1',
            phase2_data_status='VALIDATED_ONCE', capsule_id='synthetic', split_count=300))
    save(path, spec)
    calls = install_workers(monkeypatch, spec)
    runner.run(path, 'synthetic-commit')
    assert len([c for c in calls if c[0] == candidate[2]]) == 4
    assert not any(c[0] == 'predict_d92_support_cv.py' for c in calls)
    assert calls[-1][0] == 'score_d92_confirmation.py'
    assert any(c[0] == 'build_d92_confirmation_data.py' for c in calls) is (not reuse)
    root = Path(spec['execution']['remote_run_root'])
    assert all(r['status'] == 'PREDICTIONS_COMPLETE' for r in runner.read(root / 'state.json').values())


@pytest.mark.parametrize('candidate_index', [1, 2])
def test_trained_candidate_failure_blocks_score_while_healthy_rows_finish(tmp_path, monkeypatch, candidate_index):
    path, spec = specification(tmp_path)
    candidate=runner.CANDIDATES[candidate_index]
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS, candidate)))
    spec['confirmation']['expected_split_count'] = 300
    save(path, spec)
    calls = install_workers(monkeypatch, spec, fail=(candidate[2], '2026092701'))
    with pytest.raises(RuntimeError, match='synthetic failure'):
        runner.run(path, 'synthetic-commit')
    assert not any(c[0] == 'score_d92_confirmation.py' for c in calls)
    root = Path(spec['execution']['remote_run_root'])
    state = runner.read(root / 'state.json')
    assert state['seed-2026092701']['status'] == 'TECHNICAL_FAILURE'
    assert all(state[f'seed-{seed}']['status'] == 'PREDICTIONS_COMPLETE' for seed in runner.MODEL_SEEDS[1:])


@pytest.mark.parametrize('key,value', [
    ('candidate_method', 'arbitrary'), ('candidate_folder', '../escape'),
    ('candidate_predictor', '../other.py'), ('candidate_mode', 'd92_scv_registration'),
])
def test_illegal_candidate_combination_rejected_before_run_creation(tmp_path, key, value):
    path, spec = specification(tmp_path)
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS, runner.CANDIDATES[1])))
    spec['confirmation'][key] = value
    save(path, spec)
    with pytest.raises(ValueError, match='combination is not allowed'):
        runner.run(path, 'synthetic-commit')
    assert not Path(spec['execution']['remote_run_root']).exists()


def test_fresh_capsule_split_count_must_match_preregistered_count(tmp_path, monkeypatch):
    path, spec = specification(tmp_path)
    spec['confirmation']['expected_split_count'] = 300
    save(path, spec)
    calls = install_workers(monkeypatch, spec)
    original = runner.invoke
    def wrong_count(tool, arguments, log, **kwargs):
        original(tool, arguments, log, **kwargs)
        if tool == 'build_d92_confirmation_data.py':
            manifest_path = Path(spec['confirmation']['capsule']) / 'manifest.json'
            manifest = runner.read(manifest_path)
            manifest['split_count'] = 900
            save(manifest_path, manifest)
    monkeypatch.setattr(runner, 'invoke', wrong_count)
    with pytest.raises(ValueError, match='expected_split_count'):
        runner.run(path, 'synthetic-commit')
    assert [c[0] for c in calls] == ['build_d92_confirmation_data.py']


@pytest.mark.parametrize('candidate_index', [0, 1, 2, 3, 4, 5, 6, 7])
def test_candidate_release_paths_and_readback_use_exact_registered_folder(tmp_path,capfd,candidate_index):
    from collect_d92_fit_logs import fit_log_paths
    from publish_d92_confirmation import release_tool_paths
    from read_d92_run import readback_script
    candidate=runner.CANDIDATES[candidate_index]
    confirmation=dict(zip(runner.CANDIDATE_FIELDS,candidate))
    assert fit_log_paths(confirmation)==[('compact.jsonl',candidate[1]+'/compact.jsonl'),('d92.jsonl','d92.log')]
    paths=release_tool_paths(confirmation)
    assert 'code' in paths and 'tools/'+candidate[2] in paths
    assert ('tools/predict_d92_summary_joint.py' in paths) is (candidate_index==2)
    root,release=tmp_path/'run',tmp_path/'release'
    release.mkdir()
    marker=root/'row'/candidate[1]/'predictions_complete.json'
    save(marker,dict(status='PREDICTIONS_COMPLETE'))
    log=root/'row'/(candidate[1]+'.log')
    log.write_text(json.dumps(dict(split_id='synthetic',steps=[dict(loss=1)],optimizer_status='done'))+'\n',encoding='utf-8')
    script=readback_script(dict(confirmation=confirmation,execution=dict(remote_run_root=str(root)),code=dict(cwd=str(release))))
    # Execute read-only artifact inspection on synthetic local paths; no SSH or scores.
    exec(compile(script,'synthetic_readback','exec'),{})
    data=json.loads(capfd.readouterr().out)
    assert data[str(marker)]['status']=='PREDICTIONS_COMPLETE'
    if candidate_index:
        assert 'steps' not in json.loads(data[str(log)]['tail'][0])


def test_sgjoint_rejects_sfhead_mode_in_release_and_readback(tmp_path):
    from collect_d92_fit_logs import fit_log_paths
    from publish_d92_confirmation import release_tool_paths
    from read_d92_run import readback_script
    confirmation=dict(zip(runner.CANDIDATE_FIELDS,runner.CANDIDATES[2]))
    confirmation['candidate_mode']='d92_sfhead_registration'
    with pytest.raises(ValueError,match='combination'):
        release_tool_paths(confirmation)
    with pytest.raises(ValueError,match='combination'):
        readback_script(dict(confirmation=confirmation))
    with pytest.raises(ValueError,match='combination'):
        fit_log_paths(confirmation)


def reused_specification(tmp_path):
    from test_d92_confirmation_score import fixture as scoring_fixture
    from test_d92_ground_summary import fixture_files
    sys.path.insert(0,str(runner.RELEASE/'code'))
    from cvsrffi import phase1_center_lowrank_prototype_bundle as codec
    path,spec=specification(tmp_path)
    old_spec,old_row,_=scoring_fixture(tmp_path/'capsule-test')
    data_root=Path(old_spec['confirmation']['capsule']).parent
    cap_manifest=runner.read(data_root/'capsule/manifest.json')
    cap_manifest.update(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE')
    save(data_root/'capsule/manifest.json',cap_manifest)
    truth=data_root/'score_only/truth.json';truth.parent.mkdir()
    shutil.copyfile(old_spec['confirmation']['truth'],truth)
    save(Path(spec['confirmation']['data_config']),dict(output_root=str(data_root)))
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS,runner.CANDIDATES[2])))
    spec['confirmation'].update(capsule=old_spec['confirmation']['capsule'],truth=str(truth),
        reuse_validated_capsule_id='synthetic',expected_split_count=2,
        old_classes=[f'class-{i}' for i in range(6)])
    spec['permissions']=dict(claim_scope='Repeated synthetic benchmark; not independent confirmation')
    ids=np.asarray([f'physical-{i}' for i in range(16)])
    for row in spec['rows']:
        origin=tmp_path/'old-run'/row['row_id'];origin.mkdir(parents=True)
        row.update(reuse_row_root=str(origin),expected_checkpoint_sha256='a'*64)
        for name in ('predictions.jsonl','predictions_complete.json'):
            shutil.copyfile(old_row/name,origin/name)
        complete=runner.read(origin/'predictions_complete.json');complete['truth_read']=False
        save(origin/'predictions_complete.json',complete)
        ground=origin/'ground';ground.mkdir()
        payload,manifest=fixture_files(ground)
        for key in ('core_q','core_scale','residual_basis_q','residual_basis_scale','radius_scale'):
            payload[key]=np.repeat(payload[key][:1],6,axis=0)
        for key in ('residual_coeff_q','residual_coeff_scale','radius_q'):
            payload[key]=np.repeat(payload[key][:,:1],6,axis=1)
        payload['class_registry']=np.asarray(spec['confirmation']['old_classes'])
        manifest['resource_audit']=dict(codec._numeric_resource_audit(payload),reconstruction_rmse=0.001)
        np.savez(ground/codec.NPZ_NAME,**payload);save(ground/'manifest.json',manifest)
        features=origin/'received_features';features.mkdir()
        np.savez(features/'received_features.npz',identity160=np.ones((16,160)),logits=np.ones((16,6)),ids=ids)
        save(features/'features_complete.json',dict(status='FROZEN_FEATURES_COMPLETE',capsule_id='synthetic',count=16,query_used_for_fitting=False))
        source=tmp_path/'DO_NOT_OPEN_SOURCE_OR_CHECKPOINT'/row['row_id']
        save(features/'checkpoint_provenance.json',dict(verdict='MATCHED_SOURCE_ONLY_SCRATCH',checkpoint=str(source/'final_ssdg.pth'),
            checkpoint_epoch=200,initialization='scratch',checkpoint_inheritance=[],target_access_before_freeze=False,
            source_role_comparison='EXACT_MATCH',classes=spec['confirmation']['old_classes'],model_seed=row['seeds']['model'],selection='fixed_final_epoch200'))
        save(features/'startup.json',dict(source_arguments=dict(seed=row['seeds']['model'])))
        save(origin/'ground_provenance.json',dict(source_root=str(source),checkpoint_sha256='a'*64))
        save(origin/'d92_startup.json',dict(checkpoint_sha256='a'*64,seed=row['seeds']['model'],
            capsule=spec['confirmation']['capsule'],features=str(features/'received_features.npz'),ground=str(ground),
            query_fit_access=False,truth_read=False))
    save(path,spec)
    return path,spec


def install_orbit_cache_reuse(tmp_path,spec):
    source=tmp_path/'frozen-orbit-origin'
    spec['confirmation']['frozen_feature_source_root']=str(source)
    for row in spec['rows']:
        cache=source/row['row_id']/'bnna_features'
        row['reuse_multiview_features_root']=str(cache)
        save(cache/'features_complete.json',dict(status='BNNA_FEATURES_COMPLETE',capsule_id='synthetic',
            checkpoint_sha256='a'*64,model_seed=row['seeds']['model'],query_used_for_fitting=False,encoder_updated=False))


@pytest.mark.parametrize('candidate_index',[2,3,4,5,6,7])
def test_reused_rows_only_run_candidate_and_score_with_old_artifacts_unchanged(tmp_path,monkeypatch,candidate_index):
    from score_d92_confirmation import score
    path,spec=reused_specification(tmp_path)
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS,runner.CANDIDATES[candidate_index])))
    if candidate_index in (5,6): install_orbit_cache_reuse(tmp_path,spec)
    save(path,spec)
    multiview=runner.needs_multiview(spec['confirmation'])
    feature=runner.feature_definition(spec['confirmation'])
    method=runner.candidate_definition(spec['confirmation'])
    old_files=list((tmp_path/'old-run').rglob('*'))
    if candidate_index in (5,6):old_files+=list((tmp_path/'frozen-orbit-origin').rglob('*'))
    old_hashes={p:runner.sha(p) for p in old_files if p.is_file()}
    calls=[]
    def invoke(tool,arguments,log,*,gpu=False):
        args=list(map(str,arguments));calls.append(tool)
        if feature is not None and tool==feature[0]:
            assert multiview and gpu
            assert method['candidate_predictor'] not in calls
            output=Path(args[args.index('--output')+1])
            save(output/'features_complete.json',dict(status=feature[3],capsule_id='synthetic',
                checkpoint_sha256='a'*64,query_used_for_fitting=False))
        elif tool==method['candidate_predictor']:
            assert not gpu
            if candidate_index in (5,6):
                cache=Path(args[args.index('--orbit-features')+1])
                assert cache.parent.parent==tmp_path/'frozen-orbit-origin'
                assert not any('export_d92' in call for call in calls)
            if multiview:
                assert calls.count(feature[0])==4
                assert Path(args[args.index(feature[2])+1]).name==feature[1]
            origin=Path(args[args.index('--row-root')+1]);output=Path(args[args.index('--output')+1])
            assert origin.parent==tmp_path/'old-run'
            assert output.parent.parent==Path(spec['execution']['remote_run_root'])
            entries=[json.loads(line) for line in (origin/'predictions.jsonl').read_text().splitlines()]
            candidate=[dict(r,mode=method['candidate_mode']) for r in entries if r['mode']=='d92_registration']
            output.mkdir()
            (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in candidate),encoding='utf-8')
            save(output/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='synthetic',split_count=2,predictions=2,truth_read=False))
        elif tool=='score_d92_confirmation.py':
            root=Path(spec['execution']['remote_run_root'])
            assert all(r['status']=='PREDICTIONS_COMPLETE' for r in runner.read(root/'state.json').values())
            score(spec,args[args.index('--output')+1])
        else:
            raise AssertionError('No builder, encoder or D92 inference allowed: '+tool)
    monkeypatch.setattr(runner,'invoke',invoke)
    monkeypatch.setattr(runner,'prepare_ground',lambda *_:pytest.fail('Must reuse existing ground'))
    runner.run(path,'synthetic-commit')
    assert calls.count(method['candidate_predictor'])==4 and calls[-1]=='score_d92_confirmation.py'
    assert {p:runner.sha(p) for p in old_hashes}==old_hashes
    root=Path(spec['execution']['remote_run_root'])
    assert len(runner.read(root/'scores.json')['results'])==20
    assert runner.read(root/'scores.json')['claim_scope']==spec['permissions']['claim_scope']
    policy=runner.read(root/'startup.json')['worker_policy']
    assert policy['feature_gpu']==(0 if multiview else None) and policy['max_feature_workers']==int(multiview)
    for row in spec['rows']:
        output=Path(row['output_root'])
        assert not (output/'ground').exists() and not (output/'received_features').exists()
        assert not (output/'predictions.jsonl').exists()
        evidence=runner.read(output/'artifact_reuse.json')
        assert evidence['old_scores_read'] is False and evidence['baseline_executed'] is False
        if candidate_index in (5,6):
            cached=runner.read(output/'frozen_feature_reuse.json')
            assert cached['cache_recomputed'] is False and cached['checkpoint_reloaded'] is False
            assert cached['adapted_state_reused'] is False


@pytest.mark.parametrize('fault',['missing','mixed_source','duplicate','overlap','wrong_candidate'])
def test_orbit_cache_reuse_requires_complete_explicit_model_binding(tmp_path,fault):
    path,spec=reused_specification(tmp_path)
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS,runner.CANDIDATES[5])))
    install_orbit_cache_reuse(tmp_path,spec)
    if fault=='missing':spec['rows'][0].pop('reuse_multiview_features_root')
    if fault=='mixed_source':spec['rows'][0]['reuse_multiview_features_root']=str(tmp_path/'other'/'bnna_features')
    if fault=='duplicate':spec['rows'][0]['reuse_multiview_features_root']=spec['rows'][1]['reuse_multiview_features_root']
    if fault=='overlap':spec['confirmation']['frozen_feature_source_root']=spec['execution']['remote_run_root']
    if fault=='wrong_candidate':spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS,runner.CANDIDATES[2])))
    save(path,spec)
    with pytest.raises(ValueError,match='cache|OSC|overlap'):runner.run(path,'synthetic-commit')
    assert not Path(spec['execution']['remote_run_root']).exists()


def test_orbit_bad_cache_metadata_blocks_prediction_and_score(tmp_path,monkeypatch):
    path,spec=reused_specification(tmp_path)
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS,runner.CANDIDATES[5])))
    install_orbit_cache_reuse(tmp_path,spec);save(path,spec)
    marker=Path(spec['rows'][0]['reuse_multiview_features_root'])/'features_complete.json'
    data=runner.read(marker);data['encoder_updated']=True;save(marker,data)
    monkeypatch.setattr(runner,'invoke',lambda *_args,**_kwargs:pytest.fail('No prediction or scoring on incompatible cache'))
    with pytest.raises(ValueError,match='cache metadata'):runner.run(path,'synthetic-commit')


@pytest.mark.parametrize('fault',['partial','mixed_parent','duplicate_origin','bad_sha','wrong_seed',
                                  'wrong_capsule','wrong_features','wrong_classes','incomplete_baseline','missing_baseline_record'])
def test_reuse_binding_failures_prevent_any_child_launch(tmp_path,monkeypatch,fault):
    path,spec=reused_specification(tmp_path)
    row=spec['rows'][1];origin=Path(row['reuse_row_root'])
    if fault=='partial':
        del row['reuse_row_root'];del row['expected_checkpoint_sha256']
    elif fault=='mixed_parent':row['reuse_row_root']=str(tmp_path/'different-parent'/row['row_id'])
    elif fault=='duplicate_origin':row['reuse_row_root']=spec['rows'][0]['reuse_row_root']
    elif fault=='bad_sha':row['expected_checkpoint_sha256']='c'*64
    elif fault in ('wrong_seed','wrong_capsule','wrong_features'):
        document=runner.read(origin/'d92_startup.json')
        document[{'wrong_seed':'seed','wrong_capsule':'capsule','wrong_features':'features'}[fault]]='wrong'
        save(origin/'d92_startup.json',document)
    elif fault=='wrong_classes':
        document=runner.read(origin/'received_features/checkpoint_provenance.json');document['classes']=list(reversed(document['classes']))
        save(origin/'received_features/checkpoint_provenance.json',document)
    elif fault=='incomplete_baseline':
        document=runner.read(origin/'predictions_complete.json');document['status']='RUNNING';save(origin/'predictions_complete.json',document)
    else:
        lines=(origin/'predictions.jsonl').read_text().splitlines()
        (origin/'predictions.jsonl').write_text('\n'.join(lines[1:])+'\n',encoding='utf-8')
    save(path,spec)
    monkeypatch.setattr(runner,'invoke',lambda *_args,**_kwargs:pytest.fail('No child launch for invalid reused inputs'))
    with pytest.raises(ValueError):runner.run(path,'synthetic-commit')
    assert not (Path(spec['execution']['remote_run_root'])/'complete.json').exists()


def test_publisher_cpu_reuse_never_queries_or_waits_for_gpu(monkeypatch):
    import ast
    import subprocess
    from publish_d92_confirmation import REMOTE
    tree=ast.parse(REMOTE)
    gpu_check=next(node for node in tree.body if isinstance(node,ast.If) and 'reuse_frozen_rows' in ast.unparse(node.test))
    code=compile(ast.fix_missing_locations(ast.Module(body=[gpu_check],type_ignores=[])),'isolated_gpu_preflight','exec')
    calls=[]
    def occupied(*args,**kwargs):calls.append(args);return '20000'
    monkeypatch.setattr(subprocess,'check_output',occupied)
    exec(code,dict(c=dict(reuse_frozen_rows=True),subprocess=subprocess))
    assert calls==[]
    with pytest.raises(RuntimeError,match='occupied'):
        exec(code,dict(c=dict(reuse_frozen_rows=False),subprocess=subprocess))
    assert len(calls)==1
    with pytest.raises(RuntimeError,match='occupied'):
        exec(code,dict(c=dict(reuse_frozen_rows=True,multiview=True),subprocess=subprocess))
    assert len(calls)==2


@pytest.mark.parametrize('candidate_index',[3,4,7])
def test_multiview_release_includes_exporter_and_requires_reused_baseline(tmp_path,candidate_index):
    from publish_d92_confirmation import release_tool_paths
    path,spec=specification(tmp_path)
    spec['confirmation'].update(dict(zip(runner.CANDIDATE_FIELDS,runner.CANDIDATES[candidate_index])))
    save(path,spec)
    assert 'tools/export_d92_mv_kme_features.py' in release_tool_paths(spec['confirmation'])
    assert 'tools/'+runner.feature_definition(spec['confirmation'])[0] in release_tool_paths(spec['confirmation'])
    with pytest.raises(ValueError,match='reused baseline'):
        runner.run(path,'synthetic-commit')
    assert not Path(spec['execution']['remote_run_root']).exists()


@pytest.mark.parametrize('tool', ['cvs_d92_matched.py', 'predict_d92_support_cv.py'])
def test_prediction_failure_preserves_other_rows_without_scoring(tmp_path, monkeypatch, tool):
    path, spec = specification(tmp_path)
    calls = install_workers(monkeypatch, spec, fail=(tool, '2026092701'))
    with pytest.raises(RuntimeError, match='synthetic failure'):
        runner.run(path, 'synthetic-commit')
    assert not any(c[0] == 'score_d92_confirmation.py' for c in calls)
    root = Path(spec['execution']['remote_run_root'])
    state = runner.read(root / 'state.json')
    assert state['seed-2026092701']['status'] == 'TECHNICAL_FAILURE'
    assert all(state[f'seed-{seed}']['status'] == 'PREDICTIONS_COMPLETE' for seed in runner.MODEL_SEEDS[1:])
    assert runner.read(root / 'workflow_state.json')['status'] == 'TECHNICAL_FAILURE'
    assert not (root / 'complete.json').exists()


def test_feature_failure_never_launches_prediction_or_score(tmp_path, monkeypatch):
    path, spec = specification(tmp_path)
    calls = install_workers(monkeypatch, spec, fail=('cvs_native_artifacts.py', '2026092702'))
    with pytest.raises(RuntimeError, match='synthetic failure'):
        runner.run(path, 'synthetic-commit')
    assert [c[0] for c in calls] == ['build_d92_confirmation_data.py', 'cvs_native_artifacts.py', 'cvs_native_artifacts.py']
    root = Path(spec['execution']['remote_run_root'])
    assert runner.read(root / 'workflow_state.json')['status'] == 'TECHNICAL_FAILURE'
    assert runner.read(root / 'state.json')['seed-2026092702']['status'] == 'TECHNICAL_FAILURE'


def test_existing_run_not_overwritten(tmp_path, monkeypatch):
    path, spec = specification(tmp_path)
    root = Path(spec['execution']['remote_run_root'])
    root.mkdir()
    save(root / 'state.json', dict(original=True))
    calls = install_workers(monkeypatch, spec)
    with pytest.raises(FileExistsError):
        runner.run(path, 'synthetic-commit')
    assert calls == [] and runner.read(root / 'state.json') == dict(original=True)


def test_workers_have_bounded_cpu_and_gpu_visibility():
    cpu, gpu = runner.worker_environment(), runner.worker_environment(gpu=True)
    assert cpu['CUDA_VISIBLE_DEVICES'] == '' and gpu['CUDA_VISIBLE_DEVICES'] == '0'
    for env in (cpu, gpu):
        assert all(env[name] == '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))


@pytest.mark.parametrize('fault', [None, 'checkpoint_sha', 'source_classes', 'ground_classes'])
def test_ground_copy_bound_to_actual_source_checkpoint(tmp_path, fault):
    sys.path.insert(0, str(runner.RELEASE / 'code'))
    from cvsrffi.phase1_center_lowrank_prototype_bundle import NPZ_NAME
    source, origin, output = tmp_path / 'source', tmp_path / 'original-ground', tmp_path / 'row'
    classes = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
    # The reference source contract has no classes or native_role_comparison.
    contract = dict(role_ids=dict(L_s=['source-id']),
        source_rxs=['source-rx'], source_days=[1], ratios=[0.1], split_seed=42,
        num_classes=6)
    actual_classes = list(reversed(classes)) if fault == 'source_classes' else classes
    save(source / 'source_contract.json', dict(contract, native_role_comparison='EXACT_MATCH', classes=actual_classes))
    save(tmp_path / 'contract.json', contract)
    save(source / 'initialization.json', dict(scratch_only=True, checkpoint_sources=[],
        target_contact=False, source_roles='EXACT_MATCH'))
    save(source / 'completion.json', dict(status='TRAINING_COMPLETE', epochs=200,
        target_evaluated=False))
    save(source / 'resolved_config.json', dict(seed=2026092701, checkpoint_selection='final_only',
        a1_periodic_target_start=0, a1_final_weak_reference=False))
    (source / 'final_ssdg.pth').write_bytes(b'synthetic-checkpoint')
    expected = runner.sha(source / 'final_ssdg.pth')
    save(origin / 'manifest.json', dict(checkpoint_sha256='wrong' if fault == 'checkpoint_sha' else expected,
        target_access=False, source_role='L_s', formal_phase2_eligible=True,
        provenance_status='CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L', member_allowlist=[NPZ_NAME]))
    np.savez(origin / NPZ_NAME, class_registry=list(reversed(classes)) if fault == 'ground_classes' else classes)
    (origin / 'do_not_copy_source_sample_cache.npz').write_bytes(b'cache')
    output.mkdir()
    row = dict(source_root=str(source), ground_source=str(origin), output_root=str(output),
               seeds=dict(model=2026092701))
    confirmation = dict(source_contract=str(tmp_path / 'contract.json'), old_classes=classes)
    if fault is not None:
        with pytest.raises(ValueError, match='class registry mismatch|Ground source/checkpoint'):
            runner.prepare_ground(row, confirmation)
        assert not (output / 'ground').exists()
    else:
        assert runner.prepare_ground(row, confirmation) == expected
        assert {p.name for p in (output / 'ground').iterdir()} == {'manifest.json', NPZ_NAME}
        assert runner.read(output / 'ground_provenance.json')['checkpoint_sha256'] == expected


@pytest.mark.parametrize('fault', [None, 'capsule_id', 'split_count', 'protocol_schema', 'phase2_data_status'])
def test_reuses_validated_capsule_without_builder_or_data_revalidation(tmp_path, monkeypatch, fault):
    path, spec = specification(tmp_path)
    spec['confirmation']['reuse_validated_capsule_id'] = 'synthetic'
    save(path, spec)
    manifest = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
                    capsule_id='synthetic', split_count=900)
    if fault:
        manifest[fault] = 899 if fault == 'split_count' else 'wrong'
    capsule_manifest = Path(spec['confirmation']['capsule']) / 'manifest.json'
    save(capsule_manifest, manifest)
    original = capsule_manifest.read_bytes()
    calls = install_workers(monkeypatch, spec)
    if fault:
        with pytest.raises(ValueError, match='capsule'):
            runner.run(path, 'synthetic-commit')
        assert calls == []
    else:
        runner.run(path, 'synthetic-commit')
        assert not any(c[0] == 'build_d92_confirmation_data.py' for c in calls)
        record = runner.read(Path(spec['execution']['remote_run_root']) / 'data_reuse.json')
        assert record['capsule_id'] == 'synthetic' and record['data_revalidated'] is False
        assert record['data_rebuilt'] is False and record['truth_read'] is False
    assert capsule_manifest.read_bytes() == original
