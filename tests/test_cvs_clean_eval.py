"""Synthetic protocol fixtures only; these files are never experiment evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
import pytest
import torch
from experiments.cvs_clean_eval import contracts,predict,score,dispatch
from experiments.cvs_residual_identity.dispatch import combine_research_selection


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(512,6)
    def forward(self,x):return self.linear(x.flatten(1))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def case():
    root=Path(__file__).resolve().parents[1]/'local_artifacts/clean_eval_protocol_fixtures'/str(uuid.uuid4())
    root.mkdir(parents=True)
    (root/'SYNTHETIC_TEST_ONLY.txt').write_text('Synthetic unit-test metadata, weights and IQ;not a formal E200 run or benchmark.',encoding='utf-8')
    stats=dict(score=.90,source_accuracy=.92,worst_rx_accuracy=.88,parameters=382146,conv_linear_macs=9862436)
    first=dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,
        source_summaries={v:copy.deepcopy(stats) for v in ['native','cvcnn','real_cnn','resnet1d','orthogonal_pa','moment_pool','orthogonal_moment','shared_complex']})
    second=dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,
        source_summaries={v:dict(stats,score=.96 if v=='residual_fusion' else .94,parameters=164225,conv_linear_macs=9708836) for v in ['residual_fusion','residual_fusion_moment']})
    for name,value in [('first.json',first),('second.json',second)]:write(root/name,value)
    selection=combine_research_selection(first,second);selection['source_selection_refs']=[str(root/'first.json'),str(root/'second.json')]
    write(root/'selection.json',selection)
    capsule=root/'capsule';capsule.mkdir();ids=np.asarray(['opaque-'+str(i) for i in range(12)])
    np.savez(capsule/'index.npz',ids=ids)
    np.save(capsule/'clean.npy',np.random.default_rng(31).normal(size=(12,2,256)).astype('float32'))
    classes=['TX'+str(i) for i in range(6)]
    write(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=classes,channel='residual/post_sync/noeq'))
    truth={sid:dict(label=i%6,receiver=i%2) for i,sid in enumerate(ids.tolist())};write(root/'truth.json',truth)
    source_contract=dict(role_ids={'L_s':['L'],'U_s':['U'],'V':['V']},source_rxs=[1,3,4,6,8],source_days=[1,2,3],
        ratios={'L_s':.1},split_seed=392005,num_classes=6,classes=classes,equalized=1,out_len=256,normalize=True)
    write(root/'expected_source_contract.json',source_contract)
    runtime=root/'evaluation';runtime.mkdir();rows=[]
    for seed in sorted(contracts.SEEDS):
        for variant in ['native','cvcnn','real_cnn','resnet1d','residual_fusion']:
            rid=variant+'-s'+str(seed);out=runtime/rid/'prediction'
            cfg=dict(method='cvs_clean_eval',variant=variant,model_seed=seed,selection_file=str(root/'selection.json'),
                baseline_source_root=str(root/'first_source'),residual_source_root=str(root/'second_source'),
                source_output=str(root/('second_source' if variant=='residual_fusion' else 'first_source')/rid/'source'),
                output_root=str(out),source_contract=str(root/'expected_source_contract.json'),p1_capsule=str(capsule),device='cpu',views=['clean'])
            path=root/'configs'/(rid+'.json');write(path,cfg)
            rows.append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(path),output_root=str(out)))
    spec=dict(run_id='synthetic-fixture',runtime_root=str(runtime),log_root=str(root/'logs'),selection_file=str(root/'selection.json'),p1_truth=str(root/'truth.json'),rows=rows)
    return dict(root=root,ids=ids,classes=classes,truth=truth,selection=selection,spec=spec,contract=source_contract)


def source_fixture(case,cfg):
    folder=Path(cfg['source_output'])
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,model_seed=cfg['model_seed'])
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False)
    resolved=dict(method=contracts.source_method(cfg['variant']),variant=cfg['variant'],model_seed=cfg['model_seed'],augmentation=False,domain_backbone=False,
        extra_losses=[],target_access=False,epochs=200,steps_per_epoch=50,selection='fixed_last_epoch',source_counts={'L_s':6300,'U_s':56700,'V':27000})
    if cfg['variant'] in contracts.TENTH:resolved.update(observables_active=True,observables_actual=contracts.observable_contract(cfg['variant']),observables=contracts.observable_contract(cfg['variant']),classifier_scale=30.0)
    if cfg['variant'] in contracts.NINTH:resolved.update(rf_operator_active=True,rf_operator_actual=contracts.operator_contract(cfg['variant']),rf_operator=contracts.operator_contract(cfg['variant']),classifier_scale=30.0)
    if cfg['variant'] in contracts.EIGHTH:resolved.update(coherence_phase_active=True,dsq_active=False,coherence_epsilon=1e-6,time_stability_channels=8,freq_stability_channels=0,classifier_scale=30.0,classifier_geometry='learned_rotated_simplex' if cfg['variant']=='simplex_learned' else 'fixed_simplex')
    if cfg['variant'] in contracts.SEVENTH:resolved.update(coherence_phase_active=True,dsq_active=cfg['variant']=='coherence_dsq',coherence_epsilon=1e-6,time_stability_channels=8,freq_stability_channels=4 if cfg['variant']=='coherence_dsq' else 0)
    if cfg['variant'] in contracts.SIXTH:resolved.update(phase_delta_active=True,dsq_active=cfg['variant']=='phase_dsq',time_stability_channels=8,freq_stability_channels=4 if cfg['variant']=='phase_dsq' else 0)
    payload=dict(model=TinyClassifier().state_dict(),epoch=200,source_contract=case['contract'],initialization=initial,selection='fixed_last_epoch',
        method=resolved['method'],variant=cfg['variant'],config=resolved,classes=case['classes'],num_classes=6)
    for name,value in [('initialization.json',initial),('completion.json',done),('source_contract.json',case['contract']),('resolved_config.json',resolved)]:write(folder/name,value)
    torch.save(payload,folder/'last.pt')
    return done,initial,resolved,payload


def prediction_fixtures(case):
    for row in case['spec']['rows']:
        cfg=contracts.read(row['config']);out=Path(row['output_root']);out.mkdir(parents=True)
        p=np.arange(12)%6
        if row['variant']=='cvcnn':p[::2]=(p[::2]+1)%6
        elif row['variant']=='real_cnn':p=(p+1)%6
        np.savez(out/'clean_predictions.npz',ids=case['ids'],clean=p.astype('int64'))
        write(out/'clean_complete.json',dict(status='PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,views=['clean'],count=12))
        write(out/'resolved_config.json',dict(cfg,truth_read=False,query_fit=False))
        write(out/'provenance.json',dict(status='VERIFIED',query_fit=False))


def test_source_frozen_selection_recomputed_and_tampering_rejected(case):
    assert contracts.frozen_selection(case['root']/'selection.json')['selected_variant']=='residual_fusion'
    write(case['root']/'selection.json',dict(case['selection'],selected_variant='shared_complex'))
    with pytest.raises(ValueError):contracts.frozen_selection(case['root']/'selection.json')


def test_predict_config_rejects_nonselected_views_and_truth(case):
    cfg=contracts.read(case['spec']['rows'][-4]['config'])
    cfg=dict(cfg,variant='residual_fusion',source_output=str(case['root']/'second_source'/('residual_fusion-s'+str(cfg['model_seed']))/'source'))
    contracts.validate_predict_config(cfg,case['selection'])
    for change in [dict(variant='shared_complex'),dict(views=['clean','satellite']),dict(p1_truth='truth.json'),dict(source_output='wrong-seed/source')]:
        with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,**change),case['selection'])


def test_checkpoint_roles_seed_and_target_contamination_guards(case):
    cfg=contracts.read(case['spec']['rows'][4]['config']);done,initial,resolved,payload=source_fixture(case,cfg)
    args=[cfg,done,initial,case['contract'],case['contract'],resolved,payload]
    contracts.checkpoint_contract(*args)
    changed=copy.deepcopy(args);changed[4]['role_ids']['L_s']=['different physical data']
    # Deepcopy preserves shared references; replace expected with an independent copy.
    changed=list(args);changed[4]=copy.deepcopy(case['contract']);changed[4]['role_ids']['L_s']=['different physical data']
    with pytest.raises(ValueError):contracts.checkpoint_contract(*changed)
    changed=list(args);changed[2]=dict(initial,target_contact=True)
    with pytest.raises(ValueError):contracts.checkpoint_contract(*changed)
    changed=list(args);changed[1]=dict(done,epoch=199)
    with pytest.raises(ValueError):contracts.checkpoint_contract(*changed)
    changed=list(args);changed[6]=dict(payload,variant='residual_fusion_moment')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*changed)


def test_predict_only_clean_without_satellite_file_and_without_truth(case,monkeypatch):
    cfg=contracts.read(case['spec']['rows'][4]['config']);source_fixture(case,cfg)
    monkeypatch.setattr(predict,'build_model',lambda variant:TinyClassifier())
    opened=[];original=np.load
    def observed(path,*args,**kwargs):
        opened.append(Path(path).name)
        assert Path(path).name!='satellite.npy'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(predict.np,'load',observed)
    predict.predict(cfg)
    marker=contracts.read(Path(cfg['output_root'])/'clean_complete.json')
    assert marker['views']==['clean'] and not marker['truth_read'] and not marker['query_fit']
    assert 'clean.npy' in opened and 'satellite.npy' not in opened
    with original(Path(cfg['output_root'])/'clean_predictions.npz') as values:assert set(values.files)=={'ids','clean'}


def test_truth_stays_closed_when_one_prediction_missing(case,monkeypatch):
    prediction_fixtures(case)
    marker=Path(case['spec']['rows'][-1]['output_root'])/'clean_complete.json'
    write(marker,dict(status='INCOMPLETE',truth_read=False,query_fit=False,views=['clean'],count=12))
    opened=[];original=score.read
    def observed(path):opened.append(str(path));return original(path)
    monkeypatch.setattr(score,'read',observed)
    with pytest.raises(ValueError):score.score(case['spec'])
    assert case['spec']['p1_truth'] not in opened


def test_truth_stays_closed_for_identity_or_method_mismatch(case,monkeypatch):
    prediction_fixtures(case)
    row=case['spec']['rows'][-1];out=Path(row['output_root'])
    np.savez(out/'clean_predictions.npz',ids=case['ids'][::-1],clean=np.arange(12)%6)
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    with pytest.raises(ValueError):score.score(case['spec'])
    assert case['spec']['p1_truth'] not in opened


def test_complete_truth_last_metrics_and_paired_seeds(case,monkeypatch):
    prediction_fixtures(case)
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec'])
    assert marker['rows']==20 and marker['records']==60 and marker['view']=='clean'
    truth_position=opened.index(case['spec']['p1_truth'])
    assert sum(p.endswith('clean_complete.json') for p in opened[:truth_position])==20
    result=contracts.read(Path(case['spec']['runtime_root'])/'clean_summary.json')
    overall={r['method']:r for r in result['summary'] if r['receiver']=='ALL'}
    assert overall['residual_fusion']['accuracy_mean']==1.0 and overall['cvcnn']['accuracy_mean']==.5
    paired=next(r for r in result['paired'] if r['receiver']=='ALL' and r['baseline']=='cvcnn')
    assert paired['accuracy_delta_pp_mean']==50 and paired['positive_seeds']==4


def test_matrix_and_dispatch_reject_missing_registered_seed(case):
    dispatch.validate_spec(case['spec'])
    changed=dict(case['spec'],rows=case['spec']['rows'][:-1])
    with pytest.raises(ValueError):dispatch.validate_spec(changed)
    with pytest.raises(ValueError):score.validate_matrix(changed)


def test_one_evaluator_per_gpu_includes_external_occupancy(monkeypatch):
    monkeypatch.setattr(dispatch,'occupancy',lambda active:{0:dict(pids={999},free_mb=22000),1:dict(pids=set(),free_mb=15000),2:dict(pids=set(),free_mb=11000)})
    assert dispatch.choose_gpu({})==1


def baseline_case(case):
    rows=[r for r in case['spec']['rows'] if r['variant'] in contracts.BASELINES]
    source=dict(rows=[dict(variant=r['variant'],model_seed=r['model_seed']) for r in rows])
    write(case['root']/'source_matrix.json',source)
    selection=dict(scope='baseline_only',status='FIXED_BASELINES_FROZEN',test_variants=list(contracts.BASELINES),
        model_seeds=sorted(contracts.SEEDS),target_access=False,target_score_used=False,source_matrix_ref=str(case['root']/'source_matrix.json'))
    write(case['root']/'selection.json',selection)
    case['spec']['rows']=rows;case['selection']=selection
    return case


def test_fixed_baselines_score16_without_candidate_source_selection(case):
    case=baseline_case(case)
    assert contracts.frozen_selection(case['root']/'selection.json')['status']=='FIXED_BASELINES_FROZEN'
    dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    result=score.score(case['spec'])
    assert result['rows']==16 and result['models']==4 and result['records']==48
    summary=contracts.read(Path(case['spec']['runtime_root'])/'clean_summary.json')
    assert len(summary['paired'])==9 and all(r['candidate']=='native' for r in summary['paired'])


def test_fixed_baselines_reject_candidate_and_plan_changes(case):
    case=baseline_case(case);cfg=contracts.read(case['spec']['rows'][0]['config'])
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='residual_fusion'),case['selection'])
    write(case['root']/'selection.json',dict(case['selection'],model_seeds=[2026092701]))
    with pytest.raises(ValueError):contracts.frozen_selection(case['root']/'selection.json')


def test_original_role_contract_without_runtime_extensions(case):
    cfg=contracts.read(case['spec']['rows'][0]['config']);done,initial,resolved,payload=source_fixture(case,cfg)
    expected={k:v for k,v in case['contract'].items() if k not in {'classes','equalized','out_len','normalize'}}
    contracts.checkpoint_contract(cfg,done,initial,case['contract'],expected,resolved,payload)
    with pytest.raises(ValueError):contracts.checkpoint_contract(cfg,done,initial,case['contract'],dict(expected,num_classes=5),resolved,payload)


def reused_case(case):
    old=case['root']/contracts.REUSED_BASELINE_RUN
    first=dict(rows=[dict(variant=v,model_seed=s) for v in contracts.BASELINES for s in sorted(contracts.SEEDS)])
    write(case['root']/'baseline_source_matrix.json',first)
    fixed=dict(scope='baseline_only',status='FIXED_BASELINES_FROZEN',test_variants=list(contracts.BASELINES),model_seeds=sorted(contracts.SEEDS),
        target_access=False,target_score_used=False,source_matrix_ref=str(case['root']/'baseline_source_matrix.json'))
    write(case['root']/'fixed_baselines.json',fixed)
    write(old/'scoring_clean_complete.json',dict(status='SCORED_COMPLETE',models=4,seeds=4,rows=16))
    for row in case['spec']['rows']:
        if row['variant'] not in contracts.BASELINES:continue
        row.update(reuse_from_run=contracts.REUSED_BASELINE_RUN,output_root=str(old/row['row_id']/'prediction'))
        cfg=contracts.read(row['config']);cfg.update(output_root=row['output_root'],selection_file=str(case['root']/'fixed_baselines.json'));write(Path(row['config']),cfg)
    return case


def test_reuse16_fixed_baselines_and_score4_new_predictions(case):
    reused_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    marker=score.score(case['spec'])
    assert marker['rows']==20 and marker['models']==5
    assert sum(not r.get('reuse_from_run') for r in case['spec']['rows'])==4


def test_reuse_rejects_wrong_original_run_or_incomplete_baselines(case,monkeypatch):
    reused_case(case);prediction_fixtures(case)
    row=case['spec']['rows'][0];row['reuse_from_run']='different-run'
    with pytest.raises(ValueError):dispatch.validate_spec(case['spec'])
    opened=[];original=score.read;monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    with pytest.raises(ValueError):score.score(case['spec'])
    assert case['spec']['p1_truth'] not in opened
    row['reuse_from_run']=contracts.REUSED_BASELINE_RUN
    write(case['root']/contracts.REUSED_BASELINE_RUN/'scoring_clean_complete.json',dict(status='INCOMPLETE',models=4,seeds=4,rows=16))
    with pytest.raises(ValueError):dispatch.validate_spec(case['spec'])


def test_fresh_process_imports():
    root=Path(__file__).resolve().parents[1]
    for module in ['predict','score','dispatch']:
        subprocess.run([sys.executable,'-m','experiments.cvs_clean_eval.'+module,'--help'],cwd=root,check=True,capture_output=True)


def balanced_case(case):
    from experiments.cvs_balanced_identity.freeze import select_performance_candidate
    reused_case(case)
    old=case['root']/contracts.REUSED_RESIDUAL_RUN
    write(case['root']/'previous_residual_selection.json',case['selection'])
    write(old/'scoring_clean_complete.json',dict(status='SCORED_COMPLETE',models=5,seeds=4,rows=20))
    for row in case['spec']['rows']:
        if row['variant']!='residual_fusion':continue
        row.update(reuse_from_run=contracts.REUSED_RESIDUAL_RUN,output_root=str(old/row['row_id']/'prediction'))
        cfg=contracts.read(row['config']);cfg.update(output_root=row['output_root'],selection_file=str(case['root']/'previous_residual_selection.json'))
        write(Path(row['config']),cfg)
    records=[];source_rows=[]
    for variant in contracts.THIRD:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed);q=case['root']/'balanced_source'/rid/'source'
            write(q/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                final_source_metrics=dict(source_val_accuracy=.98,source_val_worst_rx=.95)))
            write(q/'resource_profile.json',dict(total_parameters=113665,conv_linear_macs_per_sample=9657476))
            source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_output=str(q)))
            records.append(dict(variant=variant,seed=seed,accuracy=.98,worst_rx=.95,parameters=113665,macs=9657476))
    write(case['root']/'balanced_source_matrix.json',dict(rows=source_rows))
    selection=dict(select_performance_candidate(records),scope='balanced_source',source_matrix_ref=str(case['root']/'balanced_source_matrix.json'))
    selection_path=case['root']/'balanced_selection.json';write(selection_path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config'])
        cfg.update(variant=variant,model_seed=seed,selection_file=str(selection_path),balanced_source_root=str(case['root']/'balanced_source'),
            source_output=str(case['root']/'balanced_source'/rid/'source'),output_root=str(out))
        config_path=case['root']/'configs'/(rid+'.json');write(config_path,cfg)
        case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(config_path),output_root=str(out)))
    case['spec']['selection_file']=str(selection_path);case['selection']=selection
    return case


def test_balanced_freeze_uses_actual_completed_source_and_rejects_tampering(case):
    balanced_case(case)
    selection=contracts.frozen_selection(case['spec']['selection_file'])
    assert selection['selected_variant']=='balanced_fusion'
    write(Path(case['spec']['selection_file']),dict(selection,selected_variant='signed_balanced_fusion'))
    with pytest.raises(ValueError):contracts.frozen_selection(case['spec']['selection_file'])
    write(Path(case['spec']['selection_file']),selection)
    q=case['root']/'balanced_source'/'balanced_fusion-s2026092701'/'source'/'completion.json'
    write(q,dict(contracts.read(q),epoch=199))
    with pytest.raises(ValueError):contracts.frozen_selection(case['spec']['selection_file'])


def test_balanced_reuses20_and_scores4_new_truth_last(case,monkeypatch):
    balanced_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec'])
    assert marker['models']==6 and marker['rows']==24 and marker['records']==72
    assert sum(not r.get('reuse_from_run') for r in case['spec']['rows'])==4
    at=opened.index(case['spec']['p1_truth'])
    assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24
    summary=contracts.read(Path(case['spec']['runtime_root'])/'clean_summary.json')
    assert any(r['baseline']=='residual_fusion' and r['receiver']=='ALL' for r in summary['paired'])


def test_balanced_rejects_other_old_variant_missing_marker_and_unselected_candidate(case,monkeypatch):
    balanced_case(case);prediction_fixtures(case)
    row=next(r for r in case['spec']['rows'] if r['variant']=='residual_fusion')
    row['reuse_from_run']='unregistered-run'
    with pytest.raises(ValueError):dispatch.validate_spec(case['spec'])
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    with pytest.raises(ValueError):score.score(case['spec'])
    assert case['spec']['p1_truth'] not in opened
    row['reuse_from_run']=contracts.REUSED_RESIDUAL_RUN
    write(case['root']/contracts.REUSED_RESIDUAL_RUN/'scoring_clean_complete.json',dict(status='INCOMPLETE',models=5,seeds=4,rows=20))
    with pytest.raises(ValueError):dispatch.validate_spec(case['spec'])
    cfg=contracts.read(case['spec']['rows'][-1]['config'])
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='signed_balanced_fusion'),case['selection'])


def test_balanced_checkpoint_same_physical_contract_and_clean_prediction(case,monkeypatch):
    balanced_case(case);cfg=contracts.read(case['spec']['rows'][-1]['config'])
    done,initial,resolved,payload=source_fixture(case,cfg)
    write(Path(cfg['source_output'])/'completion.json',dict(done,final_source_metrics=dict(source_val_accuracy=.98,source_val_worst_rx=.95)))
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier())
    predict.predict(cfg)
    marker=contracts.read(Path(cfg['output_root'])/'clean_complete.json')
    assert marker['count']==12 and marker['truth_read'] is False and marker['query_fit'] is False


def test_balanced_prepare_emits_only_four_new_rows_with_fixed_capsule(case,monkeypatch):
    from experiments.cvs_balanced_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1]
    prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):
        write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    selection=dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='balanced_fusion')
    prepare.main(selection)
    folder=case['root']/'experiments/cvs_balanced_clean/configs'
    runtime=contracts.read(folder/'launch_spec.json')
    new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(runtime['rows'])==24 and len(new)==4
    enriched=contracts.read(folder/'frozen_selection.json')
    for row in new:
        cfg=contracts.read(folder/(row['row_id']+'.json'))
        assert cfg['p1_capsule']==CAPSULE and cfg['views']==['clean']
        contracts.validate_predict_config(cfg,enriched)


def test_large_release_status_uses_bounded_windows_pathspecs(case,monkeypatch):
    from experiments.cvs_clean_eval import publish
    names=['experiments/cvs_balanced_identity/configs/large-name-'+str(i)+'.json' for i in range(1000)]
    calls=[]
    def output(command,**kwargs):
        if command[1:]==['rev-parse','HEAD']:return 'abc\n'
        if command[1:]==['branch','--show-current']:return 'codex/test\n'
        if 'ls-remote' in command:return 'abc refs/heads/codex/test\n'
        if command[1:]==['ls-files']:return '\n'.join(names)
        if command[1]=='status':
            calls.append(command)
            assert len(' '.join(command))<32000
            # Stop before package/remote work; the test concerns this boundary.
            return ' M experiments/cvs_balanced_identity/model.py\n'
        raise AssertionError(command)
    monkeypatch.setattr(publish.subprocess,'check_output',output)
    with pytest.raises(ValueError,match='Uncommitted release'):publish.publish(case['root']/'package')
    assert calls


def interaction_case(case):
    from experiments.cvs_interaction_identity.dispatch import select_source_candidate
    balanced_case(case)
    case['spec']['rows']=[r for r in case['spec']['rows'] if r['variant'] not in contracts.THIRD]
    records=[];source_rows=[]
    for variant in contracts.FOURTH:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed);q=case['root']/'interaction_source'/rid/'source'
            accuracy=.981 if variant=='tf_lowrank32' else .98
            write(q/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                final_source_metrics=dict(source_val_accuracy=accuracy,source_val_worst_rx=.95)))
            write(q/'resource_profile.json',dict(total_parameters=129089 if variant=='tf_lowrank32' else 113825,conv_linear_macs_per_sample=9672868 if variant=='tf_lowrank32' else 9657476))
            source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_output=str(q)))
            records.append(dict(variant=variant,seed=seed,accuracy=accuracy,worst_rx=.95,parameters=129089 if variant=='tf_lowrank32' else 113825,macs=9672868 if variant=='tf_lowrank32' else 9657476))
    matrix=case['root']/'interaction_source_matrix.json';write(matrix,dict(rows=source_rows))
    selection=dict(select_source_candidate(records),scope='interaction_source',source_matrix_ref=str(matrix))
    path=case['root']/'interaction_selection.json';write(path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config']);cfg.update(variant=variant,model_seed=seed,selection_file=str(path),
            interaction_source_root=str(case['root']/'interaction_source'),source_output=str(case['root']/'interaction_source'/rid/'source'),output_root=str(out))
        cp=case['root']/'configs'/(rid+'.json');write(cp,cfg)
        case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(cp),output_root=str(out)))
    case['selection']=selection;case['spec']['selection_file']=str(path)
    return case


def test_interaction_actual_source_freeze_and24row_truth_last(case,monkeypatch):
    interaction_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    assert contracts.frozen_selection(case['spec']['selection_file'])['selected_variant']=='tf_lowrank32'
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec']);assert marker['rows']==24 and marker['models']==6
    at=opened.index(case['spec']['p1_truth']);assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24


def test_interaction_checkpoint_guards_and_unselected_rejection(case,monkeypatch):
    interaction_case(case);cfg=contracts.read(case['spec']['rows'][-1]['config'])
    done,initial,resolved,payload=source_fixture(case,cfg)
    write(Path(cfg['source_output'])/'completion.json',dict(done,final_source_metrics=dict(source_val_accuracy=.981,source_val_worst_rx=.95)))
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier())
    predict.predict(cfg)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='tf_bilinear'),case['selection'])
    args=[cfg,done,initial,case['contract'],case['contract'],resolved,payload]
    bad=list(args);bad[-1]=dict(payload,method='cvs_balanced_identity')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*bad)


def test_interaction_prepare_only_selected4_fixed_capsule(case,monkeypatch):
    from experiments.cvs_interaction_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1];prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    prepare.main(dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='tf_lowrank32'))
    folder=case['root']/'experiments/cvs_interaction_clean/configs';runtime=contracts.read(folder/'launch_spec.json');new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(new)==4 and len(runtime['rows'])==24
    for row in new:
        c=contracts.read(folder/(row['row_id']+'.json'));assert c['p1_capsule']==CAPSULE
        contracts.validate_predict_config(c,contracts.read(folder/'frozen_selection.json'))


def attentive_case(case):
    from experiments.cvs_attentive_identity.dispatch import select_source_candidate
    balanced_case(case)
    case['spec']['rows']=[r for r in case['spec']['rows'] if r['variant'] not in contracts.THIRD]
    records=[];source_rows=[]
    for variant in contracts.FIFTH:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed);q=case['root']/'attentive_source'/rid/'source'
            accuracy=.981 if variant=='attentive_moments' else .98
            write(q/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                final_source_metrics=dict(source_val_accuracy=accuracy,source_val_worst_rx=.95)))
            write(q/'resource_profile.json',dict(total_parameters=164609 if variant=='attentive_moments' else 164417,conv_linear_macs_per_sample=9716260 if variant=='attentive_moments' else 9716260))
            source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_output=str(q)))
            records.append(dict(variant=variant,seed=seed,accuracy=accuracy,worst_rx=.95,parameters=164609 if variant=='attentive_moments' else 164417,macs=9716260 if variant=='attentive_moments' else 9716260))
    matrix=case['root']/'attentive_source_matrix.json';write(matrix,dict(rows=source_rows))
    selection=dict(select_source_candidate(records),scope='attentive_source',source_matrix_ref=str(matrix))
    path=case['root']/'attentive_selection.json';write(path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config']);cfg.update(variant=variant,model_seed=seed,selection_file=str(path),
            attentive_source_root=str(case['root']/'attentive_source'),source_output=str(case['root']/'attentive_source'/rid/'source'),output_root=str(out))
        cp=case['root']/'configs'/(rid+'.json');write(cp,cfg)
        case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(cp),output_root=str(out)))
    case['selection']=selection;case['spec']['selection_file']=str(path)
    return case


def test_attentive_actual_source_freeze_and24row_truth_last(case,monkeypatch):
    attentive_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    assert contracts.frozen_selection(case['spec']['selection_file'])['selected_variant']=='attentive_moments'
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec']);assert marker['rows']==24 and marker['models']==6
    at=opened.index(case['spec']['p1_truth']);assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24


def test_attentive_checkpoint_guards_and_unselected_rejection(case,monkeypatch):
    attentive_case(case);cfg=contracts.read(case['spec']['rows'][-1]['config'])
    done,initial,resolved,payload=source_fixture(case,cfg)
    write(Path(cfg['source_output'])/'completion.json',dict(done,final_source_metrics=dict(source_val_accuracy=.981,source_val_worst_rx=.95)))
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier())
    predict.predict(cfg)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='attentive_mean'),case['selection'])
    args=[cfg,done,initial,case['contract'],case['contract'],resolved,payload]
    bad=list(args);bad[-1]=dict(payload,method='cvs_balanced_identity')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*bad)


def test_attentive_prepare_only_selected4_fixed_capsule(case,monkeypatch):
    from experiments.cvs_attentive_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1];prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    prepare.main(dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='attentive_moments'))
    folder=case['root']/'experiments/cvs_attentive_clean/configs';runtime=contracts.read(folder/'launch_spec.json');new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(new)==4 and len(runtime['rows'])==24
    for row in new:
        c=contracts.read(folder/(row['row_id']+'.json'));assert c['p1_capsule']==CAPSULE
        contracts.validate_predict_config(c,contracts.read(folder/'frozen_selection.json'))


def stability_case(case):
    from experiments.cvs_stability_identity.dispatch import select_source_candidate
    balanced_case(case)
    case['spec']['rows']=[r for r in case['spec']['rows'] if r['variant'] not in contracts.THIRD]
    records=[];source_rows=[]
    for variant in contracts.SIXTH:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed);q=case['root']/'stability_source'/rid/'source'
            accuracy=.981 if variant=='phase_dsq' else .98
            write(q/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                final_source_metrics=dict(source_val_accuracy=accuracy,source_val_worst_rx=.95)))
            write(q/'resource_profile.json',dict(total_parameters=165521 if variant=='phase_dsq' else 165393,conv_linear_macs_per_sample=10007588 if variant=='phase_dsq' else 10003748))
            source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_output=str(q)))
            records.append(dict(variant=variant,seed=seed,accuracy=accuracy,worst_rx=.95,parameters=165521 if variant=='phase_dsq' else 165393,macs=10007588 if variant=='phase_dsq' else 10003748))
    matrix=case['root']/'stability_source_matrix.json';write(matrix,dict(rows=source_rows))
    selection=dict(select_source_candidate(records),scope='stability_source',source_matrix_ref=str(matrix))
    path=case['root']/'stability_selection.json';write(path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config']);cfg.update(variant=variant,model_seed=seed,selection_file=str(path),
            stability_source_root=str(case['root']/'stability_source'),source_output=str(case['root']/'stability_source'/rid/'source'),output_root=str(out))
        cp=case['root']/'configs'/(rid+'.json');write(cp,cfg)
        case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(cp),output_root=str(out)))
    case['selection']=selection;case['spec']['selection_file']=str(path)
    return case


def test_stability_actual_source_freeze_and24row_truth_last(case,monkeypatch):
    stability_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    assert contracts.frozen_selection(case['spec']['selection_file'])['selected_variant']=='phase_dsq'
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec']);assert marker['rows']==24 and marker['models']==6
    at=opened.index(case['spec']['p1_truth']);assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24


def test_stability_checkpoint_guards_and_unselected_rejection(case,monkeypatch):
    stability_case(case);cfg=contracts.read(case['spec']['rows'][-1]['config'])
    done,initial,resolved,payload=source_fixture(case,cfg)
    write(Path(cfg['source_output'])/'completion.json',dict(done,final_source_metrics=dict(source_val_accuracy=.981,source_val_worst_rx=.95)))
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier())
    predict.predict(cfg)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='phase_delta'),case['selection'])
    args=[cfg,done,initial,case['contract'],case['contract'],resolved,payload]
    bad=list(args);bad[-1]=dict(payload,method='cvs_balanced_identity')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*bad)
    bad=list(args);bad[5]=dict(resolved,phase_delta_active=False)
    with pytest.raises(ValueError,match='Physical cue activation'):contracts.checkpoint_contract(*bad)


def test_stability_prepare_only_selected4_fixed_capsule(case,monkeypatch):
    from experiments.cvs_stability_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1];prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    prepare.main(dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='phase_dsq'))
    folder=case['root']/'experiments/cvs_stability_clean/configs';runtime=contracts.read(folder/'launch_spec.json');new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(new)==4 and len(runtime['rows'])==24
    for row in new:
        c=contracts.read(folder/(row['row_id']+'.json'));assert c['p1_capsule']==CAPSULE
        contracts.validate_predict_config(c,contracts.read(folder/'frozen_selection.json'))


def coherence_case(case):
    from experiments.cvs_coherence_identity.dispatch import select_source_candidate
    balanced_case(case)
    case['spec']['rows']=[r for r in case['spec']['rows'] if r['variant'] not in contracts.THIRD]
    records=[];source_rows=[]
    for variant in contracts.SEVENTH:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed);q=case['root']/'coherence_source'/rid/'source'
            accuracy=.981 if variant=='coherence_dsq' else .98
            write(q/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                final_source_metrics=dict(source_val_accuracy=accuracy,source_val_worst_rx=.95)))
            write(q/'resource_profile.json',dict(total_parameters=165521 if variant=='coherence_dsq' else 165393,conv_linear_macs_per_sample=10007588 if variant=='coherence_dsq' else 10003748))
            source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_output=str(q)))
            records.append(dict(variant=variant,seed=seed,accuracy=accuracy,worst_rx=.95,parameters=165521 if variant=='coherence_dsq' else 165393,macs=10007588 if variant=='coherence_dsq' else 10003748))
    matrix=case['root']/'coherence_source_matrix.json';write(matrix,dict(rows=source_rows))
    selection=dict(select_source_candidate(records),scope='coherence_source',source_matrix_ref=str(matrix))
    path=case['root']/'coherence_selection.json';write(path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config']);cfg.update(variant=variant,model_seed=seed,selection_file=str(path),
            coherence_source_root=str(case['root']/'coherence_source'),source_output=str(case['root']/'coherence_source'/rid/'source'),output_root=str(out))
        cp=case['root']/'configs'/(rid+'.json');write(cp,cfg)
        case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(cp),output_root=str(out)))
    case['selection']=selection;case['spec']['selection_file']=str(path)
    return case


def test_coherence_actual_source_freeze_and24row_truth_last(case,monkeypatch):
    coherence_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    assert contracts.frozen_selection(case['spec']['selection_file'])['selected_variant']=='coherence_dsq'
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec']);assert marker['rows']==24 and marker['models']==6
    at=opened.index(case['spec']['p1_truth']);assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24


def test_coherence_checkpoint_guards_and_unselected_rejection(case,monkeypatch):
    coherence_case(case);cfg=contracts.read(case['spec']['rows'][-1]['config'])
    done,initial,resolved,payload=source_fixture(case,cfg)
    write(Path(cfg['source_output'])/'completion.json',dict(done,final_source_metrics=dict(source_val_accuracy=.981,source_val_worst_rx=.95)))
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier())
    predict.predict(cfg)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='coherence_phase'),case['selection'])
    args=[cfg,done,initial,case['contract'],case['contract'],resolved,payload]
    bad=list(args);bad[-1]=dict(payload,method='cvs_balanced_identity')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*bad)
    bad=list(args);bad[5]=dict(resolved,coherence_phase_active=False)
    with pytest.raises(ValueError,match='Complex coherence activation'):contracts.checkpoint_contract(*bad)


def test_coherence_prepare_only_selected4_fixed_capsule(case,monkeypatch):
    from experiments.cvs_coherence_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1];prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    prepare.main(dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='coherence_dsq'))
    folder=case['root']/'experiments/cvs_coherence_clean/configs';runtime=contracts.read(folder/'launch_spec.json');new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(new)==4 and len(runtime['rows'])==24
    for row in new:
        c=contracts.read(folder/(row['row_id']+'.json'));assert c['p1_capsule']==CAPSULE
        contracts.validate_predict_config(c,contracts.read(folder/'frozen_selection.json'))


def simplex_case(case):
    from experiments.cvs_simplex_identity.dispatch import select_source_candidate
    balanced_case(case)
    case['spec']['rows']=[r for r in case['spec']['rows'] if r['variant'] not in contracts.THIRD]
    records=[];source_rows=[]
    for variant in contracts.EIGHTH:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed);q=case['root']/'simplex_source'/rid/'source'
            accuracy=.981 if variant=='simplex_fixed' else .98
            write(q/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                final_source_metrics=dict(source_val_accuracy=accuracy,source_val_worst_rx=.95)))
            write(q/'resource_profile.json',dict(total_parameters=165521 if variant=='simplex_fixed' else 165393,conv_linear_macs_per_sample=10007588 if variant=='simplex_fixed' else 10003748))
            source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_output=str(q)))
            records.append(dict(variant=variant,seed=seed,accuracy=accuracy,worst_rx=.95,parameters=165521 if variant=='simplex_fixed' else 165393,macs=10007588 if variant=='simplex_fixed' else 10003748))
    matrix=case['root']/'simplex_source_matrix.json';write(matrix,dict(rows=source_rows))
    selection=dict(select_source_candidate(records),scope='simplex_source',source_matrix_ref=str(matrix))
    path=case['root']/'simplex_selection.json';write(path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config']);cfg.update(variant=variant,model_seed=seed,selection_file=str(path),
            simplex_source_root=str(case['root']/'simplex_source'),source_output=str(case['root']/'simplex_source'/rid/'source'),output_root=str(out))
        cp=case['root']/'configs'/(rid+'.json');write(cp,cfg)
        case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(cp),output_root=str(out)))
    case['selection']=selection;case['spec']['selection_file']=str(path)
    return case


def test_simplex_actual_source_freeze_and24row_truth_last(case,monkeypatch):
    simplex_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    assert contracts.frozen_selection(case['spec']['selection_file'])['selected_variant']=='simplex_fixed'
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec']);assert marker['rows']==24 and marker['models']==6
    at=opened.index(case['spec']['p1_truth']);assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24


def test_simplex_checkpoint_guards_and_unselected_rejection(case,monkeypatch):
    simplex_case(case);cfg=contracts.read(case['spec']['rows'][-1]['config'])
    done,initial,resolved,payload=source_fixture(case,cfg)
    write(Path(cfg['source_output'])/'completion.json',dict(done,final_source_metrics=dict(source_val_accuracy=.981,source_val_worst_rx=.95)))
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier())
    predict.predict(cfg)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='simplex_learned'),case['selection'])
    args=[cfg,done,initial,case['contract'],case['contract'],resolved,payload]
    bad=list(args);bad[-1]=dict(payload,method='cvs_balanced_identity')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*bad)
    bad=list(args);bad[5]=dict(resolved,classifier_geometry='learned_rotated_simplex')
    with pytest.raises(ValueError,match='Simplex classifier activation'):contracts.checkpoint_contract(*bad)


def test_simplex_prepare_only_selected4_fixed_capsule(case,monkeypatch):
    from experiments.cvs_simplex_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1];prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    prepare.main(dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='simplex_fixed'))
    folder=case['root']/'experiments/cvs_simplex_clean/configs';runtime=contracts.read(folder/'launch_spec.json');new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(new)==4 and len(runtime['rows'])==24
    for row in new:
        c=contracts.read(folder/(row['row_id']+'.json'));assert c['p1_capsule']==CAPSULE
        contracts.validate_predict_config(c,contracts.read(folder/'frozen_selection.json'))


def rf_operator_case(case):
    from experiments.cvs_rf_operator_identity.dispatch import select_source_candidate
    balanced_case(case)
    case['spec']['rows']=[r for r in case['spec']['rows'] if r['variant'] not in contracts.THIRD]
    records=[];source_rows=[]
    for variant in contracts.NINTH:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed);q=case['root']/'rf_operator_source'/rid/'source'
            accuracy=.981 if variant=='rf_gmp' else .98
            write(q/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                final_source_metrics=dict(source_val_accuracy=accuracy,source_val_worst_rx=.95)))
            write(q/'resource_profile.json',dict(total_parameters=165521 if variant=='rf_gmp' else 165393,conv_linear_macs_per_sample=10007588 if variant=='rf_gmp' else 10003748))
            source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_output=str(q)))
            records.append(dict(variant=variant,seed=seed,accuracy=accuracy,worst_rx=.95,parameters=165521 if variant=='rf_gmp' else 165393,macs=10007588 if variant=='rf_gmp' else 10003748))
    matrix=case['root']/'rf_operator_source_matrix.json';write(matrix,dict(rows=source_rows))
    selection=dict(select_source_candidate(records),scope='rf_operator_source',source_matrix_ref=str(matrix))
    path=case['root']/'rf_operator_selection.json';write(path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config']);cfg.update(variant=variant,model_seed=seed,selection_file=str(path),
            rf_operator_source_root=str(case['root']/'rf_operator_source'),source_output=str(case['root']/'rf_operator_source'/rid/'source'),output_root=str(out))
        cp=case['root']/'configs'/(rid+'.json');write(cp,cfg)
        case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(cp),output_root=str(out)))
    case['selection']=selection;case['spec']['selection_file']=str(path)
    return case


def test_rf_operator_actual_source_freeze_and24row_truth_last(case,monkeypatch):
    rf_operator_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    assert contracts.frozen_selection(case['spec']['selection_file'])['selected_variant']=='rf_gmp'
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec']);assert marker['rows']==24 and marker['models']==6
    at=opened.index(case['spec']['p1_truth']);assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24


def test_rf_operator_checkpoint_guards_and_unselected_rejection(case,monkeypatch):
    rf_operator_case(case);cfg=contracts.read(case['spec']['rows'][-1]['config'])
    done,initial,resolved,payload=source_fixture(case,cfg)
    write(Path(cfg['source_output'])/'completion.json',dict(done,final_source_metrics=dict(source_val_accuracy=.981,source_val_worst_rx=.95)))
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier())
    predict.predict(cfg)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='rf_mp'),case['selection'])
    args=[cfg,done,initial,case['contract'],case['contract'],resolved,payload]
    bad=list(args);bad[-1]=dict(payload,method='cvs_balanced_identity')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*bad)
    bad=list(args);bad[5]=dict(resolved,rf_operator_active=False)
    with pytest.raises(ValueError,match='RF operator activation'):contracts.checkpoint_contract(*bad)


def test_rf_operator_prepare_only_selected4_fixed_capsule(case,monkeypatch):
    from experiments.cvs_rf_operator_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1];prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    prepare.main(dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='rf_gmp'))
    folder=case['root']/'experiments/cvs_rf_operator_clean/configs';runtime=contracts.read(folder/'launch_spec.json');new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(new)==4 and len(runtime['rows'])==24
    for row in new:
        c=contracts.read(folder/(row['row_id']+'.json'));assert c['p1_capsule']==CAPSULE
        contracts.validate_predict_config(c,contracts.read(folder/'frozen_selection.json'))


def observable_case(case):
    from experiments.cvs_observable_identity.dispatch import select_source_candidate
    balanced_case(case)
    case['spec']['rows']=[r for r in case['spec']['rows'] if r['variant'] not in contracts.THIRD]
    records=[];source_rows=[]
    for variant in contracts.TENTH:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed);q=case['root']/'observable_source'/rid/'source'
            accuracy=.981 if variant=='observable_affine' else .98
            write(q/'completion.json',dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
                final_source_metrics=dict(source_val_accuracy=accuracy,source_val_worst_rx=.95)))
            write(q/'resource_profile.json',dict(total_parameters=165521 if variant=='observable_affine' else 165393,conv_linear_macs_per_sample=10007588 if variant=='observable_affine' else 10003748))
            source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_output=str(q)))
            records.append(dict(variant=variant,seed=seed,accuracy=accuracy,worst_rx=.95,parameters=165521 if variant=='observable_affine' else 165393,macs=10007588 if variant=='observable_affine' else 10003748))
    matrix=case['root']/'observable_source_matrix.json';write(matrix,dict(rows=source_rows))
    selection=dict(select_source_candidate(records),scope='observable_source',source_matrix_ref=str(matrix))
    path=case['root']/'observable_selection.json';write(path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config']);cfg.update(variant=variant,model_seed=seed,selection_file=str(path),
            observable_source_root=str(case['root']/'observable_source'),source_output=str(case['root']/'observable_source'/rid/'source'),output_root=str(out))
        cp=case['root']/'configs'/(rid+'.json');write(cp,cfg)
        case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(cp),output_root=str(out)))
    case['selection']=selection;case['spec']['selection_file']=str(path)
    return case


def test_observable_actual_source_freeze_and24row_truth_last(case,monkeypatch):
    observable_case(case);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    assert contracts.frozen_selection(case['spec']['selection_file'])['selected_variant']=='observable_affine'
    opened=[];original=score.read
    monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec']);assert marker['rows']==24 and marker['models']==6
    at=opened.index(case['spec']['p1_truth']);assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24


def test_observable_checkpoint_guards_and_unselected_rejection(case,monkeypatch):
    observable_case(case);cfg=contracts.read(case['spec']['rows'][-1]['config'])
    done,initial,resolved,payload=source_fixture(case,cfg)
    write(Path(cfg['source_output'])/'completion.json',dict(done,final_source_metrics=dict(source_val_accuracy=.981,source_val_worst_rx=.95)))
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier())
    predict.predict(cfg)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='observable_phase'),case['selection'])
    args=[cfg,done,initial,case['contract'],case['contract'],resolved,payload]
    bad=list(args);bad[-1]=dict(payload,method='cvs_balanced_identity')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*bad)
    bad=list(args);bad[5]=dict(resolved,observables_active=False)
    with pytest.raises(ValueError,match='Wholeidentity observable activation'):contracts.checkpoint_contract(*bad)


def test_observable_prepare_only_selected4_fixed_capsule(case,monkeypatch):
    from experiments.cvs_observable_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1];prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    prepare.main(dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='observable_affine'))
    folder=case['root']/'experiments/cvs_observable_clean/configs';runtime=contracts.read(folder/'launch_spec.json');new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(new)==4 and len(runtime['rows'])==24
    for row in new:
        c=contracts.read(folder/(row['row_id']+'.json'));assert c['p1_capsule']==CAPSULE
        contracts.validate_predict_config(c,contracts.read(folder/'frozen_selection.json'))


def gauge_case(case,monkeypatch):
    """Full synthetic source provenance and fixed twenty control predictions."""
    from experiments.cvs_gauge_identity import dispatch as gd
    from experiments.cvs_gauge_identity.model import gauge_contract
    balanced_case(case)
    case['spec']['rows']=[r for r in case['spec']['rows'] if r['variant'] not in contracts.THIRD]
    project=case['root']/'synthetic_project';monkeypatch.setattr(gd,'PROJECT',project.as_posix())
    original=copy.deepcopy(case['contract']);original['classes']=['14-10','14-7','20-15','20-19','6-15','8-20']
    case['classes']=original['classes'];case['contract']=dict(original,physical_roles='EXACT_MATCH',dataset_path=(project/'Dataset_WigSig/ManySig.pkl').as_posix())
    capsule=case['root']/'capsule';manifest=contracts.read(capsule/'manifest.json');manifest['classes']=case['classes'];write(capsule/'manifest.json',manifest)
    original_path=project/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json';write(original_path,original)
    new_root=project/'runs/gauge_source';source_rows=[]
    for variant in gd.CANDIDATES:
        for seed in sorted(contracts.SEEDS):
            rid=variant+'-s'+str(seed)
            q=project/'runs'/gd.CONTROL_RUN/rid/'source' if variant=='residual_fusion' else new_root/rid/'source'
            c=dict(method='cvs_residual_identity' if variant=='residual_fusion' else 'cvs_gauge_identity',variant=variant,model_seed=seed,
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,
                source_contract=original_path.as_posix(),dataset=(project/'Dataset_WigSig/ManySig.pkl').as_posix(),output_root=str(q))
            if variant!='residual_fusion':c['gauge']=gauge_contract(variant)
            resolved=dict(c,steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},U_s_use='unused',target_access=False,
                precision='float32',gradient_clipping=None,optimizer='AdamW+CosineAnnealingLR',loader_seed=seed)
            if variant!='residual_fusion':resolved.update(gauge=c['gauge'],gauge_actual=c['gauge'],gauge_active=True,classifier_scale=30.)
            initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
            accuracy={'residual_fusion':.97,'gauge_peak':.98,'gauge_coherent':.981}[variant]
            done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=accuracy,source_val_worst_rx=.95,source_val_rx_accuracy={str(rx):.95 for rx in [1,3,4,6,8]}))
            for n,d in [('completion.json',done),('initialization.json',initial),('source_contract.json',case['contract']),('resolved_config.json',resolved),('resource_profile.json',dict(total_parameters=164225,conv_linear_macs_per_sample=9708836))]:write(q/n,d)
            torch.save(dict(model=TinyClassifier().state_dict(),epoch=200,source_contract=case['contract'],initialization=initial,selection='fixed_last_epoch',method=c['method'],variant=variant,config=resolved,classes=case['classes'],num_classes=6),q/'last.pt')
            if variant!='residual_fusion':
                cp=case['root']/'source_configs'/(rid+'.json');write(cp,c)
                source_rows.append(dict(row_id=rid,variant=variant,model_seed=seed,source_config=str(cp),source_output=str(q)))
    matrix=dict(rows=source_rows,source_controls=gd.control_rows(),runtime_root=str(new_root));matrix_path=case['root']/'gauge_matrix.json';write(matrix_path,matrix)
    records=[gd.read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]+[gd.read_source_record(r,original,'cvs_gauge_identity') for r in matrix['rows']]
    selection=dict(gd.select_source_candidate(records),scope='gauge_source',source_matrix_ref=str(matrix_path));path=case['root']/'gauge_selection.json';write(path,selection)
    for seed in sorted(contracts.SEEDS):
        variant=selection['selected_variant'];rid=variant+'-s'+str(seed);out=Path(case['spec']['runtime_root'])/rid/'prediction'
        cfg=contracts.read(case['spec']['rows'][0]['config']);cfg.update(variant=variant,model_seed=seed,selection_file=str(path),gauge_source_root=str(new_root),source_output=str(new_root/rid/'source'),output_root=str(out),source_contract=str(original_path))
        cp=case['root']/'configs'/(rid+'.json');write(cp,cfg);case['spec']['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=str(cp),output_root=str(out)))
    case['selection']=selection;case['spec']['selection_file']=str(path)
    return matrix


def test_gauge_freeze_recomputes_control_and24row_truth_last(case,monkeypatch):
    gauge_case(case,monkeypatch);dispatch.validate_spec(case['spec']);prediction_fixtures(case)
    assert contracts.frozen_selection(case['spec']['selection_file'])['selected_variant']=='gauge_coherent'
    opened=[];original=score.read;monkeypatch.setattr(score,'read',lambda p:(opened.append(str(p)),original(p))[1])
    marker=score.score(case['spec']);assert marker['rows']==24 and marker['models']==6
    at=opened.index(case['spec']['p1_truth']);assert sum(Path(p).name=='clean_complete.json' for p in opened[:at])==24


def test_gauge_prediction_payload_activation_and_nonselected_guard(case,monkeypatch):
    gauge_case(case,monkeypatch);cfg=contracts.read(case['spec']['rows'][-1]['config']);q=Path(cfg['source_output'])
    monkeypatch.setattr(predict,'build_model',lambda v:TinyClassifier());predict.predict(cfg)
    assert contracts.read(Path(cfg['output_root'])/'clean_complete.json')['truth_read'] is False
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(cfg,variant='gauge_peak'),case['selection'])
    payload=torch.load(q/'last.pt',weights_only=False)
    args=[cfg,contracts.read(q/'completion.json'),contracts.read(q/'initialization.json'),case['contract'],contracts.read(cfg['source_contract']),contracts.read(q/'resolved_config.json'),payload]
    bad=list(args);bad[5]=dict(args[5],gauge_active=False)
    with pytest.raises(ValueError,match='phase gauge activation'):contracts.checkpoint_contract(*bad)
    bad=list(args);bad[-1]=dict(payload,method='cvs_residual_identity')
    with pytest.raises(ValueError):contracts.checkpoint_contract(*bad)


def test_gauge_control_corruption_and_ranking_tampering_close_query(case,monkeypatch):
    matrix=gauge_case(case,monkeypatch);path=Path(case['spec']['selection_file'])
    write(path,dict(case['selection'],selected_variant='gauge_peak'))
    with pytest.raises(ValueError):contracts.frozen_selection(path)
    write(path,case['selection']);q=Path(matrix['source_controls'][0]['source_output'])/'initialization.json'
    write(q,dict(contracts.read(q),target_contact=True))
    with pytest.raises(ValueError):contracts.frozen_selection(path)


def test_gauge_baseline_wins_blocks_all_new_query(case,monkeypatch):
    from experiments.cvs_gauge_identity import dispatch as gd
    matrix=gauge_case(case,monkeypatch);original=contracts.read(Path(gd.PROJECT)/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')
    for row in matrix['source_controls']:
        q=Path(row['source_output'])/'completion.json';d=contracts.read(q);d['final_source_metrics']['source_val_accuracy']=.999;write(q,d)
    records=[gd.read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]+[gd.read_source_record(r,original,'cvs_gauge_identity') for r in matrix['rows']]
    selection=dict(gd.select_source_candidate(records),scope='gauge_source',source_matrix_ref=case['selection']['source_matrix_ref']);assert selection['new_candidate_selected'] is False
    write(Path(case['spec']['selection_file']),selection)
    with pytest.raises(ValueError,match='baseline retained'):contracts.frozen_selection(case['spec']['selection_file'])
    from experiments.cvs_gauge_clean import prepare
    with pytest.raises(ValueError,match='No selected new gauge'):prepare.main(selection)


def test_gauge_prepare_only_selected4_fixed_capsule(case,monkeypatch):
    from experiments.cvs_gauge_clean import prepare
    from experiments.cvs_clean_eval.prepare import CAPSULE
    root=Path(__file__).resolve().parents[1];prefix='experiments/cvs_selected_clean/configs/'
    for name in ('launch_spec.json','experiment_spec.json'):write(case['root']/prefix/name,contracts.read(root/prefix/name))
    monkeypatch.setattr(prepare,'ROOT',case['root'])
    prepare.main(dict(status='SOURCE_SELECTION_FROZEN',target_access=False,target_score_used=False,selected_variant='gauge_coherent',new_candidate_selected=True))
    folder=case['root']/'experiments/cvs_gauge_clean/configs';runtime=contracts.read(folder/'launch_spec.json');new=[r for r in runtime['rows'] if not r.get('reuse_from_run')]
    assert len(new)==4 and len(runtime['rows'])==24
    for row in new:
        cfg=contracts.read(folder/(row['row_id']+'.json'));assert cfg['p1_capsule']==CAPSULE
        contracts.validate_predict_config(cfg,contracts.read(folder/'frozen_selection.json'))
