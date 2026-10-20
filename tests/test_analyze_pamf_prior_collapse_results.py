import copy
import pytest
from experiments.pamf_r46_prior_collapse import design as d
from tools import analyze_pamf_prior_collapse_results as a
from test_analyze_pamf_r46_cuda_graph_results import summary,agreement


def audit():
    return dict(mode='prior_only_linear_pair_collapse',before_parameters=278180,after_parameters=299297,
        factor_modules_replaced=84,unique_kernel_pairs=28,context_unchanged=True,normalization_unchanged=True,
        source_model_unchanged=True,new_training_updates=0,IQ_reads=0,F_reads=0,TX_labels_read=0,seconds=.1)


def profile(arm):
    results={}
    for name,size,ms in [('original_R17',4665073,20.),('parent_R46_350521',350521,8.),('lowrank_R22',278180,10.),('compact',299297,6.)]:
        cases={}
        for mode in ('G_only','G_plus_F'):
            for batch in (1,32):
                cases[mode+'_B'+str(batch)]=dict(batch=batch,rounds=3,warmup_per_round=16 if batch==1 else 4,
                    iterations_per_round=64 if batch==1 else 16,source_V_unique_IQ=64,median_ms=ms,p90_ms=ms,p95_ms=ms,
                    throughput_samples_per_second=1000*batch/ms,peak_allocated_bytes=30,resident_before_forward_bytes=20,
                    incremental_peak_bytes=10,round_median_ms=[ms]*3)
        results[name]=dict(parameters=size,ConvLinear_MAC_per_IQ=35000000 if name=='compact' else 29727404,cases=cases)
    return dict(status='PROFILE_COMPLETE',physical_GPU=d.GPUS[arm],includes_full_context_3prior_6CG=True,
        checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,source_V_IQ=64,TX_labels_read=0,state_updates=0,training_F_access=False,results=results)


def collection():
    rows={};smoke_arms={}
    for arm in d.ARMS:
        views={v:dict(restorer_close=True,strict_reload_exact=True,F_equivalence_gate=False,argmax_disagreements=0,
            outputs={k:dict(max_abs=0.,relative_L2=0.,elementwise_close=True) for k in ('IQ','variance','features','logits')}) for v in ('clean','received01')}
        smoke_arms[arm]=dict(parameters=299297,source_parameters=278180,conversion=audit(),views=views,strict_reload_exact=True,
            source_unchanged=True,new_training_updates=0,training_F_access=False)
        row=dict(complete=dict(status='ANALYZED',arm=arm,all_predictions_before_truth=True,target_feedback=False,new_training_updates=0),failure=None,
            checkpoint_source=dict(source_run=d.COMPLEX_RUN,source_row=d.SOURCE_ROWS[arm],source_commit=d.COMPLEX_COMMIT,
                inherited_epoch=20,inherited_updates=19700,new_training_updates=0,source_parameters=278180,parameters=299297),
            runtime_ready=dict(status='FROZEN_RUNTIME_READY',arm=arm,parameters=299297,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0),
            collapse=audit(),profile=profile(arm),checkpoint_bytes=1500000)
        for role in ('source','target'):
            row[role+'_summary']=summary(role);row['native_'+role+'_summary']=summary(role);row[role+'_agreement']=agreement(role)
            row[role+'_prediction_complete']=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role=role,count=1800 if role=='source' else 168000,
                views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED,
                original_default_head=True,native_fallback=False,batch=128,execution='prior_collapsed_native_FP32',setup=[])
        rows[arm]=row
    return dict(run_id=d.RUN,method_commit=a.METHOD,complete=dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS)),failure=None,
        actual_live_owned_PIDs=[],controller_smoke=dict(status='PASS',real_L_s_IQ=2,source_views=['clean','received01'],arms=smoke_arms,
            TX_labels_read=0,V_reads=0,target_reads=0,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0),rows=rows,
        baselines={k:{r:summary(r) for r in ('source','target')} for k in ('R17','R46')})


def test_all_fixed_versions_have_deployed_size_and_two_cost_comparisons():
    got=a.completed_metrics(collection());row=got['rows'][d.ARMS[0]]
    assert row['parameters']==299297 and row['new_training_updates']==0
    assert row['agreement']['target']['all_argmax_equal']
    assert row['profile']['comparisons']['G_plus_F_B1']['vs_lowrank_R22']['median_speedup']==pytest.approx(10/6)
    assert row['profile']['comparisons']['G_plus_F_B1']['vs_parent_R46']['median_speedup']==pytest.approx(8/6)
    assert got['long_term_goal_complete'] is False


@pytest.mark.parametrize('bad',('live','missing_row','old_commit','F_knowledge','new_training','parameters','missing_scene','changed_batch','wrong_source','smoke_IQ','reload','collapse','false_native_equality'))
def test_invalid_delivery_does_not_become_complete(bad):
    raw=collection();arm=d.ARMS[0];r=raw['rows'][arm]
    if bad=='live':raw['actual_live_owned_PIDs']=[1]
    elif bad=='missing_row':del raw['rows'][d.ARMS[-1]]
    elif bad=='old_commit':raw['method_commit']='0686ed878fe16c1eb53995749fa9fb41cd1e3371'
    elif bad=='F_knowledge':raw['controller_smoke']['arms'][arm]['training_F_access']=True
    elif bad=='new_training':r['checkpoint_source']['new_training_updates']=1
    elif bad=='parameters':r['runtime_ready']['parameters']=278180
    elif bad=='missing_scene':r['target_prediction_complete']['views']=list(d.VIEWS[:-1])
    elif bad=='changed_batch':r['target_prediction_complete']['batch']=256
    elif bad=='wrong_source':r['checkpoint_source']['source_row']='complex625_identity'
    elif bad=='smoke_IQ':raw['controller_smoke']['arms'][arm]['views']['clean']['restorer_close']=False
    elif bad=='reload':raw['controller_smoke']['arms'][arm]['views']['clean']['strict_reload_exact']=False
    elif bad=='collapse':r['collapse']['unique_kernel_pairs']=84
    else:r['native_target_summary']=summary('target',wrong=True)
    with pytest.raises(ValueError):a.completed_metrics(raw)


def test_F_numerical_difference_is_reported_without_source_selection():
    raw=collection();v=raw['controller_smoke']['arms'][d.ARMS[0]]['views']['clean']['outputs']['features']
    v.update(max_abs=1e-5,relative_L2=1e-6,elementwise_close=False)
    got=a.completed_metrics(raw)
    assert got['rows'][d.ARMS[0]]['smoke']['views']['clean']['outputs']['features']['elementwise_close'] is False


@pytest.mark.parametrize('bad',('missing_lowrank','parameters','rounds','memory','MAC'))
def test_cost_matrix_cannot_hide_expanded_weights_or_missing_controls(bad):
    p=profile(d.ARMS[0]);r=p['results']['compact'];c=r['cases']['G_only_B1']
    if bad=='missing_lowrank':del p['results']['lowrank_R22']
    elif bad=='parameters':r['parameters']=278180
    elif bad=='rounds':c['round_median_ms']=[6.]
    elif bad=='memory':c['incremental_peak_bytes']=-1
    else:r['ConvLinear_MAC_per_IQ']=20000000
    with pytest.raises(ValueError):a.profile_metrics(p,d.ARMS[0])


def test_completed_report_contains_all_versions_views_costs_and_full_group_csv(tmp_path):
    import csv,json
    from tools.complete_pamf_prior_collapse import report
    analysis=a.completed_metrics(collection());local=tmp_path/'local';work=tmp_path/'work';output=tmp_path/'artifacts'
    folder=local/'automation_reports/CV-SincNet'/d.RUN;folder.mkdir(parents=True)
    (folder/'report.md').write_text('# Frozen prior collapse\n状态：LOCAL_VERIFIED。\n',encoding='utf-8')
    (folder/'experiment.json').write_text(json.dumps(dict(execution={})),encoding='utf-8')
    (folder/'events.jsonl').write_text('{}\n',encoding='utf-8')
    models={arm:dict(bytes=1500000,components=dict(coarse=106658,action=80045,prior=112594)) for arm in d.ARMS}
    report(local,output,analysis,dict(models=models),record_events=False,worktree=work)
    body=(folder/'report.md').read_text(encoding='utf-8')
    assert all(arm in body for arm in d.ARMS) and all(view in body for view in d.VIEWS)
    assert '299297' in body and '278180' in body and '350521' in body
    records=list(csv.DictReader((folder/'row_metrics.csv').open(encoding='utf-8',newline='')))
    assert len([r for r in records if r['role']=='target' and r['dimension']=='all'])==14
    assert {r['dimension'] for r in records}=={'all','TX','RX','day'}
    assert json.loads((folder/'experiment.json').read_text(encoding='utf-8'))['scientific_outcome']['long_term_goal_complete'] is False


def test_pending_actual_owned_process_never_starts_collection_or_weights(tmp_path,monkeypatch):
    import json
    from tools import complete_pamf_prior_collapse as c
    (tmp_path/'deployment.json').write_text(json.dumps(dict(startup=dict(run='/immutable/run',pid=123))),encoding='utf-8')
    monkeypatch.setattr(c,'remote',lambda *args:dict(status='PENDING',failure=None,actual_live_owned_PIDs=[123]))
    assert c.collect(tmp_path,tmp_path) is None
    assert not (tmp_path/'analysis.json').exists() and not (tmp_path/'models').exists()
