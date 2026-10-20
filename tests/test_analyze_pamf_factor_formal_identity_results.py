import copy,json
from pathlib import Path
import pytest
from experiments.pamf_factor_formal_identity import design as d
from tools.analyze_pamf_factor_formal_identity_results import completed_metrics,METHOD


def fixture():
    groups=[('overall','all',168000)]+[('TX',str(k),28000) for k in range(6)]+[('RX','rx'+str(k),24000) for k in range(7)]+[('day','day'+str(k),42000) for k in range(4)]
    records=[];controls=[];pairs=[]
    for a in d.CANDIDATES:
        for group in d.GROUPS:
            for view in d.VIEWS:
                for dim,key,n in groups:
                    stat=dict(count=n,valid_count=n,zero_query_count=0,mean_same_TX_cosine=.4,mean_other_TX_cosine=.1,mean_margin=.3,positive_margin_percent=85.)
                    records.append(dict(candidate=a,group=group,view=view,dimension=dim,value=key,baseline=dict(stat),restored=dict(stat),clean=dict(stat),paired_delta=dict(count=n),paired_descriptor_error=dict(count=n,raw=.1,restored=.08)))
                for readout in ('raw','restored','clean'):
                    controls.append(dict(candidate=a,group=group,view=view,readout=readout,count=168000,permutations=199,seed=2026102034,
                        reference_labels_fixed=True,is_p_value=False,observed_mean=.3,null_mean=0.,null_p05=-.01,null_p95=.01,above_null_p95=True))
    for a,b in d.COMPARISONS:
        for group in d.GROUPS:
            for view in d.VIEWS:pairs.append(dict(candidate=a,control=b,group=group,view=view,common_count=168000,margin_change=.0,descriptor_error_change=.0))
    summary=dict(status='ANALYZED',run=d.RUN,count=d.COUNT,views=list(d.VIEWS),groups=d.GROUPS,candidates=list(d.CANDIDATES),
        configuration=d.configuration(METHOD),records=records,query_label_negative_controls=controls,comparisons=pairs,
        all_features_before_labels=True,target_truth_reads=1,target_truth_for_evaluation_only=True,F_access=0,F_weights_read=0,F_embeddings_logits_read=0,
        model_checkpoint_loads=0,extra_model_predictions=0,neural_training_updates=0,head_fit_updates=0,new_calibration_samples=0,
        parent_training_or_decisions_changed=False,source_selection_changed=False,new_source_thresholds=False,target_decisions_changed=False,selected_candidate=None)
    return copy.deepcopy(dict(status='COLLECTED',failure=None,actual_live_owned_PIDs=[],complete=dict(status='ANALYZED'),launch=dict(commit=METHOD),summary=summary,config=d.configuration(METHOD),
        feature_complete=dict(status='FEATURES_COMPLETE',count=d.COUNT,views=list(d.VIEWS),groups=d.GROUPS,readouts=list(d.READOUTS),cells=280,
        TX_labels_read=0,F_access=0,model_checkpoint_loads=0,extra_model_predictions=0,new_calibration_samples=0,reused_feature_cells=56,new_feature_cells=224,analytic_transform_samples=9408000),
        calibration_reference=dict(run=d.REFERENCE,commit=d.REFERENCE_COMMIT,count=8192,L_count=1000,U_count=7192,new_calibration_samples=0,new_parameters_fitted=0),
        smoke=dict(status='PASS',source_Ls_IQ=2,model_checkpoint_loads=0,F_access=0,TX_labels_read=0,V_reads=0,target_reads=0,parameter_updates=0),
        timing=dict(analytic_extract_seconds=1.,score_process_seconds=2.,total_seconds_including_parent_wait=3.)))


def test_complete_fixed_matrix_is_descriptive_and_never_goal_completion():
    result=completed_metrics(fixture())
    assert (len(result['records']),len(result['query_label_negative_controls']),len(result['comparisons']))==(4536,756,364)
    assert not result['goal_complete'] and result['selected_candidate'] is None
    assert result['verification']['additional_truth_IQ_F_reads']==0


@pytest.mark.parametrize('bad',['live','F','truthbefore','featurecost','calibration','parentbudget','missingrow','duplicaterow','margin',
    'permutations','pvalue','inventedzero','missingpair','nancontrol','groupcount','pairnan','missingcontrol'])
def test_incomplete_permissions_or_incorrect_arithmetic_cannot_be_reported_complete(bad):
    value=fixture();s=value['summary']
    if bad=='live':value['actual_live_owned_PIDs']=[123]
    elif bad=='F':s['F_access']=1
    elif bad=='truthbefore':s['all_features_before_labels']=False
    elif bad=='featurecost':value['feature_complete']['new_feature_cells']=196
    elif bad=='calibration':value['calibration_reference']['new_calibration_samples']=1
    elif bad=='parentbudget':s['configuration']['rows'][d.NEW_ROWS[0]]['updates']=4940
    elif bad=='missingrow':s['records'].pop()
    elif bad=='duplicaterow':s['records'][-1]=s['records'][0]
    elif bad=='margin':s['records'][0]['restored']['mean_margin']=.5
    elif bad=='permutations':s['query_label_negative_controls'][0]['permutations']=19
    elif bad=='pvalue':s['query_label_negative_controls'][0]['is_p_value']=True
    elif bad=='inventedzero':s['records'][0]['restored']['valid_count']=0
    elif bad=='missingpair':s['comparisons'].pop()
    elif bad=='nancontrol':s['query_label_negative_controls'][0]['null_mean']=float('nan')
    elif bad=='groupcount':s['records'][0]['restored']['count']=1800
    elif bad=='pairnan':s['comparisons'][0]['margin_change']=float('nan')
    else:s['query_label_negative_controls'].pop()
    with pytest.raises(ValueError):completed_metrics(value)


def test_report_preserves_all_versions_scenes_groups_and_claim_boundary(tmp_path,monkeypatch):
    from tools import complete_pamf_factor_formal_identity as delivery
    local=tmp_path/'local';work=tmp_path/'work';folder=local/'automation_reports/CV-SincNet'/d.RUN;folder.mkdir(parents=True)
    (local/'docs').mkdir();(work/'docs').mkdir(parents=True)
    (folder/'experiment.json').write_text('{"execution":{}}',encoding='utf-8');(folder/'events.jsonl').write_text('',encoding='utf-8')
    monkeypatch.setattr(delivery,'WORK',work);result=completed_metrics(fixture());delivery.report(local,local/'artifacts',result,record_events=False)
    text=(folder/'report.md').read_text(encoding='utf-8')
    assert all('|'+view+'|'+name+'|' in text for view in d.VIEWS for name in d.CANDIDATES)
    assert '不是分类准确率或逐细节真实性证明' in text and '整体目标未完成' in text
    assert len((folder/'row_metrics.csv').read_text(encoding='utf-8').splitlines())==4537
    assert json.loads((folder/'summary.json').read_text(encoding='utf-8'))['verification']['paired_comparisons']==364
