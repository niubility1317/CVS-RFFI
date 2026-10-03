"""Complete synthetic physical-count matrix and independent terminal readback."""
import copy
import json
import statistics
import numpy as np
import pytest

from experiments.cvs_crosspath_relation_clean import analyze as report, collect
from experiments.cvs_crosspath_relation_clean.analyze import BASELINES, SEEDS, validate_results
from experiments.cvs_crosspath_relation_clean.prepare import SOURCE, RUN
from experiments.cvs_crosspath_relation_identity.model import VARIANTS, readout_contract
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

RX=('1-1','14-7','2-1','20-1','7-14','7-7','8-8')


def metrics(cm):
    cm=np.asarray(cm);tp=np.diag(cm);n=int(cm.sum());den=cm.sum(0)+cm.sum(1)
    return dict(confusion=cm.tolist(),query_count=n,accuracy=float(tp.sum()/n),
        macro_accuracy=float((tp/cm.sum(1)).mean()),macro_f1=float((2*tp/den).mean()))


def source_fixture(candidate=VARIANTS[0]):
    """Synthetic complete source metadata; does not claim actual training occurred."""
    controls=[dict(variant=v,seed=s,accuracy=.7,worst_rx=.6,parameters=220987,macs=900,
                   source_output='synthetic_prior_source') for v in report.SOURCE_CANDIDATES[:2] for s in SEEDS]
    rows=[]
    for variant in VARIANTS:
        architecture=readout_contract(variant);parameters=233275
        for seed in SEEDS:
            source=SOURCE+'/'+variant+'-s'+str(seed)+'/source'
            accuracy=.9 if variant==candidate else .8;worst=.85 if variant==candidate else .75
            profile=dict(total_parameters=parameters,trainable_parameters=parameters,gradient_used_parameters=parameters,
                readout_contract=copy.deepcopy(architecture),resident_state_bytes=parameters*4,
                conv_linear_macs_per_sample=1000,mac_scope='synthetic only',hardware='fixture CPU',
                torch_version='fixture',inference_batch1_ms=1.,training_batch128_ms=2.)
            final=dict(source_val_accuracy=accuracy,source_val_worst_rx=worst)
            rows.append(dict(resolved=dict(variant=variant,model_seed=seed,output_root=source,
                commit=report.SOURCE_COMMIT,cwd=report.PROJECT+'/releases/'+report.SOURCE_RELEASE,
                readout=copy.deepcopy(architecture),readout_actual=copy.deepcopy(architecture),learned_readout_active=True,
                total_parameters=parameters,trainable_parameters=parameters),profile=profile,
                epochs=[dict(epoch=e,peak_cuda_allocated_bytes=200000,**final) for e in range(1,201)],
                completion=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,final_source_metrics=final),
                source_record=dict(variant=variant,seed=seed,accuracy=accuracy,worst_rx=worst,
                    parameters=parameters,macs=1000,source_output=source)))
    actual=report.select_source_candidate(controls+[r['source_record'] for r in rows])
    source=dict(run_id=report.SOURCE_RUN,source_controls=controls,rows=rows,source_selection=actual)
    selection=dict(actual,scope='crosspath_relation_source',source_release_commit=report.SOURCE_COMMIT,source_matrix_ref=collect.SOURCE_MATRIX)
    return source,selection


def fixture(candidate=VARIANTS[0],candidate_error_index=None):
    records=[];terminal=[]
    for index,method in enumerate((*BASELINES,candidate)):
        for j,seed in enumerate(SEEDS):
            rid=f'{method}-s{seed}';overall=np.zeros((6,6),dtype=np.int64)
            for k,rx in enumerate(RX):
                measured_index=4 if method=='response_anchor_mean' else candidate_error_index if method==candidate and candidate_error_index is not None else index
                cm=np.eye(6,dtype=np.int64)*4000;error=10*(measured_index+j+k)
                for c in range(6):cm[c,c]-=error;cm[c,(c+1)%6]+=error
                overall+=cm
                records.append(dict(row_id=rid,method=method,receiver=rx,model_seed=seed,view='clean',**metrics(cm)))
            records.append(dict(row_id=rid,method=method,receiver='ALL',model_seed=seed,view='clean',**metrics(overall)))
            source=SOURCE+'/'+rid+'/source'
            resolved=dict(variant=method,model_seed=seed,views=['clean'],truth_read=False,query_fit=False,source_output=source,
                crosspath_source_root=SOURCE,source_contract=collect.SOURCE_CONTRACT,source_resolved_ref=source+'/resolved_config.json',
                backend_flags=copy.deepcopy(FULL_FP32_POLICY),numerical_policy=copy.deepcopy(FULL_FP32_POLICY))
            terminal.append(dict(row_id=rid,status='PREDICTIONS_COMPLETE',process=None,resolved=resolved,
                completion=dict(status='PREDICTIONS_COMPLETE',count=168000,views=['clean'],truth_read=False,query_fit=False,prediction_seconds=10.,peak_cuda_allocated_bytes=123456),
                provenance=dict(status='VERIFIED',query_fit=False,source_contract_roles='EXACT_MATCH',initialization='SCRATCH',ancestors=[],selection='E200',checkpoint=source+'/last.pt',architecture_selection='SOURCE_SELECTION_FROZEN')))
    lookup={(r['method'],r['receiver'],r['model_seed']):r for r in records};summary=[];pairs=[]
    for method in (*BASELINES,candidate):
        for rx in ('ALL',*RX):
            row=dict(method=method,receiver=rx,view='clean',model_seeds=list(SEEDS),query_count_per_seed=168000 if rx=='ALL' else 24000)
            for key in ('accuracy','macro_accuracy','macro_f1'):
                values=[lookup[method,rx,seed][key] for seed in SEEDS]
                row.update({key+'_mean':statistics.mean(values),key+'_seed_sd':statistics.stdev(values)})
            summary.append(row)
    for method in BASELINES:
        for rx in ('ALL',*RX):
            values=[100*(lookup[candidate,rx,seed]['accuracy']-lookup[method,rx,seed]['accuracy']) for seed in SEEDS]
            pairs.append(dict(candidate=candidate,baseline=method,receiver=rx,view='clean',model_seeds=list(SEEDS),accuracy_delta_pp_by_seed=values,
                accuracy_delta_pp_mean=statistics.mean(values),accuracy_delta_pp_seed_sd=statistics.stdev(values),positive_seeds=sum(v>0 for v in values)))
    data=dict(run_id=RUN,pipeline=dict(status='ANALYZED',independent_scoring_complete=True),dispatcher_process=None,classes=collect.CLASSES.copy(),rows=terminal,
        marker=dict(status='SCORED_COMPLETE',rows=48,models=12,seeds=4,records=384,view='clean',query_count=168000,target_feedback_forbidden=True),
        results=dict(results=records),summary=dict(summary=summary,paired=pairs))
    _,selection=source_fixture(candidate)
    previous=dict(results=dict(results=copy.deepcopy([r for r in records if r['method'] in BASELINES])))
    return data,selection,previous


@pytest.mark.parametrize('candidate',VARIANTS)
def test_complete384_matrix_and48_terminal_rows(candidate):
    data,selection,previous=fixture(candidate);lookup,receivers=validate_results(data,selection,previous)
    assert len(lookup)==384 and set(receivers)==set(RX)
    assert len(collect.validate_terminal(data,selection))==48


@pytest.mark.parametrize('case',('missing','duplicate','fractional_cm','metric','nan','RX_partition','old_control','old_anchor','old_duplicate',
    'summary_mean','summary_sd','missing_summary','summary_view','summary_seed','paired_seed','paired_sign','paired_duplicate',
    'target_access','unselected','stale_source','non_clean','classmap','marker_models','row_id','per_tx','source_commit','source_release','control_selected'))
def test_analysis_rejects_corrupt_or_out_of_scope_results(case):
    data,selection,previous=fixture()
    if case=='missing':data['results']['results'].pop()
    elif case=='duplicate':data['results']['results'][-1]=copy.deepcopy(data['results']['results'][0])
    elif case=='fractional_cm':data['results']['results'][0]['confusion']=np.asarray(data['results']['results'][0]['confusion'],dtype=float).tolist()
    elif case=='metric':data['results']['results'][0]['macro_f1']+=.001
    elif case=='nan':data['results']['results'][0]['accuracy']=float('nan')
    elif case in ('RX_partition','per_tx'):
        row=data['results']['results'][0];cm=np.asarray(row['confusion']);cm[0,0]-=1;cm[0 if case=='RX_partition' else 1,1]+=1;row.update(metrics(cm))
    elif case=='old_control':previous['results']['results'][0]['accuracy']+=.001
    elif case=='old_anchor':next(r for r in previous['results']['results'] if r['method']=='response_anchor_mean')['accuracy']+=.001
    elif case=='old_duplicate':previous['results']['results'][-1]=copy.deepcopy(previous['results']['results'][0])
    elif case=='summary_mean':data['summary']['summary'][0]['accuracy_mean']+=.001
    elif case=='summary_sd':data['summary']['summary'][0]['macro_f1_seed_sd']+=.001
    elif case=='missing_summary':data['summary']['summary'].pop()
    elif case=='summary_view':data['summary']['summary'][0]['view']='leo'
    elif case=='summary_seed':data['summary']['summary'][0]['model_seeds']=list(reversed(SEEDS))
    elif case=='paired_seed':data['summary']['paired'][0]['model_seeds']=list(reversed(SEEDS))
    elif case=='paired_sign':data['summary']['paired'][0]['positive_seeds']=4
    elif case=='paired_duplicate':data['summary']['paired'][-1]=copy.deepcopy(data['summary']['paired'][0])
    elif case=='target_access':selection['target_access']=True
    elif case=='unselected':selection['new_candidate_selected']=False
    elif case=='stale_source':selection['candidate_universe'].remove('response_anchor_mean')
    elif case=='non_clean':data['results']['results'][0]['view']='practical_mid'
    elif case=='classmap':data['classes'].reverse()
    elif case=='marker_models':data['marker']['models']=9
    elif case=='row_id':data['results']['results'][0]['row_id']='wrong'
    elif case=='source_commit':selection['source_release_commit']='wrong'
    elif case=='source_release':selection['source_matrix_ref']='wrong'
    elif case=='control_selected':selection['selected_variant']='response_anchor_mean'
    with pytest.raises(ValueError):validate_results(data,selection,previous)


@pytest.mark.parametrize('case',('dispatcher','unknown_process','old_process','missing_row','duplicate_row','missing_completion','provenance',
    'ancestor','truth','classmap','not_analyzed','not_scored','fp32','source_path','wrong_run','old_count','missing_process','missing_dispatcher','stale_source',
    'source_ref','source_contract','source_root','architecture_selection','source_commit','source_release','control_selected'))
def test_terminal_readback_checks_old_and_new_rows(case):
    data,selection,_=fixture()
    if case=='dispatcher':data['dispatcher_process']={'pid':1}
    elif case=='unknown_process':data['rows'][-1]['process']={'state':'UNKNOWN'}
    elif case=='old_process':data['rows'][0]['process']={'pid':2}
    elif case=='missing_row':data['rows'].pop(0)
    elif case=='duplicate_row':data['rows'][-1]=copy.deepcopy(data['rows'][0])
    elif case=='missing_completion':data['rows'][0]['completion']=None
    elif case=='provenance':data['rows'][0]['provenance']['status']='UNKNOWN'
    elif case=='ancestor':data['rows'][0]['provenance']['ancestors']=['old.pt']
    elif case=='truth':data['rows'][0]['resolved']['truth_read']=True
    elif case=='classmap':data['classes'].reverse()
    elif case=='not_analyzed':data['pipeline']['status']='PREDICTING'
    elif case=='not_scored':data['pipeline']['independent_scoring_complete']=False
    elif case=='fp32':data['rows'][-1]['resolved']['backend_flags']['cudnn_allow_tf32']=True
    elif case=='source_path':data['rows'][-1]['resolved']['source_output']='wrong'
    elif case=='wrong_run':data['run_id']='wrong'
    elif case=='old_count':data['rows'][0]['completion']['count']=167999
    elif case=='missing_process':data['rows'][0].pop('process')
    elif case=='missing_dispatcher':data.pop('dispatcher_process')
    elif case=='stale_source':selection['candidate_universe'].remove('response_anchor_mean')
    elif case=='source_ref':data['rows'][-1]['resolved']['source_resolved_ref']='wrong'
    elif case=='source_contract':data['rows'][-1]['resolved']['source_contract']='wrong'
    elif case=='source_root':data['rows'][-1]['resolved']['crosspath_source_root']='wrong'
    elif case=='architecture_selection':data['rows'][-1]['provenance']['architecture_selection']='UNKNOWN'
    elif case=='source_commit':selection['source_release_commit']='wrong'
    elif case=='source_release':selection['source_matrix_ref']='wrong'
    elif case=='control_selected':selection['selected_variant']='response_anchor_mean'
    with pytest.raises(ValueError):collect.validate_terminal(data,selection)


@pytest.mark.parametrize('candidate',VARIANTS)
def test_full_report_resources_and_shared_actual_parameter_count(tmp_path,candidate):
    data,selection,previous=fixture(candidate,candidate_error_index=6)
    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
    e=tmp_path/'automation_reports/CV-SincNet'/report.RUN/'evidence'
    save(e/'final_readback.json',data);save(e/'performance_selection.json',selection)
    save(tmp_path/'automation_reports/CV-SincNet'/report.OLD_RUN/'evidence/final_readback.json',previous)
    parameters=233275;source,_=source_fixture(candidate)
    save(tmp_path/'automation_reports/CV-SincNet'/report.SOURCE_RUN/'evidence/source_research_complete.json',
        source)
    verdict=report.analyze(tmp_path)
    assert verdict['all384_confusions_recomputed'] and verdict['old352_metrics_exactly_unchanged']
    assert verdict['all96_summaries_recomputed'] and verdict['all88_pairs_recomputed']
    assert verdict['paired_vs_response_anchor']['baseline']=='response_anchor_mean'
    assert verdict['paired_vs_neural_shallow']['accuracy_delta_pp_mean']>0
    assert verdict['paired_vs_response_anchor']['accuracy_delta_pp_mean']<0
    assert verdict['scientific_verdict']=='NO_MEAN_IMPROVEMENT_ON_FIXED_CLEAN_BENCHMARK'
    resources=report.read(e/'resource_summary.json');assert all(r['parameters']==parameters for r in resources['rows'])
    assert len((e/'per_transmitter_summary.csv').read_text(encoding='utf-8').splitlines())==73
    assert '小幅均值改善不证明' in (e.parent/'report.md').read_text(encoding='utf-8')
    text=(e.parent/'report.md').read_text(encoding='utf-8')
    assert 'rank≤16 是架构上界' in text and '新增 12288 个参数' in text
    assert '注意力' not in text and 'attention' not in text
    assert resources['readout_contract']==readout_contract(candidate)


def test_collect_reads_only_verified_source_class_map(tmp_path,monkeypatch):
    data,selection,_=fixture();data.pop('classes');output=tmp_path/'readback';output.mkdir()
    (output/'readback.json').write_text(json.dumps(data),encoding='utf-8')
    config=tmp_path/'experiments/cvs_crosspath_relation_clean/configs';config.mkdir(parents=True)
    (config/'frozen_selection.json').write_text(json.dumps(selection),encoding='utf-8')
    rows=[dict(row_id=r['row_id'],variant=r['resolved']['variant'],model_seed=r['resolved']['model_seed'],output_root=None) for r in data['rows']]
    (config/'launch_spec.json').write_text(json.dumps(dict(rows=rows)),encoding='utf-8')
    monkeypatch.setattr(collect,'inspect',lambda *a,**kw:None);scripts=[]
    def ssh(script):scripts.append(script);return json.dumps(collect.CLASSES)
    monkeypatch.setattr(collect,'ssh',ssh)
    assert collect.collect(tmp_path,output)
    assert len(scripts)==1 and 'source_contract.json' in scripts[0] and 'truth' not in scripts[0]
    assert report.read(tmp_path/'automation_reports/CV-SincNet'/RUN/'evidence/final_readback.json')['classes']==collect.CLASSES


@pytest.mark.parametrize('case',('run','missing_new','duplicate_new','missing_control','duplicate_control','record_seed',
    'record_path','frozen_selection','selected_variant','commit','release','source_path','readout','actual_readout',
    'inactive','parameter_count','gradient_count','profile_contract','missing_epoch','duplicate_epoch','epoch_budget',
    'target','final_score','profile_macs'))
def test_source_resources_reject_nonfrozen_or_inconsistent_metadata(case):
    source,selection=source_fixture();row=next(r for r in source['rows'] if r['resolved']['variant']==VARIANTS[0])
    if case=='run':source['run_id']='wrong'
    elif case=='missing_new':source['rows'].pop()
    elif case=='duplicate_new':source['rows'][-1]=copy.deepcopy(source['rows'][0])
    elif case=='missing_control':source['source_controls'].pop()
    elif case=='duplicate_control':source['source_controls'][-1]=copy.deepcopy(source['source_controls'][0])
    elif case=='record_seed':row['source_record']['seed']=0
    elif case=='record_path':row['source_record']['source_output']='wrong'
    elif case=='frozen_selection':source['source_selection']['source_summaries'][VARIANTS[0]]['score']-=.01
    elif case=='selected_variant':selection['selected_variant']=VARIANTS[1]
    elif case=='commit':row['resolved']['commit']='wrong'
    elif case=='release':row['resolved']['cwd']='wrong'
    elif case=='source_path':row['resolved']['output_root']='wrong'
    elif case=='readout':row['resolved']['readout']['relation_linear_rank']=32
    elif case=='actual_readout':row['resolved']['readout_actual']['relation_pairs_per_lag']=64
    elif case=='inactive':row['resolved']['learned_readout_active']=False
    elif case=='parameter_count':row['profile']['total_parameters']=344891
    elif case=='gradient_count':row['profile']['gradient_used_parameters']-=1
    elif case=='profile_contract':row['profile']['readout_contract']['relation_normalization']='wrong'
    elif case=='missing_epoch':row['epochs'].pop(0)
    elif case=='duplicate_epoch':row['epochs'][0]['epoch']=2
    elif case=='epoch_budget':row['completion']['steps']=9999
    elif case=='target':row['completion']['target_access']=True
    elif case=='final_score':row['epochs'][-1]['source_val_accuracy']-=.001
    elif case=='profile_macs':row['profile']['conv_linear_macs_per_sample']+=1
    with pytest.raises(ValueError):report.validate_source_resources(source,selection)


def test_source_resource_summary_uses_exact_registered_seed_pairs_without_mutation():
    source,selection=source_fixture(VARIANTS[1]);source['rows'].reverse();before=copy.deepcopy(source)
    models,architecture=report.validate_source_resources(source,selection)
    assert source==before and architecture==readout_contract(VARIANTS[1])
    assert {r['resolved']['model_seed'] for r in models}==set(SEEDS)
    assert all(r['resolved']['variant']==VARIANTS[1] for r in models)


def test_collect_rejects_wrong_source_classes_before_artifact_delivery(tmp_path,monkeypatch):
    data,selection,_=fixture();data.pop('classes');output=tmp_path/'readback';output.mkdir()
    (output/'readback.json').write_text(json.dumps(data),encoding='utf-8')
    config=tmp_path/'experiments/cvs_crosspath_relation_clean/configs';config.mkdir(parents=True)
    (config/'frozen_selection.json').write_text(json.dumps(selection),encoding='utf-8')
    rows=[dict(row_id=r['row_id'],variant=r['resolved']['variant'],model_seed=r['resolved']['model_seed'],output_root=None) for r in data['rows']]
    (config/'launch_spec.json').write_text(json.dumps(dict(rows=rows)),encoding='utf-8')
    monkeypatch.setattr(collect,'inspect',lambda *a,**kw:None)
    monkeypatch.setattr(collect,'ssh',lambda script:json.dumps(list(reversed(collect.CLASSES))))
    with pytest.raises(ValueError,match='class map'):collect.collect(tmp_path,output)
    assert not (tmp_path/'automation_reports/CV-SincNet'/RUN/'evidence').exists()
