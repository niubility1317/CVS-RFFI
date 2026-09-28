"""Full MVKME audit from synthetic fit logs only; no SSH or target artifacts."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import collect_d92_mvkme_audit as tool
from cvsrffi.stage2_d92_mv_kme import FROZEN_CONFIG,fit_mv_kme


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=True),encoding='utf-8')


def lines(path,values):
    path.write_text(''.join(json.dumps(v,allow_nan=True)+'\n' for v in values),encoding='utf-8')


@pytest.fixture(scope='module')
def measured():
    result=[]
    with threadpool_limits(limits=1):
        for k in [1,2,5,20]:
            c=4;classes=['old-a','old-b','new-a','new-b']
            labels=np.repeat(np.arange(c),k);ids=[f'physical-{i:04d}' for i in range(c*k)]
            blocks=np.random.default_rng(782+k).normal(size=(c*k,3,256))/20
            state=fit_mv_kme(support_blocks=blocks,support_labels=labels,support_ids=ids,classes=classes,old_classes=classes[:2])
            trace=state.audit_dict();lookup={id_:classes[int(label)] for id_,label in zip(ids,labels)}
            for entry in trace['physical_fold_assignment']:entry['class_id']=lookup[entry['physical_id']]
            trace.update(split_id=f'synthetic-k{k}',registered_classes=classes,old_classes=classes[:2],
                receiver='rx',scenario='scene',support_seed=701,new_count=2,
                fit_call_seconds=trace['fit_seconds']+.001,query_score_seconds=.02,prediction_write_seconds=.01)
            result.append(trace)
    return result


def make_run(tmp_path,measured):
    root=tmp_path/'run';root.mkdir();rows=[]
    for model in (1,2):
        row=dict(row_id=f'model-{model}',output_root=str(root/f'model-{model}'),expected_checkpoint_sha256='a'*64,seeds=dict(model=model))
        rows.append(row);out=Path(row['output_root']);folder=out/'mvkme';feature=out/'mv_features'
        folder.mkdir(parents=True);feature.mkdir()
        (feature/'received_mv_features.npz').write_bytes(b'metadata-audit-must-not-read-this')
        size=(feature/'received_mv_features.npz').stat().st_size
        payload=dict(new_ground_statistics_bytes=0,model_file_bytes=7654321,
                     received_feature_array_bytes=100*3*256*4,received_feature_file_bytes=size)
        write(feature/'features_complete.json',dict(status='MULTIVIEW_FEATURES_COMPLETE',schema='d92_mvkme_received_blocks_v1',
            capsule_id='capsule',checkpoint_sha256='a'*64,algorithm=FROZEN_CONFIG,dtype='float32',classes=['old-a','old-b'],model_seed=model,
            query_used_for_fitting=False,source_data_access=False,truth_read=False,encoder_updated=False,native_eval=True,
            native_buffers_unchanged=True,view_count_per_observation=16,new_ground_statistics_bytes=0,count=100,shape=[100,3,256],
            feature_array_bytes=100*3*256*4,feature_file_bytes=size,model_file_bytes=7654321,feature_seconds=1.25,
            peak_process_rss_bytes=9876543))
        write(feature/'checkpoint_provenance.json',dict(checkpoint_sha256='a'*64,model_file_bytes=7654321))
        write(folder/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',split_count=4,predictions=4,capsule_id='capsule',
            truth_read=False,source_data_access=False,new_ground_statistics_bytes=0,payload_audit=payload,peak_process_rss_bytes=12345678))
        write(folder/'startup.json',dict(config=dict(algorithm=FROZEN_CONFIG),payload_audit=payload,query_fit_access=False,
            truth_read=False,source_data_access=False,ground_summary_access=False,cross_row_adapted_state_reuse=False))
        lines(folder/'fit_trace.jsonl',measured)
        compacts=[]
        for index,trace in enumerate(measured):
            small={key:trace[key] for key in ('split_id','k','classes','selected','selection','candidate_count','fold_count',
                'selected_objective','selected_macro_nll','selected_old_nll','selected_new_nll','fit_seconds','head_bytes',
                'fourier_matrix_bytes','persistent_state_bytes','receiver','scenario','support_seed','new_count')}
            small['fit_call_seconds']=trace['fit_call_seconds']
            small.update(query_rows_used_for_fit=0,source_rows_used_for_fit=0,learning_rate=None,gradient=None,source_validation=None,
                source_validation_reason='No source input',unavailable_reason='Closed form',query_score_seconds=.02,prediction_write_seconds=.01,
                total_seconds=trace['fit_seconds']+.04,peak_process_rss_bytes=12345678,completed=index+1,total=4)
            compacts.append(small)
        lines(folder/'compact.jsonl',compacts)
    write(root/'complete.json',dict(status='SCORED'))
    write(root/'state.json',{r['row_id']:dict(status='PREDICTIONS_COMPLETE') for r in rows})
    return dict(root=str(root),rows=rows,splits=4,receivers=['rx'],scenarios=['scene'],ks=[1,2,5,20],new_counts=[2],
                support_seeds=[701],old_classes=['old-a','old-b'],algorithm=copy.deepcopy(FROZEN_CONFIG),capsule_id='capsule')


def test_complete_fit_resources_without_data_or_score_reads(tmp_path,measured,monkeypatch):
    config=make_run(tmp_path,measured);original=Path.open;opened=[]
    def guard(path,*args,**kwargs):
        assert path.name not in ('scores.json','truth.json','predictions.jsonl','received_mv_features.npz','received.npz','final_ssdg.pth')
        opened.append(path);return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    result=tool.audit_run(config)
    assert result['status']=='VERIFIED' and result['total_fits']==8
    assert result['fixed_k1_fits']==2 and result['support_cv_fits']==6
    assert result['new_source_payload_bytes']==0 and result['ground_summary_used'] is False
    assert result['parameter_feedback_forbidden'] and result['raw_traces_preserved']
    for model in result['models']:
        assert [g['folds_per_fit'] for g in model['per_k']]==[0,2,3,3]
        assert [g['candidates_per_fit'] for g in model['per_k']]==[0,9,9,9]
        assert model['resources']['received_cache_array_bytes']==307200
        assert model['resources']['existing_frozen_model_file_bytes']==7654321
        assert model['resources']['model_incremental_transfer_bytes'] is None
        assert model['resources']['feature_extraction_seconds']==1.25
        assert model['timing']['query_score_seconds']['total']==pytest.approx(.08)
        assert model['numeric_state_byte_ranges']['fourier_matrix_bytes']['min']==262144
    json.dumps(result,allow_nan=False);assert opened


@pytest.mark.parametrize('fault',['missing','duplicate','nonfinite','candidate','fold','class_fold','physical_duplicate','k1',
    'selected','head_bytes','query','source','summary','time','cell','compact','cache_bytes','model_bytes','nonterminal'])
def test_invalid_evidence_fails_closed(tmp_path,measured,fault):
    config=make_run(tmp_path,measured);row=Path(config['rows'][0]['output_root']);folder=row/'mvkme'
    traces=list(tool.json_lines(folder/'fit_trace.jsonl'));small=list(tool.json_lines(folder/'compact.jsonl'))
    if fault=='missing':traces.pop()
    elif fault=='duplicate':traces[-1]=copy.deepcopy(traces[0])
    elif fault=='nonfinite':traces[1]['steps'][0]['objective']=float('nan')
    elif fault=='candidate':traces[1]['candidates'].pop()
    elif fault=='fold':traces[1]['steps'][0]['train_k']+=1
    elif fault=='class_fold':traces[1]['physical_fold_assignment'][0]['class_id']='old-b'
    elif fault=='physical_duplicate':traces[1]['physical_fold_assignment'][1]['physical_id']=traces[1]['physical_fold_assignment'][0]['physical_id']
    elif fault=='k1':traces[0]['selected']['gamma']=.01;small[0]['selected']['gamma']=.01
    elif fault=='selected':traces[1]['selected_objective']+=1;small[1]['selected_objective']+=1
    elif fault=='head_bytes':traces[1]['head_bytes']+=8
    elif fault=='query':traces[1]['query_rows_used_for_fit']=1
    elif fault=='source':traces[1]['source_rows_used_for_fit']=1
    elif fault=='summary':traces[1]['ground_summary_used']=True
    elif fault=='time':small[1]['query_score_seconds']=1000
    elif fault=='cell':traces[1]['receiver']='other';small[1]['receiver']='other'
    elif fault=='compact':small.pop()
    elif fault=='cache_bytes':
        path=row/'mv_features/features_complete.json';value=tool.read_json(path);value['feature_file_bytes']+=1;write(path,value)
    elif fault=='model_bytes':
        path=row/'mv_features/checkpoint_provenance.json';value=tool.read_json(path);value['model_file_bytes']+=1;write(path,value)
    else:write(Path(config['root'])/'complete.json',dict(status='PREDICTIONS_COMPLETE'))
    lines(folder/'fit_trace.jsonl',traces);lines(folder/'compact.jsonl',small)
    with pytest.raises(ValueError):tool.audit_run(config)


def test_scored_guard_precedes_fit_access(tmp_path,measured,monkeypatch):
    config=make_run(tmp_path,measured)
    write(Path(config['root'])/'complete.json',dict(status='RUNNING'))
    monkeypatch.setattr(tool,'json_lines',lambda *_:pytest.fail('No trace before SCORED'))
    with pytest.raises(ValueError,match='SCORED'):tool.audit_run(config)


def test_old_only_registration_keeps_new_nll_null(tmp_path,measured):
    config=make_run(tmp_path,measured)
    classes=['old-a','old-b'];k=5;labels=np.repeat(np.arange(2),k)
    ids=[f'old-physical-{i}' for i in range(10)]
    with threadpool_limits(limits=1):
        state=fit_mv_kme(support_blocks=np.random.default_rng(52).normal(size=(10,3,256))/20,
            support_labels=labels,support_ids=ids,classes=classes,old_classes=classes)
    trace=state.audit_dict()
    lookup={identifier:classes[int(label)] for identifier,label in zip(ids,labels)}
    for entry in trace['physical_fold_assignment']:entry['class_id']=lookup[entry['physical_id']]
    trace.update(split_id='old-only',registered_classes=classes,old_classes=classes,receiver='rx',scenario='scene',
        support_seed=701,new_count=0,fit_call_seconds=trace['fit_seconds']+.001,query_score_seconds=.02,prediction_write_seconds=.01)
    template=list(tool.json_lines(Path(config['rows'][0]['output_root'])/'mvkme/compact.jsonl'))[0]
    compact={key:trace[key] if key in trace else value for key,value in template.items()}
    compact['total_seconds']=trace['fit_seconds']+.04
    config['new_counts']=[0]
    tool.validate_fit(trace,compact,config)
    assert trace['selected_new_nll'] is None


def test_remote_script_same_parser_locally(tmp_path,measured,capsys):
    config=make_run(tmp_path,measured)
    exec(compile(tool.remote_script(config),'synthetic_only','exec'),{})
    value=json.loads(capsys.readouterr().out)
    assert value['status']=='VERIFIED' and value['total_fits']==8


def test_existing_output_rejected_before_remote(tmp_path,monkeypatch):
    out=tmp_path/'existing.json';out.write_text('preserve',encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['audit','--spec','unused.json','--output',str(out)])
    monkeypatch.setattr(tool.subprocess,'run',lambda *a,**k:pytest.fail('No remote action'))
    with pytest.raises(FileExistsError):tool.main()
    assert out.read_text()=='preserve'
