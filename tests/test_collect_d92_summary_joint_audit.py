"""Audit real core diagnostics from synthetic support only, with no SSH."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import collect_d92_summary_joint_audit as tool
from cvsrffi.d92_ground_summary import load_ground_summary
from cvsrffi.stage2_d92_summary_joint import FROZEN_CONFIG,build_summary_operator,fit_summary_joint
from test_d92_ground_summary import fixture_files,SHA,CLASSES


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=True),encoding='utf-8')


def lines(path,values):
    path.write_text(''.join(json.dumps(v,allow_nan=True)+'\n' for v in values),encoding='utf-8')


@pytest.fixture(scope='module')
def measured(tmp_path_factory):
    root=tmp_path_factory.mktemp('sgjoint_measured_synthetic')
    ground=root/'ground';ground.mkdir();fixture_files(ground)
    summary=load_ground_summary(ground,expected_checkpoint_sha256=SHA,expected_classes=CLASSES,already_deployed=False)
    traces=[]
    with threadpool_limits(limits=1):
        operator=build_summary_operator(summary)
        for k in [1,2,5,20]:
            rng=np.random.default_rng(91+k)
            labels=np.repeat(np.arange(4),k)
            features=rng.normal(size=(4*k,256))
            state=fit_summary_joint(features,labels,[f'physical-{i}' for i in range(4*k)],
                [*CLASSES,'new-0','new-1'],CLASSES,operator)
            traces.append(dict(split_id=f'synthetic-k{k}',**state.audit_dict()))
    return ground,traces


def make_run(tmp_path,measured):
    ground,traces=measured
    root=tmp_path/'run';root.mkdir()
    rows=[];payload=traces[0]['summary']['payload']
    for model in (1,2):
        rid=f'model-{model}';output=root/rid;folder=output/'sgjoint';folder.mkdir(parents=True)
        origin=tmp_path/'existing'/rid;origin.mkdir(parents=True)
        shutil.copytree(ground,origin/'ground')
        rows.append(dict(row_id=rid,output_root=str(output),reuse_row_root=str(origin)))
        write(folder/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',split_count=4,predictions=4,
            capsule_id='synthetic',truth_read=False,source_data_access=False,payload_audit=payload,
            peak_process_rss_bytes=model*1000000))
        write(folder/'startup.json',dict(config=dict(algorithm=FROZEN_CONFIG),payload_audit=payload))
        lines(folder/'fit_trace.jsonl',traces)
        compact=[]
        for trace in traces:
            risk=trace['selected_oof_risk'] or {}
            compact.append(dict(split_id=trace['split_id'],k=trace['k'],classes=trace['class_count'],selected=trace['selected'],
                selection=trace['selection'],fit_seconds=trace['fit_seconds'],persistent_state_bytes=trace['persistent_state_bytes'],
                summary_operator_bytes=trace['summary_operator_bytes'],fold_count=trace['fold_count'],candidate_count=len(trace['candidate_trace']),
                query_rows_used_for_fit=0,source_rows_used_for_fit=0,learning_rate=None,gradient=None,source_validation=None,
                source_validation_reason='User forbids source samples',unavailable_reason='Closed form',
                **{'selected_'+name:risk.get(name) for name in ('objective','macro_nll','old_nll','new_nll')}))
        lines(folder/'compact.jsonl',compact)
    write(root/'complete.json',dict(status='SCORED'))
    write(root/'state.json',{r['row_id']:dict(status='PREDICTIONS_COMPLETE') for r in rows})
    return dict(root=str(root),rows=rows,splits=4,cells_per_k_new=1,ks=[1,2,5,20],new_counts=[2],
                old_classes=list(CLASSES),algorithm=copy.deepcopy(FROZEN_CONFIG),capsule_id='synthetic')


def test_complete_core_traces_summarized_without_query_or_score_access(tmp_path,measured,monkeypatch):
    config=make_run(tmp_path,measured)
    original=Path.open;opened=[]
    def guarded(path,*args,**kwargs):
        assert path.name not in ('scores.json','truth.json','predictions.jsonl')
        assert path.name not in ('received_features.npz','received.npz','final_ssdg.pth')
        opened.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    result=tool.audit_run(config)
    assert {r['feature_dim'] for r in measured[1]}=={160,256}
    assert result['status']=='VERIFIED' and result['total_fits']==8
    assert result['fixed_k1_fits']==2 and result['support_cv_fits']==6
    assert result['optimizer_convergence_claim'] is False
    assert result['parameter_feedback_forbidden'] is True
    assert len(result['models'])==2
    for model in result['models']:
        assert model['fits']==4
        assert model['total_fit_seconds']==pytest.approx(model['mean_fit_seconds']*4)
        assert model['max_fit_seconds']>=model['mean_fit_seconds']
        assert [g['folds_per_fit'] for g in model['per_k']]==[0,2,3,3]
        assert [g['candidates_per_fit'] for g in model['per_k']]==[0,16,16,16]
        assert model['per_k'][0]['selected_oof_loss']['objective'] is None
        assert model['per_k'][1]['selected_oof_loss']['objective']['mean'] is not None
        assert model['payload_files']['incremental_transfer_bytes']==model['payload_files']['total_file_bytes']
        assert {v['feature_dim'] for v in model['state_bytes_by_classes'].values()}=={160,256}
        assert sum(v['fits'] for v in model['state_bytes_by_classes'].values())==4
    assert len(json.dumps(result,allow_nan=False))<25000
    assert result['process_peak_rss_bytes']==dict(min=1000000,mean=1500000,max=2000000)
    assert result['payload_byte_ranges']['total_file_bytes']['min']>0
    assert opened


@pytest.mark.parametrize('fault',['missing','duplicate','nonfinite','candidate_missing','fold_missing','temperature',
                                  'selected_mismatch','byte_mismatch','query_access','source_access','compact_missing'])
def test_incomplete_or_invalid_trace_rejected(tmp_path,measured,fault):
    config=make_run(tmp_path,measured)
    folder=Path(config['rows'][-1]['output_root'])/'sgjoint'
    path=folder/'fit_trace.jsonl';traces=list(tool.json_lines(path))
    if fault=='missing':traces.pop()
    elif fault=='duplicate':traces[-1]=copy.deepcopy(traces[0])
    elif fault=='nonfinite':traces[1]['candidate_trace'][0]['folds'][0]['held_after_temperature']['old_nll']=float('nan')
    elif fault=='candidate_missing':traces[1]['candidate_trace'].pop()
    elif fault=='fold_missing':traces[1]['candidate_trace'][0]['folds'].pop()
    elif fault=='temperature':traces[1]['candidate_trace'][0]['temperature']=100
    elif fault=='selected_mismatch':traces[1]['selected_oof_risk']['objective']+=1
    elif fault=='byte_mismatch':traces[1]['head_bytes']+=8
    elif fault=='query_access':traces[1]['query_rows_used_for_fit']=1
    elif fault=='source_access':traces[1]['source_runtime_access']=True
    else:
        compact=list(tool.json_lines(folder/'compact.jsonl'));compact.pop();lines(folder/'compact.jsonl',compact)
    lines(path,traces)
    with pytest.raises(ValueError):tool.audit_run(config)


def test_requires_scored_completion_before_fit_files_open(tmp_path,measured,monkeypatch):
    config=make_run(tmp_path,measured)
    write(Path(config['root'])/'complete.json',dict(status='PREDICTIONS_COMPLETE'))
    monkeypatch.setattr(tool,'json_lines',lambda *_:pytest.fail('Incomplete run must not open fit logs'))
    with pytest.raises(ValueError,match='SCORED'):
        tool.audit_run(config)


def test_remote_payload_uses_same_checked_parser_and_returns_only_summary(tmp_path,measured,capsys):
    config=make_run(tmp_path,measured)
    # The exact code sent over SSH runs here on synthetic temporary files.
    exec(compile(tool.remote_script(config),'synthetic_remote_audit','exec'),{})
    result=json.loads(capsys.readouterr().out)
    assert result['total_fits']==8 and result['status']=='VERIFIED'
    assert all('candidate_trace' not in model for model in result['models'])


def test_output_overwrite_rejected_before_remote_action(tmp_path,monkeypatch):
    output=tmp_path/'existing.json';output.write_text('original',encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['collect','--spec',str(tmp_path/'unused.json'),'--output',str(output)])
    monkeypatch.setattr(tool.subprocess,'run',lambda *_args,**_kwargs:pytest.fail('Must not contact remote'))
    with pytest.raises(FileExistsError):tool.main()
    assert output.read_text()=='original'
