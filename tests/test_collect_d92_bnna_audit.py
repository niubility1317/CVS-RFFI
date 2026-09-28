"""Full BNNA metadata audit, with actual synthetic optimizer records only."""
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'tools')]
import collect_d92_bnna_audit as tool
from cvsrffi.stage2_d92_bnna import FROZEN_CONFIG,fit_bnna


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def lines(path,records):
    path.write_text(''.join(json.dumps(v)+'\n' for v in records),encoding='utf-8')


def csvwrite(path,records,fields=None):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(records[0]));writer.writeheader()
        for v in records:
            writer.writerow({k:'N/A' if x is None else json.dumps(x,sort_keys=True) if isinstance(x,(dict,list)) else x for k,x in v.items()})


def synthetic(k,zero=False,old_only=False):
    classes=['old-a','old-b'] if old_only else ['old-a','new-b'];c=2
    rng=np.random.default_rng(519+k);z=rng.normal(size=(c*k,4,160));fft=rng.normal(size=(c*k,96))
    fft/=np.linalg.norm(fft,axis=1,keepdims=True)
    if zero:z=np.repeat(z[:,:1],4,axis=1)
    state=fit_bnna(support_identity_views=z,support_fft=fft,support_labels=np.repeat(np.arange(c),k),
        support_ids=[f'physical-{i:04}' for i in range(c*k)],classes=classes,old_classes=classes if old_only else classes[:1])
    trace=state.audit_dict();trace.update(split_id=f'fit-{k}',registered_classes=classes,old_classes=classes if old_only else classes[:1],
        receiver='rx',scenario='scene',support_seed=1,new_count=0 if old_only else 1,
        fit_call_seconds=trace['fit_seconds']+.001,query_score_seconds=.002,prediction_write_seconds=.003)
    return trace


def compact(trace,index=1,total=3):
    small={k:trace[k] for k in ('split_id','k','classes','selected','selection','candidate_count','fold_count',
        'selected_objective','selected_macro_nll','selected_old_nll','selected_new_nll','fit_seconds','active_rank',
        'head_bytes','basis_bytes','gate_bytes','persistent_state_bytes','final_optimizer_steps',
        'receiver','scenario','support_seed','new_count','fit_call_seconds','query_score_seconds','prediction_write_seconds')}
    finals=[s for s in trace['steps'] if s['scope']=='final']
    small.update(completed=index,total=total,optimizer_steps_recorded=len(trace['steps']),
        final_step=tool.scalars(finals[-1]) if finals else None,final_fit=tool.final_summary(trace['final_fit']),
        total_seconds=trace['fit_seconds']+.01,peak_process_rss_bytes=123456,
        query_rows_used_for_fit=0,source_rows_used_for_fit=0,source_validation=None,source_validation_reason='No source input',
        epoch=None,epoch_reason='Step-based budget')
    return small


@pytest.fixture(scope='module')
def measured():
    with threadpool_limits(limits=1):
        return [synthetic(k) for k in (1,2,5)]


def rewrite(folder,traces):
    small=[compact(t,i+1,len(traces)) for i,t in enumerate(traces)]
    steps=[dict(split_id=t['split_id'],**tool.scalars(s)) for t in traces for s in t['steps']]
    lines(folder/'fit_trace.jsonl',traces);lines(folder/'compact.jsonl',small);csvwrite(folder/'compact.csv',small)
    lines(folder/'training_steps.jsonl',steps);csvwrite(folder/'training_steps.csv',steps)


def make_run(tmp_path,measured):
    root=tmp_path/'run';root.mkdir();rows=[]
    for model in (1,2):
        output=root/f'model-{model}';folder=output/'bnna';feature=output/'bnna_features'
        folder.mkdir(parents=True);feature.mkdir()
        row=dict(row_id=f'model-{model}',output_root=str(output),expected_checkpoint_sha256='a'*64,seeds=dict(model=model));rows.append(row)
        cache=feature/'received_bnna_features.npz';cache.write_bytes(b'never-read-features')
        payload=dict(new_ground_statistics_bytes=0,model_file_bytes=123456,received_feature_array_bytes=100*2944,received_feature_file_bytes=cache.stat().st_size)
        dump(feature/'features_complete.json',dict(status='BNNA_FEATURES_COMPLETE',schema='d92_bnna_received_views_v1',algorithm=FROZEN_CONFIG,
            capsule_id='cap',checkpoint_sha256='a'*64,model_seed=model,classes=['old-a'],dtype='float32',
            query_used_for_fitting=False,source_data_access=False,truth_read=False,encoder_updated=False,native_eval=True,native_buffers_unchanged=True,
            view_count_per_observation=4,new_ground_statistics_bytes=0,count=100,identity_views_shape=[100,4,160],fft_shape=[100,96],
            identity_views_bytes=100*4*160*4,fft_bytes=100*96*4,feature_array_bytes=100*2944,feature_file_bytes=cache.stat().st_size,
            model_file_bytes=123456,feature_seconds=1.25,peak_process_rss_bytes=7654321,peak_process_rss_reason=None))
        dump(feature/'checkpoint_provenance.json',dict(checkpoint_sha256='a'*64,model_file_bytes=123456))
        dump(folder/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',split_count=3,predictions=3,capsule_id='cap',truth_read=False,
            source_data_access=False,new_ground_statistics_bytes=0,payload_audit=payload,peak_process_rss_bytes=123456))
        dump(folder/'startup.json',dict(config={'algorithm':FROZEN_CONFIG},checkpoint_sha256='a'*64,capsule_id='cap',model_seed=model,
            payload_audit=payload,query_fit_access=False,truth_read=False,source_data_access=False,ground_summary_access=False,cross_row_adapted_state_reuse=False))
        rewrite(folder,measured)
    dump(root/'complete.json',dict(status='SCORED'));dump(root/'state.json',{r['row_id']:dict(status='PREDICTIONS_COMPLETE') for r in rows})
    return dict(root=str(root),rows=rows,splits=3,receivers=['rx'],scenarios=['scene'],ks=[1,2,5],new_counts=[1],support_seeds=[1],
        old_classes=['old-a'],algorithm=copy.deepcopy(FROZEN_CONFIG),capsule_id='cap')


def test_full_logs_no_score_truth_features_or_checkpoint_reads(tmp_path,measured,monkeypatch):
    config=make_run(tmp_path,measured);original=Path.open;opened=[]
    def guarded(path,*a,**kw):
        assert path.name not in ('truth.json','scores.json','predictions.jsonl','received_bnna_features.npz','received.npz','final_ssdg.pth')
        opened.append(path);return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    audit=tool.audit_run(config)
    assert audit['status']=='VERIFIED' and audit['total_fits']==6 and audit['fixed_k1_fits']==2 and audit['support_cv_fits']==4
    assert audit['optimizer_steps']==2*sum(len(r['steps']) for r in measured)
    assert not audit['optimizer_convergence_claim'] and audit['new_source_payload_bytes']==0
    assert audit['models'][0]['resources']['model_incremental_transfer_bytes'] is None
    assert audit['models'][0]['resources']['received_cache_array_bytes']==294400
    assert opened


@pytest.mark.parametrize('fault',['missing','duplicate','formula','nonfinite','k1','step_budget','lr','weights','grad','init','gate','final_state',
    'fold_leak','basis_access','fold_class','fold_k','candidate','selected','bytes','query','encoder','compact','stepstream','stepcsv','compactcsv',
    'cachebytes','marker','time','progress','model_deployed'])
def test_corrupt_metadata_rejected(tmp_path,measured,fault):
    config=make_run(tmp_path,measured);out=Path(config['rows'][0]['output_root']);folder=out/'bnna'
    traces=copy.deepcopy(measured);r=traces[1];step=r['steps'][0]
    if fault=='missing':traces.pop()
    elif fault=='duplicate':traces[-1]=copy.deepcopy(traces[0])
    elif fault=='formula':r['config']['learning_rate']=.05
    elif fault=='nonfinite':step['gradient_norm']=float('nan')
    elif fault=='k1':traces[0]['candidate_count']=2
    elif fault=='step_budget':r['steps'].pop(1)
    elif fault=='lr':step['learning_rate']=.05
    elif fault=='weights':step['loss_view_weight']=.2
    elif fault=='grad':step['gradient_norm']=-1
    elif fault=='init':step['gate_before_max']=.2
    elif fault=='gate':step['gate_after_max']=.7
    elif fault=='final_state':traces[0]['final_fit']['final_gate'][0]+=.1
    elif fault=='fold_leak':r['folds'][0]['training']['training_physical_ids'][0]=r['folds'][0]['heldout_ids'][0]
    elif fault=='basis_access':r['folds'][0]['training']['basis_estimated_from_trainfold_only']=False
    elif fault=='fold_class':r['physical_fold_assignment'][0]['class_id']='bad'
    elif fault=='fold_k':r['folds'][0]['train_k']=20
    elif fault=='candidate':r['candidates'].pop()
    elif fault=='selected':r['selected_objective']+=1
    elif fault=='bytes':r['head_bytes']+=8
    elif fault=='query':r['query_rows_used_for_fit']=1
    elif fault=='encoder':r['encoder_updated']=True
    rewrite(folder,traces)
    if fault=='compact':
        records=list(tool.json_lines(folder/'compact.jsonl'));records[1]['final_fit']['status']='invented';lines(folder/'compact.jsonl',records)
    elif fault=='stepstream':
        records=list(tool.json_lines(folder/'training_steps.jsonl'));records.pop();lines(folder/'training_steps.jsonl',records)
    elif fault in ('stepcsv','compactcsv'):
        path=folder/('training_steps.csv' if fault=='stepcsv' else 'compact.csv');path.write_text(path.read_text(encoding='utf-8')+'extra\n',encoding='utf-8')
    elif fault=='cachebytes':
        path=out/'bnna_features/features_complete.json';value=tool.read_json(path);value['feature_array_bytes']+=1;dump(path,value)
    elif fault=='marker':
        path=folder/'predictions_complete.json';value=tool.read_json(path);value['split_count']=1;dump(path,value)
    elif fault=='model_deployed':
        for name in ('startup.json','predictions_complete.json'):
            path=folder/name;value=tool.read_json(path);value['payload_audit']['model_already_deployed']=True;dump(path,value)
    elif fault in ('time','progress'):
        records=list(tool.json_lines(folder/'compact.jsonl'))
        records[1]['total_seconds' if fault=='time' else 'completed']=-1
        lines(folder/'compact.jsonl',records);csvwrite(folder/'compact.csv',records)
    with pytest.raises(ValueError):tool.audit_run(config)


@pytest.mark.parametrize('k',[1,2])
def test_rank_zero_and_identity_selection_are_not_fake_64_steps(k):
    with threadpool_limits(limits=1):trace=synthetic(k,zero=True)
    cfg=dict(ks=[k],new_counts=[1],old_classes=['old-a'],algorithm=FROZEN_CONFIG)
    tool.validate_fit(trace,compact(trace),cfg)
    assert not trace['steps'] and trace['final_optimizer_steps']==0
    assert trace['final_fit']['status']==('NO_VIEW_VARIATION' if k==1 else 'IDENTITY_SELECTED')


def test_old_only_group_has_no_invented_new_loss():
    with threadpool_limits(limits=1):trace=synthetic(2,old_only=True)
    tool.validate_fit(trace,compact(trace),dict(ks=[2],new_counts=[0],old_classes=['old-a','old-b'],algorithm=FROZEN_CONFIG))
    assert trace['selected_new_nll'] is None


def test_remote_serialization_runs_same_audit(tmp_path,measured,capsys):
    config=make_run(tmp_path,measured)
    exec(compile(tool.remote_script(config),'synthetic_metadata_only','exec'),{})
    assert json.loads(capsys.readouterr().out)['total_fits']==6


def test_nonterminal_rejected_before_logs(tmp_path,measured,monkeypatch):
    config=make_run(tmp_path,measured);dump(Path(config['root'])/'complete.json',dict(status='RUNNING'))
    monkeypatch.setattr(tool,'json_lines',lambda *_:pytest.fail('No logs before terminal guard'))
    with pytest.raises(ValueError,match='SCORED'):tool.audit_run(config)


def test_existing_output_rejected_before_ssh(tmp_path,monkeypatch):
    output=tmp_path/'audit.json';output.write_text('preserve',encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['collector','--spec',str(tmp_path/'unused'),'--output',str(output)])
    monkeypatch.setattr(tool.subprocess,'run',lambda *_a,**_k:pytest.fail('No remote operation'))
    with pytest.raises(FileExistsError):tool.main()
    assert output.read_text()=='preserve'


def test_actual_exporter_predictor_logs_match_auditor(tmp_path,monkeypatch):
    from test_export_d92_bnna_features import fixture
    import export_d92_bnna_features as exporter
    import predict_d92_bnna as predictor
    args,_,_=fixture(tmp_path,monkeypatch,k=1)
    with threadpool_limits(limits=1):
        exporter.export(**args)
        predictor.predict(**{k:args[k] for k in ('row_root','capsule','config','expected_capsule_id','expected_checkpoint_sha256')},
            bnna_features=args['output'],output=tmp_path/'bnna')
    dump(tmp_path/'complete.json',dict(status='SCORED'))
    dump(tmp_path/'state.json',{'model':dict(status='PREDICTIONS_COMPLETE')})
    config=dict(root=str(tmp_path),rows=[dict(row_id='model',output_root=str(tmp_path),expected_checkpoint_sha256='a'*64,seeds=dict(model=123))],
        splits=1,receivers=['rx'],scenarios=['synthetic'],ks=[1],new_counts=[2],support_seeds=[1],
        old_classes=['old-z','old-a'],algorithm=FROZEN_CONFIG,capsule_id='synthetic')
    result=tool.audit_run(config)
    assert result['total_fits']==1 and result['optimizer_steps']==64
