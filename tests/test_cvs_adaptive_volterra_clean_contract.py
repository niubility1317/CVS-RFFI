import copy,json
from pathlib import Path
import pytest
from experiments.cvs_clean_eval import contracts
from experiments.cvs_adaptive_volterra_identity.model import VARIANTS,adaptive_contract
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

def fixture(v):
    from test_cvs_coupled_clean_contract import fixture as old_fixture
    args=list(old_fixture('coupled_lag4'))
    cfg=json.loads((Path(__file__).resolve().parents[1]/'experiments/cvs_adaptive_volterra_identity/configs'/f'{v}-s2026092701.json').read_text(encoding='utf-8'))
    cfg.update(steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},target_access=False,adaptive_active=True,
               adaptive_actual=adaptive_contract(v),classifier_scale=30.,total_parameters=202555,trainable_parameters=202555,backend_flags=copy.deepcopy(FULL_FP32_POLICY))
    args[0]=dict(variant=v,model_seed=2026092701);args[5]=cfg
    args[6].update(method=cfg['method'],variant=v,config=cfg)
    return args

@pytest.mark.parametrize('v',VARIANTS)
def test_actual_source_payload_and_model(v):
    contracts.checkpoint_contract(*fixture(v))
    assert contracts.source_method(v)=='cvs_adaptive_volterra_identity'
    assert contracts.build_model(v).contract()==adaptive_contract(v)

@pytest.mark.parametrize('bad',['roles','target','ancestor','policy','tf32','inactive','lag','envelope','mix','lift','budget','payload'])
def test_corrupt_provenance_rejected(bad):
    a=fixture(VARIANTS[0]);_,done,initial,c,expected,resolved,payload=a
    if bad=='roles':c['role_ids']['V']=['wrong']
    elif bad=='target':done['target_evaluated']=True
    elif bad=='ancestor':initial['ancestors']=['other.pt']
    elif bad=='policy':resolved.pop('numerical_policy')
    elif bad=='tf32':done['backend_flags']['cudnn_allow_tf32']=True
    elif bad=='inactive':resolved['adaptive_active']=False
    elif bad=='lag':resolved['adaptive_actual']['actual_phase_lag']=4
    elif bad=='envelope':resolved['adaptive_actual']['actual_envelope_lag']=1
    elif bad=='mix':resolved['adaptive_actual']['mixture_parameter_count']=3
    elif bad=='lift':resolved['adaptive_actual']['adaptive_lift_active']=False
    elif bad=='budget':done['steps']=9999
    elif bad=='payload':payload['method']='cvs_coupled_identity'
    with pytest.raises(ValueError):contracts.checkpoint_contract(*a)

def test_selected_only_and_source_path():
    sel=dict(scope='adaptive_volterra_source',selected_variant=VARIANTS[0])
    c=dict(method='cvs_clean_eval',views=['clean'],variant=VARIANTS[0],model_seed=2026092701,
           adaptive_source_root='/source',source_output='/source/'+VARIANTS[0]+'-s2026092701/source')
    assert contracts.validate_predict_config(c,sel)==c
    assert contracts.evaluation_variants(sel)==[*contracts.BASELINES,'residual_fusion','energy_equivariant','coupled_lag4',VARIANTS[0]]
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(c,variant=VARIANTS[1]),sel)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(c,source_output='/other'),sel)
    with pytest.raises(ValueError):contracts.validate_predict_config(dict(c,truth='forbidden'),sel)

def test_recompute_all12_source_records_and_reject_control_retention(monkeypatch):
    from experiments.cvs_adaptive_volterra_identity import dispatch
    records=[dict(variant=v,seed=s,accuracy=.93 if v==VARIANTS[0] else .91,worst_rx=.9,parameters=202555,macs=100)
             for v in ['coupled_lag4',*VARIANTS] for s in dispatch.SEEDS]
    matrix=dict(source_controls=[dict(variant='coupled_lag4',model_seed=s) for s in dispatch.SEEDS],
                rows=[dict(variant=v,model_seed=s) for v in VARIANTS for s in dispatch.SEEDS])
    monkeypatch.setattr(dispatch,'validate_spec',lambda _:matrix);called=[]
    def record(row,original,method):
        called.append(method);return next(r for r in records if r['variant']==row['variant'] and r['seed']==row['model_seed'])
    monkeypatch.setattr(dispatch,'read_source_record',record)
    selection=dict(dispatch.select_source_candidate(records),scope='adaptive_volterra_source',source_matrix_ref='synthetic')
    monkeypatch.setattr(contracts,'read',lambda p:selection if str(p)=='selection' else {})
    assert contracts.frozen_selection('selection')==selection
    assert called.count('cvs_coupled_identity')==4 and called.count('cvs_adaptive_volterra_identity')==8
    selection['selected_variant']=VARIANTS[1]
    with pytest.raises(ValueError):contracts.frozen_selection('selection')
    for r in records:r['accuracy']=.99 if r['variant']=='coupled_lag4' else .91
    selection.clear();selection.update(dispatch.select_source_candidate(records),scope='adaptive_volterra_source',source_matrix_ref='synthetic')
    with pytest.raises(ValueError,match='baseline retained'):contracts.frozen_selection('selection')

def synthetic32(tmp_path):
    import numpy as np
    from test_cvs_coupled_clean_contract import synthetic_28
    from experiments.cvs_clean_eval import score
    spec=synthetic_28(tmp_path)
    def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d),encoding='utf-8')
    sel=score.read(spec['selection_file']);sel.update(scope='adaptive_volterra_source',selected_variant=VARIANTS[0]);save(Path(spec['selection_file']),sel)
    for row in spec['rows']:
        if row['variant']=='coupled_lag1':
            cfg=score.read(row['config']);cfg.update(variant=VARIANTS[0]);row.update(variant=VARIANTS[0]);save(Path(row['config']),cfg)
            resolved=score.read(Path(row['output_root'])/'resolved_config.json');resolved.update(variant=VARIANTS[0]);save(Path(row['output_root'])/'resolved_config.json',resolved)
    for s in sorted(score.SEEDS):
        rid=f'coupled_lag4-s{s}';out=tmp_path/score.REUSED_COUPLED_RUN/rid/'prediction';out.mkdir(parents=True)
        oldsel=out.parents[1]/'selection.json';save(oldsel,dict(scope='coupled_source',status='SOURCE_SELECTION_FROZEN',selected_variant='coupled_lag4',new_candidate_selected=True,target_access=False,target_score_used=False))
        cfg=dict(method='cvs_clean_eval',variant='coupled_lag4',model_seed=s,output_root=str(out),selection_file=str(oldsel),views=['clean'],p1_capsule=str(tmp_path/'capsule'),
                 coupled_source_root=str(tmp_path/'coupled-source'),source_output=str(tmp_path/'coupled-source'/rid/'source'))
        cf=tmp_path/'configs'/f'{rid}.json';save(cf,cfg)
        save(out.parents[1]/'scoring_clean_complete.json',dict(status='SCORED_COMPLETE',models=7,seeds=4,rows=28))
        ids=np.array(['synthetic-'+str(i) for i in range(12)]);pred=np.arange(12)%6;pred[0]=1
        np.savez(out/'clean_predictions.npz',ids=ids,clean=pred.astype('int64'))
        save(out/'clean_complete.json',dict(status='PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,views=['clean'],count=12))
        save(out/'resolved_config.json',dict(cfg,truth_read=False,query_fit=False));save(out/'provenance.json',dict(status='VERIFIED',query_fit=False))
        spec['rows'].append(dict(row_id=rid,variant='coupled_lag4',model_seed=s,config=str(cf),output_root=str(out),reuse_from_run=score.REUSED_COUPLED_RUN))
    return spec

@pytest.mark.parametrize('bad',[None,'selection','marker','reuse','missing'])
def test_all32_reuse_and_truth_last(tmp_path,monkeypatch,bad):
    from experiments.cvs_clean_eval import score
    spec=synthetic32(tmp_path);row=next(r for r in spec['rows'] if r['variant']=='coupled_lag4');cfg=score.read(row['config'])
    monkeypatch.setattr(contracts,'frozen_selection',lambda p:score.read(p))
    assert contracts.validate_reused_row(row)==cfg
    if bad=='selection':Path(cfg['selection_file']).write_text(json.dumps(dict(score.read(cfg['selection_file']),selected_variant='coupled_lag1')),encoding='utf-8')
    elif bad=='marker':(Path(row['output_root']).parents[1]/'scoring_clean_complete.json').write_text(json.dumps(dict(status='SCORED_COMPLETE',models=7,seeds=4,rows=27)),encoding='utf-8')
    elif bad=='reuse':row['reuse_from_run']='unknown'
    elif bad=='missing':(Path(row['output_root'])/'clean_complete.json').unlink()
    opened=[];original=score.read
    def observed(p):opened.append(str(p));return original(p)
    monkeypatch.setattr(score,'read',observed)
    if bad:
        with pytest.raises((ValueError,FileNotFoundError)):score.score(spec)
        assert spec['p1_truth'] not in opened
    else:
        result=score.score(spec);assert result['models']==8 and result['rows']==32 and result['records']==96
        assert sum(Path(p).name=='clean_complete.json' for p in opened[:opened.index(spec['p1_truth'])])==32
        summary=original(Path(spec['runtime_root'])/'clean_summary.json')
        pair=next(p for p in summary['paired'] if p['receiver']=='ALL' and p['baseline']=='coupled_lag4')
        assert pair['accuracy_delta_pp_mean']==pytest.approx(100/12)

@pytest.mark.parametrize('v',VARIANTS)
def test_prepare32_preserves28_original_paths(tmp_path,monkeypatch,v):
    import shutil
    from experiments.cvs_adaptive_volterra_clean import prepare
    root=Path(__file__).resolve().parents[1];base=tmp_path/'experiments/cvs_coupled_clean/configs';base.mkdir(parents=True)
    for name in ('launch_spec.json','experiment_spec.json'):shutil.copyfile(root/'experiments/cvs_coupled_clean/configs'/name,base/name)
    monkeypatch.setattr(prepare,'ROOT',tmp_path)
    sel=dict(status='SOURCE_SELECTION_FROZEN',selected_variant=v,new_candidate_selected=True,target_access=False,target_score_used=False)
    prepare.main(sel);launch=json.loads((tmp_path/'experiments/cvs_adaptive_volterra_clean/configs/launch_spec.json').read_text(encoding='utf-8'))
    old=json.loads((base/'launch_spec.json').read_text(encoding='utf-8'))
    assert len(launch['rows'])==32 and sum(bool(r.get('reuse_from_run')) for r in launch['rows'])==28
    assert all({k:r[k] for k in prior}==prior for r,prior in zip(launch['rows'][:28],old['rows']))
    assert {r['variant'] for r in launch['rows'] if not r.get('reuse_from_run')}=={v}
    for row in launch['rows']:
        if row.get('reuse_from_run'):continue
        config=json.loads((tmp_path/'experiments/cvs_adaptive_volterra_clean/configs'/(row['row_id']+'.json')).read_text(encoding='utf-8'))
        contracts.validate_predict_config(config,dict(sel,scope='adaptive_volterra_source'))
    with pytest.raises(ValueError):prepare.main(dict(sel,new_candidate_selected=False))
    with pytest.raises(FileExistsError):prepare.main(sel)

@pytest.mark.parametrize('v',VARIANTS)
def test_disposable_actual_checkpoint_inference_without_truth(tmp_path,monkeypatch,v):
    import numpy as np,torch
    from experiments.cvs_clean_eval import predict
    torch.set_num_threads(2)
    cfg,done,initial,contract,expected,resolved,payload=fixture(v)
    source=tmp_path/'source'/f'{v}-s2026092701'/'source';source.mkdir(parents=True)
    def save(p,d):p.write_text(json.dumps(d),encoding='utf-8')
    model=contracts.build_model(v).eval()
    with torch.no_grad():model.core.behavior[0].mix_raw.copy_(torch.tensor([.2,-.1]))
    payload['model']=model.state_dict()
    for name,value in [('completion.json',done),('initialization.json',initial),('source_contract.json',contract),('resolved_config.json',resolved)]:save(source/name,value)
    torch.save(payload,source/'last.pt');save(tmp_path/'expected.json',expected)
    capsule=tmp_path/'capsule';capsule.mkdir();ids=np.asarray(['public0','public1','public2'])
    x=np.random.default_rng(42).normal(size=(3,2,256)).astype('float32')
    np.savez(capsule/'index.npz',ids=ids);np.save(capsule/'clean.npy',x)
    save(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=contract['classes'],channel='residual/post_sync/noeq'))
    monkeypatch.setattr(predict,'frozen_selection',lambda _:dict(scope='adaptive_volterra_source',status='SOURCE_SELECTION_FROZEN',selected_variant=v))
    cfg.update(method='cvs_clean_eval',views=['clean'],selection_file='public-only',adaptive_source_root=str(tmp_path/'source'),source_output=str(source),source_contract=str(tmp_path/'expected.json'),p1_capsule=str(capsule),output_root=str(tmp_path/'prediction'),device='cpu')
    before=copy.deepcopy(model.state_dict())
    with torch.no_grad():expected_pred=model(torch.tensor(x)).argmax(1).tolist()
    predict.predict(cfg)
    with np.load(tmp_path/'prediction/clean_predictions.npz') as d:assert d['clean'].tolist()==expected_pred and d['ids'].tolist()==ids.tolist()
    assert all(torch.equal(before[k],model.state_dict()[k]) for k in before)
    actual=json.loads((tmp_path/'prediction/resolved_config.json').read_text(encoding='utf-8'))
    assert actual['backend_flags']==FULL_FP32_POLICY and actual['truth_read'] is False and actual['query_fit'] is False

@pytest.mark.parametrize('count',[202553,202556])
def test_actual_gate_parameter_count_cannot_change(count):
    a=fixture(VARIANTS[0]);a[5]['total_parameters']=count
    with pytest.raises(ValueError):contracts.checkpoint_contract(*a)

@pytest.mark.parametrize('bad',['missing','shape'])
def test_incomplete_gate_checkpoint_rejected_before_query(tmp_path,monkeypatch,bad):
    import torch
    from experiments.cvs_clean_eval import predict
    v=VARIANTS[0];cfg,done,initial,contract,expected,resolved,payload=fixture(v)
    source=tmp_path/'source'/f'{v}-s2026092701'/'source';source.mkdir(parents=True)
    def save(p,d):p.write_text(json.dumps(d),encoding='utf-8')
    model=contracts.build_model(v);payload['model']=model.state_dict()
    if bad=='missing':del payload['model']['core.behavior.0.mix_raw']
    else:payload['model']['core.behavior.0.mix_raw']=torch.zeros(3)
    for name,value in [('completion.json',done),('initialization.json',initial),('source_contract.json',contract),('resolved_config.json',resolved)]:save(source/name,value)
    torch.save(payload,source/'last.pt');save(tmp_path/'expected.json',expected)
    monkeypatch.setattr(predict,'frozen_selection',lambda _:dict(scope='adaptive_volterra_source',status='SOURCE_SELECTION_FROZEN',selected_variant=v))
    cfg.update(method='cvs_clean_eval',views=['clean'],selection_file='public-only',adaptive_source_root=str(tmp_path/'source'),source_output=str(source),source_contract=str(tmp_path/'expected.json'),p1_capsule=str(tmp_path/'query-that-must-not-be-read'),output_root=str(tmp_path/'prediction'),device='cpu')
    with pytest.raises(RuntimeError):predict.predict(cfg)
    assert not (tmp_path/'prediction/provenance.json').exists()
