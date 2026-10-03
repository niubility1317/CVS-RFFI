import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.cvs_reference_residual_clean import contracts as c,score as scorer
from experiments.cvs_reference_residual_clean import provenance as provenance
from experiments.cvs_reference_residual_identity.model import build
from experiments.cvs_reference_residual_identity.contracts import architecture_contract as dual_contract,PRECISION

def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data),encoding='utf-8')

@pytest.fixture
def fixture_matrix(tmp_path,monkeypatch):
    def make(count=42):
        for key,value in dict(PROJECT=str(tmp_path),RUNTIME=str(tmp_path/'run'),LOGS=str(tmp_path/'logs'),SOURCE_ROOT=str(tmp_path/'source'),CAPSULE=str(tmp_path/'capsule'),TRUTH=str(tmp_path/'truth.json'),QUERY_COUNT=count).items():
            monkeypatch.setattr(c,key,value)
        monkeypatch.setattr(scorer,'CAPSULE',c.CAPSULE);monkeypatch.setattr(scorer,'QUERY_COUNT',count)
        cap=Path(c.CAPSULE);cap.mkdir();put(cap/'manifest.json',dict(status='VALIDATED_ONCE',classes=c.CLASSES,channel='residual/post_sync/noeq'))
        ids=np.asarray([f'id-{i:06}' for i in range(count)]);np.savez(cap/'index.npz',ids=ids)
        receivers=['1-1','14-7','2-1','20-1','7-14','7-7','8-8']
        put(Path(c.TRUTH),{i:dict(label=n%6,receiver=receivers[n%7]) for n,i in enumerate(ids)})
        spec=c.make_spec()
        for row in spec['rows']:
            config=c.prediction_config(row['row_id']);put(Path(row['config']),config)
            out=Path(row['output_root']);out.mkdir(parents=True)
            put(out/'resolved_config.json',dict(config,truth_read=False,query_fit=False))
            put(out/'provenance.json',dict(status='VERIFIED',query_fit=False,checkpoint=row['source_output']+'/last.pt'))
            put(out/'clean_complete.json',dict(status='PREDICTIONS_COMPLETE',count=count,views=['clean'],truth_read=False,query_fit=False))
            pred=np.arange(count)%6
            if row['variant']=='reference_residual_scalar':pred=(pred+1)%6
            np.savez(out/'clean_predictions.npz',ids=ids,clean=pred)
        return spec,ids
    return make

def test_complete_score_and_paired_difference(fixture_matrix):
    spec,ids=fixture_matrix();result=scorer.score(spec)
    assert result['records']==64 and result['class_records']==48 and result['decisions']==8*len(ids)
    data=c.read(Path(spec['runtime_root'])/'clean_summary.json')
    assert all(row['mean_pp']==100 and row['positive_seeds']==4 for row in data['paired'])
    assert len(data['summary'])==16
    from experiments.cvs_reference_residual_clean.recount import recount
    check=recount(spec);assert check['max_metric_absolute_error']==0 and check['decisions']==8*len(ids)
    with pytest.raises(FileExistsError):scorer.score(spec)

def test_independent_recount_rejects_corrupted_score(fixture_matrix):
    from experiments.cvs_reference_residual_clean.recount import recount
    spec,_=fixture_matrix();scorer.score(spec);p=Path(spec['runtime_root'])/'clean_scored_results.json'
    data=c.read(p);data['results'][0]['accuracy']+=.1;put(p,data)
    with pytest.raises(ValueError):recount(spec)

def test_full168000_preflight_and_last_missing_row_keeps_truth_closed(fixture_matrix,monkeypatch):
    spec,ids=fixture_matrix(168000);actual,rows=scorer.preflight_predictions(spec)
    assert np.array_equal(actual,ids) and len(rows)==8
    last=Path(spec['rows'][-1]['output_root'])/'clean_complete.json';last.unlink()
    original=scorer.read;reads=[]
    def guarded(path):
        if str(path)==spec['p1_truth']:reads.append(path);raise AssertionError('Truth opened prematurely')
        return original(path)
    monkeypatch.setattr(scorer,'read',guarded)
    with pytest.raises(FileNotFoundError):scorer.score(spec)
    assert reads==[]

@pytest.mark.parametrize('change',['duplicate_row','swapped_ids','invalid_class','wrong_variant','query_fit'])
def test_bad_predictions_rejected_before_truth(fixture_matrix,monkeypatch,change):
    spec,ids=fixture_matrix();row=spec['rows'][0];out=Path(row['output_root'])
    if change=='duplicate_row':spec['rows'][-1]=spec['rows'][0]
    elif change=='swapped_ids':np.savez(out/'clean_predictions.npz',ids=ids[::-1],clean=np.arange(len(ids))%6)
    elif change=='invalid_class':np.savez(out/'clean_predictions.npz',ids=ids,clean=np.full(len(ids),6))
    else:
        config=c.read(out/'resolved_config.json');config['variant' if change=='wrong_variant' else 'query_fit']='wrong' if change=='wrong_variant' else True;put(out/'resolved_config.json',config)
    original=scorer.read
    def guarded(path):
        if str(path)==spec['p1_truth']:raise AssertionError('Truth opened prematurely')
        return original(path)
    monkeypatch.setattr(scorer,'read',guarded)
    with pytest.raises(ValueError):scorer.score(spec)

@pytest.mark.parametrize('name',scorer.OUTPUTS)
def test_all_scoring_outputs_preserved(fixture_matrix,name):
    spec,_=fixture_matrix();path=Path(spec['runtime_root'])/name;path.write_bytes(b'existing evidence')
    with pytest.raises(FileExistsError):scorer.score(spec)
    assert path.read_bytes()==b'existing evidence'

@pytest.fixture
def source_payload():
    variant='reference_residual_scalar';rid=variant+'-s2026092701';cfg=c.prediction_config(rid)
    root=Path(__file__).resolve().parents[1]
    resolved=c.read(root/'experiments/cvs_reference_residual_identity/source_configs'/f'{rid}.json')
    arch=dual_contract(variant)
    resolved.update(commit=c.SOURCE_COMMIT,source_counts={'L_s':6300,'U_s':56700,'V':27000},steps_per_epoch=50,U_s_use='unused',target_access=False,
        total_parameters=arch['total_parameters'],trainable_parameters=arch['total_parameters'],response_actual=arch,response_active=True,classifier_scale=30.,precision=PRECISION,backend_flags=resolved['numerical_policy'])
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,
        model_seed=cfg['model_seed'],physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    contract=dict(role_ids={'L_s':['a'],'U_s':['b'],'V':['c']},classes=c.CLASSES,equalized=1,out_len=256,normalize=True)
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,checkpoint=cfg['source_output']+'/last.pt',backend_flags=resolved['numerical_policy'])
    payload=dict(method='cvs_reference_residual_identity',variant=variant,epoch=200,num_classes=6,classes=c.CLASSES,source_contract=contract,initialization=initial,selection='fixed_last_epoch',config=resolved,model=build(variant).state_dict())
    return cfg,done,initial,contract,copy.deepcopy(contract),resolved,payload

def test_frozen_payload_validates_and_rejects_inheritance_role_precision_and_architecture(source_payload):
    args=source_payload;provenance.validate_payload(*args)
    for change in ('ancestor','roles','precision','architecture','commit','epoch','payload'):
        q=copy.deepcopy(args)
        if change=='ancestor':q[2]['ancestors']=['unverified']
        elif change=='roles':q[3]['role_ids']['L_s']=['different']
        elif change=='precision':q[6]['model']['projection.weight']=q[6]['model']['projection.weight'].half()
        elif change=='architecture':q[5]['response_actual']['window']=[0,80]
        elif change=='commit':q[5]['commit']='unverified'
        elif change=='epoch':q[1]['epoch']=199
        else:q[6]['method']='another_model'
        with pytest.raises(ValueError):provenance.validate_payload(*q)

def test_publish_remote_scripts_compile_and_no_query_smoke():
    from experiments.cvs_reference_residual_clean import publish
    for name in ('PREFLIGHT','REMOTE','INSPECT'):compile(getattr(publish,name),name,'exec')
    assert 'experiments.cvs_reference_residual_clean.cpu_smoke' in publish.REMOTE
    assert "cfg['freeze_file']" in publish.REMOTE
    assert "'source_contract'" in publish.REMOTE
    assert "row['config']" in publish.REMOTE and "row['source_config']" not in publish.REMOTE
