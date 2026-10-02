import copy
import json
from pathlib import Path
import pytest
from experiments.cvs_clean_eval.contracts import checkpoint_contract,validate_predict_config,source_method,build_model,evaluation_variants
from experiments.cvs_energy_identity.model import VARIANTS,energy_contract
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY


def fixture(variant):
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'experiments/cvs_energy_identity/configs'/f'{variant}-s2026092701.json').read_text(encoding='utf-8'))
    contract=dict(role_ids={'L_s':['l'],'U_s':['u'],'V':['v']},source_rxs=[1,3,4,6,8],source_days=[1,2,3],ratios=[.1,.9],split_seed=392005,
                  num_classes=6,classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,
                 model_seed=c['model_seed'],physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    c.update(steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},target_access=False,energy_active=True,
             energy_actual=energy_contract(variant),classifier_scale=30.,backend_flags=copy.deepcopy(FULL_FP32_POLICY))
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,backend_flags=copy.deepcopy(FULL_FP32_POLICY))
    payload=dict(epoch=200,source_contract=contract,initialization=initial,selection='fixed_last_epoch',method=c['method'],variant=variant,config=c,classes=contract['classes'],num_classes=6)
    return dict(variant=variant,model_seed=c['model_seed']),done,initial,contract,copy.deepcopy(contract),c,payload


@pytest.mark.parametrize('variant',VARIANTS)
def test_clean_load_requires_actual_registered_source(variant):
    args=fixture(variant);checkpoint_contract(*args)
    assert source_method(variant)=='cvs_energy_identity'
    assert build_model(variant).contract()==energy_contract(variant)


@pytest.mark.parametrize('corruption',['physical_ids','target','ancestor','missing_policy','resolved_tf32','completion_tf32','inactive','changed_frontend','payload','budget'])
def test_corrupted_source_rejected_before_query(corruption):
    args=fixture(VARIANTS[0]);_,done,initial,contract,expected,resolved,payload=args
    if corruption=='physical_ids':contract['role_ids']['V']=['other']
    elif corruption=='target':done['target_evaluated']=True
    elif corruption=='ancestor':initial['ancestors']=['unverified.pt']
    elif corruption=='missing_policy':del resolved['numerical_policy']
    elif corruption=='resolved_tf32':resolved['backend_flags']['cudnn_allow_tf32']=True
    elif corruption=='completion_tf32':done['backend_flags']['cudnn_allow_tf32']=True
    elif corruption=='inactive':resolved['energy_active']=False
    elif corruption=='changed_frontend':resolved['energy_actual']['estimator_window']=[0,80]
    elif corruption=='payload':payload['method']='cvs_coordinate_identity'
    elif corruption=='budget':done['steps']=9999
    with pytest.raises(ValueError):checkpoint_contract(*args)


def test_unselected_variant_and_wrong_source_seed_path_rejected():
    chosen=VARIANTS[0];selection=dict(scope='energy_source',selected_variant=chosen)
    c=dict(method='cvs_clean_eval',views=['clean'],variant=chosen,model_seed=2026092701,energy_source_root='/source',source_output='/source/'+chosen+'-s2026092701/source')
    assert validate_predict_config(c,selection)==c
    assert len(evaluation_variants(selection))==6
    wrong=dict(c,variant=VARIANTS[1]);
    with pytest.raises(ValueError):validate_predict_config(wrong,selection)
    wrong=dict(c,source_output='/source/'+chosen+'-s2026092702/source')
    with pytest.raises(ValueError):validate_predict_config(wrong,selection)


@pytest.mark.parametrize('variant',VARIANTS)
def test_independent_scorer_accepts_all24_and_rejects_incomplete_matrix(tmp_path,variant):
    from experiments.cvs_clean_eval.score import validate_matrix,BASELINES,SEEDS
    selection=dict(scope='energy_source',status='SOURCE_SELECTION_FROZEN',selected_variant=variant,target_access=False,target_score_used=False)
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection),encoding='utf-8')
    rows=[dict(row_id=f'{v}-s{s}',variant=v,model_seed=s,output_root=str(tmp_path/f'{v}-s{s}')) for v in [*BASELINES,'residual_fusion',variant] for s in SEEDS]
    spec=dict(selection_file=str(path),rows=rows)
    assert validate_matrix(spec)==selection
    with pytest.raises(ValueError):validate_matrix(dict(spec,rows=rows[:-1]))


@pytest.mark.parametrize('variant',VARIANTS)
def test_real_checkpoint_predictor_roundtrip_without_truth(tmp_path,monkeypatch,variant):
    """Disposable synthetic E200 metadata; proves loader behavior, not training evidence."""
    import numpy as np
    import torch
    from experiments.cvs_clean_eval import predict
    torch.set_num_threads(2)
    cfg,done,initial,contract,expected,resolved,payload=fixture(variant)
    source=tmp_path/'source'/f'{variant}-s2026092701'/'source';source.mkdir(parents=True)
    def save(p,d):p.write_text(json.dumps(d),encoding='utf-8')
    model=build_model(variant)
    with torch.no_grad():next(model.parameters()).add_(.01)
    payload['model']=model.state_dict()
    for name,value in [('completion.json',done),('initialization.json',initial),('source_contract.json',contract),('resolved_config.json',resolved)]:save(source/name,value)
    torch.save(payload,source/'last.pt');save(tmp_path/'expected.json',expected)
    capsule=tmp_path/'capsule';capsule.mkdir();ids=np.asarray(['synthetic0','synthetic1','synthetic2'])
    x=np.random.default_rng(42).normal(size=(3,2,256)).astype('float32')
    np.savez(capsule/'index.npz',ids=ids);np.save(capsule/'clean.npy',x)
    save(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=contract['classes'],channel='residual/post_sync/noeq'))
    selection=dict(scope='energy_source',status='SOURCE_SELECTION_FROZEN',selected_variant=variant,target_access=False,target_score_used=False)
    # Source ranking is tested separately; this fixture isolates the actual model loader and inference.
    monkeypatch.setattr(predict,'frozen_selection',lambda _:selection)
    cfg.update(method='cvs_clean_eval',views=['clean'],selection_file='synthetic_selection_only',energy_source_root=str(tmp_path/'source'),source_output=str(source),source_contract=str(tmp_path/'expected.json'),p1_capsule=str(capsule),output_root=str(tmp_path/'prediction'),device='cpu')
    before=copy.deepcopy(model.state_dict());model.eval()
    with torch.no_grad():expected_pred=model(torch.tensor(x)).argmax(1).tolist()
    predict.predict(cfg)
    with np.load(tmp_path/'prediction/clean_predictions.npz') as d:assert d['clean'].tolist()==expected_pred and d['ids'].tolist()==ids.tolist()
    assert all(torch.equal(before[k],model.state_dict()[k]) for k in before)
    actual=json.loads((tmp_path/'prediction/resolved_config.json').read_text(encoding='utf-8'))
    assert actual['backend_flags']==FULL_FP32_POLICY and actual['truth_read'] is False and actual['query_fit'] is False
