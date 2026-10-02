import copy
import json
from pathlib import Path
import pytest
from experiments.cvs_clean_eval.contracts import checkpoint_contract,validate_predict_config,source_method,build_model,evaluation_variants
from experiments.cvs_orthopoly_identity.model import VARIANTS,orthopoly_contract
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY


def fixture(variant):
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'experiments/cvs_orthopoly_identity/configs'/f'{variant}-s2026092701.json').read_text(encoding='utf-8'))
    contract=dict(role_ids={'L_s':['l'],'U_s':['u'],'V':['v']},source_rxs=[1,3,4,6,8],source_days=[1,2,3],ratios=[.1,.9],split_seed=392005,
                  num_classes=6,classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,
                 model_seed=c['model_seed'],physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    c.update(steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},target_access=False,orthopoly_active=True,total_parameters=202553,trainable_parameters=202553,
             orthopoly_actual=orthopoly_contract(variant),classifier_scale=30.,backend_flags=copy.deepcopy(FULL_FP32_POLICY))
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,backend_flags=copy.deepcopy(FULL_FP32_POLICY))
    payload=dict(epoch=200,source_contract=contract,initialization=initial,selection='fixed_last_epoch',method=c['method'],variant=variant,config=c,classes=contract['classes'],num_classes=6)
    return dict(variant=variant,model_seed=c['model_seed']),done,initial,contract,copy.deepcopy(contract),c,payload


@pytest.mark.parametrize('variant',VARIANTS)
def test_clean_load_requires_actual_registered_source(variant):
    args=fixture(variant);checkpoint_contract(*args)
    assert source_method(variant)=='cvs_orthopoly_identity'
    assert build_model(variant).contract()==orthopoly_contract(variant)


@pytest.mark.parametrize('corruption',['physical_ids','target','ancestor','missing_policy','resolved_tf32','completion_tf32','inactive','changed_frontend','changed_lag','inactive_lift','payload','budget'])
def test_corrupted_source_rejected_before_query(corruption):
    args=fixture(VARIANTS[0]);_,done,initial,contract,expected,resolved,payload=args
    if corruption=='physical_ids':contract['role_ids']['V']=['other']
    elif corruption=='target':done['target_evaluated']=True
    elif corruption=='ancestor':initial['ancestors']=['unverified.pt']
    elif corruption=='missing_policy':del resolved['numerical_policy']
    elif corruption=='resolved_tf32':resolved['backend_flags']['cudnn_allow_tf32']=True
    elif corruption=='completion_tf32':done['backend_flags']['cudnn_allow_tf32']=True
    elif corruption=='inactive':resolved['orthopoly_active']=False
    elif corruption=='changed_frontend':resolved['orthopoly_actual']['estimator_window']=[0,80]
    elif corruption=='changed_lag':resolved['orthopoly_actual']['actual_envelope_lag']=4
    elif corruption=='inactive_lift':resolved['orthopoly_actual']['orthopoly_active']=False
    elif corruption=='payload':payload['method']='cvs_coordinate_identity'
    elif corruption=='budget':done['steps']=9999
    with pytest.raises(ValueError):checkpoint_contract(*args)


def test_unselected_variant_and_wrong_source_seed_path_rejected():
    chosen=VARIANTS[0];selection=dict(scope='orthopoly_source',selected_variant=chosen)
    c=dict(method='cvs_clean_eval',views=['clean'],variant=chosen,model_seed=2026092701,orthopoly_source_root='/source',source_output='/source/'+chosen+'-s2026092701/source')
    assert validate_predict_config(c,selection)==c
    assert len(evaluation_variants(selection))==9
    wrong=dict(c,variant=VARIANTS[1]);
    with pytest.raises(ValueError):validate_predict_config(wrong,selection)
    wrong=dict(c,source_output='/source/'+chosen+'-s2026092702/source')
    with pytest.raises(ValueError):validate_predict_config(wrong,selection)


@pytest.mark.parametrize('variant',VARIANTS)
def test_independent_scorer_accepts_all36_and_rejects_incomplete_matrix(tmp_path,variant):
    from experiments.cvs_clean_eval.score import validate_matrix,BASELINES,SEEDS
    selection=dict(scope='orthopoly_source',status='SOURCE_SELECTION_FROZEN',selected_variant=variant,new_candidate_selected=True,target_access=False,target_score_used=False)
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection),encoding='utf-8')
    rows=[dict(row_id=f'{v}-s{s}',variant=v,model_seed=s,output_root=str(tmp_path/f'{v}-s{s}')) for v in [*BASELINES,'residual_fusion','energy_equivariant','coupled_lag4','adaptive_volterra_lag4',variant] for s in SEEDS]
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
    selection=dict(scope='orthopoly_source',status='SOURCE_SELECTION_FROZEN',selected_variant=variant,target_access=False,target_score_used=False)
    # Source ranking is tested separately; this fixture isolates the actual model loader and inference.
    monkeypatch.setattr(predict,'frozen_selection',lambda _:selection)
    cfg.update(method='cvs_clean_eval',views=['clean'],selection_file='synthetic_selection_only',orthopoly_source_root=str(tmp_path/'source'),source_output=str(source),source_contract=str(tmp_path/'expected.json'),p1_capsule=str(capsule),output_root=str(tmp_path/'prediction'),device='cpu')
    before=copy.deepcopy(model.state_dict());model.eval()
    with torch.no_grad():expected_pred=model(torch.tensor(x)).argmax(1).tolist()
    predict.predict(cfg)
    with np.load(tmp_path/'prediction/clean_predictions.npz') as d:assert d['clean'].tolist()==expected_pred and d['ids'].tolist()==ids.tolist()
    assert all(torch.equal(before[k],model.state_dict()[k]) for k in before)
    actual=json.loads((tmp_path/'prediction/resolved_config.json').read_text(encoding='utf-8'))
    assert actual['backend_flags']==FULL_FP32_POLICY and actual['truth_read'] is False and actual['query_fit'] is False


def test_source_freeze_recomputed_and_baseline_retention_rejects_query(tmp_path,monkeypatch):
    from experiments.cvs_clean_eval import contracts
    from experiments.cvs_orthopoly_identity import dispatch
    variants=['adaptive_volterra_lag4',*VARIANTS]
    records=[dict(variant=v,seed=s,accuracy=.93 if v==VARIANTS[0] else .91,worst_rx=.90,
                  parameters=202553,macs=100) for v in variants for s in dispatch.SEEDS]
    matrix=dict(source_controls=[dict(variant='adaptive_volterra_lag4',model_seed=s) for s in dispatch.SEEDS],
                rows=[dict(variant=v,model_seed=s) for v in VARIANTS for s in dispatch.SEEDS])
    monkeypatch.setattr(dispatch,'validate_spec',lambda _:matrix)
    called=[]
    def record(row,original,method):
        called.append(method)
        return next(r for r in records if r['variant']==row['variant'] and r['seed']==row['model_seed'])
    monkeypatch.setattr(dispatch,'read_source_record',record)
    selection=dict(dispatch.select_source_candidate(records),scope='orthopoly_source',source_matrix_ref='synthetic-matrix')
    monkeypatch.setattr(contracts,'read',lambda p: selection if str(p)=='selection' else {})
    assert contracts.frozen_selection('selection')==selection
    assert called.count('cvs_adaptive_volterra_identity')==4 and called.count('cvs_orthopoly_identity')==8
    selection['selected_variant']=VARIANTS[1]
    with pytest.raises(ValueError):contracts.frozen_selection('selection')
    for r in records:r['accuracy']=.99 if r['variant']=='adaptive_volterra_lag4' else .91
    selection.clear();selection.update(dispatch.select_source_candidate(records),scope='orthopoly_source',source_matrix_ref='synthetic-matrix')
    with pytest.raises(ValueError,match='baseline retained'):contracts.frozen_selection('selection')


def synthetic_36(tmp_path):
    import numpy as np
    from experiments.cvs_clean_eval import contracts,score
    def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d),encoding='utf-8')
    selection=dict(scope='orthopoly_source',status='SOURCE_SELECTION_FROZEN',selected_variant=VARIANTS[0],
                   new_candidate_selected=True,target_access=False,target_score_used=False)
    save(tmp_path/'selection.json',selection)
    capsule=tmp_path/'capsule';capsule.mkdir();ids=np.array(['synthetic-'+str(i) for i in range(12)])
    np.savez(capsule/'index.npz',ids=ids)
    save(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=list('abcdef')))
    save(tmp_path/'truth.json',{sid:dict(label=i%6,receiver=i%2) for i,sid in enumerate(ids.tolist())})
    runtime=tmp_path/'new-run';runtime.mkdir();rows=[]
    for v in contracts.evaluation_variants(selection):
        for seed in sorted(contracts.SEEDS):
            rid=f'{v}-s{seed}';old=score.REUSED_BASELINE_RUN if v in score.BASELINES else score.REUSED_RESIDUAL_RUN if v=='residual_fusion' else score.REUSED_ENERGY_RUN if v=='energy_equivariant' else score.REUSED_COUPLED_RUN if v=='coupled_lag4' else score.REUSED_ADAPTIVE_RUN if v=='adaptive_volterra_lag4' else None
            out=(tmp_path/old if old else runtime)/rid/'prediction';out.mkdir(parents=True)
            original=dict(scope='baseline_only',status='FIXED_BASELINES_FROZEN',test_variants=list(score.BASELINES),model_seeds=sorted(score.SEEDS),target_access=False,target_score_used=False) if v in score.BASELINES else dict(scope='energy_source' if v=='energy_equivariant' else 'coupled_source' if v=='coupled_lag4' else 'adaptive_volterra_source' if v=='adaptive_volterra_lag4' else 'historical_residual_source',new_candidate_selected=True,status='SOURCE_SELECTION_FROZEN',selected_variant=v,target_access=False,target_score_used=False)
            sel=tmp_path/(old or 'new-run')/'selection.json'
            if old:save(sel,original)
            else:sel=tmp_path/'selection.json'
            cfg=dict(method='cvs_clean_eval',variant=v,model_seed=seed,output_root=str(out),selection_file=str(sel),p1_capsule=str(capsule),views=['clean'],
                     energy_source_root=str(tmp_path/'energy-source'),adaptive_source_root=str(tmp_path/'energy-source'),coupled_source_root=str(tmp_path/'energy-source'),source_output=str(tmp_path/'energy-source'/rid/'source'))
            cf=tmp_path/'configs'/f'{rid}.json';save(cf,cfg)
            row=dict(row_id=rid,variant=v,model_seed=seed,config=str(cf),output_root=str(out))
            if old:
                row['reuse_from_run']=old;models=4 if v in score.BASELINES else 5 if v=='residual_fusion' else 7 if v=='coupled_lag4' else 8 if v=='adaptive_volterra_lag4' else 6
                save(out.parents[1]/'scoring_clean_complete.json',dict(status='SCORED_COMPLETE',models=models,seeds=4,rows=models*4))
            pred=np.arange(12)%6
            if v=='energy_equivariant':pred[0]=1
            np.savez(out/'clean_predictions.npz',ids=ids,clean=pred.astype('int64'))
            save(out/'clean_complete.json',dict(status='PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,views=['clean'],count=12))
            save(out/'resolved_config.json',dict(cfg,truth_read=False,query_fit=False))
            save(out/'provenance.json',dict(status='VERIFIED',query_fit=False))
            rows.append(row)
    return dict(rows=rows,selection_file=str(tmp_path/'selection.json'),runtime_root=str(runtime),p1_truth=str(tmp_path/'truth.json'))


@pytest.mark.parametrize('corruption',[None,'energy_selection','energy_marker','wrong_reuse','missing_prediction'])
def test_all36_truth_last_energy_reuse_and_paired_score(tmp_path,monkeypatch,corruption):
    from experiments.cvs_clean_eval import contracts,score
    spec=synthetic_36(tmp_path)
    energy=next(r for r in spec['rows'] if r['variant']=='energy_equivariant')
    cfg=score.read(energy['config'])
    # Isolate historical ranking; source recomputation is independently exercised above.
    monkeypatch.setattr(contracts,'frozen_selection',lambda p:score.read(p))
    assert contracts.validate_reused_row(energy)==cfg
    if corruption=='energy_selection':Path(cfg['selection_file']).write_text(json.dumps(dict(score.read(cfg['selection_file']),selected_variant='energy_half')),encoding='utf-8')
    elif corruption=='energy_marker':(Path(energy['output_root']).parents[1]/'scoring_clean_complete.json').write_text(json.dumps(dict(status='SCORED_COMPLETE',models=6,seeds=4,rows=23)),encoding='utf-8')
    elif corruption=='wrong_reuse':energy['reuse_from_run']='unknown'
    elif corruption=='missing_prediction':(Path(spec['rows'][-1]['output_root'])/'clean_complete.json').unlink()
    opened=[];original=score.read
    def observed(p):opened.append(str(p));return original(p)
    monkeypatch.setattr(score,'read',observed)
    if corruption:
        with pytest.raises((ValueError,FileNotFoundError)):score.score(spec)
        assert spec['p1_truth'] not in opened
        if corruption!='missing_prediction':
            with pytest.raises(ValueError):contracts.validate_reused_row(energy)
    else:
        result=score.score(spec)
        assert result['models']==9 and result['rows']==36 and result['records']==108
        position=opened.index(spec['p1_truth'])
        assert sum(Path(p).name=='clean_complete.json' for p in opened[:position])==36
        summary=original(Path(spec['runtime_root'])/'clean_summary.json')
        pair=next(p for p in summary['paired'] if p['receiver']=='ALL' and p['baseline']=='energy_equivariant')
        assert pair['accuracy_delta_pp_mean']==pytest.approx(100/12) and pair['positive_seeds']==4


@pytest.mark.parametrize('variant',VARIANTS)
def test_prepare_only_four_new_configs_and_preserves32_controls(tmp_path,monkeypatch,variant):
    import shutil
    from experiments.cvs_orthopoly_clean import prepare
    root=Path(__file__).resolve().parents[1]
    base=tmp_path/'experiments/cvs_adaptive_volterra_clean/configs';base.mkdir(parents=True)
    for name in ('launch_spec.json','experiment_spec.json'):shutil.copyfile(root/'experiments/cvs_adaptive_volterra_clean/configs'/name,base/name)
    monkeypatch.setattr(prepare,'ROOT',tmp_path)
    selection=dict(status='SOURCE_SELECTION_FROZEN',selected_variant=variant,new_candidate_selected=True,target_access=False,target_score_used=False)
    prepare.main(selection)
    output=tmp_path/'experiments/cvs_orthopoly_clean/configs'
    launch=json.loads((output/'launch_spec.json').read_text(encoding='utf-8'))
    assert len(launch['rows'])==36 and sum(bool(r.get('reuse_from_run')) for r in launch['rows'])==32
    assert {r['variant'] for r in launch['rows'] if not r.get('reuse_from_run')}=={variant}
    old=json.loads((base/'launch_spec.json').read_text(encoding='utf-8'))
    assert all({k:r[k] for k in prior}==prior for r,prior in zip(launch['rows'][:32],old['rows']))
    with pytest.raises(ValueError):prepare.main(dict(selection,new_candidate_selected=False))

@pytest.mark.parametrize('corruption',[None,'adaptive_selection','adaptive_marker','adaptive_path'])
def test_adaptive_reuse_and_truth_closed_on_corruption(tmp_path,monkeypatch,corruption):
    from experiments.cvs_clean_eval import contracts,score
    spec=synthetic_36(tmp_path)
    adaptive=next(r for r in spec['rows'] if r['variant']=='adaptive_volterra_lag4')
    cfg=score.read(adaptive['config'])
    monkeypatch.setattr(contracts,'frozen_selection',lambda p:score.read(p))
    assert contracts.validate_reused_row(adaptive)==cfg
    if corruption=='adaptive_selection':Path(cfg['selection_file']).write_text(json.dumps(dict(score.read(cfg['selection_file']),selected_variant='adaptive_volterra_lag1')),encoding='utf-8')
    elif corruption=='adaptive_marker':(Path(adaptive['output_root']).parents[1]/'scoring_clean_complete.json').write_text(json.dumps(dict(status='SCORED_COMPLETE',models=8,seeds=4,rows=31)),encoding='utf-8')
    elif corruption=='adaptive_path':adaptive['output_root']=adaptive['output_root'].replace(score.REUSED_ADAPTIVE_RUN,'outside')
    opened=[];original=score.read
    def observed(p):opened.append(str(p));return original(p)
    monkeypatch.setattr(score,'read',observed)
    if corruption:
        with pytest.raises((ValueError,FileNotFoundError)):score.score(spec)
        assert spec['p1_truth'] not in opened
        with pytest.raises(ValueError):contracts.validate_reused_row(adaptive)
    else:
        result=score.score(spec)
        assert result['models']==9 and result['rows']==36 and result['records']==108
        position=opened.index(spec['p1_truth'])
        assert sum(Path(p).name=='clean_complete.json' for p in opened[:position])==36
