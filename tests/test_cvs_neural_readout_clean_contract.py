"""Synthetic-only coverage: no registered IQ, target scores, or remote access."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from experiments.cvs_clean_eval import contracts, score
from experiments.cvs_neural_readout_identity.model import VARIANTS, readout_contract
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def fixture(variant):
    architecture=readout_contract(variant);parameters=architecture['base_trainable_parameters']+architecture['new_trainable_parameters']
    c=dict(method='cvs_neural_readout_identity',variant=variant,model_seed=2026092701,epochs=200,batch_size=128,
        lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],
        selection='fixed_last_epoch',split_seed=392005,readout=readout_contract(variant),numerical_policy=copy.deepcopy(FULL_FP32_POLICY),
        steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},target_access=False,learned_readout_active=True,
        total_parameters=parameters,trainable_parameters=parameters,readout_actual=readout_contract(variant),classifier_scale=30.,backend_flags=copy.deepcopy(FULL_FP32_POLICY))
    contract=dict(role_ids={'L_s':['l'],'U_s':['u'],'V':['v']},source_rxs=[1,3,4,6,8],source_days=[1,2,3],ratios=[.1,.9],split_seed=392005,
        num_classes=6,classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,
        model_seed=c['model_seed'],physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,backend_flags=copy.deepcopy(FULL_FP32_POLICY))
    payload=dict(epoch=200,source_contract=contract,initialization=initial,selection='fixed_last_epoch',method=c['method'],variant=variant,config=c,classes=contract['classes'],num_classes=6)
    return dict(variant=variant,model_seed=c['model_seed']),done,initial,contract,copy.deepcopy(contract),c,payload


@pytest.mark.parametrize('variant',VARIANTS)
def test_registered_source_contract_and_distinct_parameter_count(variant):
    args=fixture(variant);contracts.checkpoint_contract(*args)
    model=contracts.build_model(variant)
    assert contracts.source_method(variant)=='cvs_neural_readout_identity'
    assert model.contract()==readout_contract(variant)
    assert sum(p.numel() for p in model.parameters())==(306147 if variant==VARIANTS[0] else 310243)


@pytest.mark.parametrize('corruption',('physical_ids','full_contract','target','ancestor','inheritance','missing_policy','resolved_tf32','completion_tf32',
    'inactive','architecture','lag','payload','steps','total_parameters','trainable_parameters','lr','batch','scale'))
def test_corruption_rejected_before_query(corruption):
    args=fixture(VARIANTS[0]);_,done,initial,contract,expected,resolved,payload=args
    if corruption=='physical_ids':contract['role_ids']['V']=['other']
    elif corruption=='full_contract':expected['physical_identity']='required'
    elif corruption=='target':done['target_evaluated']=True
    elif corruption=='ancestor':initial['ancestors']=['old.pt']
    elif corruption=='inheritance':initial['checkpoint_sources']=['teacher.pt']
    elif corruption=='missing_policy':del resolved['numerical_policy']
    elif corruption=='resolved_tf32':resolved['backend_flags']['cudnn_allow_tf32']=True
    elif corruption=='completion_tf32':done['backend_flags']['cudnn_allow_tf32']=True
    elif corruption=='inactive':resolved['learned_readout_active']=False
    elif corruption=='architecture':resolved['readout_actual']['attention_heads']=8
    elif corruption=='lag':resolved['readout_actual']['phase_lag']=1
    elif corruption=='payload':payload['variant']=VARIANTS[1]
    elif corruption=='steps':done['steps']=9999
    elif corruption in ('total_parameters','trainable_parameters'):resolved[corruption]+=1
    elif corruption=='lr':resolved['lr']=.001
    elif corruption=='batch':resolved['batch_size']=64
    elif corruption=='scale':resolved['classifier_scale']=16.
    with pytest.raises(ValueError):contracts.checkpoint_contract(*args)


def selection(variant=VARIANTS[0]):
    return dict(scope='neural_readout_source',status='SOURCE_SELECTION_FROZEN',selected_variant=variant,
                new_candidate_selected=True,target_access=False,target_score_used=False)


def test_unselected_variant_and_source_path_rejected():
    chosen=VARIANTS[0];s=selection()
    c=dict(method='cvs_clean_eval',views=['clean'],variant=chosen,model_seed=2026092701,readout_source_root='/source',source_output='/source/'+chosen+'-s2026092701/source')
    assert contracts.validate_predict_config(c,s)==c
    assert len(contracts.evaluation_variants(s))==10
    for wrong in (dict(c,variant=VARIANTS[1]),dict(c,variant='neural_residual_shallow'),dict(c,source_output='/other'),dict(c,teacher='old.pt'),dict(c,views=['leo'])):
        with pytest.raises(ValueError):contracts.validate_predict_config(wrong,s)


def test_source_freeze_recomputed_with_twelve_records_and_control_retention(monkeypatch):
    from experiments.cvs_neural_readout_identity import dispatch
    records=[dict(variant=v,seed=s,accuracy=.93 if v==VARIANTS[0] else .91,worst_rx=.90,parameters=306147,macs=100)
             for v in ('neural_residual_shallow',*VARIANTS) for s in contracts.SEEDS]
    matrix=dict(source_controls=[dict(variant='neural_residual_shallow',model_seed=s) for s in contracts.SEEDS],
                rows=[dict(variant=v,model_seed=s) for v in VARIANTS for s in contracts.SEEDS])
    monkeypatch.setattr(dispatch,'validate_spec',lambda _:matrix);called=[]
    def record(row,original,method):
        called.append(method);return next(r for r in records if r['variant']==row['variant'] and r['seed']==row['model_seed'])
    monkeypatch.setattr(dispatch,'read_source_record',record)
    frozen=dict(dispatch.select_source_candidate(records),scope='neural_readout_source',source_matrix_ref='fixture')
    monkeypatch.setattr(contracts,'read',lambda p:frozen if p=='selection' else {})
    assert contracts.frozen_selection('selection')==frozen
    assert called.count('cvs_neural_residual_identity')==4 and called.count('cvs_neural_readout_identity')==8
    frozen['selected_variant']=VARIANTS[1]
    with pytest.raises(ValueError):contracts.frozen_selection('selection')
    for r in records:r['accuracy']=.99 if r['variant']=='neural_residual_shallow' else .9
    frozen.clear();frozen.update(dispatch.select_source_candidate(records),scope='neural_readout_source',source_matrix_ref='fixture')
    with pytest.raises(ValueError,match='baseline retained'):contracts.frozen_selection('selection')


def synthetic_40(tmp_path,variant=VARIANTS[0]):
    frozen=selection(variant);save(tmp_path/'selection.json',frozen)
    capsule=tmp_path/'capsule';capsule.mkdir();ids=np.array(['synthetic-'+str(i) for i in range(12)])
    np.savez(capsule/'index.npz',ids=ids);save(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=list('abcdef')))
    save(tmp_path/'truth.json',{sid:dict(label=i%6,receiver=i%2) for i,sid in enumerate(ids.tolist())})
    runtime=tmp_path/'new-run';runtime.mkdir();rows=[]
    old_runs={'residual_fusion':(score.REUSED_RESIDUAL_RUN,5,'historical_residual_source'),
              'energy_equivariant':(score.REUSED_ENERGY_RUN,6,'energy_source'),
              'coupled_lag4':(score.REUSED_COUPLED_RUN,7,'coupled_source'),
              'adaptive_volterra_lag4':(score.REUSED_ADAPTIVE_RUN,8,'adaptive_volterra_source'),
              'neural_residual_shallow':(score.REUSED_NEURAL_RUN,9,'neural_source')}
    for v in contracts.evaluation_variants(frozen):
        old,models,scope=(score.REUSED_BASELINE_RUN,4,'baseline_only') if v in score.BASELINES else old_runs.get(v,(None,None,None))
        for seed in sorted(contracts.SEEDS):
            rid=f'{v}-s{seed}';out=(tmp_path/old if old else runtime)/rid/'prediction';out.mkdir(parents=True)
            sel=tmp_path/old/'selection.json' if old else tmp_path/'selection.json'
            if old:
                original=dict(scope=scope,status='SOURCE_SELECTION_FROZEN',selected_variant=v,new_candidate_selected=True,target_access=False,target_score_used=False)
                if v in score.BASELINES:original.update(status='FIXED_BASELINES_FROZEN',test_variants=list(score.BASELINES),model_seeds=sorted(score.SEEDS))
                save(sel,original);save(out.parents[1]/'scoring_clean_complete.json',dict(status='SCORED_COMPLETE',models=models,seeds=4,rows=models*4))
            cfg=dict(method='cvs_clean_eval',variant=v,model_seed=seed,output_root=str(out),selection_file=str(sel),p1_capsule=str(capsule),views=['clean'],
                neural_source_root=str(tmp_path/'source'),source_output=str(tmp_path/'source'/rid/'source'))
            cf=tmp_path/'configs'/f'{rid}.json';save(cf,cfg)
            row=dict(row_id=rid,variant=v,model_seed=seed,config=str(cf),output_root=str(out))
            if old:row['reuse_from_run']=old
            np.savez(out/'clean_predictions.npz',ids=ids,clean=(np.arange(12)%6).astype('int64'))
            save(out/'clean_complete.json',dict(status='PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,views=['clean'],count=12))
            save(out/'resolved_config.json',dict(cfg,truth_read=False,query_fit=False));save(out/'provenance.json',dict(status='VERIFIED',query_fit=False))
            rows.append(row)
    return dict(rows=rows,selection_file=str(tmp_path/'selection.json'),runtime_root=str(runtime),p1_truth=str(tmp_path/'truth.json'))


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('corruption',(None,'missing_row','missing_prediction','wrong_ids','neural_selection','neural_marker','neural_reuse','unmarked_control','reused_candidate'))
def test_all40_fixed_before_first_truth_and_neural_control_reuse(tmp_path,monkeypatch,variant,corruption):
    spec=synthetic_40(tmp_path,variant);neural=next(r for r in spec['rows'] if r['variant']=='neural_residual_shallow');cfg=score.read(neural['config'])
    monkeypatch.setattr(contracts,'frozen_selection',lambda p:score.read(p))
    assert contracts.validate_reused_row(neural)==cfg
    if corruption=='missing_row':spec['rows'].pop()
    elif corruption=='missing_prediction':(Path(spec['rows'][-1]['output_root'])/'clean_complete.json').unlink()
    elif corruption=='wrong_ids':np.savez(Path(spec['rows'][-1]['output_root'])/'clean_predictions.npz',ids=np.array(['wrong']*12),clean=np.zeros(12,dtype='int64'))
    elif corruption=='neural_selection':save(Path(cfg['selection_file']),dict(score.read(cfg['selection_file']),selected_variant='neural_residual_deep'))
    elif corruption=='neural_marker':save(Path(neural['output_root']).parents[1]/'scoring_clean_complete.json',dict(status='SCORED_COMPLETE',models=9,seeds=4,rows=35))
    elif corruption=='neural_reuse':neural['reuse_from_run']='unknown'
    elif corruption=='unmarked_control':neural.pop('reuse_from_run')
    elif corruption=='reused_candidate':spec['rows'][-1]['reuse_from_run']=score.REUSED_NEURAL_RUN
    opened=[];original=score.read
    def observed(p):opened.append(str(p));return original(p)
    monkeypatch.setattr(score,'read',observed)
    if corruption:
        with pytest.raises((ValueError,FileNotFoundError)):score.score(spec)
        assert spec['p1_truth'] not in opened
        if corruption.startswith('neural'):
            with pytest.raises(ValueError):contracts.validate_reused_row(neural)
    else:
        result=score.score(spec);assert (result['models'],result['rows'],result['records'])==(10,40,120)
        assert sum(Path(p).name=='clean_complete.json' for p in opened[:opened.index(spec['p1_truth'])])==40
        assert len(original(Path(spec['runtime_root'])/'clean_summary.json')['paired'])==27


@pytest.mark.parametrize('variant',VARIANTS)
def test_checkpoint_loader_strict_roundtrip_and_fp32(tmp_path,monkeypatch,variant):
    import torch
    from experiments.cvs_clean_eval import predict
    torch.set_num_threads(2)
    c,done,initial,contract,expected,resolved,payload=fixture(variant)
    source=tmp_path/'source'/f'{variant}-s2026092701'/'source';source.mkdir(parents=True)
    model=contracts.build_model(variant)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'project.weight' in name:p.add_(.01)
    payload['model']=model.state_dict()
    for name,value in [('completion.json',done),('initialization.json',initial),('source_contract.json',contract),('resolved_config.json',resolved)]:save(source/name,value)
    torch.save(payload,source/'last.pt');save(tmp_path/'expected.json',expected)
    capsule=tmp_path/'capsule';capsule.mkdir();ids=np.array(['fixture0','fixture1','fixture2'])
    x=np.random.default_rng(42).normal(size=(3,2,256)).astype('float32');np.savez(capsule/'index.npz',ids=ids);np.save(capsule/'clean.npy',x)
    save(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=contract['classes'],channel='residual/post_sync/noeq'))
    monkeypatch.setattr(predict,'frozen_selection',lambda _:selection(variant))
    c.update(method='cvs_clean_eval',views=['clean'],selection_file='synthetic-only',readout_source_root=str(tmp_path/'source'),source_output=str(source),
        source_contract=str(tmp_path/'expected.json'),p1_capsule=str(capsule),output_root=str(tmp_path/'prediction'),device='cpu')
    model.eval()
    with torch.no_grad():expected_prediction=model(torch.tensor(x)).argmax(1).tolist()
    predict.predict(c)
    with np.load(tmp_path/'prediction/clean_predictions.npz') as d:assert d['clean'].tolist()==expected_prediction and d['ids'].tolist()==ids.tolist()
    actual=score.read(tmp_path/'prediction/resolved_config.json');assert actual['backend_flags']==FULL_FP32_POLICY
    payload['model'].pop(next(iter(payload['model'])));torch.save(payload,source/'last.pt');c['output_root']=str(tmp_path/'strict-failure')
    with pytest.raises(RuntimeError,match='Missing key'):predict.predict(c)
    assert not (tmp_path/'strict-failure/clean_predictions.npz').exists()


@pytest.mark.parametrize('variant',VARIANTS)
def test_prepare_only_four_new_configs_and_preserves36_controls(tmp_path,monkeypatch,variant):
    import shutil
    from experiments.cvs_neural_readout_clean import prepare
    root=Path(__file__).resolve().parents[1];base=tmp_path/'experiments/cvs_neural_residual_clean/configs';base.mkdir(parents=True)
    for name in ('launch_spec.json','experiment_spec.json'):shutil.copyfile(root/'experiments/cvs_neural_residual_clean/configs'/name,base/name)
    monkeypatch.setattr(prepare,'ROOT',tmp_path);runtime=prepare.main(selection(variant))
    assert len(runtime['rows'])==40 and sum(bool(r.get('reuse_from_run')) for r in runtime['rows'])==36
    old=score.read(base/'launch_spec.json')
    assert all({k:r[k] for k in prior}==prior for r,prior in zip(runtime['rows'][:36],old['rows']))
    assert {r['variant'] for r in runtime['rows'][36:]}=={variant}
    for seed in contracts.SEEDS:
        cfg=score.read(tmp_path/'experiments/cvs_neural_readout_clean/configs'/f'{variant}-s{seed}.json')
        assert cfg['readout_source_root']==prepare.SOURCE and 'neural_source_root' not in cfg
    with pytest.raises(ValueError):prepare.main(dict(selection(variant),new_candidate_selected=False))
    with pytest.raises(FileExistsError):prepare.main(selection(variant))
