"""Synthetic-only coverage: no registered IQ, target scores, or remote access."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.cvs_clean_eval import contracts, score
from experiments.cvs_mirror_subspace_identity.model import VARIANTS, relation_contract
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def fixture(variant):
    architecture=relation_contract(variant);parameters=architecture['base_trainable_parameters']+architecture['new_trainable_parameters']
    source='/source/'+variant+'-s2026092701/source'
    c=dict(method='cvs_mirror_subspace_identity',variant=variant,model_seed=2026092701,epochs=200,batch_size=128,
        lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],
        selection='fixed_last_epoch',split_seed=392005,mirror_relation=relation_contract(variant),numerical_policy=copy.deepcopy(FULL_FP32_POLICY),
        steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},target_access=False,mirror_relation_active=True,
        total_parameters=parameters,trainable_parameters=parameters,mirror_relation_actual=relation_contract(variant),classifier_scale=30.,backend_flags=copy.deepcopy(FULL_FP32_POLICY),
        dataset='/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl',
        source_contract='/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
        commit=contracts.MIRROR_RELATION_SOURCE_COMMIT,output_root=source,precision='float32')
    contract=dict(role_ids={'L_s':['l'],'U_s':['u'],'V':['v']},source_rxs=[1,3,4,6,8],source_days=[1,2,3],ratios=[.1,.9],split_seed=392005,
        num_classes=6,classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,
        model_seed=c['model_seed'],physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,backend_flags=copy.deepcopy(FULL_FP32_POLICY),checkpoint=str(Path(source)/'last.pt'))
    payload=dict(epoch=200,source_contract=contract,initialization=initial,selection='fixed_last_epoch',method=c['method'],variant=variant,config=c,classes=contract['classes'],num_classes=6,
        model={'core.mirror_relation.window':torch.hann_window(64,periodic=True)})
    return dict(variant=variant,model_seed=c['model_seed'],source_output=source,source_release_commit=contracts.MIRROR_RELATION_SOURCE_COMMIT),done,initial,contract,copy.deepcopy(contract),c,payload


@pytest.mark.parametrize('variant',VARIANTS)
def test_registered_source_contract_and_matched_parameter_count(variant):
    args=fixture(variant);contracts.checkpoint_contract(*args)
    model=contracts.build_model(variant)
    assert contracts.source_method(variant)=='cvs_mirror_subspace_identity'
    assert model.contract()==relation_contract(variant)
    assert sum(p.numel() for p in model.parameters())==relation_contract(variant)["total_parameters"]


@pytest.mark.parametrize('corruption',('physical_ids','full_contract','target','ancestor','inheritance','missing_policy','resolved_tf32','completion_tf32',
    'inactive','architecture','lag','payload','steps','total_parameters','trainable_parameters','lr','batch','scale',
    'source_commit','declared_commit','source_path','checkpoint_path','precision','source_data','relative_floor','gain_qualifier','fp16_state','nonfinite_state','window'))
def test_corruption_rejected_before_query(corruption):
    args=fixture(VARIANTS[0]);cfg,done,initial,contract,expected,resolved,payload=args
    if corruption=='physical_ids':contract['role_ids']['V']=['other']
    elif corruption=='full_contract':expected['physical_identity']='required'
    elif corruption=='target':done['target_evaluated']=True
    elif corruption=='ancestor':initial['ancestors']=['old.pt']
    elif corruption=='inheritance':initial['checkpoint_sources']=['teacher.pt']
    elif corruption=='missing_policy':del resolved['numerical_policy']
    elif corruption=='resolved_tf32':resolved['backend_flags']['cudnn_allow_tf32']=True
    elif corruption=='completion_tf32':done['backend_flags']['cudnn_allow_tf32']=True
    elif corruption=='inactive':resolved['mirror_relation_active']=False
    elif corruption=='architecture':resolved['mirror_relation_actual']['spectral_hop']=16
    elif corruption=='relative_floor':resolved['mirror_relation_actual']['relation_relative_power_floor']=0.
    elif corruption=='gain_qualifier':resolved['mirror_relation_actual']['relation_weak_frequency_gain_invariance']=True
    elif corruption=='fp16_state':payload['model']['core.mirror_relation.window']=payload['model']['core.mirror_relation.window'].half()
    elif corruption=='nonfinite_state':payload['model']['core.mirror_relation.window'][0]=float('nan')
    elif corruption=='window':payload['model']['core.mirror_relation.window'][0]=.1
    elif corruption=='lag':resolved['mirror_relation_actual']['phase_lag']=1
    elif corruption=='payload':payload['variant']=VARIANTS[1]
    elif corruption=='steps':done['steps']=9999
    elif corruption in ('total_parameters','trainable_parameters'):resolved[corruption]+=1
    elif corruption=='lr':resolved['lr']=.001
    elif corruption=='batch':resolved['batch_size']=64
    elif corruption=='scale':resolved['classifier_scale']=16.
    elif corruption=='source_commit':resolved['commit']='wrong-source'
    elif corruption=='declared_commit':cfg['source_release_commit']='wrong-source'
    elif corruption=='source_path':resolved['output_root']='/another-source'
    elif corruption=='checkpoint_path':done['checkpoint']='/another/last.pt'
    elif corruption=='precision':resolved['precision']='float16'
    elif corruption=='source_data':resolved['dataset']='different.pkl'
    with pytest.raises(ValueError):contracts.checkpoint_contract(*args)


def selection(variant=VARIANTS[0]):
    return dict(scope='mirror_relation_source',status='SOURCE_SELECTION_FROZEN',selected_variant=variant,
                new_candidate_selected=True,target_access=False,target_score_used=False,
                candidate_universe=['neural_residual_shallow','response_anchor_mean','relation_frequency_energy',*VARIANTS],source_release_commit=contracts.MIRROR_RELATION_SOURCE_COMMIT)


def test_unselected_variant_and_source_path_rejected():
    chosen=VARIANTS[0];s=selection()
    c=dict(method='cvs_clean_eval',views=['clean'],variant=chosen,model_seed=2026092701,mirror_relation_source_root='/source',source_output='/source/'+chosen+'-s2026092701/source',source_release_commit=contracts.MIRROR_RELATION_SOURCE_COMMIT)
    assert contracts.validate_predict_config(c,s)==c
    assert len(contracts.evaluation_variants(s))==13
    for wrong in (dict(c,variant=VARIANTS[1]),dict(c,variant='neural_residual_shallow'),dict(c,variant='response_anchor_mean'),dict(c,source_output='/other'),dict(c,teacher='old.pt'),dict(c,views=['leo']),dict(c,source_release_commit='wrong')):
        with pytest.raises(ValueError):contracts.validate_predict_config(wrong,s)
    for retained in ('neural_residual_shallow','response_anchor_mean','relation_frequency_energy'):
        with pytest.raises(ValueError,match='Retained source controls'):
            contracts.validate_predict_config(dict(c,variant=retained),dict(s,selected_variant=retained,new_candidate_selected=False))


@pytest.mark.parametrize('retained',('neural_residual_shallow','response_anchor_mean','relation_frequency_energy'))
def test_source_freeze_recomputed_with_twenty_records_and_control_retention(monkeypatch,retained):
    from experiments.cvs_mirror_subspace_identity import dispatch
    records=[dict(variant=v,seed=s,accuracy=.93 if v==VARIANTS[0] else .91,worst_rx=.90,
                  parameters=relation_contract(v)['total_parameters'] if v in VARIANTS else 220987,macs=100)
             for v in ('neural_residual_shallow','response_anchor_mean','relation_frequency_energy',*VARIANTS) for s in contracts.SEEDS]
    matrix=dict(source_controls=[dict(variant=v,model_seed=s) for v in ('neural_residual_shallow','response_anchor_mean','relation_frequency_energy') for s in contracts.SEEDS],
                rows=[dict(variant=v,model_seed=s,source_output='synthetic/'+v+'-'+str(s)) for v in VARIANTS for s in contracts.SEEDS])
    monkeypatch.setattr(dispatch,'validate_spec',lambda _:matrix);called=[]
    def record(row,original,method):
        called.append(method);return next(r for r in records if r['variant']==row['variant'] and r['seed']==row['model_seed'])
    monkeypatch.setattr(dispatch,'read_source_record',record)
    frozen=dict(dispatch.select_source_candidate(records),scope='mirror_relation_source',source_matrix_ref='fixture',source_release_commit=contracts.MIRROR_RELATION_SOURCE_COMMIT)
    actual_commit=[contracts.MIRROR_RELATION_SOURCE_COMMIT]
    monkeypatch.setattr(contracts,'read',lambda p:frozen if p=='selection' else {'commit':actual_commit[0]})
    assert contracts.frozen_selection('selection')==frozen
    assert called.count('cvs_neural_residual_identity')==4 and called.count('cvs_response_fusion_identity')==4 and called.count('cvs_mirror_subspace_identity')==8
    actual_commit[0]='wrong-source-release'
    with pytest.raises(ValueError,match='Actual mirror_relation source commit'):contracts.frozen_selection('selection')
    actual_commit[0]=contracts.MIRROR_RELATION_SOURCE_COMMIT
    frozen['selected_variant']=VARIANTS[1]
    with pytest.raises(ValueError):contracts.frozen_selection('selection')
    for r in records:r['accuracy']=.99 if r['variant']==retained else .9
    frozen.clear();frozen.update(dispatch.select_source_candidate(records),scope='mirror_relation_source',source_matrix_ref='fixture',source_release_commit=contracts.MIRROR_RELATION_SOURCE_COMMIT)
    with pytest.raises(ValueError,match='baseline retained'):contracts.frozen_selection('selection')


def synthetic_52(tmp_path,variant=VARIANTS[0]):
    frozen=selection(variant);save(tmp_path/'selection.json',frozen)
    capsule=tmp_path/'capsule';capsule.mkdir();ids=np.array(['synthetic-'+str(i) for i in range(12)])
    np.savez(capsule/'index.npz',ids=ids);save(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=list('abcdef')))
    save(tmp_path/'truth.json',{sid:dict(label=i%6,receiver=i%2) for i,sid in enumerate(ids.tolist())})
    runtime=tmp_path/'new-run';runtime.mkdir();rows=[]
    old_runs={'residual_fusion':(score.REUSED_RESIDUAL_RUN,5,'historical_residual_source'),
              'energy_equivariant':(score.REUSED_ENERGY_RUN,6,'energy_source'),
              'coupled_lag4':(score.REUSED_COUPLED_RUN,7,'coupled_source'),
              'adaptive_volterra_lag4':(score.REUSED_ADAPTIVE_RUN,8,'adaptive_volterra_source'),
              'neural_residual_shallow':(score.REUSED_NEURAL_RUN,9,'neural_source'),
              'channel_dual':(score.REUSED_CHANNEL_RUN,10,'channel_order_source'),
              'response_anchor_mean':(score.REUSED_FUSION_RUN,11,'response_fusion_source'),
              'relation_frequency_energy':(score.REUSED_SPECTRAL_RUN,12,'spectral_relation_source')}
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
                neural_source_root=str(tmp_path/'source'),channel_source_root=str(tmp_path/'source'),fusion_source_root=str(tmp_path/'source'),source_output=str(tmp_path/'source'/rid/'source'))
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
def test_all52_fixed_before_first_truth_and_neural_control_reuse(tmp_path,monkeypatch,variant,corruption):
    spec=synthetic_52(tmp_path,variant);neural=next(r for r in spec['rows'] if r['variant']=='neural_residual_shallow');cfg=score.read(neural['config'])
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
        result=score.score(spec);assert (result['models'],result['rows'],result['records'])==(13,52,156)
        assert sum(Path(p).name=='clean_complete.json' for p in opened[:opened.index(spec['p1_truth'])])==52
        assert len(original(Path(spec['runtime_root'])/'clean_summary.json')['paired'])==36


@pytest.mark.parametrize('corruption',(None,'selection','marker','path','variant','provenance','missing_fusion_row'))
def test_frozen_response_anchor_reuse_and_all52_truth_gate(tmp_path,monkeypatch,corruption):
    spec=synthetic_52(tmp_path);anchor=next(r for r in spec['rows'] if r['variant']=='response_anchor_mean')
    cfg=score.read(anchor['config']);out=Path(anchor['output_root'])
    monkeypatch.setattr(contracts,'frozen_selection',lambda p:score.read(p))
    assert contracts.validate_reused_row(anchor)==cfg
    if corruption=='selection':save(Path(cfg['selection_file']),dict(score.read(cfg['selection_file']),selected_variant='response_span_mean'))
    elif corruption=='marker':save(out.parents[1]/'scoring_clean_complete.json',dict(status='SCORED_COMPLETE',models=11,seeds=4,rows=43))
    elif corruption=='path':anchor['output_root']=anchor['output_root'].replace(score.REUSED_FUSION_RUN,'outside-registered-run')
    elif corruption=='variant':anchor['variant']='response_span_mean'
    elif corruption=='provenance':save(out/'provenance.json',dict(status='UNKNOWN',query_fit=False))
    elif corruption=='missing_fusion_row':spec['rows'].remove(anchor)
    opened=[];original=score.read
    def observed(p):opened.append(str(p));return original(p)
    monkeypatch.setattr(score,'read',observed)
    if corruption:
        with pytest.raises((ValueError,FileNotFoundError)):score.score(spec)
        assert spec['p1_truth'] not in opened
        if corruption in ('selection','marker','path','variant'):
            with pytest.raises(ValueError):contracts.validate_reused_row(anchor)
    else:
        assert score.score(spec)['rows']==52
        assert sum(Path(p).name=='clean_complete.json' for p in opened[:opened.index(spec['p1_truth'])])==52


@pytest.mark.parametrize('variant',VARIANTS)
def test_checkpoint_loader_strict_roundtrip_and_fp32(tmp_path,monkeypatch,variant):
    import torch
    from experiments.cvs_clean_eval import predict
    torch.set_num_threads(2)
    c,done,initial,contract,expected,resolved,payload=fixture(variant)
    source=tmp_path/'source'/f'{variant}-s2026092701'/'source';source.mkdir(parents=True)
    resolved['output_root']=str(source);done['checkpoint']=str(source/'last.pt')
    model=contracts.build_model(variant)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'project.weight' in name:p.add_(.01)
        model.core.mirror_relation.mix_real.add_(.15)
    payload['model']=model.state_dict()
    for name,value in [('completion.json',done),('initialization.json',initial),('source_contract.json',contract),('resolved_config.json',resolved)]:save(source/name,value)
    torch.save(payload,source/'last.pt');save(tmp_path/'expected.json',expected)
    capsule=tmp_path/'capsule';capsule.mkdir();ids=np.array(['fixture0','fixture1','fixture2'])
    x=np.random.default_rng(42).normal(size=(3,2,256)).astype('float32');np.savez(capsule/'index.npz',ids=ids);np.save(capsule/'clean.npy',x)
    save(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=contract['classes'],channel='residual/post_sync/noeq'))
    monkeypatch.setattr(predict,'frozen_selection',lambda _:selection(variant))
    c.update(method='cvs_clean_eval',views=['clean'],selection_file='synthetic-only',mirror_relation_source_root=str(tmp_path/'source'),source_output=str(source),
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
def test_prepare_only_four_new_configs_and_preserves48_controls(tmp_path,monkeypatch,variant):
    import shutil
    from experiments.cvs_mirror_subspace_clean import prepare
    root=Path(__file__).resolve().parents[1];base=tmp_path/'experiments/cvs_spectral_relation_clean/configs';base.mkdir(parents=True)
    for name in ('launch_spec.json','experiment_spec.json'):shutil.copyfile(root/'experiments/cvs_spectral_relation_clean/configs'/name,base/name)
    monkeypatch.setattr(prepare,'ROOT',tmp_path);runtime=prepare.main(selection(variant))
    assert len(runtime['rows'])==52 and sum(bool(r.get('reuse_from_run')) for r in runtime['rows'])==48
    old=score.read(base/'launch_spec.json')
    assert all({k:r[k] for k in prior}==prior for r,prior in zip(runtime['rows'][:48],old['rows']))
    assert {r['variant'] for r in runtime['rows'][48:]}=={variant}
    assert runtime['run_id']=='20261003-phase1-cvs-mirror-subspace-clean-manysig-m52-r01'
    assert {r['reuse_from_run'] for r in runtime['rows'] if r['variant']=='relation_frequency_energy'}=={prepare.OLD_RUN}
    for seed in contracts.SEEDS:
        cfg=score.read(tmp_path/'experiments/cvs_mirror_subspace_clean/configs'/f'{variant}-s{seed}.json')
        assert cfg['mirror_relation_source_root']==prepare.SOURCE and 'neural_source_root' not in cfg
    with pytest.raises(ValueError):prepare.main(dict(selection(variant),new_candidate_selected=False))
    with pytest.raises(FileExistsError):prepare.main(selection(variant))


def test_stale_source_freeze_without_anchor_cannot_generate_configs(tmp_path,monkeypatch):
    from experiments.cvs_mirror_subspace_clean import prepare
    monkeypatch.setattr(prepare,'ROOT',tmp_path)
    stale=dict(selection(),candidate_universe=['neural_residual_shallow',*VARIANTS])
    with pytest.raises(ValueError,match='Source-only freeze'):prepare.main(stale)
    assert not (tmp_path/'experiments/cvs_mirror_subspace_clean/configs').exists()


@pytest.mark.parametrize('blocked',(False,True))
def test_clean_publisher_uses_preflight_before_scp_without_changing_legacy_default(tmp_path,monkeypatch,blocked):
    from experiments.cvs_clean_eval import publish
    oid='c'*40;events=[]
    def git(command,**kwargs):
        if 'rev-parse' in command:return oid+'\n'
        if 'branch' in command:return 'synthetic\n'
        if 'ls-remote' in command:return oid+'\trefs/heads/synthetic\n'
        if 'ls-files' in command or 'status' in command:return ''
        raise AssertionError(command)
    def preflight(cfg):
        events.append('preflight')
        assert cfg['matrix_rows']==52 and cfg['run'].endswith('mirror-subspace-clean-manysig-m52-r01')
        if blocked:raise FileExistsError('synthetic existing archive')
        return dict(synthetic=True,existing=[])
    monkeypatch.setattr(publish.subprocess,'check_output',git)
    monkeypatch.setattr(publish.subprocess,'run',lambda command,**kwargs:events.append(command[0]))
    monkeypatch.setattr(publish,'ssh',lambda script:events.append('submit') or b'{"status":"SUBMITTED","synthetic":true}')
    from experiments.cvs_mirror_subspace_clean.prepare import RUN,RELEASE
    if blocked:
        with pytest.raises(FileExistsError):publish.publish(tmp_path,run=RUN,release=RELEASE,matrix_rows=52,preflight=preflight)
        assert events==['preflight']
    else:
        publish.publish(tmp_path,run=RUN,release=RELEASE,matrix_rows=52,preflight=preflight)
        assert events==['preflight','scp','submit']
        assert json.loads((tmp_path/'preflight.json').read_text())==dict(synthetic=True,existing=[])


def test_mirror_relation_clean_cli_supplies_frozen_source_publisher_preflight(monkeypatch,tmp_path):
    import runpy,sys
    from experiments.cvs_clean_eval import publish
    from experiments.cvs_mirror_subspace_identity.publish import preflight
    called=[]
    monkeypatch.setattr(publish,'publish',lambda *a,**kw:called.append(kw))
    monkeypatch.setattr(sys,'argv',['publish','--output',str(tmp_path)])
    runpy.run_module('experiments.cvs_mirror_subspace_clean.publish',run_name='__main__')
    assert len(called)==1 and called[0]['preflight'] is preflight and called[0]['matrix_rows']==52


@pytest.mark.parametrize('corruption',(None,'selection','marker','path','variant','missing'))
def test_original_spectral_winner_reuse_is_checked_before_truth(tmp_path,monkeypatch,corruption):
    spec=synthetic_52(tmp_path)
    row=next(r for r in spec['rows'] if r['variant']=='relation_frequency_energy')
    cfg=score.read(row['config']);original=score.read(cfg['selection_file'])
    original['source_release_commit']=contracts.SPECTRAL_RELATION_SOURCE_COMMIT
    cfg.update(spectral_relation_source_root=str(tmp_path/'source'),source_release_commit=contracts.SPECTRAL_RELATION_SOURCE_COMMIT)
    save(Path(row['config']),cfg);save(Path(cfg['selection_file']),original)
    save(Path(row['output_root'])/'resolved_config.json',dict(cfg,truth_read=False,query_fit=False))
    monkeypatch.setattr(contracts,'frozen_selection',lambda p:score.read(p))
    assert contracts.validate_reused_row(row)==cfg
    if corruption=='selection':save(Path(cfg['selection_file']),dict(original,selected_variant='relation_packet_energy'))
    elif corruption=='marker':save(Path(row['output_root']).parents[1]/'scoring_clean_complete.json',dict(status='SCORED_COMPLETE',models=12,seeds=4,rows=47))
    elif corruption=='path':row['output_root']=row['output_root'].replace(score.REUSED_SPECTRAL_RUN,'wrong-run')
    elif corruption=='variant':row['variant']='relation_packet_energy'
    elif corruption=='missing':spec['rows'].remove(row)
    opened=[];read=score.read
    def observed(p):opened.append(str(p));return read(p)
    monkeypatch.setattr(score,'read',observed)
    if corruption:
        with pytest.raises((ValueError,FileNotFoundError)):score.score(spec)
        assert spec['p1_truth'] not in opened
        if corruption!='missing':
            with pytest.raises(ValueError):contracts.validate_reused_row(row)
    else:assert score.score(spec)['rows']==52
