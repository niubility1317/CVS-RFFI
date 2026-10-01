import copy
import json
import pytest
from experiments.cvs_gauge_identity.dispatch import CANDIDATES,SEEDS,select_source_candidate,read_source_record,PROJECT
from experiments.cvs_gauge_identity.source import validate_config
from experiments.cvs_gauge_identity.model import gauge_contract


def records():
    return [dict(variant=v,seed=s,accuracy=.97 if v=='residual_fusion' else .96,worst_rx=.94 if v=='residual_fusion' else .92,parameters=164225,macs=9708836) for v in CANDIDATES for s in sorted(SEEDS)]


def test_existing_source_control_can_win_without_new_query():
    d=select_source_candidate(records())
    assert d['selected_variant']=='residual_fusion' and d['new_candidate_selected'] is False
    assert d['target_access'] is False and d['target_score_used'] is False
    r=records()
    for x in r:
        if x['variant']=='gauge_coherent':x.update(accuracy=.971,worst_rx=.941)
    d=select_source_candidate(r)
    assert d['selected_variant']=='gauge_coherent' and d['new_candidate_selected'] is True
    with pytest.raises(ValueError):select_source_candidate(r[:-1])
    r[-1]['seed']=r[-2]['seed']
    with pytest.raises(ValueError):select_source_candidate(r)


def test_performance_precedes_cost_without_tolerance_band():
    r=records()
    for x in r:
        x.update(accuracy=.97,worst_rx=.94)
        if x['variant']=='gauge_peak':x.update(accuracy=.97001,parameters=999999,macs=999999999)
    assert select_source_candidate(r)['selected_variant']=='gauge_peak'
    for x in r:x.update(accuracy=.97,parameters=164225,macs=9708836)
    assert select_source_candidate(r)['selected_variant']=='residual_fusion'


def fixture(folder):
    seed=min(SEEDS);folder.mkdir()
    original=dict(schema='core90_game_source_roles_v1',role_ids={'L_s':['L'], 'U_s':['U'],'V':['V']},counts={'L_s':6300,'U_s':56700,'V':27000})
    resolved=dict(method='cvs_residual_identity',variant='residual_fusion',model_seed=seed,epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,steps_per_epoch=50,source_counts=original['counts'],U_s_use='unused',target_access=False,precision='float32',gradient_clipping=None,optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,source_contract=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',output_root=str(folder))
    data={'completion.json':dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.97,source_val_worst_rx=.94,source_val_rx_accuracy={str(rx):.94 for rx in [1,3,4,6,8]})),
          'initialization.json':dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch'),
          'resolved_config.json':resolved,'source_contract.json':dict(copy.deepcopy(original),physical_roles='EXACT_MATCH',dataset_path=resolved['dataset'],classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True),
          'resource_profile.json':dict(total_parameters=164225,conv_linear_macs_per_sample=9708836)}
    def save():
        for name,d in data.items():(folder/name).write_text(json.dumps(d),encoding='utf-8')
    save()
    return dict(variant='residual_fusion',model_seed=seed,source_output=str(folder)),original,data,save


@pytest.mark.parametrize('corruption',['roles','aug','ancestor','target','budget','representation','worst'])
def test_actual_control_contract_corruption_rejected(tmp_path,corruption):
    row,original,data,save=fixture(tmp_path/'source')
    assert read_source_record(row,original,'cvs_residual_identity')['variant']=='residual_fusion'
    if corruption=='roles':data['source_contract.json']['role_ids']['V']=['different_physical_id']
    elif corruption=='aug':data['resolved_config.json']['augmentation']=True
    elif corruption=='ancestor':data['initialization.json']['ancestors']=['unknown']
    elif corruption=='target':data['completion.json']['target_evaluated']=True
    elif corruption=='budget':data['completion.json']['steps']=9999
    elif corruption=='representation':data['source_contract.json']['equalized']=0
    elif corruption=='worst':data['completion.json']['final_source_metrics']['source_val_worst_rx']=.99
    save()
    with pytest.raises(ValueError):read_source_record(row,original,'cvs_residual_identity')


def test_source_gauge_rejects_extra_training_and_checkpoint_inputs():
    c=dict(method='cvs_gauge_identity',variant='gauge_peak',epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',split_seed=392005,gauge=gauge_contract('gauge_peak'))
    assert validate_config(c)==c
    for k,v in [('checkpoint','old.pt'),('target_truth','truth.json'),('augmentation',True),('domain_backbone',True),('extra_losses',['physics_loss'])]:
        bad=copy.deepcopy(c);bad[k]=v
        with pytest.raises(ValueError):validate_config(bad)
