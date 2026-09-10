import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_evidence_integration import model_args,source_batch
from post_stage_common import build_baseline_model
from cvsrffi.evidence_head_training import initialize_from_source,validate_evidence_training_args
from cvsrffi.evidence_pipeline import verify_checkpoint_contract,fit_frozen_head,export_deployment,seal_predictions,score_predictions
from cvsrffi.evidence_observation import normalize_observed
from cvsrffi.pairwise_evidence import SharedPairResidual
from cvsrffi.evidence_decision import ConditionAwareCalibrator


def test_missing_coordinates_do_not_renormalize_or_enter_expert():
    z=torch.tensor([[3.,4.]])
    full=normalize_observed(z,torch.tensor([[True,True]]),(2,))
    partial=normalize_observed(z,torch.tensor([[True,False]]),(2,))
    assert full[0,0]==partial[0,0]
    pair=SharedPairResidual(2,2)
    torch.nn.init.normal_(pair.network[-1].weight)
    means=torch.randn(1,3,2);quality=torch.rand(1,2);cov=torch.rand(1,3,2)
    mask=torch.tensor([[True,False]])
    baseline=pair(z,means,torch.zeros(1,2),quality,cov,mask)
    z[:,1]=float('nan');means[:,:,1]=1e15;cov[:,:,1]=float('nan');quality[:,1]=1e30
    torch.testing.assert_close(baseline,pair(z,means,torch.zeros(1,2),quality,cov,mask))


def test_frozen_fit_deploy_register_prediction_truth_last(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(14)
    args=model_args();base=build_baseline_model(args,torch.device('cpu'))
    x,y=source_batch()
    ids=[f'{int(y[i])}:0:0:{i}' for i in range(len(x))]
    contract={'dataset_id':'synthetic_test_only','source_receivers':[0],'target_receivers':[1],
              'roles':{'L_s':ids,'U_s':[],'V':[],'target':[]}}
    payload={'model':base.state_dict(),'baseline_args':vars(args),'training_data_contract':contract,
             'checkpoint_lineage':{'from_scratch':True,'upstream':[],'selection':'final_only','target_feedback':False}}
    for key,value,error in [('selection','joint_safe','CONTAMINATED'),('target_feedback',True,'CONTAMINATED'),('upstream',['unknown'],'UNVERIFIED')]:
        bad=copy.deepcopy(payload);bad['checkpoint_lineage'][key]=value
        with pytest.raises(ValueError,match=error): verify_checkpoint_contract(bad,contract)
    with pytest.raises(ValueError,match='MISMATCH'): verify_checkpoint_contract(payload,{**contract,'dataset_id':'different'})
    cfg={'variant':'H4','covariance_rank':2}
    model,trained=fit_frozen_head(payload,{'x':x,'y':y,'physical_ids':ids},contract,cfg,epochs=1,batch_size=9)
    assert trained['history'][0]['evidence/support_queries']>0
    exported=export_deployment(model,SimpleNamespace(**trained['baseline_args']),tmp_path/'bundle.pt')
    assert 'training_data_contract' not in exported
    restored=build_baseline_model(SimpleNamespace(**exported['baseline_args']),torch.device('cpu'))
    restored.load_state_dict(exported['model'],strict=True);restored.eval()
    with torch.no_grad(): torch.testing.assert_close(model(x),restored(x))
    # Unknown/new class is an arbitrary label, support-only; no true query roles.
    received={'x':x[:3],'query_ids':['q1','q2','q3'],'protocol_schema':'p2_min_v1',
              'phase2_data_status':'VALIDATED_ONCE','capsule_id':'fixture','split_id':'split'}
    support={'x':x[3:6],'labels':[0,1,'new'],'physical_ids':['s1','s2','s3'],'class_labels':[0,1,'new'],
             'capsule_id':'fixture','split_id':'split'}
    before={k:v.clone() for k,v in restored.state_dict().items()}
    path=tmp_path/'predictions.json'
    pred=seal_predictions(restored,received,path,support=support,source_class_labels=[0,1,2])
    assert len(pred['rows'])==3 and len(pred['rows'][0]['scores'])==3
    assert all(torch.equal(before[k],v) for k,v in restored.state_dict().items())
    truth={'capsule_id':'fixture','split_id':'split','labels':{'q1':0,'q2':1,'q3':'new'}}
    (tmp_path/'truth.json').write_text(json.dumps(truth),encoding='utf-8')
    scored=score_predictions(path,tmp_path/'truth.json',tmp_path/'metrics.json')
    assert scored['prediction_rows']==3
    with pytest.raises(FileExistsError): seal_predictions(restored,received,path)
    with pytest.raises(ValueError,match='whitelist'): seal_predictions(restored,{**received,'truth':y},tmp_path/'bad.json')
    with pytest.raises(ValueError,match='overlap'): seal_predictions(restored,received,tmp_path/'bad.json',support={**support,'physical_ids':['q1','s2','s3']},source_class_labels=[0,1,2])


def test_outside_state_is_explicit_and_uncertainty_not_zero():
    torch.set_num_threads(2)
    m=build_baseline_model(model_args('H3'),torch.device('cpu'));x,y=source_batch()
    initialize_from_source(m,[(x,y)]);m.eval()
    # Force a source support box that excludes the received query shape.
    m.evidence_head.response.state_lower.fill_(0)
    m.evidence_head.response.state_upper.fill_(.01)
    with torch.no_grad(): result=m(x,return_aux=True)['evidence']
    assert (~result['state_in_domain']).all()
    assert (result['state_outside_distance']>0).all()
    assert (result['state_covariance'].diagonal(dim1=-2,dim2=-1).sum(-1)>0).all()


@pytest.mark.parametrize('variant',[None,'H3'])
def test_completed_training_rng_safe_export_predict_and_score(tmp_path,variant):
    import numpy as np
    from cvsrffi.evidence_pipeline import export_completed_training
    torch.set_num_threads(2)
    args=model_args(variant)
    model=build_baseline_model(args,torch.device('cpu'))
    x,y=source_batch()
    if variant is not None: initialize_from_source(model,[(x,y)])
    model.eval()
    contract={'dataset_id':'synthetic_export'}
    payload={'model':model.state_dict(),'baseline_args':vars(args),
             'training_data_contract':contract,
             'checkpoint_lineage':{'from_scratch':True,'upstream':[],
                                   'selection':'final_only','target_feedback':False},
             'rng_state':{'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state()}}
    torch.save(payload,tmp_path/'final_ssdg.pth')
    export_completed_training(tmp_path)
    ground=torch.load(tmp_path/'ground_checkpoint.pt',weights_only=True)
    assert set(ground)=={'model','baseline_args','training_data_contract','checkpoint_lineage'}
    assert verify_checkpoint_contract(ground,contract)
    bundle=torch.load(tmp_path/'deployment.pt',weights_only=True)
    restored=build_baseline_model(SimpleNamespace(**bundle['baseline_args']),torch.device('cpu'))
    restored.load_state_dict(bundle['model'],strict=True);restored.eval()
    received={'x':x[:2],'query_ids':['q1','q2'],'protocol_schema':'p2_min_v1',
              'phase2_data_status':'VALIDATED_ONCE','capsule_id':'export','split_id':'split'}
    pred=seal_predictions(restored,received,tmp_path/'pred.json')
    with torch.no_grad():
        for i,row in enumerate(pred['rows']):
            expected=model(x[i:i+1],return_aux=True)['tx_logits'][0]
            torch.testing.assert_close(torch.tensor(row['scores']),expected)
    assert pred['class_labels']==[0,1,2]
    if variant is None:
        assert all(row['decision_status']=='H0_identity_only' for row in pred['rows'])
    truth={'capsule_id':'export','split_id':'split','labels':{'q1':0,'q2':1}}
    (tmp_path/'truth.json').write_text(json.dumps(truth),encoding='utf-8')
    assert score_predictions(tmp_path/'pred.json',tmp_path/'truth.json',tmp_path/'metrics.json')['prediction_rows']==2


def test_actual_target_contract_is_checked(tmp_path):
    from cvsrffi.evidence_head_training import validate_data_contract
    def dataset(rx,sig):
        return SimpleNamespace(index=[SimpleNamespace(tx_i=0,rx_i=rx,day_i=0,sig_i=sig)])
    ctx={'dataset_id':'synthetic_contract',
         'train_loader':SimpleNamespace(dataset=dataset(0,0)),
         'unlabeled_loader':SimpleNamespace(dataset=dataset(0,1)),
         'val_loader':SimpleNamespace(dataset=dataset(0,2)),
         'split_info':{'test':{'test_rxs_idx':[1]}},
         'named_test_loaders':{'target':SimpleNamespace(dataset=dataset(1,3))}}
    contract={'dataset_id':'synthetic_contract','source_receivers':[0],'target_receivers':[1],
              'roles':{'L_s':['0:0:0:0'],'U_s':['0:0:0:1'],'V':['0:0:0:2'],'target':['0:1:0:3']}}
    path=tmp_path/'contract.json'
    path.write_text(json.dumps(contract),encoding='utf-8')
    assert validate_data_contract(ctx,path)==contract
    ctx['named_test_loaders']['target'].dataset=dataset(1,4)
    with pytest.raises(ValueError,match='target physical'): validate_data_contract(ctx,path)
    ctx['named_test_loaders']['target'].dataset=dataset(1,3)
    ctx['split_info']['test']['test_rxs_idx']=[2]
    with pytest.raises(ValueError,match='target receiver'): validate_data_contract(ctx,path)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_autocast_actual_leo_scene_routes():
    from cvsrffi.eval import apply_sat_channel_for_scenario
    torch.set_num_threads(2)
    m=build_baseline_model(model_args('H5'),torch.device('cuda'));x,y=source_batch();x=x.cuda();y=y.cuda()
    initialize_from_source(m,[(x,y)]);m.eval()
    for scene in ('leo_clear_weak','leo_low_elev_weak','leo_rain_weak'):
        view,_=apply_sat_channel_for_scenario(x[:2],scene,SimpleNamespace())
        assert not torch.equal(view,x[:2])
        with torch.autocast('cuda',dtype=torch.float16):
            result=m(view,y_tx=y[:2],return_aux=True)
            loss=torch.nn.functional.cross_entropy(result['tx_logits'],y[:2])
        m.zero_grad(set_to_none=True);loss.backward()
        assert torch.isfinite(result['tx_logits']).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
