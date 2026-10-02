import copy
import json
import numpy as np
import pytest
from experiments.cvs_sixscene_eval.common import VIEWS,SEEDS,write
from experiments.cvs_sixscene_eval.predict import check
from experiments.cvs_sixscene_eval.score import preflight
from experiments.cvs_sixscene_eval.views import configuration

def fixture():
    c=dict(condition='clean_train',model_seed=SEEDS[0]);contract=dict(equalized=1,out_len=256,normalize=True,classes=['14-10','14-7','20-15','20-19','6-15','8-20'],role_ids={'L_s':['a'],'U_s':['b'],'V':['c']})
    done=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],target_access=False,target_contact=False,physical_roles='EXACT_MATCH',model_seed=SEEDS[0])
    resolved=dict(method='cvs_residual_identity',variant='residual_fusion',model_seed=SEEDS[0],selection='fixed_last_epoch',target_access=False,source_counts={'L_s':6300,'U_s':56700,'V':27000},steps_per_epoch=50,domain_backbone=False,extra_losses=[],augmentation=False)
    payload=dict(method='cvs_residual_identity',variant='residual_fusion',config=resolved,source_contract=contract,initialization=initial,epoch=200,selection='fixed_last_epoch')
    return c,done,initial,contract,copy.deepcopy(contract),resolved,payload

@pytest.mark.parametrize('part,key,value',[(0,'truth','forbidden'),(1,'target_evaluated',True),(2,'checkpoint','bad.pt'),(2,'ancestors',['bad.pt']),(2,'physical_roles','UNKNOWN'),(4,'role_ids',{}),(5,'augmentation',True),(6,'epoch',199)])
def test_rejects_contaminated_or_incompatible_checkpoint(part,key,value):
    args=list(fixture());args[part][key]=value
    with pytest.raises(ValueError):check(*args)

def test_valid_scratch_frozen_checkpoint():check(*fixture())

def test_original_role_contract_accepts_verified_runtime_representation():
    args=list(fixture());args[4]={'role_ids':copy.deepcopy(args[3]['role_ids'])};check(*args)

def test_all_rows_fixed_before_truth(tmp_path):
    views=tmp_path/'views';views.mkdir();ids=np.asarray(['a','b']);np.savez(views/'index.npz',ids=ids);rows=[]
    for condition in ('clean_train','mid_low_aug'):
        for seed in SEEDS:
            out=tmp_path/(condition+str(seed));out.mkdir()
            write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,count=2,views=list(VIEWS)))
            np.savez(out/'predictions.npz',ids=ids,**{v:np.asarray([0,1]) for v in VIEWS})
            rows.append(dict(condition=condition,model_seed=seed,output_root=str(out)))
    spec=dict(rows=rows,views_root=str(views),truth=str(tmp_path/'NEVER_OPENED_TRUTH.json'))
    got,p=preflight(spec);assert got.tolist()==ids.tolist() and len(p)==8
    (tmp_path/(rows[-1]['condition']+str(rows[-1]['model_seed']))/'complete.json').unlink()
    with pytest.raises(FileNotFoundError):preflight(spec)

def test_six_fixed_residual_configs():
    for scene in VIEWS[1:]:
        cfg=configuration(scene)
        assert cfg.processing_route=='residual' and cfg.mode=='post_sync' and cfg.equalization_enabled is False and cfg.fs_hz==25e6

def test_model_state_roundtrip_both_forms(tmp_path):
    import torch
    from experiments.cvs_residual_identity.model import build as plain
    from experiments.cvs_selected_concat.model import build as wrapped
    torch.set_num_threads(2)
    for factory in (lambda:plain('residual_fusion'),wrapped):
        a=factory().eval();path=tmp_path/'smoke.pt';torch.save(a.state_dict(),path);b=factory().eval();b.load_state_dict(torch.load(path,weights_only=False),strict=True)
        x=torch.randn(2,2,256)
        with torch.no_grad():assert torch.equal(a(x),b(x)) and torch.isfinite(b(x)).all()
