import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.pamf_quality_preserve import design as d
from experiments.pamf_quality_preserve.model import QualityGate,select_iq
from experiments.pamf_quality_preserve.runner import pair_batch,observation_epochs,save_checkpoint,load_checkpoint


def test_synthetic_targets_and_route_do_not_use_TX_or_F():
    clean=np.ones((3,2,256),np.float32)
    residual=clean*10;full=clean*20
    observations=[dict(residual=residual,full=full)]
    values,targets,route=pair_batch(clean,observations,np.zeros(3,dtype=int),np.array([True,False,True]),np.array([0,1]))
    assert route.tolist()==[True,False]
    assert targets.tolist()==[0.,0.,1.,1.]
    assert np.array_equal(values[:2],clean[:2])
    assert np.all(values[2]==10) and np.all(values[3]==20)


def test_clean_cache_slots_reuse_valid_same_physical_observation():
    clean=np.ones((3,2,256),np.float32)
    available=np.array([[False,True,True],[True,True,False],[True,True,True]])
    plan=observation_epochs(available,0)
    assert plan.tolist()==[1,0,0]
    observations=[]
    for epoch in range(3):
        residual=np.full_like(clean,10*(epoch+1));full=residual*2
        residual[~available[epoch]]=0;full[~available[epoch]]=0
        observations.append(dict(residual=residual,full=full))
    values,_,_=pair_batch(clean,observations,plan,np.ones(3,bool),np.arange(3))
    assert np.all(values[3]==20) and np.all(values[4:]==10)
    assert np.all(values!=0)
    available[:,0]=False
    with pytest.raises(ValueError,match='valid damaged'):
        observation_epochs(available,0)


def test_current_packet_only_routing_and_fixed_boundary():
    received=torch.ones(3,2,256);restored=received*3
    value,keep=select_iq(received,restored,torch.tensor([-1.,0.,1.]))
    assert keep.tolist()==[True,False,False]
    assert torch.equal(value[0],received[0]) and torch.equal(value[1:],restored[1:])
    with pytest.raises(FloatingPointError):
        select_iq(received,restored,torch.full((3,),float('nan')))


def test_quality_model_batch_independence_and_finite_training():
    torch.set_num_threads(2);torch.manual_seed(4)
    model=QualityGate()
    x=torch.randn(4,2,256)
    batch=model.eval()(x)
    individual=torch.cat([model(v[None]) for v in x])
    torch.testing.assert_close(batch,individual,atol=1e-5,rtol=1e-5)
    assert not any(isinstance(m,torch.nn.BatchNorm1d) for m in model.modules())
    loss=F.binary_cross_entropy_with_logits(model.train()(x),torch.tensor([0.,0.,1.,1.]))
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


@pytest.mark.parametrize('key,value',[('TX_labels_read',1),('F_training_access',True),
                                      ('target_training',True),('ancestors',['unknown']),('updates',d.UPDATES+1)])
def test_quality_checkpoint_rejects_permission_or_budget_change(tmp_path,key,value):
    model=QualityGate();path=tmp_path/'parent.pt'
    save_checkpoint(path,model,None,0,0)
    fresh,item=load_checkpoint(path,'cpu')
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
    invalid=copy.deepcopy(item);invalid[key]=value
    bad=tmp_path/'bad.pt';torch.save(invalid,bad)
    with pytest.raises(ValueError,match='provenance/budget'):
        load_checkpoint(bad,'cpu')
