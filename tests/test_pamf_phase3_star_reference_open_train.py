import copy
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_phase3_open_world import star_reference_open_train as t
from tests.test_pamf_phase3_star_open_world_train import config as old_config


def config(arm='C1'):
    c=copy.deepcopy(old_config());out='synthetic-reference/'+arm+'/train'
    c.update(run_id=t.RUN_ID,arm=arm,output_root=out);c['native_config'].update(run_id=t.RUN_ID,row_id=arm+'-s2026092701',output_root=out)
    return c


def test_new_identity_and_frozen_budget_permissions():
    c=config();assert t.validate_config(c) is c and t.MODEL_PARAMETERS==315106
    for key,value in [('arm','M'),('steps',222),('target_truth','forbidden'),('reference_mode','fit-on-Q')]:
        bad=copy.deepcopy(c);bad[key]=value
        with pytest.raises(ValueError):t.validate_config(bad)


@pytest.mark.parametrize('arm',['C0','C1'])
def test_actual_row_uses_unchanged_rotating_M_loop_and_conditioner_gradients(monkeypatch,arm):
    model=nn.Module();model.open_world=nn.Module();model.open_world.conditioner=nn.Sequential(nn.Linear(29,32),nn.GELU(),nn.Linear(32,640))
    reference=torch.randn(4,29);reference[:,27]=0.
    x=reference if arm=='C1' else torch.zeros_like(reference)
    model.open_world.conditioner(x).square().mean().backward()
    calls=[]
    def loop(m,L,OE,c,device,optimizer,callback):
        assert c['arm']=='M' and c['steps']==2220 and c['sampling_seed']==2026103017
        row=dict(step=1,arm='M');callback(row);return [row],dict(native=True)
    monkeypatch.setattr(t.previous_open,'_loop',loop)
    rows,state=t._loop(model,object(),object(),config(arm),'cpu',None,calls.append)
    assert rows[0]['arm']==arm and rows[0]['episode_membership_arm']=='M' and state==dict(native=True)
    assert rows[0]['reference_mode']==('reference' if arm=='C1' else 'constant')
    assert rows[0]['conditioner_CFO_column_grad_norm']==0.
    if arm=='C0':assert rows[0]['conditioner_first_weight_grad_norm']==0.
    else:assert rows[0]['conditioner_first_weight_grad_norm']>0


def test_wrong_constant_or_CFO_gradient_is_not_hidden(monkeypatch):
    model=nn.Module();model.open_world=nn.Module();model.open_world.conditioner=nn.Sequential(nn.Linear(29,32),nn.GELU(),nn.Linear(32,640))
    model.open_world.conditioner(torch.ones(3,29)).sum().backward()
    monkeypatch.setattr(t.previous_open,'_loop',lambda m,L,OE,c,device,optimizer,callback:callback(dict(step=1)))
    with pytest.raises(ValueError):t._loop(model,None,None,config('C0'),'cpu',None)
