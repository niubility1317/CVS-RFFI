import numpy as np
import torch
import pytest
from torch import nn

from experiments.pamf_phase2_practical import clean_additional_worker as worker
from experiments.pamf_phase2_practical.external_execution import infer


def test_m1_query_decisions_are_one_record_each():
    model=nn.Linear(2,6).eval(); calls=[]
    def score(net,x):
        calls.append(len(x));return net(x[:, :, 0])
    state=dict(model=model,score_fn=score,classes=tuple(range(6)))
    pred,_=infer(state,np.ones((4,2,256),np.float32),6,batch=1)
    assert len(pred)==4 and calls==[1,1,1,1]


def test_issl_fixed_v1_uses_long_budget_and_no_target_old(monkeypatch):
    model=nn.Linear(2,6);seen={}
    bundle=dict(profile='issl_fixed_v1',model=model,native_state={
        'source_replay_x':torch.ones(6,2,256), 'source_replay_y':torch.arange(6)})
    monkeypatch.setattr(worker.extra,'_issl_expand',lambda m,count:seen.update(classes=count))
    events=[]
    def emit(event):events.append(event)
    def ssl(m,x,old,spec,seed,logger):
        seen.update(ssl_epochs=spec['C_SSL_epochs'],ssl_rows=len(x),old_count=old)
        logger({'stage':'C_ssl','loss':1.})
    def train(m,x,y,epochs,batch,lr,seed,logger,**kw):
        seen.update(transfer_epochs=epochs,transfer_labels=y.tolist(),stage=kw['stage'])
        logger({'stage':'C_transfer','loss':.5})
    monkeypatch.setattr(worker.extra,'_issl_ssl',ssl)
    monkeypatch.setattr(worker.extra,'_train_ce',train)
    result=worker.direct_issl(bundle,torch.ones(4,2,256),torch.tensor([6,6,7,7]),123,logger=emit)
    assert seen==dict(classes=8,ssl_epochs=2000,ssl_rows=16,old_count=6,transfer_epochs=300,
                      transfer_labels=[0,1,2,3,4,5,6,6,7,7],stage='C_transfer')
    assert result['resources']['actual_target_old_labels']==0
    assert result['resources']['actual_target_new_labels']==4
    assert [x['stage'] for x in events]==['C_ssl','C_transfer']


@pytest.mark.parametrize('budget',[(50,25),(100,50),(200,100),(400,100)])
def test_issl_short_budgets_preserve_source_and_support_contract(monkeypatch,budget):
    model=nn.Linear(2,6);calls=[]
    bundle=dict(profile='issl_fixed_v1',model=model,native_state={
        'source_replay_x':torch.ones(6,2,256),'source_replay_y':torch.arange(6)})
    monkeypatch.setattr(worker.extra,'_issl_expand',lambda m,count:None)
    monkeypatch.setattr(worker.extra,'_issl_ssl',lambda m,x,old,spec,seed,logger:
                        calls.append(('ssl',spec['C_SSL_epochs'],len(x))))
    monkeypatch.setattr(worker.extra,'_train_ce',lambda m,x,y,epochs,batch,lr,seed,logger,**kw:
                        calls.append(('transfer',epochs,y.tolist())))
    result=worker.direct_issl(bundle,torch.ones(4,2,256),torch.tensor([6,6,7,7]),123,budget=budget)
    assert calls==[('ssl',budget[0],16),('transfer',budget[1],[0,1,2,3,4,5,6,6,7,7])]
    assert result['resources']['actual_target_old_labels']==0
    assert (result['resources']['C_SSL_epochs'],result['resources']['C_transfer_epochs'])==budget


def test_issl_rejects_unregistered_budget(monkeypatch):
    bundle=dict(profile='issl_fixed_v1',model=nn.Linear(2,6),native_state={
        'source_replay_x':torch.ones(6,2,256),'source_replay_y':torch.arange(6)})
    monkeypatch.setattr(worker.extra,'_issl_expand',lambda m,count:None)
    with pytest.raises(ValueError,match='Unregistered'):
        worker.direct_issl(bundle,torch.ones(1,2,256),torch.tensor([6]),1,budget=(3,3))


@pytest.mark.parametrize('lr',[3e-5,1e-4])
def test_issl_short_lr_changes_only_ssl_lr(monkeypatch,lr):
    bundle=dict(profile='issl_fixed_v1',model=nn.Linear(2,6),native_state={
        'source_replay_x':torch.ones(6,2,256),'source_replay_y':torch.arange(6)})
    seen={}
    monkeypatch.setattr(worker.extra,'_issl_expand',lambda m,count:None)
    monkeypatch.setattr(worker.extra,'_issl_ssl',lambda m,x,old,spec,seed,logger:seen.update(spec=spec))
    monkeypatch.setattr(worker.extra,'_train_ce',lambda m,x,y,epochs,batch,rate,seed,logger,**kw:seen.update(transfer_lr=rate))
    state=worker.direct_issl(bundle,torch.ones(1,2,256),torch.tensor([6]),1,budget=(50,25),ssl_lr=lr)
    assert seen['spec']['C_SSL_lr']==lr and seen['transfer_lr']==0.03
    assert state['resources']['C_SSL_lr']==lr
    with pytest.raises(ValueError,match='learning rate'):
        worker.direct_issl(bundle,torch.ones(1,2,256),torch.tensor([6]),1,budget=(2000,300),ssl_lr=lr)
