import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_phase3_open_world import star_open_world_train as t


def config():
    out=str(Path('synthetic-open-world')/'train')
    n=dict(run_id=t.RUN_ID,row_id='M-s2026092701',code_commit='a'*40,output_root=out,model_seed=2026092701,
        base_release='original-release',frozen_sources={k:k for k in ('F','AF','BASE','FOLD','G')},
        source_output='original-F0/source',source_contract='contract.json',source_cache='legacy-cache-MUST-NOT-OPEN',
        registered_classes=t.native.CLASSES.copy(),batch_size=128)
    return dict(run_id=t.RUN_ID,code_commit='a'*40,arm='M',native_config=n,L_root='synthetic-cache/L',OE_root='synthetic-cache/OE',output_root=out,
                steps=2220,lr=.0001,sampling_seed=2026103017,slot_seed=2026103018)


def test_config_rejects_target_truth_and_budget_changes():
    c=config();assert t.validate_config(c) is c
    for key,value in [('Q_truth','private'),('steps',4440),('sampling_seed',2026092701),('arm','choose_best'),('output_root',c['L_root'])]:
        changed=copy.deepcopy(c);changed[key]=value
        with pytest.raises(ValueError):t.validate_config(changed)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Linear(2,64)
        self.head=nn.Linear(64,6);self.open_world=nn.Module();self.open_world.relation_scorer=nn.Linear(128,1)
        self._phase3_parent_args={}


@pytest.mark.parametrize('arm',['S0','S1','M'])
def test_one_step_full_anchor_member_integration(monkeypatch,arm):
    torch.set_num_threads(1);torch.manual_seed(8);model=Tiny();c=config();c.update(arm=arm,steps=1)
    cells=np.asarray([(y,r,d) for y in range(6) for r in (1,3,4,6,8) for d in (1,2,3) for _ in range(70)],np.int64)
    rng=np.random.default_rng(9);L=dict(ids=np.asarray(['L'+str(i) for i in range(6300)]),restored=rng.normal(size=(6300,2,256)).astype(np.float32),
        teacher_logits=np.zeros((6300,6),np.float32),metadata=dict(y=cells[:,0],rx_i=cells[:,1],day_i=cells[:,2],domain=np.searchsorted([1,3,4,6,8],cells[:,1])*3+cells[:,2]-1))
    OE=dict(restored=rng.normal(size=(150,2,256)).astype(np.float32),metadata=dict(domain=np.repeat(np.arange(15,dtype=np.int64),10)))
    def forward(m,x):
        z=torch.nn.functional.normalize(m.id_backbone.encoder(x.mean(2)),dim=1)
        return dict(open64=z,logits=m.head(z))
    def rel(m,z,p):return m.open_world.relation_scorer(torch.cat(((z[:,None]-p[None]).abs(),z[:,None]*p[None]),-1)).squeeze(-1)
    def sup(m,x,y,domains,meta,aug,no_pa,step):
        assert m.training and not m.id_backbone.encoder.training and step==1
        assert len(x)==128 and set(meta)=={'rx_i','day_i','domain'}
        return nn.functional.cross_entropy(forward(m,x)['logits'],y),dict(T1_L_packets=768,CRIC_supcon_weight=.1,T1_PL_seen=0,T1_sat_seen=0,T1_As_KL_weight=0.)
    monkeypatch.setattr(t.native,'_augmentor',lambda args:(None,True,{}));monkeypatch.setattr(t.native,'supervised_loss',sup)
    monkeypatch.setattr(t.ow,'forward',forward);monkeypatch.setattr(t.ow,'relation',rel);monkeypatch.setattr(t.ow,'clear',lambda m:None)
    monkeypatch.setattr(t,'_clip',lambda optimizer,params:nn.utils.clip_grad_norm_(params,5.))
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    rows,_=t._loop(model,L,OE,c,'cpu',torch.optim.AdamW(model.parameters(),lr=.001))
    row=rows[0];assert row['total_F_packet_views']==1152 and row['As_new_channel_generated_samples']==0
    assert row['member_counts']['prototype_count']==(5 if arm=='M' else 6)
    assert row['gradient']['auxiliary_parameter_count']==2
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n.startswith('open_world.relation_scorer.'))
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters() if n.startswith('id_backbone.encoder.'))
    if arm=='S0':assert row['gradient']['shared_member_norm_before']==0
    else:assert row['gradient']['shared_member_norm_before']>0
