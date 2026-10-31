"""Synthetic source-only checks of exact receiver-held-out meta gradients."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_meta_train')


def core_api():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_meta_core')


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(2.,dtype=torch.float64))
        self.unused=nn.Parameter(torch.tensor(7.,dtype=torch.float64))
        self.fixed=nn.Parameter(torch.tensor(3.,dtype=torch.float64),requires_grad=False)
        self.register_buffer('buffer',torch.tensor(5.,dtype=torch.float64))

    def forward(self,x):return self.weight*x


def test_exact_second_order_factor_alpha_zero_and_virtual_state_not_committed():
    c=core_api();m=Scalar();initial={k:v.clone() for k,v in m.state_dict().items()};x=torch.ones(1,dtype=torch.float64)
    alpha=.001;beta=.5
    result=c.bilevel_loss(m,lambda call:.5*(call(x)-1).square().sum(),lambda call:.5*(call(x)-4).square().sum(),alpha=alpha,beta=beta)
    gradient=torch.autograd.grad(result['total'],m.weight)[0]
    virtual=2-alpha*(2-1)
    expected=(2-1)+beta*(virtual-4)*(1-alpha)
    first_order=(2-1)+beta*(virtual-4)
    assert torch.allclose(gradient,torch.tensor(expected,dtype=torch.float64),rtol=0,atol=1e-14)
    assert abs(float(gradient)-first_order)>1e-4
    assert result['diagnostics']['unused_inner_parameters']==['unused']
    assert result['diagnostics']['full_second_order'] and not result['diagnostics']['virtual_updates_committed']
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items())
    result=c.bilevel_loss(m,lambda call:.5*(call(x)-1).square().sum(),lambda call:.5*(call(x)-4).square().sum(),alpha=0.,beta=beta)
    g=torch.autograd.grad(result['total'],m.weight)[0]
    ordinary=.5*(m(x)-1).square().sum()+beta*.5*(m(x)-4).square().sum()
    assert torch.equal(g,torch.autograd.grad(ordinary,m.weight)[0])
    assert m.fixed.grad is None and m.unused.grad is None


def test_sampler_complete_matched_444_receiver_splits_and_public_OE_only(subtests):
    c=core_api();rx=np.repeat(np.asarray(c.RECEIVERS,np.int64),90)
    L=dict(metadata=dict(rx_i=rx,y=np.tile(np.repeat(np.arange(6),15),5).astype(np.int64),day_i=np.tile(np.arange(90)%3+1,5).astype(np.int64)))
    OE=dict(metadata=dict(rx_i=rx,day_i=L['metadata']['day_i'].copy()))
    a,b=c.ReceiverSampler(L,OE),c.ReceiverSampler(L,OE)
    for step in range(1,445):
        left,right=a.sample(step),b.sample(step);held=c.RECEIVERS[(step-1)%5]
        assert left['held_rx']==held
        for key in ('train_L','train_OE','held_L','held_OE'):assert np.array_equal(left[key],right[key])
        for key,count in (('train_L',32),('train_OE',16)):
            source=L if key.endswith('_L') else OE
            selected=source['metadata']['rx_i'][left[key]]
            assert set(selected.tolist())==set(c.RECEIVERS)-{held}
            assert all(int((selected==r).sum())==count for r in c.RECEIVERS if r!=held)
        assert len(left['held_L'])==64 and len(left['held_OE'])==32
        assert np.all(rx[left['held_L']]==held) and np.all(rx[left['held_OE']]==held)
    for change in ({'y':np.zeros(len(rx),np.int64)}, {'rx_i':np.ones(len(rx),np.int64)}):
        with subtests.test(change=list(change)),pytest.raises(ValueError):
            bad=deepcopy(OE);bad['metadata'].update(change);c.ReceiverSampler(L,bad)


def test_selected_anchor_losses_equal_old_all_anchor_values_gradients_and_held_candidates_live():
    c=core_api();old=c.metric;torch.manual_seed(31)
    rx=torch.tensor(np.repeat(np.asarray(c.RECEIVERS),18).tolist());day=torch.tensor(np.tile([1,2,3],30).tolist())
    y=torch.tensor(np.tile(np.repeat(np.arange(6),3),5).tolist())
    z=torch.nn.functional.normalize(torch.randn(90,160),dim=1).requires_grad_()
    oe=torch.nn.functional.normalize(torch.randn(90,160),dim=1).requires_grad_();teacher=torch.nn.functional.normalize(torch.randn(90,160),dim=1)
    current=c.anchor_losses(z,oe,y,rx,day,rx,day,teacher)
    expected=old.crossset_metric(z,oe,y,rx,day,rx,day)['loss']+old.oe_structure(oe,teacher,rx,day)['loss']
    actual=current['crossset_metric']+current['OE_KL']
    assert torch.allclose(actual,expected,rtol=0,atol=2e-6)
    ga=torch.autograd.grad(actual,(z,oe),retain_graph=True);gb=torch.autograd.grad(expected,(z,oe),retain_graph=True)
    assert all(torch.allclose(a,b,rtol=2e-5,atol=2e-6) for a,b in zip(ga,gb))
    held=torch.nonzero(rx==1).flatten()
    with pytest.raises(ValueError,match='cross|both'):old.crossset_metric(z[held],oe[held],y[held],rx[held],day[held],rx[held],day[held])
    selected=c.anchor_losses(z,oe,y,rx,day,rx,day,teacher,L_rows=held,OE_rows=held)
    gradients=torch.autograd.grad(selected['crossset_metric']+selected['OE_KL'],(z,oe))
    assert all(bool(torch.isfinite(g).all()) and g[rx!=1].abs().sum()>0 for g in gradients)


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_source_style_train import fixture as old
    with old(root,mp) as (style,sc,pm,b,parent,banks,events):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(style.COUNTS))
        c=dict(sc,run_id=m.RUN_ID,output_root=str(root/'meta/train'))
        mp.setattr(m,'_reference',lambda c:(pm,{},np.zeros((2,6)),{}))
        mp.setattr(m,'_backend',lambda *args:b)
        mp.setattr(m,'_model',lambda c,pm,p,device:style._model(sc,pm,p,device))
        def source(*args):
            assert events==['checkpoint'];events.append('L_OE')
            return banks,dict(parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_only=True),{
                'L':dict(teacher_logits=parent.id_backbone.encoder.classify_features(parent.id_backbone.encoder.features(torch.tensor(banks['L']['iq'].tolist()))).detach().tolist()),'OE':{}}
        mp.setattr(m,'_source_cache',source)
        yield m,c,pm,b,parent,banks,events


def test_native_samepass_two_arms_second_order_training_strict_checkpoint_and_terminal(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,banks,events):
        before=m._snapshot(parent);done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L_OE']
        assert done['total_optimizer_updates']==4 and done['full_second_order']
        assert done['source_sampling_matched'] and done['all_receiver_splits_disjoint']
        assert not done['virtual_updates_committed'] and done['target_U_reads']==0
        assert all(torch.equal(v,parent.state_dict()[k]) for k,v in before.items())
        for arm in m.ARMS:
            model,i=m.load_trained_model(c,arm,'cpu')
            assert i['signal_parameters']==335681 and i['identity_head_updates']==2
            assert i['meta_enabled']==(arm=='META_NATIVE')
            assert all(not p.requires_grad for p in model.parameters()) and not model.training
            rows=[json.loads(line) for line in (out/arm/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(rows)==2 and all(r['full_second_order'] for r in rows)
        a=[json.loads(x) for x in (out/'ERM_NATIVE/steps.jsonl').read_text().splitlines()]
        z=[json.loads(x) for x in (out/'META_NATIVE/steps.jsonl').read_text().splitlines()]
        assert all(all(x[k]==y[k] for k in ('train_L_indices','train_OE_indices','held_L_indices','held_OE_indices')) for x,y in zip(a,z))
        assert done['real_student_forward_rows']==2*2*480
        with pytest.raises(FileExistsError):m.train(c,'cpu')
        path=out/'META_NATIVE/final.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
        for key in ('identity','sinc','buffer'):
            with subtests.test(key=key):
                bad=deepcopy(payload)
                if key=='identity':bad['identity']['meta_alpha']=0.
                else:
                    name=next(n for n in bad['network'] if (n.startswith(m.core.SINC_PREFIX) if key=='sinc' else n.endswith('unchanged')))
                    bad['network'][name]=bad['network'][name]+1
                torch.save(bad,path)
                with pytest.raises(ValueError):m.load_trained_model(c,'META_NATIVE','cpu')
        torch.save(payload,path)
        # The launcher has the production 444-step budget; preserve every
        # actual toy-produced flag and adapt only its two numeric budgets.
        from experiments.pamf_phase3_open_world import source_meta_launch as launch
        handoff=deepcopy(done);handoff['updates_per_arm']=444;handoff['total_optimizer_updates']=888
        root=tmp_path/'terminal_handoff';(root/'train').mkdir(parents=True)
        (root/'train/complete.json').write_text(json.dumps(handoff),encoding='utf-8')
        assert launch._terminal(dict(run_root=str(root),code_commit=c['code_commit']),'train')==handoff


def test_private_config_and_failed_second_order_smoke_precede_source_values(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,parent,banks,events):
        for delta in (dict(truth='private'),dict(U_root='not allowed'),dict(K=4),dict(steps=1)):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        monkeypatch.setattr(m,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('second-order unsupported')))
        with pytest.raises(ValueError,match='second-order unsupported'):m.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root']).exists()
