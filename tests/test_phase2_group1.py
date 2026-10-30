import importlib.util
import pytest
import torch


def module():
    name='experiments.pamf_phase2_practical.group1'
    assert importlib.util.find_spec(name) is not None, 'Group1 execution contract is missing'
    return __import__(name,fromlist=['*'])


def test_metric_softmax_reaches_threshold_without_classifier():
    g=module()
    centers=torch.eye(6)
    p=g.metric_probabilities(centers[:2],centers,temperature=.1)
    assert p.shape==(2,6)
    assert (p.max(1).values>=.95).all()
    assert torch.equal(p.argmax(1),torch.tensor([0,1]))


def test_midpoint_and_whole_batch_PL_reduction():
    g=module()
    assert g.pl_enabled(25,50) is False
    assert g.pl_enabled(26,50) is True
    logits=torch.tensor([[4.,0.],[0.,0.]],requires_grad=True)
    confidence=torch.tensor([.99,.8]);labels=torch.tensor([0,1])
    value=g.masked_ce(logits,labels,confidence)
    expected=torch.nn.functional.cross_entropy(logits[:1],labels[:1])/2
    assert torch.allclose(value,expected)
    value.backward()
    assert torch.count_nonzero(logits.grad[1])==0


def test_source_boundary_and_native_paths():
    g=module()
    with pytest.raises(ValueError,match='target'):
        g.validate_source_config({'data_root':'target-bank'})
    for row in g.catalog():
        if row['category']=='CIL': assert row['B'] is None
        if row['category']=='DA': assert row['C'] is None
    assert g.workpoints('MIX01')==([1,5,10,20],[(k,n) for k in [1,5,10,20] for n in [2,5,10,20]])
    assert g.workpoints('MIX02')==([5],[(5,20)])


def test_direct_registration_anchor_never_calls_B(monkeypatch):
    g=module()
    from experiments.pamf_phase2_practical.vendor_native.comparison_suite import registration as cil
    def forbidden(*args,**kwargs): raise AssertionError('generic B must never execute for CIL')
    monkeypatch.setattr(cil,'prepare_registration_state',forbidden)
    model=torch.nn.Linear(3,6)
    bundle=dict(model=model,method='orthogonal',classes=tuple(range(6)),base_metadata={},provenance={})
    anchor=g.registration_anchor(bundle)
    assert anchor.classes==tuple(range(6))
    assert anchor.model is not model
    assert anchor.protocol_metadata['phase']=='A_FROZEN_DIRECT_C'


def test_native_source_optimizes_satellite_and_enables_PL_at_midpoint(tmp_path,monkeypatch):
    import json
    import numpy as np
    from experiments.pamf_phase2_practical import group1_source as source
    from experiments.practical_full_2x2 import cache
    from types import SimpleNamespace
    y=torch.arange(6).repeat_interleave(3)
    x=torch.randn(len(y),2,256)
    class Data:
        def __init__(self,hidden=False):
            self.x=x.repeat(2,1,1) if hidden else x
            self.metadata={'rx_i':np.ones(len(self.x),dtype=int)}
            if not hidden:self.metadata['y']=y.numpy()
        def __len__(self):return len(self.x)
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.fc=torch.nn.Linear(512,6)
        def forward(self,iq):return {'tx_logits':self.fc(iq.flatten(1)),'features':iq.flatten(1)}
    class Reader:
        wait_seconds=0.
        def __init__(self,c):pass
        def get(self,indices,*args):return (x[indices]*1.5).numpy(),{'applied_count':len(indices)}
    monkeypatch.setattr(source,'source_data',lambda c,role:Data(role=='U_s'))
    monkeypatch.setattr(source,'model_for',lambda method,device:Tiny().to(device))
    monkeypatch.setattr(source,'row',lambda name:dict(method='mrior',epochs=2,PL_start_epoch=2))
    monkeypatch.setattr(cache,'Reader',Reader)
    contract=tmp_path/'contract.json';contract.write_text(json.dumps({'source_rxs':[1]}))
    out=tmp_path/'source'
    source._train_native({'source_contract':str(contract)},'mrior_sda',42,out,'cpu')
    records=[json.loads(v) for v in (out/'epochs.jsonl').read_text().splitlines()]
    assert [r['PL_enabled'] for r in records]==[False,True]
    assert all(r['satellite_loss']>0 for r in records)
    assert records[0]['PL_seen']==0 and records[1]['PL_seen']==36
    resources=json.loads((out/'A_resources.json').read_text())
    assert resources['optimizer_steps']==2
    assert resources['satellite_actually_applied']==36
    assert resources['source_U_true_labels']==0
    assert resources['CPU_RSS_sampled_peak']>0


def test_issl_direct_C_keeps_all_classes_without_old_target_support(monkeypatch):
    from experiments.pamf_phase2_practical import group1_target as target
    from experiments.pamf_phase2_practical import external_acceleration as acceleration
    calls={}
    def no_B(*args,**kwargs):raise AssertionError('B is forbidden for direct C')
    monkeypatch.setattr(target.extra,'fit_B',no_B)
    monkeypatch.setattr(target.extra,'_issl_expand',lambda model,count:calls.update(classes=count))
    monkeypatch.setattr(acceleration,'issl_ssl',lambda model,x,old,*a:calls.update(ssl_rows=len(x),old=old))
    monkeypatch.setattr(acceleration,'issl_transfer',lambda model,x,y,epochs,*a,**kw:calls.update(transfer_rows=len(x),epochs=epochs))
    source_x=torch.randn(6,2,256);source_y=torch.arange(6)
    bundle=dict(profile='issl',method='issl_author_behavior',model=torch.nn.Linear(3,6),
        native_state={'source_replay_x':source_x,'source_replay_y':source_y})
    result=target.fit_C({},bundle,torch.randn(4,2,256),torch.tensor([6,6,7,7]),(6,7),42,None)
    assert calls==dict(classes=8,ssl_rows=16,old=6,transfer_rows=10,epochs=300)
    assert result['classes']==tuple(range(8))
    assert result['resources']['actual_target_old_labels']==0


def test_optimizer_counter_includes_scheduler_wrapped_steps(tmp_path):
    import json
    from experiments.pamf_phase2_practical.group1_resources import Measure
    model=torch.nn.Linear(3,2);optimizer=torch.optim.Adam(model.parameters())
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.5)
    with Measure(tmp_path/'resources.json',model=model,optimizers=True,optimizer_instances=[optimizer]):
        model(torch.ones(2,3)).square().mean().backward();optimizer.step();scheduler.step()
    result=json.loads((tmp_path/'resources.json').read_text())
    assert result['optimizer_steps']==1
    assert result['actual_updated_parameter_count']==8


def test_native_source_bundle_exposes_path_required_by_B(tmp_path,monkeypatch):
    import json
    from experiments.pamf_phase2_practical import group1_source as source
    folder=tmp_path/'source/mrior_sda-s42';folder.mkdir(parents=True)
    contract={'classes':[str(i) for i in range(6)]}
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(contract))
    model=torch.nn.Linear(3,6)
    torch.save(dict(schema='group1_A_v1',recipe=source.recipe('mrior_sda'),source_contract=contract,
        seed=42,target_contact=False,ancestors=[],native_state={},model=model.state_dict()),folder/'source.pt')
    monkeypatch.setattr(source,'model_for',lambda *args:torch.nn.Linear(3,6))
    bundle=source.load_source(dict(run_root=str(tmp_path),source_contract=str(cp),source_cache=str(tmp_path)),
        'mrior_sda',42,'cpu')
    assert bundle['source_path']==str(folder)
    assert bundle['source_checkpoint']==str(folder/'source.pt')


def test_PAMF_rejects_trainer_without_actual_supervised_hook(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical.group1_source import bind_pamf_trainer
    from experiments.cvs_phase1_overlay import model as overlay
    path=tmp_path/'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
    path.parent.mkdir(parents=True)
    def load(text):
        path.write_text(text)
        spec=importlib.util.spec_from_file_location('test_group1_bound_trainer',path)
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
        return value
    old=load('def train(args):\n    return 0\n')
    monkeypatch.setattr(overlay,'native_modules',lambda:old)
    monkeypatch.setattr(overlay,'NATIVE',tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError,match='full supervised objective'):
        bind_pamf_trainer({'frozen_sources':{'BASE':str(tmp_path)}})
    current=load("def train(args):\n    return getattr(args, '_source_labeled_feature_hook')\n")
    import functools
    implementation=current.train
    @functools.wraps(implementation)
    def profiled(*args,**kwargs):return implementation(*args,**kwargs)
    current.train=profiled
    monkeypatch.setattr(overlay,'native_modules',lambda:current)
    assert bind_pamf_trainer({'frozen_sources':{'BASE':str(tmp_path)}}) is current
    assert overlay.NATIVE==path.parents[1]


def test_recovery_reuses_external_jobs_and_replaces_only_PAMF(tmp_path):
    from experiments.pamf_phase2_practical.group1_recovery import matrix
    common=dict(model_seeds=[2026092701],capsules=['P01/MIX01/RX1-1'])
    old=dict(common,run_root=str(tmp_path/'old'),runtime_file='old/runtime.json',release_root='old/release')
    new=dict(common,run_root=str(tmp_path/'new'),runtime_file='new/runtime.json',release_root='new/release',
        method_profiles=['pamf_pl_milkr'])
    jobs=matrix(new,old)
    assert len({j['id'] for j in jobs})==len(jobs)
    assert {j['profile'] for j in jobs if j['origin']=='replacement'}=={'pamf_pl_milkr'}
    assert all(j['profile']!='pamf_pl_milkr' for j in jobs if j['origin']=='legacy')
    assert len([j for j in jobs if j['kind']=='source'])==11
    for job in jobs:
        cfg=old if job['origin']=='legacy' else new
        assert job['output'].startswith(cfg['run_root'])
        assert job['runtime_file']==cfg['runtime_file']
        assert all(v.startswith(job['origin']+'-') for v in job['dependencies'])


def test_current_PAMF_requires_full_Tri_SIG_TIME_SSL_not_ordinary_PL(tmp_path):
    import copy,json
    from pathlib import Path
    from experiments.pamf_phase2_practical.trisig_source import validate_done,SOURCE_RUN
    from experiments.pamf_phase2_practical import group1_source
    definition=json.loads((Path(__file__).parents[1]/'configs/pamf_ssl_trisigtime.json').read_text())
    contract={'role_ids':{'L_s':['L1'],'U_s':['U1']}}
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(contract))
    c=dict(run_id=SOURCE_RUN,model_seed=42,view_comparison={'view_count':3},source_contract=str(cp),
        U_start_epoch=14,satellite_concat=False,own_EMA=False,
        repl_row=dict(structure='SHUFFLE_TIME',L_A=True,L_B=True,
            blocks=dict(style=True,guard=True,joint=True,extrapolation=True)))
    c.update(copy.deepcopy(definition['frozen_REPL_recipe']))
    done=dict(status='SOURCE_FROZEN',epoch=50,updates=11100,target_access=False,U_truth_read=False,
        head_fit_updates=0,config=c,actual_REPL=dict(calls=6660,seen=1701000,student_views=10206000))
    assert validate_done(done,definition,42,contract) is c
    for key,value in [('weight',1.),('LS',0.)]:
        bad=copy.deepcopy(done);bad['config']['hard_pl'][key]=value
        with pytest.raises(ValueError,match='mechanism differs'):
            validate_done(bad,definition,42,contract)
    bad=copy.deepcopy(done);bad['config']['view_comparison']['view_count']=1
    with pytest.raises(ValueError,match='views'):validate_done(bad,definition,42,contract)
    assert module().catalog()[0]['id']=='pamf_tri_sig_time_ssl_milkr'
    recipe=group1_source.recipe('pamf_tri_sig_time_ssl_milkr')
    assert recipe['full_REPL_epochs']==[21,50] and recipe['REPL_weight']==6.
    assert recipe['temporal_permutation']==[0,5,3,6,2,1,4,7]
