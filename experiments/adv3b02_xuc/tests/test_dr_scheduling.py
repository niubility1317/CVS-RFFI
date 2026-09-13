import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from SSDG.train_ssdg import _apply_fasttrust_lr
from cvsrffi.xuc_fusion.dr_objective import options

def test_native_lr_boundaries_and_all_seven_rows():
    params=[torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    opt=torch.optim.AdamW([dict(params=[params[0]],fasttrust_role='backbone'),dict(params=[params[1]],fasttrust_role='other')])
    for epoch,backbone,other in [(1,4e-5,4e-5),(5,2e-4,2e-4),(6,2e-4,2e-4),(160,2e-5,2e-5),(161,4e-6,2e-5),(181,1e-6,2e-5),(200,1e-6,2e-5)]:
        _apply_fasttrust_lr(opt,base_lr=2e-4,epoch=epoch)
        assert [g['lr'] for g in opt.param_groups]==pytest.approx([backbone,other])
    m=json.loads((ROOT/'configs/matrix_dr.json').read_text())
    assert len(m['runs'])==7
    assert len({r['id'] for r in m['runs']})==7
    assert all(r['daot_rc4'] and r['steps_per_epoch']==49 and r['unlabeled_batch']==256 for r in m['runs'])
    original=json.loads((ROOT/'configs/matrix15.json').read_text())
    by={r['id']:r for r in original['runs']}
    for r in m['runs']:
        for key in ['x_enabled','u_enabled','control','curriculum','ticket_curriculum_enabled','cstar_action_enabled','carrier']:
            assert r[key]==by[r['parent_row']][key]
    a=options(json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text()),'unused')
    assert a.daot_teacher_view_count==2 and a.daot_aggregation=='mean'
    assert a.daot_lambda_tangent==a.daot_lambda_nuisance==a.daot_lambda_fingerprint==0
    assert a.fasttrust_rc4 and a.rc4_enable_hard and a.rc4_enable_partial
    assert not a.rc4_enable_negative and not a.rc4_use_anchor

def test_dispatcher_caps_two_workers_and_uses_dr_matrix():
    spec=importlib.util.spec_from_file_location('drdispatch',ROOT/'code/scripts/dispatch_xuc_dr.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    with patch.object(mod,'occupancy',return_value={0:dict(pids={123},free_mb=20000),1:dict(pids=set(),free_mb=20000)}):
        assert mod.available_gpu({})==1
    with patch.object(mod,'occupancy',return_value={0:dict(pids={123},free_mb=20000)}):
        assert mod.available_gpu({})==0
    with patch.object(mod,'occupancy',return_value={0:dict(pids={123,456},free_mb=20000),1:dict(pids={789},free_mb=6400)}):
        assert mod.available_gpu({}) is None
    command=mod.train_command(dict(id='DR-M08',pipeline='core90'),Path('/project'),Path('/run'))
    assert 'matrix_dr.json' in command[command.index('--matrix')+1]

def test_rc4_identity_gradients_after_warmup_controlled_routes():
    # A source-only branch diagnostic, not a claim that untrained synthetic heads
    # naturally pass the calibrated confidence threshold.
    from SSDG.train_ssdg import _compute_rc4_unlabeled_losses
    from cvsrffi.muse_ssdg import MUSETrainingHeads
    a=options(json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text()),'unused')
    logits=torch.randn(4,6,requires_grad=True)
    z=torch.randn(4,160,requires_grad=True)
    route=SimpleNamespace(pseudo=torch.tensor([0,1,2,3]),hard=torch.tensor([True,True,False,False]),
        partial=torch.tensor([False,False,True,True]),negative=torch.zeros(4,dtype=torch.bool),
        candidate_mask=torch.tensor([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,1,1,0]],dtype=torch.bool),
        weights=torch.full((4,),.15),fused_probability=torch.softmax(torch.randn(4,6),-1),
        risk=torch.ones(4),agreement=torch.ones(4,dtype=torch.bool),disagreement=torch.zeros(4))
    views=dict(clean=dict(tx_logits=logits,z_id=z,dom_logits=torch.randn(4,15,requires_grad=True)),satellite=None,satellite_indices=torch.empty(0,dtype=torch.long))
    kw=dict(route=route,ema_outputs=dict(z_id=z.detach()),anchor_outputs=None,student_views=views,
        domains=torch.arange(4),model=SimpleNamespace(adv_head=None),muse_state=dict(args=a,heads=MUSETrainingHeads(160,160,6,15,6)))
    cold=_compute_rc4_unlabeled_losses(**kw,epoch=1)
    assert cold['identity'].item()==0 and cold['u_identity_selected_count'].item()==0
    hot=_compute_rc4_unlabeled_losses(**kw,epoch=21)
    assert hot['u_identity_selected_count'].item()==4
    assert hot['hard'].item()>0 and hot['rc4_partial_set'].item()>0 and hot['rc4_partial_conditional'].item()>0
    grad=torch.autograd.grad(hot['identity'],logits)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
