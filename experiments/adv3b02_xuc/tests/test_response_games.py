"""Bounded solver acceptance only; never invokes train()."""
import sys,json
from pathlib import Path
from copy import deepcopy
import pytest
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.response_config import resolve,make_row,schedule
from cvsrffi.xuc_fusion.response_solver import ResponseSolver
from cvsrffi.xuc_fusion.response_fields import procrustes,cross_tx_fields,tx_partition
from cvsrffi.xuc_fusion.response_dric import constrained_game,flat_grad
from cvsrffi.game_tracking.solvers import GameSolver

@pytest.mark.parametrize('beta',[0.,1.])
def test_cf_endpoints_with_history_and_clipping(beta):
    torch.manual_seed(11);model=nn.Linear(3,2);ref=deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.1)
    ropt=torch.optim.AdamW(ref.parameters(),lr=.02,weight_decay=.1)
    x=torch.randn(5,3)
    def loss(m,on):return m(x).square().mean()+float(on)*m(x).sin().sum()
    for m,o in [(model,opt),(ref,ropt)]:loss(m,True).backward();o.step();o.zero_grad()
    config=resolve(dict(fixed_beta=beta))
    solver=ResponseSolver(model,opt,config,max_grad_norm=.05)
    result=solver.cf_step(lambda on:loss(model,on),lambda:model(x).mean(0))
    GameSolver(ref,ropt,mode='extragradient',max_grad_norm=.05).step(lambda:loss(ref,bool(beta)))
    assert result.telemetry['beta']==beta and solver.steps==1
    for p,q in zip(model.parameters(),ref.parameters()):
        torch.testing.assert_close(p,q,atol=0,rtol=0)
        for k in opt.state[p]:torch.testing.assert_close(opt.state[p][k],ropt.state[q][k],atol=0,rtol=0)

def test_cf_failure_rolls_back():
    model=nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters());before=deepcopy(model.state_dict());rng=torch.get_rng_state()
    solver=ResponseSolver(model,opt,resolve({}))
    with pytest.raises(FloatingPointError):solver.cf_step(lambda on:model(torch.randn(3,2)).square().mean(),lambda:torch.tensor([float('nan')]))
    assert not opt.state and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())

def test_cf_selects_largest_feasible_beta():
    model=nn.Linear(1,1,bias=False);model.weight.data.zero_()
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    solver=ResponseSolver(model,opt,resolve(dict(margin_tolerance=0.)))
    result=solver.cf_step(lambda on:model.weight.sum()*(-1 if on else 1),lambda:model.weight.flatten())
    assert result.telemetry['beta']==0
    assert [c['feasible'] for c in result.telemetry['candidates']]==[False,False,True]

def test_nonmodel_seeds_match_previous_contract():
    rows=[make_row('seed',{},s)['joint'] for s in (392005,392006,392007)]
    for key in ('data_seed','augmentation_seed','evaluation_seed'):assert [r[key] for r in rows]==[392005]*3
    with pytest.raises(ValueError):resolve(dict(encoder_adversary='false'))

def test_local_replay_preserves_selected_gradients_and_rng():
    from cvsrffi.xuc_fusion.response_replay import LocalReplay
    torch.manual_seed(5)
    model=nn.Sequential(nn.Linear(3,7),nn.Dropout(.2),nn.Linear(7,2),nn.Tanh())
    selected=list(model[2].parameters());cache=LocalReplay(model,selected);x=torch.randn(8,3)
    start=torch.get_rng_state()
    with cache.session('same_batch'):model(x).sum()
    with torch.no_grad():model[2].weight.add_(.2)
    torch.set_rng_state(start)
    expected=model(x);g_expected=torch.autograd.grad(expected.square().sum(),selected)
    rng_expected=torch.get_rng_state()
    torch.set_rng_state(start)
    with cache.session('same_batch'):actual=model(x)
    g_actual=torch.autograd.grad(actual.square().sum(),selected)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for a,b in zip(g_actual,g_expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert cache.replayed>0 and torch.equal(torch.get_rng_state(),rng_expected)

def test_rotation_exact_and_degenerate():
    torch.manual_seed(12);q,_=torch.linalg.qr(torch.randn(4,4,dtype=torch.double));q[:,-1]*=torch.linalg.det(q)
    z=torch.randn(30,4,dtype=torch.double);R=procrustes(z,z@q.T,0.)
    torch.testing.assert_close(R,q,rtol=1e-8,atol=1e-8)
    A=torch.randn(3,4,dtype=torch.double)
    torch.testing.assert_close((z@q.T)@(A@R.T).T,z@A.T,rtol=1e-8,atol=1e-8)
    R=procrustes(torch.zeros(3,4),torch.zeros(3,4))
    torch.testing.assert_close(R.T@R,torch.eye(4));assert torch.linalg.det(R)>0

def test_xt_gradient_ownership_and_partition():
    torch.manual_seed(2);z=torch.randn(90,4,requires_grad=True)
    y=torch.arange(6).repeat_interleave(15);d=torch.arange(15).repeat(6)
    head=nn.Sequential(nn.Linear(4,8),nn.Tanh(),nn.Linear(8,15))
    meta,diag=cross_tx_fields(head,z,y,d,1,.01,.1,0.)
    zg=torch.autograd.grad(meta,z,retain_graph=True,allow_unused=True)[0]
    assert zg is None or zg.count_nonzero()==0
    assert sum(float(g.norm()) for g in torch.autograd.grad(meta,head.parameters()))>0
    adv,_=cross_tx_fields(head,z,y,d,1,.01,0.,.35)
    assert torch.autograd.grad(adv,z)[0].norm()>0
    assert not set(diag['tx_A'])&set(diag['tx_B'])
    a,_=tx_partition(y,d,0);b,_=tx_partition(y,d,1);assert not torch.equal(a,b)

def test_drift_counterexample_and_asymmetric_cross_derivatives():
    theta=torch.tensor(2.,requires_grad=True);phi=torch.tensor(6.,requires_grad=True)
    dl=.5*(phi-3*theta)**2;old=torch.autograd.grad(dl,phi)[0]
    moved=.5*(phi-3*(theta+.2))**2;new=torch.autograd.grad(moved,phi)[0]
    assert old==0 and new.item()==pytest.approx(-.6,abs=1e-6)
    dl=.5*(phi-3*theta)**2;du=.5*(phi+2*theta)**2
    B=flat_grad(flat_grad(-.35*dl,[theta],True).sum(),[phi])
    C=flat_grad(flat_grad(.35*dl+.5*du,[phi],True).sum(),[theta])
    assert B.item()==pytest.approx(1.05) and C.item()==pytest.approx(-.05,abs=1e-6)
    assert not torch.allclose(B,-C)

@pytest.mark.parametrize('radius',[10.,.2])
def test_local_joint_constraint_and_residual(radius):
    J=torch.tensor([[1.,-.2],[.3,1.]])
    z,info=constrained_game(J,torch.tensor([-1.,.2]),1,torch.tensor([[1.]]),torch.tensor([.1]),radius)
    assert z[0]<=.100001 and z[0].abs()<=radius+1e-6 and info['residual']<1e-5
    assert z[1].item()==pytest.approx(-.2-.3*z[0].item(),abs=1e-6)

@pytest.mark.parametrize('method',['DRIC','CF_EG','TR_EG','XT_DANN'])
def test_config_stage_boundaries(method):
    c=resolve(dict(method=method))
    assert not schedule(c,20,19)['active']
    assert schedule(c,40,19)['active']
    assert schedule(c,40,19)['base']==('simultaneous' if method=='DRIC' else 'extragradient')
    row=make_row('test',c);assert row['joint']['normalization_scales']=='reuse_origin_used_scales'
    assert row['joint']['outer_adv_weight']==.35 and not row['joint']['disable_adversarial_head_loss']

@pytest.mark.parametrize('method,epoch,step',[('CF_EG',80,3),('TR_EG',80,3),('XT_DANN',80,19),('DRIC',40,3),('DRIC_OFF',40,3),('DRIC_ZERO',21,3),('DRIC_EMPTY',40,3),('DRIC_TINY',40,3),('FR',40,3),('CGD',40,3),('TASK_PROJECT',40,3),('DRIC_DECOMPOSE',40,3)])
def test_real_native_response_path(method,epoch,step,monkeypatch):
    from cvsrffi.xuc_fusion.runtime import resolve_args,synthetic_source
    from cvsrffi.game_tracking.runtime import build_model
    from cvsrffi.xuc_fusion.dr_objective import DROT
    from cvsrffi.xuc_fusion.tickets import TicketStream
    from cvsrffi.game_tracking.step_context import prepare_context
    from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
    from cvsrffi.game_tracking.legacy.options import _loss_weights
    from cvsrffi.schedule import build_stage_state
    from cvsrffi.xuc_fusion.objective import FusionObjective
    from cvsrffi.xuc_fusion.response_context import SourceMonitor
    from cvsrffi.xuc_fusion.response_runtime import response_step
    torch.set_num_threads(2);torch.manual_seed(32)
    actual_method='DRIC' if method.startswith('DRIC_') else method
    settings=dict(method=actual_method)
    if method=='DRIC_OFF':settings['encoder_adversary']=False
    if method=='DRIC_DECOMPOSE':settings['decompose_interval']=1
    row=make_row('local',settings)
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    args=resolve_args(recipe,row,dataset='NOT_READ',output='NOT_CREATED',device='cpu',synthetic=True)
    args.num_classes=6;model=build_model(args,15,torch.device('cpu'));ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    source=synthetic_source();reference=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    dr=DROT(model,ema,source,args,reference);dr.calibrate(epoch)
    stream=TicketStream(source,args,row);ticket,_=stream.choose();batch,ubatch=stream.batches(ticket)
    def small(b):return (b[0][:12],b[1][:12],b[2][:12],{k:v[:12] for k,v in b[3].items()})
    batch,ubatch=small(batch),small(ubatch)
    ctx=prepare_context(batch,None,model,ema,args,epoch,1,_loss_weights(args,build_stage_state(epoch,args)),torch.Generator().manual_seed(1))
    ctx.grid_plan=None;ctx.rx=batch[3]['rx_i'];ctx.day=batch[3]['day_i'];ctx.audit_gradients=False;ctx.field_components=[];ctx.fusion_telemetry={}
    dr.prepare(ctx,ubatch);proto=PrototypeMemoryBank(6,15);proto._lazy_init(160,torch.device('cpu'),torch.float32)
    objective=FusionObjective(model,args,proto,row,dr)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    solver=ResponseSolver(model,optimizer,args.response,max_grad_norm=5.)
    if method=='DRIC_TINY':
        from cvsrffi.xuc_fusion import response_runtime
        original_schedule=response_runtime.schedule
        monkeypatch.setattr(response_runtime,'schedule',lambda *a:dict(original_schedule(*a),strength=1e-8))
    if method=='DRIC_EMPTY':
        from cvsrffi.xuc_fusion import response_dric
        monkeypatch.setattr(response_dric,'basis',lambda directions,maxdim=4:directions[0].new_zeros((directions[0].numel(),0)))
    head_before=[p.detach().clone() for p in model.adv_head.parameters()]
    result=response_step(solver,objective,ctx,args,dr,SourceMonitor(source),step)
    assert result.accepted and solver.steps==1
    assert all(state['step']==1 for state in optimizer.state.values())
    assert all(div==ctx.dr_field_divisors[0] for div in ctx.dr_field_divisors)
    assert result.telemetry['schedule']['active']
    assert result.telemetry['head_LU_supervision']
    if method=='CF_EG':assert result.field_evaluations==4
    if method=='XT_DANN':assert len(result.telemetry['xt_fields'])==2
    if method=='DRIC':assert result.telemetry['extra_identity_displacement']>0
    if method=='DRIC':assert result.telemetry['local_leaf_replays']>0
    if method in ('DRIC_OFF','DRIC_ZERO','DRIC_EMPTY'):
        assert result.telemetry['extra_identity_displacement']==0
        assert any(not torch.equal(p,q) for p,q in zip(model.adv_head.parameters(),head_before))
    if method=='DRIC_DECOMPOSE':assert set(result.telemetry['component_drift_norms'])=={'TX','DAOT','RC4'}
    if method in ('DRIC','DRIC_TINY','DRIC_OFF'):assert result.telemetry['nonselected_parameter_error']==0
    if method=='DRIC_OFF':assert result.telemetry['deviation_from_ordinary']==0
    if method=='DRIC_TINY':assert result.telemetry['deviation_from_ordinary']<1e-5
