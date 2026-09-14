"""No datasets, no train loop, no external processes: local semantic acceptance."""
import sys
import json
from pathlib import Path
from copy import deepcopy
import pytest
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.joint_config import resolve,make_row,hp_multiplier,epoch_for_accepted_step
from cvsrffi.xuc_fusion.runtime import resolve_args,synthetic_source
from cvsrffi.xuc_fusion.tickets import TicketStream
from cvsrffi.xuc_fusion.joint_normalization import FieldNormalizer
from cvsrffi.orbit_teacher import EMALossScaleNormalizer
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.runtime import build_model
from cvsrffi.game_tracking.step_context import prepare_context
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.xuc_fusion.dr_objective import DROT
from cvsrffi.xuc_fusion.objective import FusionObjective
from cvsrffi.game_tracking.source_audit import isolated_rng

def args_for(settings=None):
    row=make_row('acceptance',settings or {})
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    return resolve_args(recipe,row,dataset='NOT_READ',output='NOT_CREATED',device='cpu',synthetic=True,test_steps=1),row

def test_config_rejects_silent_drift():
    for change in ({'lr':4e-4},{'controller_mode':'C2'},{'launch':True},{'typo':1},{'native_dr':'false'},{'outer_adv_weight':0.}):
        with pytest.raises(ValueError):resolve(change)

@pytest.mark.parametrize('e',[1,20,21,40,41,60,61,79,80,90,91,160,161,200])
def test_clock(e):
    assert epoch_for_accepted_step((e-1)*222)==e
    assert epoch_for_accepted_step(e*222-1)==e

def test_hp_ramp():
    assert [hp_multiplier(e,True) for e in (1,20,21,40,41,200)]==[0,0,0,1,1,1]
    assert hp_multiplier(30,True)==9/19

def test_divisors_are_actual_and_lu_order_preserved():
    base=EMALossScaleNormalizer(momentum=.95);base._scales={'x':2.}
    first=FieldNormalizer(deepcopy(base))
    first.normalize({'x':torch.tensor(4.,requires_grad=True)})
    first.normalize({'x':torch.tensor(8.,requires_grad=True)})
    assert first.used[0]['x']==pytest.approx(2.1)
    assert first.used[1]['x']==pytest.approx(2.395)
    second=FieldNormalizer(deepcopy(base),first.used)
    x=torch.tensor(100.,requires_grad=True);out,_=second.normalize({'x':x})
    assert torch.autograd.grad(out['x'],x)[0].item()==pytest.approx(1/2.1)
    second.normalize({'x':x});second.finish()
    assert second.state_dict()==base.state_dict()
    assert second.used==first.used

@pytest.mark.parametrize('kappa',[.25,.5,1.])
def test_adamw_predictor_ratio_one_commit_and_reference(kappa):
    model=nn.Linear(2,1);ref=deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    ropt=torch.optim.AdamW(ref.parameters(),lr=.01,weight_decay=.1)
    x=torch.tensor([[1.,2.]]);y=torch.tensor([[3.]])
    before=deepcopy(ref.state_dict());state=deepcopy(ropt.state_dict())
    ((ref(x)-y)**2).sum().backward();ropt.param_groups[0]['lr']*=kappa;ropt.step()
    ropt.zero_grad();((ref(x)-y)**2).sum().backward();corrector=[p.grad.clone() for p in ref.parameters()]
    ref.load_state_dict(before);ropt.load_state_dict(state)
    for p,g in zip(ref.parameters(),corrector):p.grad=g
    ropt.step()
    solver=GameSolver(model,opt,mode='extragradient',predictor_lr_ratio=kappa)
    result=solver.step(lambda:((model(x)-y)**2).sum())
    assert result.field_evaluations==2 and solver.steps==1
    assert all(v['step']==1 for v in opt.state.values())
    assert opt.param_groups[0]['lr']==.01
    for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

def test_exception_rolls_back_parameters_optimizer_rng_buffers():
    model=nn.Sequential(nn.BatchNorm1d(2),nn.Linear(2,1));opt=torch.optim.AdamW(model.parameters())
    initial=deepcopy(model.state_dict());rng=torch.random.get_rng_state();calls=0
    def closure():
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('corrector rejected')
        return model(torch.randn(4,2)).square().mean()
    with pytest.raises(RuntimeError):GameSolver(model,opt,mode='extragradient',predictor_lr_ratio=.5).step(closure)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
    assert not opt.state and torch.equal(torch.random.get_rng_state(),rng)

def test_pairing_across_dr_and_model_seeds():
    records=[];weights=[]
    for dr,seed in [(False,392005),(True,392005),(True,392006)]:
        args,row=args_for(dict(native_dr=dr,model_seed=seed));args.num_classes=6
        with isolated_rng(seed):model=build_model(args,15,torch.device('cpu'))
        weights.append({k:v.clone() for k,v in model.state_dict().items()})
        stream=TicketStream(synthetic_source(),args,row);ticket,_=stream.choose()
        records.append((ticket.labeled,ticket.unlabeled,ticket.seed))
    assert records[0]==records[1]==records[2]
    assert len(records[0][1])==256
    assert all(torch.equal(weights[0][k],weights[1][k]) for k in weights[0])
    assert any(not torch.equal(weights[1][k],weights[2][k]) for k in weights[1])

@pytest.mark.parametrize('epoch,dr_on,kappa,norm',[(1,False,1.,'legacy_reestimate'),(21,True,1.,'legacy_reestimate'),(80,True,.5,'reuse_origin_used_scales')])
def test_real_model_joint_field(epoch,dr_on,kappa,norm,monkeypatch):
    torch.set_num_threads(2)
    args,row=args_for(dict(native_dr=dr_on,solver_mode='full_EG',predictor_lr_ratio=kappa,
        normalization_scales=norm));args.num_classes=6
    with isolated_rng(123):model=build_model(args,15,torch.device('cpu'))
    ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    source=synthetic_source();reference=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    dr=DROT(model,ema,source,args,reference);dr.calibrate(epoch)
    from SSDG import train_ssdg as native
    calls=[];original=native._forward_daot_teacher_views
    def counted(*a,**kw):
        calls.append(len(a[1]));return original(*a,**kw)
    monkeypatch.setattr(native,'_forward_daot_teacher_views',counted)
    stream=TicketStream(source,args,row);ticket,_=stream.choose();batch,ubatch=stream.batches(ticket)
    # Small real-model tensors; source role labels remain hidden for U.
    from cvsrffi.game_tracking.step_context import slice_batch
    def small(b):return (b[0][:12],b[1][:12],b[2][:12],{k:v[:12] for k,v in b[3].items()})
    batch,ubatch=small(batch),small(ubatch)
    from cvsrffi.schedule import build_stage_state
    from cvsrffi.game_tracking.legacy.options import _loss_weights
    weights=_loss_weights(args,build_stage_state(epoch,args))
    ctx=prepare_context(batch,None,model,ema,args,epoch,1,weights,torch.Generator().manual_seed(1))
    ctx.grid_plan=None;ctx.rx=batch[3]['rx_i'];ctx.day=batch[3]['day_i'];ctx.audit_gradients=True;ctx.field_components=[]
    dr.prepare(ctx,ubatch)
    if epoch==80:
        # Controlled branch reachability only; never modify routes in runtime.
        from dataclasses import replace
        hard=torch.arange(len(ubatch[0]))<6;partial=~hard
        candidate=torch.zeros((len(hard),6),dtype=torch.bool);candidate[:,0:2]=True
        ctx.dr_route=replace(ctx.dr_route,hard=hard,partial=partial,negative=torch.zeros_like(hard),
            representation=torch.zeros_like(hard),candidate_mask=candidate,
            weights=torch.full((len(hard),),.15))
    proto=PrototypeMemoryBank(6,15);proto._lazy_init(160,torch.device('cpu'),torch.float32)
    objective=FusionObjective(model,args,proto,row,dr)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    before=deepcopy(ema.state_dict());solver=GameSolver(model,opt,mode='extragradient',predictor_lr_ratio=kappa,max_grad_norm=5.,telemetry_interval=1)
    result=solver.step(lambda:objective(ctx));dr.commit(ctx)
    assert result.field_evaluations==2 and len(ctx.field_components)==2
    assert ctx.field_components[0]==ctx.field_components[1]
    assert dr.commits==1 and all(v['step']==1 for v in opt.state.values())
    assert all(torch.equal(v,before[k]) for k,v in ema.state_dict().items())
    assert not ctx.dr_route.negative.any()
    assert len(set(ctx.satellite_selected_mask.tolist()))==1
    if norm=='reuse_origin_used_scales':assert ctx.dr_field_divisors[0]==ctx.dr_field_divisors[1]
    if not dr_on:assert ctx.dr_telemetry['weighted_identity']==0 and not ctx.dr_l_cache and not ctx.dr_u_cache
    if dr_on and epoch>=21:assert 'teacher' in ctx.dr_l_cache and 'teacher' in ctx.dr_u_cache
    assert calls==([2,1] if dr_on and epoch>=21 else [])
    if epoch==80:
        assert ctx.dr_telemetry['weighted_identity']>0
        assert ctx.fusion_telemetry['identity_gradients']['norms']['RC4_id']>0
        from cvsrffi.xuc_fusion.joint_diagnostics import run_probes
        state=deepcopy(model.state_dict());rng=torch.random.get_rng_state()
        result=run_probes(model,ctx,args,dr,opt,1000)
        assert result['controls_training'] is False and result['action']=='OFF'
        assert 'probe_A' in result and 'probe_B' in result and 'representation' in result
        assert torch.equal(rng,torch.random.get_rng_state())
        assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())

def test_u_grl_and_encoder_multiplier_keep_head_training_strength():
    args,_=args_for();args.num_classes=6
    with isolated_rng(1):model=build_model(args,15,torch.device('cpu'))
    model.eval();x=torch.randn(4,2,256);domain=torch.arange(4)
    encoder=[p for n,p in model.named_parameters() if n.startswith('id_backbone.')]
    head=list(model.adv_head.parameters());vectors=[]
    for multiplier in (0.,.5,1.):
        out=model(x,return_aux=True,domain_labels=domain,grl_lambda=multiplier)
        loss=torch.nn.functional.cross_entropy(out['adv_dom_logits'],domain)
        gradients=torch.autograd.grad(loss,encoder+head,allow_unused=True)
        vectors.append([torch.zeros_like(p) if g is None else g for p,g in zip(encoder+head,gradients)])
    assert all(torch.count_nonzero(g)==0 for g in vectors[0][:len(encoder)])
    assert any(torch.count_nonzero(g)>0 for g in vectors[0][len(encoder):])
    for half,full in zip(vectors[1][:len(encoder)],vectors[2][:len(encoder)]):torch.testing.assert_close(half,full*.5)
    for a,b,c in zip(*(v[len(encoder):] for v in vectors)):
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(b,c,rtol=0,atol=0)

def test_r1_effective_options_not_overridden():
    from cvsrffi.xuc_fusion.dr_objective import resolved_options
    args,_=args_for(dict(daot_logit=.1,hp_ramp=True,predictor_lr_ratio=.5))
    reference=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    dr=resolved_options(reference,args)
    assert dr.daot_lambda_orbit_logit==.1 and dr.daot_lambda_orbit_z==.5
    assert dr.daot_teacher_view_count==2 and dr.daot_aggregation=='mean'
    assert all(getattr(dr,'daot_lambda_'+name)==0 for name in ('orbit_proto','orbit_relation','tangent','nuisance','fingerprint'))
    assert not dr.rc4_enable_negative and not dr.rc4_use_anchor and not dr.rc4_satellite_hard_only

def test_metrics_and_pairing_statistics():
    from cvsrffi.xuc_fusion.joint_metrics import task_metrics,seed_contrasts
    scenes=['leo_clear_weak','leo_low_elev_weak','leo_rain_weak']*2
    result=task_metrics([0,0,0,1,1,1],[0,0,1,1,0,1],[0]*3+[1]*3,[1]*6,scenes,2,
        reference=[1,0,0,1,1,0])
    assert result['paired']==dict(rescue=2,harm=2,net=0)
    assert result['worst_RX_leo_mean']==pytest.approx(2/3)
    result=seed_contrasts({1:{'00':.5,'10':.6,'01':.6,'11':.75},2:{'00':.5,'10':.6,'01':.6,'11':.7}})
    assert result['summary']['interaction']['mean']==pytest.approx(.025)

def test_prepared_matrix_exact_and_no_launch():
    import importlib.util
    spec=importlib.util.spec_from_file_location('prepare_joint',ROOT/'tools/prepare_native_dr_eg.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    matrix=mod.build()
    assert len(matrix['rows'])==18 and matrix['launch'] is False
    assert len({r['id'] for r in matrix['rows']})==18
    assert all(not r['joint']['launch'] and r['joint']['controller_mode']=='off' for r in matrix['rows'])
    assert len(matrix['r1_templates'])==9

def test_entry_defaults_to_resolution_without_train(monkeypatch,tmp_path,capsys):
    import importlib.util
    spec=importlib.util.spec_from_file_location('joint_entry',ROOT/'code/scripts/train_native_dr_eg.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'train',lambda *a,**kw:pytest.fail('unexpected train call'))
    config=tmp_path/'config.json';output=tmp_path/'must_not_exist'
    config.write_text(json.dumps({'row':make_row('test',{'solver_mode':'full_EG'})}),encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['entry','--config',str(config),'--output',str(output)])
    assert module.main()==0 and not output.exists()
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='VALIDATED_NOT_LAUNCHED'
    assert result['resolved']['game_solver']=='extragradient'
