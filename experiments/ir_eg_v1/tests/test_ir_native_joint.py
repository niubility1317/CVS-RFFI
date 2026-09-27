"""Reach the copied production FusionObjective + DAOT + FastTrust assembly."""
import json
import os
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import torch
from cvsrffi.xuc_fusion.response_config import make_row
from cvsrffi.xuc_fusion.ir_solver import IREGSolver
from cvsrffi.xuc_fusion.response_runtime import response_step
from cvsrffi.game_tracking.state import RNGState
from ir_fixtures import assert_nested

ROOT=Path(__file__).resolve().parents[1]


def native_case(implementation='cached',epoch=80,method='IR_EG',refresh=8):
    from cvsrffi.xuc_fusion.runtime import resolve_args,synthetic_source
    from cvsrffi.game_tracking.runtime import build_model
    from cvsrffi.xuc_fusion.dr_objective import DROT
    from cvsrffi.xuc_fusion.tickets import TicketStream
    from cvsrffi.game_tracking.step_context import prepare_context
    from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
    from cvsrffi.game_tracking.legacy.options import _loss_weights
    from cvsrffi.schedule import build_stage_state
    from cvsrffi.xuc_fusion.objective import FusionObjective
    device=torch.device(os.environ.get('IR_ACCEPTANCE_DEVICE','cpu'))
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(32)
    row=make_row('ir_local',dict(method=method,implementation=implementation,br_refresh_interval=refresh))
    recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    args=resolve_args(recipe,row,dataset='NOT_READ',output='NOT_CREATED',device=str(device),synthetic=True)
    args.num_classes=6
    model=build_model(args,15,device);ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    source=synthetic_source();reference=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text(encoding='utf-8'))
    dr=DROT(model,ema,source,args,reference);dr.calibrate(epoch)
    stream=TicketStream(source,args,row);ticket,_=stream.choose();batch,ubatch=stream.batches(ticket)
    def small(b):return (b[0][:6],b[1][:6],b[2][:6],{k:v[:6] for k,v in b[3].items()})
    batch,ubatch=small(batch),small(ubatch)
    ctx=prepare_context(batch,None,model,ema,args,epoch,1,_loss_weights(args,build_stage_state(epoch,args)),torch.Generator().manual_seed(1))
    ctx.grid_plan=None;ctx.rx=batch[3]['rx_i'];ctx.day=batch[3]['day_i'];ctx.audit_gradients=False;ctx.field_components=[];ctx.fusion_telemetry={}
    dr.prepare(ctx,ubatch)
    # Controlled source-only fixture reaches both native identity branches.
    from dataclasses import replace
    hard=torch.arange(6,device=device)<3;partial=~hard
    candidate=torch.zeros((6,6),dtype=torch.bool,device=device);candidate[:,:2]=True
    ctx.dr_route=replace(ctx.dr_route,hard=hard,partial=partial,negative=torch.zeros_like(hard),
        representation=torch.zeros_like(hard),candidate_mask=candidate,weights=torch.full((6,),.15,device=device))
    proto=PrototypeMemoryBank(6,15);proto._lazy_init(160,device,torch.float32)
    objective=FusionObjective(model,args,proto,row,dr)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    solver=IREGSolver(model,opt,args.response,max_grad_norm=5.)
    def commit():
        from cvsrffi.game_tracking.legacy.options import _update_ema_model
        dr.commit(ctx);_update_ema_model(ema,model,args.ema_decay);proto.update(ctx.origin_features,ctx.y,ctx.domain)
    return SimpleNamespace(model=model,ema=ema,args=args,dr=dr,proto=proto,ctx=ctx,objective=objective,opt=opt,solver=solver,commit=commit)


def step(case):
    return response_step(case.solver,case.objective,case.ctx,case.args,case.dr,None,case.solver.steps,
                         commit=case.commit,external_stateful=(case.dr,case.ema,case.proto))


def test_native_joint_cached_reference_alignment():
    a=native_case('reference');b=native_case('cached');rng=RNGState.capture()
    ra=step(a);after=RNGState.capture();rng.restore();rb=step(b)
    assert abs(ra.loss_replay-rb.loss_replay)<1e-6
    assert_nested(a.solver.last_trace['corrector'],b.solver.last_trace['corrector'],1e-6,1e-5)
    assert_nested(a.model.state_dict(),b.model.state_dict(),1e-6,1e-5)
    assert_nested(a.opt.state_dict(),b.opt.state_dict(),1e-6,1e-5)
    assert_nested(a.ema.state_dict(),b.ema.state_dict(),1e-6,1e-5)
    assert_nested(vars(a.proto),vars(b.proto),1e-6,1e-5)
    assert a.dr.commits==b.dr.commits==1
    assert a.dr.scale.state_dict()==b.dr.scale.state_dict()
    assert a.dr.route_history==b.dr.route_history
    assert torch.equal(after.cpu,torch.get_rng_state())
    assert all(x==b.ctx.dr_field_divisors[0] for x in b.ctx.dr_field_divisors)
    assert ra.telemetry['model_forward_calls']*2==rb.telemetry['model_forward_calls']*3
    assert [c.key for c in b.solver.last_trace['head_batch'].calls]==['L/clean/main/adv','U/strong/rc4/adv']
    assert b.ctx.dr_weighted_identity.item()>0
