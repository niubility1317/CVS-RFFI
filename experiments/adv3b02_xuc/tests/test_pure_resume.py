"""Actual active-method updates, not merely checkpoint tensor reconstruction."""
from copy import deepcopy
from pathlib import Path
import json
import os
import random
import numpy as np
import pytest
import torch
from cvsrffi.xuc_fusion.runtime import resolve_args,synthetic_source,train
from cvsrffi.xuc_fusion.response_config import make_row
from cvsrffi.game_tracking.runtime import build_model
from cvsrffi.xuc_fusion.dr_objective import DROT
from cvsrffi.xuc_fusion.tickets import TicketStream
from cvsrffi.game_tracking.step_context import prepare_context
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.legacy.options import _update_ema_model
from cvsrffi.game_tracking.source_audit import isolated_rng
from cvsrffi.xuc_fusion.objective import FusionObjective
from cvsrffi.xuc_fusion.response_context import SourceMonitor
from cvsrffi.xuc_fusion.response_runtime import response_step
from cvsrffi.xuc_fusion.response_solver import ResponseSolver
from cvsrffi.xuc_fusion.resume import validate_checkpoint,restore_checkpoint,capture_resume_state
from cvsrffi.schedule import configure_mixstyle_for_epoch

ROOT=Path(__file__).resolve().parents[1]
RECIPE=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text())
REFERENCE=json.loads((ROOT/'configs/a1_native_recipe_reference.json').read_text())

@pytest.fixture(autouse=True)
def deterministic_kernels():
    # Only this diagnostic asks for bitwise replay. Formal existing jobs retain
    # their original CUDA settings and can have normal kernel roundoff variation.
    old=torch.are_deterministic_algorithms_enabled();cudnn=torch.backends.cudnn.deterministic
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
    yield
    torch.use_deterministic_algorithms(old);torch.backends.cudnn.deterministic=cudnn

def equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def setup(row,output):
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    args=resolve_args(RECIPE,row,dataset='NOT_READ',output=output,device=device,synthetic=True,test_epochs=82,test_steps=1)
    args.num_classes=6;args.joint['diagnostic_interval']=0;args.joint['probe_interval']=0
    model=build_model(args,15,torch.device(device));ema=deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    source=synthetic_source();dr=DROT(model,ema,source,args,REFERENCE)
    stream=TicketStream(source,args,row);stream.next_epoch=80;stream.consumed=set(range(79))
    proto=PrototypeMemoryBank(6,15);proto._lazy_init(160,torch.device(device),torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    solver=ResponseSolver(model,opt,args.response,max_grad_norm=5.);solver.steps=79;dr.commits=79
    return args,source,model,ema,dr,stream,proto,opt,solver

def update(env,completed):
    args,source,model,ema,dr,stream,proto,opt,solver=env
    ticket,_=stream.choose();batch,ubatch=stream.batches(ticket)
    model.train();configure_mixstyle_for_epoch(model,args,ticket.epoch)
    with isolated_rng(ticket.seed):
        context_args=deepcopy(args);context_args.game_evidence_version=2
        ctx=prepare_context(batch,None,model,ema,context_args,ticket.epoch,ticket.batch_index,ticket.weights,
            torch.Generator(device=args.device).manual_seed(ticket.seed),exposure_record=ticket.exposure(batch[3]['sample_id']))
        ctx.grid_plan=None;ctx.rx=batch[3]['rx_i'].to(args.device);ctx.day=batch[3]['day_i'].to(args.device)
        ctx.audit_gradients=False;ctx.field_components=[];ctx.fusion_telemetry={};dr.prepare(ctx,ubatch)
        result=response_step(solver,FusionObjective(model,args,proto,args.xuc_row,dr),ctx,args,dr,SourceMonitor(source),completed)
        assert result.accepted
        dr.commit(ctx);_update_ema_model(ema,model,args.ema_decay);proto.update(ctx.origin_features,ctx.y,ctx.domain)
    stream.commit(ticket,dict(scenario=ctx.satellite_scenario,count=ctx.satellite_mask_count))
    return result.loss,result.field_evaluations,ctx.dr_telemetry

def payload(env,row):
    args,source,model,ema,dr,stream,proto,opt,solver=env
    return deepcopy(dict(schema='adv3b02_xuc_v1',model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
        prototype=vars(proto),args=vars(args),row=row,epoch=80,step=80,source_info=source.info,
        initialization='scratch_only',from_scratch=True,target_contact=False,solver=solver.state_dict(),
        tickets=stream.state_dict(),daot_rc4=dr.state_dict(),cstar=None,legacy_controller=None,legacy_curriculum=None))

@pytest.mark.parametrize('seed',[392005,392007])
@pytest.mark.parametrize('method',['SIM','EG','CF_EG','TR_EG','XT_DANN','DRIC'])
def test_legacy_exact_active_update(tmp_path,method,seed):
    torch.manual_seed(seed)
    row=make_row('pure_resume_'+method,dict(method=method),seed,pure_game=True)
    env=setup(row,tmp_path/'first');update(env,79)
    saved=payload(env,row);checkpoint=tmp_path/'snapshot.pth';torch.save(saved,checkpoint)
    # Replay active cadence after the checkpoint as well (CF/TR always, XT every
    # 20, DRIC every 4): algorithm clock may be separately chosen in this bounded
    # source-only diagnostic, while the checkpoint accepted clock remains real.
    expected=update(env,99)
    random.seed(981);np.random.seed(222);torch.manual_seed(719)
    other=setup(row,tmp_path/'resumed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    args,source,model,ema,dr,stream,proto,opt,solver=other
    assert validate_checkpoint(saved,args,row,source,1,allow_legacy=True)==80
    with pytest.raises(ValueError,match='MISSING_RNG'):validate_checkpoint(saved,args,row,source,1)
    restore_checkpoint(saved,model,ema,opt,proto,solver,stream,dr)
    actual=update(other,99)
    assert expected[:2]==actual[:2]
    for i in (2,3):equal(env[i].state_dict(),other[i].state_dict())
    equal(env[7].state_dict(),opt.state_dict());equal(vars(env[6]),vars(proto));equal(env[8].state_dict(),solver.state_dict())
    equal(env[5].state_dict(),stream.state_dict());equal(env[4].state_dict()['scale'],dr.state_dict()['scale'])
    assert env[4].commits==dr.commits==81
    assert actual[2]['daot_executed'] is False and actual[2]['rc4_executed'] is False

def test_full_entrypoint_pause_and_resume(tmp_path):
    row=make_row('pure_resume_entrypoint',dict(method='SIM'),pure_game=True)
    def args(path):
        a=resolve_args(RECIPE,row,dataset='NOT_READ',output=path,device='cpu',synthetic=True,test_epochs=2,test_steps=1)
        a.joint['diagnostic_interval']=0;a.joint['probe_interval']=0
        return a
    full=args(tmp_path/'full');train(full,row)
    part=args(tmp_path/'part');part.xuc_stop_after_epoch=1;train(part,row)
    assert (tmp_path/'part/pause_complete.json').exists()
    resumed=args(tmp_path/'resumed');resumed.xuc_resume=str(tmp_path/'part/latest_ssdg.pth');train(resumed,row)
    a=torch.load(tmp_path/'full/final_ssdg.pth',weights_only=False)
    b=torch.load(tmp_path/'resumed/final_ssdg.pth',weights_only=False)
    for k in ('model','ema','optimizer','prototype','solver','tickets'):equal(a[k],b[k])
    assert b['step']==2 and b['resume_lineage']['restored_step']==1
    assert json.loads((tmp_path/'resumed/initialization.json').read_text())['scratch_only'] is False

def test_rejects_wrong_row_source_config_and_clock(tmp_path):
    row=make_row('pure_resume_reject',dict(method='SIM'),pure_game=True)
    env=setup(row,tmp_path);update(env,79);saved=payload(env,row)
    for key,value,error in [('target_contact',True,'CONTAMINATED'),('step',79,'BOUNDARY'),
                            ('row',{},'ROW_MISMATCH'),('source_info',{},'DATA_CONTRACT')]:
        broken=deepcopy(saved);broken[key]=value
        with pytest.raises(ValueError,match=error):validate_checkpoint(broken,env[0],row,env[1],1,allow_legacy=True)
    broken=deepcopy(saved);del broken['optimizer']
    with pytest.raises(ValueError,match='MISSING_STATE'):validate_checkpoint(broken,env[0],row,env[1],1,allow_legacy=True)
    broken=deepcopy(saved);broken['args']['lr']=.1
    with pytest.raises(ValueError,match='CONFIG_MISMATCH'):validate_checkpoint(broken,env[0],row,env[1],1,allow_legacy=True)
    broken=deepcopy(saved);finished_args=deepcopy(env[0]);finished_args.epochs=80;broken['args']['epochs']=80
    with pytest.raises(ValueError,match='ALREADY_COMPLETE'):validate_checkpoint(broken,finished_args,row,env[1],1,allow_legacy=True)
