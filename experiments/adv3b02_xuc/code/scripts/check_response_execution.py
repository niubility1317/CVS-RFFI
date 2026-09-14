"""Four actual response paths and scratch checkpoint roundtrip, without query."""
import argparse,json,sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cvsrffi.xuc_fusion.runtime import resolve_args,synthetic_source
from cvsrffi.xuc_fusion.response_config import make_row
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
from cvsrffi.xuc_fusion.response_solver import ResponseSolver
from cvsrffi.xuc_fusion.checkpoints import load_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
    recipe=json.loads((root/'configs/core90_recipe_reference.json').read_text());reference=json.loads((root/'configs/a1_native_recipe_reference.json').read_text())
    torch.set_num_threads(2);audit=[];device=torch.device(a.device)
    for method,epoch,step in [('CF_EG',80,3),('TR_EG',80,3),('XT_DANN',80,19),('DRIC',40,3)]:
        torch.manual_seed(32);row=make_row('diagnostic_'+method,dict(method=method))
        args=resolve_args(recipe,row,dataset='NOT_READ',output=a.output,device=a.device,synthetic=True);args.num_classes=6
        model=build_model(args,15,device);ema=deepcopy(model).eval()
        for par in ema.parameters():par.requires_grad_(False)
        source=synthetic_source();dr=DROT(model,ema,source,args,reference);dr.calibrate(epoch)
        stream=TicketStream(source,args,row);ticket,_=stream.choose();batch,ubatch=stream.batches(ticket)
        # Full configured L128/U256 exposes CUDA memory and batch-dependent errors.
        ctx=prepare_context(batch,None,model,ema,args,epoch,1,_loss_weights(args,build_stage_state(epoch,args)),torch.Generator(device=device).manual_seed(1))
        ctx.grid_plan=None;ctx.rx=batch[3]['rx_i'].to(device);ctx.day=batch[3]['day_i'].to(device)
        ctx.audit_gradients=False;ctx.field_components=[];ctx.fusion_telemetry={};dr.prepare(ctx,ubatch)
        proto=PrototypeMemoryBank(6,15);proto._lazy_init(160,device,torch.float32)
        objective=FusionObjective(model,args,proto,row,dr);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        solver=ResponseSolver(model,optimizer,args.response,max_grad_norm=5.)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        result=response_step(solver,objective,ctx,args,dr,SourceMonitor(source),step)
        assert result.accepted and solver.steps==1
        assert all(s['step']==1 for s in optimizer.state.values())
        assert all(v==ctx.dr_field_divisors[0] for v in ctx.dr_field_divisors)
        payload=dict(schema='adv3b02_xuc_v1',model=model.state_dict(),args=vars(args),epoch=200,target_contact=False,synthetic_diagnostic_only=True)
        checkpoint=a.output/(method+'_diagnostic.pth');torch.save(payload,checkpoint)
        rebuilt,_=load_model(checkpoint,device);model.eval()
        with torch.no_grad():torch.testing.assert_close(model(ctx.x[:2],return_aux=True)['tx_logits'],rebuilt(ctx.x[:2],return_aux=True)['tx_logits'],rtol=0,atol=0)
        audit.append(dict(method=method,epoch=epoch,accepted=True,fields=result.field_evaluations,telemetry=result.telemetry,
            strict_checkpoint_rebuild=True,peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
        print(json.dumps(dict(method=method,status='PASS')),flush=True)
        del model,ema,rebuilt,optimizer,solver,objective,dr,ctx,payload
        if device.type=='cuda':torch.cuda.empty_cache()
    (a.output/'acceptance.json').write_text(json.dumps(dict(status='PASS',source='synthetic_only',query_access=False,torch=torch.__version__,audit=audit),indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
