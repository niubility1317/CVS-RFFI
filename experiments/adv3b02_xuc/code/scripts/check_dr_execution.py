"""Bounded synthetic source-only check of actual DR objective, EG and checkpoint."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cvsrffi.xuc_fusion.runtime import resolve_args,synthetic_source
from cvsrffi.game_tracking.runtime import build_model,json_write
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.step_context import prepare_context
from cvsrffi.game_tracking.source_audit import isolated_rng
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.xuc_fusion.tickets import TicketStream
from cvsrffi.xuc_fusion.dr_objective import DROT
from cvsrffi.xuc_fusion.objective import FusionObjective
from cvsrffi.xuc_fusion.checkpoints import load_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    release=Path(__file__).resolve().parents[2]
    matrix=json.loads((release/'configs/matrix_dr.json').read_text())
    assert len(matrix['runs'])==7
    assert {r['parent_row'] for r in matrix['runs']}=={'M14','M11','M05','M08','M07','M12','M13'}
    row=next(r for r in matrix['runs'] if r['id']=='DR-M13')
    args=resolve_args(json.loads((release/'configs/core90_recipe_reference.json').read_text()),row,dataset='synthetic',output=a.output,device=a.device)
    args.game_synthetic=True;args.game_max_steps_per_epoch=1;args.num_classes=6
    torch.set_num_threads(2);torch.manual_seed(392005)
    source=synthetic_source();device=torch.device(a.device)
    model=build_model(args,15,device)
    ema=deepcopy(model).eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    groups={role:[] for role in ('backbone','other')}
    for name,parameter in model.named_parameters():groups['backbone' if 'backbone' in name.lower() else 'other'].append(parameter)
    optimizer=torch.optim.AdamW([dict(params=v,fasttrust_role=k) for k,v in groups.items()],lr=args.lr,weight_decay=args.weight_decay)
    solver=GameSolver(model,optimizer,max_grad_norm=5.)
    proto=PrototypeMemoryBank(6,15);proto._lazy_init(160,device,torch.float32)
    dr=DROT(model,ema,source,args,json.loads((release/'configs/a1_native_recipe_reference.json').read_text()))
    objective=FusionObjective(model,args,proto,row,dr)
    audit=[]
    for epoch in (1,21,131,181):
        from SSDG.train_ssdg import _apply_fasttrust_lr
        _apply_fasttrust_lr(optimizer,base_lr=args.lr,epoch=epoch,tail_mode=dr.args.a1_tail_lr)
        dr.calibrate(epoch)
        stream=TicketStream(source,args,row);stream.next_epoch=epoch
        ticket,_=stream.choose();batch,ubatch=stream.batches(ticket)
        assert len(ubatch[0])==256 and (ubatch[1]==-1).all()
        before_teacher={k:v.clone() for k,v in ema.state_dict().items()}
        with torch.no_grad():
            x=batch[0][:4].to(device)
            f=ema(x,return_aux=True);i=ema.forward_identity_only(x)
            torch.testing.assert_close(f['z_id'],i['z_id'],rtol=0,atol=0)
            torch.testing.assert_close(f['tx_logits'],i['tx_logits'],rtol=0,atol=0)
        model.train()
        with isolated_rng(ticket.seed):
            ca=deepcopy(args);ca.game_evidence_version=2
            ctx=prepare_context(batch,None,model,ema,ca,epoch,ticket.batch_index,ticket.weights,
                torch.Generator(device=device).manual_seed(ticket.seed),exposure_record=ticket.exposure(batch[3]['sample_id']))
            ctx.grid_plan=ticket.grid_plan;ctx.rx=batch[3]['rx_i'].to(device);ctx.day=batch[3]['day_i'].to(device)
            ctx.audit_gradients=True;ctx.field_components=[];ctx.fusion_telemetry={}
            dr.prepare(ctx,ubatch)
            before_scale=deepcopy(dr.scale.state_dict());before_commits=dr.commits
            solver.mode='simultaneous' if epoch==1 else 'extragradient'
            result=solver.step(lambda:objective(ctx))
            assert result.accepted
            assert dr.scale.state_dict()==before_scale and dr.commits==before_commits
            assert all(torch.equal(v,ema.state_dict()[k]) for k,v in before_teacher.items())
            assert len(ctx.field_components)==result.field_evaluations
            assert all(v==ctx.field_components[0] for v in ctx.field_components)
            assert {'daot_labeled','daot_unlabeled','rc4_total','x_cross_rx','u_normalized'}<=set(ctx.field_components[0])
            dr.commit(ctx);assert dr.commits==before_commits+1
            if epoch==1:assert ctx.dr_telemetry['daot_labeled']==0 and ctx.dr_telemetry['rc4_identity']==0
            else:assert ctx.dr_telemetry['daot_grad_norm']>0
            assert ctx.dr_telemetry['rc4_grad_norm']>0
            assert ctx.fusion_telemetry['normalized_valid_blocks']==4
            audit.append(dict(epoch=epoch,fields=result.field_evaluations,loss=result.loss,dr=ctx.dr_telemetry,fusion=ctx.fusion_telemetry,
                              lr={g['fasttrust_role']:g['lr'] for g in optimizer.param_groups}))
    # This generated diagnostic weight is never a formal training initialization.
    payload=dict(schema='adv3b02_xuc_v1',model=model.state_dict(),args=vars(args),epoch=200,
                 from_scratch=True,target_contact=False,synthetic_diagnostic_only=True)
    checkpoint=a.output/'generated_diagnostic.pth';torch.save(payload,checkpoint)
    rebuilt,_=load_model(checkpoint,device)
    model.eval()
    with torch.no_grad():torch.testing.assert_close(model(x,return_aux=True)['tx_logits'],rebuilt(x,return_aux=True)['tx_logits'],rtol=0,atol=0)
    json_write(a.output/'acceptance.json',dict(status='PASS',synthetic_source_only=True,audit=audit,
               strict_checkpoint_rebuild=True,teacher_unchanged_in_fields=True,one_scale_commit_per_main_step=True))
    print(json.dumps(dict(status='PASS',output=str(a.output),epochs=[1,21,131,181])))

if __name__=='__main__':main()
