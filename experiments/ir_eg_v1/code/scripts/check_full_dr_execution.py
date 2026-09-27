"""Bounded synthetic source-only check of actual DR objective, EG and checkpoint."""
import argparse
from copy import deepcopy
from dataclasses import replace
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
from cvsrffi.xuc_fusion.full_dr_objective import FullDROT
from cvsrffi.xuc_fusion.objective import FusionObjective
from cvsrffi.xuc_fusion.checkpoints import load_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--controlled-routes',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    release=Path(__file__).resolve().parents[2]
    matrix=json.loads((release/'configs/matrix_dr_full.json').read_text())
    assert len(matrix['runs'])==9
    assert all(r['steps_per_epoch']==222 for r in matrix['runs'])
    row=next(r for r in matrix['runs'] if r['id']=='F-M13')
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
    dr=FullDROT(model,ema,source,args,json.loads((release/'configs/a1_native_recipe_reference.json').read_text()),proto)
    objective=FusionObjective(model,args,proto,row,dr)
    audit=[]
    for epoch in (1,21,61,131,181):
        from SSDG.train_ssdg import _apply_fasttrust_lr
        _apply_fasttrust_lr(optimizer,base_lr=args.lr,epoch=epoch,tail_mode=dr.args.a1_tail_lr)
        model.eval()
        with torch.no_grad():
            for sx,sy,sd,_ in source.loader('train',128):
                so=model(sx.to(device),return_aux=True)
                proto.update(so['z_id'],sy.to(device),sd.to(device))
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
            if a.controlled_routes and epoch>=21:
                # Deliberately injected diagnostic H/P/N masks. Never used by train().
                n=len(ctx.dr_x);hard=torch.zeros(n,dtype=torch.bool,device=device)
                partial=hard.clone();negative=hard.clone()
                hard[:8]=True;partial[8:16]=True;negative[16:24]=True
                candidate=torch.zeros(n,6,dtype=torch.bool,device=device);candidate[:,:2]=True
                ctx.dr_route=replace(ctx.dr_route,hard=hard,partial=partial,negative=negative,
                    representation=~(hard|partial|negative),candidate_mask=candidate,
                    excluded_mask=~candidate,weights=torch.full((n,),.1,device=device))
                dr.prepare_satellite(ctx)
            before_scale=deepcopy(dr.scale.state_dict());before_commits=dr.commits
            before_anchor={k:v.clone() for k,v in dr.anchor.state_dict().items()} if dr.anchor else {}
            before_proto=ctx.dr_prototypes.clone() if ctx.dr_prototypes is not None else None
            solver.mode='simultaneous' if epoch==1 else 'extragradient'
            result=solver.step(lambda:objective(ctx))
            assert result.accepted
            assert dr.scale.state_dict()==before_scale and dr.commits==before_commits
            assert all(torch.equal(v,ema.state_dict()[k]) for k,v in before_teacher.items())
            assert all(torch.equal(v,dr.anchor.state_dict()[k]) for k,v in before_anchor.items())
            if before_proto is not None:assert torch.equal(before_proto,ctx.dr_prototypes)
            assert len(ctx.field_components)==result.field_evaluations
            assert all(v==ctx.field_components[0] for v in ctx.field_components)
            assert {'daot_labeled','daot_unlabeled','rc4_total','x_cross_rx','u_normalized'}<=set(ctx.field_components[0])
            dr.commit(ctx);assert dr.commits==before_commits+1
            if epoch==1:assert ctx.dr_telemetry['daot_labeled']==0 and ctx.dr_telemetry['rc4_identity']==0
            else:assert ctx.dr_telemetry['daot_grad_norm']>0
            assert ctx.dr_telemetry['rc4_grad_norm']>0
            assert ctx.dr_telemetry['rc4_u_grl_lambda']==0
            if epoch>=21:
                assert ctx.dr_telemetry['prototype_rows']==6 and ctx.dr_telemetry['anchor_active']
                assert ctx.dr_telemetry['nuisance_head_grad_norm']>0
            if a.controlled_routes and epoch>=21:
                for key in ['rc4_hard','rc4_partial_set','rc4_partial_conditional','rc4_negative','rc4_satellite','rc4_anchor']:
                    assert ctx.dr_telemetry[key+'_grad_norm']>0,(epoch,key)
                assert ctx.dr_telemetry['satellite_samples']==8
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
    json_write(a.output/'acceptance.json',dict(status='PASS',synthetic_source_only=True,controlled_routes=a.controlled_routes,audit=audit,
               strict_checkpoint_rebuild=True,teacher_unchanged_in_fields=True,anchor_and_prototypes_unchanged_in_fields=True,
               cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
               one_scale_commit_per_main_step=True))
    print(json.dumps(dict(status='PASS',output=str(a.output),epochs=[1,21,61,131,181])))

if __name__=='__main__':main()
