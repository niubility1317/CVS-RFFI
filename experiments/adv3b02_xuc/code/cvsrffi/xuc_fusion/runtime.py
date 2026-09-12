"""One source-only row. Every persistent update is committed outside its closure."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from cvsrffi.game_tracking.config import parser as game_parser, validate as validate_game
from cvsrffi.game_tracking.data import build_source, SourceData, audit_indices
from cvsrffi.game_tracking.runtime import build_model, plain, json_write, append, audit, calibrate, head_catchup
from cvsrffi.game_tracking.legacy.options import _update_ema_model
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.source_audit import SourceAuditor, AuditConfig, isolated_rng
from cvsrffi.game_tracking.budget import ComputeBudget, BudgetConfig
from cvsrffi.game_tracking.solvers import GameSolver
from cvsrffi.game_tracking.step_context import prepare_context, satellite_stage
from cvsrffi.schedule import build_aug_base_cfg, make_augmentor, configure_augmentor_for_epoch, configure_mixstyle_for_epoch
from cvsrffi.tensors import set_seed
from .objective import FusionObjective
from .tickets import TicketStream
from .control import SourceObserver, ReliableController


class SyntheticSource(Dataset):
    def __init__(self,role,seed):
        self.role=role;self.items=[];gen=torch.Generator().manual_seed(seed)
        for tx in range(6):
            for rx in range(5):
                for day in range(3):
                    for k in range(4):
                        x=torch.randn(2,256,generator=gen)*.2
                        t=torch.arange(256.)
                        x[0]+=torch.sin(t*(.05+.02*tx));x[1]+=torch.cos(t*(.05+.02*tx))+rx*.1
                        i=len(self.items)
                        meta=dict(rx_i=rx,day_i=day,eq_i=1,sig_i=k,base_index=i,sample_id=f'{role}-{i}',role=role)
                        if role!='unlabeled':meta['capture_group']=f'{tx}:{rx}:{day}'
                        self.items.append((x,-1 if role=='unlabeled' else tx,rx*3+day,meta))
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]


def synthetic_source():
    data=[SyntheticSource(r,s) for r,s in [('train',1),('unlabeled',2),('val',3)]]
    return SourceData(*data,{(r,d):r*3+d for r in range(5) for d in range(3)},
                      dict(synthetic=True,counts={'L_s':360,'U_s':360,'V':360},target_access_before_freeze=False,checkpoint_init='scratch_only'))


def resolve_args(recipe,row,*,dataset,output,device='cuda:0',synthetic=False,test_epochs=1,test_steps=1):
    p=game_parser();known={a.dest for a in p._actions}
    base=dict(recipe['baseline_args'])
    ignored={k:v for k,v in base.items() if k not in known}
    defaults={k:v for k,v in base.items() if k in known}
    defaults.update(seed=392005,game_split_seed=392005,game_data_order_seed=392005,
        wisig_pkl=str(dataset),wisig_equalized='1',wisig_train_rxs='1,3,4,6,8',wisig_train_days='1,2,3',
        wisig_test_rxs='0,2,5,7,9,10,11',wisig_test_days='0,1,2,3',wisig_domain='rx_day',
        epochs=200,batch_size=128,eval_batch_size=256,num_workers=0,device=device,amp=False,
        from_scratch=True,baseline_ckpt='',best_metric='clean_val_tx',enable_joint_safe_guard=False,
        paic_guard_enabled=False,game_resume='',game_skip_final_eval=True,game_evidence_version=1,
        game_control='both' if row['control']=='legacy_both' else 'off',
        game_curriculum='capability' if row['curriculum']=='legacy_capability' else 'fixed',
        game_no_audit=row['control']!='legacy_both',game_telemetry_interval=0,
        game_synthetic=synthetic,game_max_grad_norm=5.,game_max_steps_per_epoch=0)
    if synthetic:defaults.update(epochs=test_epochs,game_max_steps_per_epoch=test_steps)
    p.set_defaults(**defaults)
    args=p.parse_args(['--output_dir',str(output)])
    validate_game(args)
    args.xuc_unconsumed_reference_options=ignored
    args.xuc_row=deepcopy(row)
    return args


def validate_source_contract(source, path=None):
    if source.info.get('synthetic'):return
    if source.info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('actual source role sizes differ')
    if source.info['num_classes']!=6:raise ValueError('TX universe differs')
    if path:
        expected=json.loads(Path(path).read_text(encoding='utf-8'))
        if expected['role_ids']!=source.info['role_ids']:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: shared physical roles')


def evaluate_source(model,source,args):
    flags={m:m.training for m in model.modules()};model.eval()
    correct=total=0;ce=0.
    try:
        with torch.no_grad():
            for x,y,_,_ in source.loader('val',args.eval_batch_size,workers=0):
                x=x.to(args.device);y=y.to(args.device)
                logits=model(x,return_aux=True)['tx_logits']
                correct+=int((logits.argmax(1)==y).sum());total+=len(y)
                ce+=float(torch.nn.functional.cross_entropy(logits,y,reduction='sum'))
    finally:
        for m,flag in flags.items():m.training=flag
    return dict(correct=correct,total=total,accuracy=correct/total,ce=ce/total,source_V_read_only=True)


def train(args,row,source_contract=None):
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Refusing existing output '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    if not args.game_synthetic and (args.epochs!=200 or args.seed!=392005 or args.amp or args.baseline_ckpt):raise ValueError('formal row contract changed')
    set_seed(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('requested CUDA unavailable')
    torch.set_num_threads(2)
    source=synthetic_source() if args.game_synthetic else build_source(args)
    if args.game_synthetic:args.num_classes=6
    validate_source_contract(source,source_contract)
    json_write(output/'source_contract.json',source.info)
    json_write(output/'resolved_config.json',vars(args))
    model=build_model(args,len(source.domains),device)
    ema=deepcopy(model).eval() if args.use_ema_teacher else None
    if ema is not None:
        for parameter in ema.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    solver=GameSolver(model,optimizer,nonfinite='raise',max_grad_norm=5.)
    proto=PrototypeMemoryBank(args.num_classes,len(source.domains),momentum=args.proto_momentum,
        margin=args.proto_margin,domain_align_weight=args.proto_domain_align_weight,push_weight=args.proto_push_weight,min_count=args.proto_min_count)
    proto._lazy_init(160,device,torch.float32)
    objective=FusionObjective(model,args,proto,row)
    stream=TicketStream(source,args,row)
    audit_enabled=row['source_audit_policy']=='cstar_isolated_passive_or_active'
    observer=SourceObserver(source,args) if audit_enabled else None
    reliable=ReliableController() if audit_enabled else None
    legacy=row['control']=='legacy_both'
    indexes=audit_indices(source,args.game_audit_samples_per_capture) if legacy else None
    auditor=SourceAuditor(AuditConfig(steps=40,independent_interval=4)) if legacy else None
    budget=ComputeBudget(BudgetConfig(window_steps=100,max_head_steps=20,max_field_evaluations=20,
                                     max_audit_seconds=120.,max_correction_fraction=.2))
    legacy_controller=legacy_curriculum=None
    observations=[];metrics={};next_audit=0
    aug_cfg=build_aug_base_cfg(args) if args.use_aug else None
    augmentor=make_augmentor(aug_cfg) if aug_cfg else None
    if augmentor is not None and hasattr(augmentor,'to'):augmentor=augmentor.to(device)
    started=time.perf_counter();epoch_rows=[];completed=0;actions={'NORMAL':0,'CATCHUP':0,'CORRECT':0};head_total=0
    print(f'[XUC-INIT] row={row["id"]} scratch_only=true seed=392005 domains={len(source.domains)} steps_per_epoch={stream.steps}',flush=True)
    json_write(output/'initialization.json',dict(scratch_only=True,checkpoint_sources=[],seed=args.seed,
        parameter_count=sum(p.numel() for p in model.parameters()),batchnorm_modules=[n for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)],
        mixstyle_modules=[n for n,m in model.named_modules() if m.__class__.__name__=='MixStyle1D'],target_contact=False))
    while completed<args.epochs*stream.steps:
        live_epoch=completed//stream.steps+1
        if reliable is not None and completed%250==0:
            # Audit is independent of which remaining ticket the controller selects.
            from cvsrffi.schedule import build_stage_state
            from cvsrffi.game_tracking.legacy.options import _loss_weights
            weights=_loss_weights(args,build_stage_state(live_epoch,args))
            evidence=observer.observe(model,step=completed,epoch=live_epoch,adv_weight=weights['adv'])
            reliable.observe(evidence);append(output/'cstar_audit.jsonl',evidence)
        ticket,coverage=stream.choose(capability=reliable.capability(completed) if reliable else False,
                                     difficulty=reliable.latest.get('difficulty',{}) if reliable and reliable.latest else {})
        batch,ubatch=stream.batches(ticket)
        model.train();configure_mixstyle_for_epoch(model,args,ticket.epoch)
        if augmentor is not None:configure_augmentor_for_epoch(augmentor,aug_cfg,min(ticket.epoch,args.label_epochs),args)
        decision=dict(action='NORMAL',catchup_steps=0,reason='fixed')
        sets=None
        if legacy and completed>=next_audit:
            forecast=min(120.,max(1.,metrics.get('elapsed_seconds',1.) or 1.))
            if budget.can_afford(completed,audit_seconds=forecast)[0]:
                metrics,sets=audit(model,source,indexes,auditor,args,completed,completed,ticket.epoch,proto)
                append(output/'legacy_audit.jsonl',metrics);budget.commit(completed,audit_seconds=metrics['elapsed_seconds'])
                if legacy_controller is None:
                    observations.append(metrics)
                    if completed>=500:legacy_controller,legacy_curriculum=calibrate(observations,args)
            else:metrics=dict(valid=False,step=completed,encoder_version=completed,reason='audit_budget_exhausted')
            next_audit=completed+250
        if legacy_controller is not None:
            from cvsrffi.game_tracking.controller import signal_fresh
            c=legacy_controller.config
            if signal_fresh(metrics,completed,completed,c.max_age_steps,c.max_version_lag):
                decision=legacy_controller.decide(metrics,step=completed,encoder_version=completed,budget=budget)
                if sets is not None:next_audit=completed+decision['next_audit_interval']
        if reliable is not None:decision=reliable.decide(completed,row['cstar_action_enabled'])
        exposure=ticket.exposure(batch[3]['sample_id'])
        if legacy:
            level=legacy_curriculum.level if legacy_curriculum else 0.
            scenes,p=satellite_stage(ticket.epoch,level)
            exposure['scenario']=scenes[(ticket.epoch+ticket.batch_index-2)%len(scenes)]
            g=torch.Generator().manual_seed(ticket.seed);torch.randint(0,2**31-1,(),generator=g)
            exposure['selected_mask']=(torch.rand(len(batch[0]),generator=g)<p).tolist()
        with isolated_rng(ticket.seed):
            # Version2 here selects deterministic exposure replay, not the GAME V2 controller.
            context_args=deepcopy(args);context_args.game_evidence_version=2
            gen=torch.Generator(device=device).manual_seed(ticket.seed)
            ctx=prepare_context(batch,ubatch,model,ema,context_args,ticket.epoch,ticket.batch_index,ticket.weights,gen,augmentor,
                                exposure_record=exposure)
            ctx.grid_plan=ticket.grid_plan;ctx.rx=batch[3]['rx_i'].to(device);ctx.day=batch[3]['day_i'].to(device)
            ctx.audit_gradients=completed%250==0;ctx.field_components=[];ctx.fusion_telemetry={}
            count=decision.get('catchup_steps',0)
            if legacy and count and not budget.can_afford(completed,head_steps=count)[0]:count=0
            head_steps,head_enc=head_catchup(model,optimizer,ctx,count,args)
            solver.mode='extragradient' if decision['action']=='CORRECT' else 'simultaneous'
            result=solver.step(lambda:objective(ctx))
            if result.accepted:
                if ema is not None:_update_ema_model(ema,model,args.ema_decay)
                proto.update(ctx.origin_features,ctx.y,ctx.domain)
        if not result.accepted:raise RuntimeError('main update not accepted')
        if result.field_evaluations==2 and (len(ctx.field_components)!=2 or ctx.field_components[0]!=ctx.field_components[1]):raise RuntimeError('EG objective mismatch')
        budget.commit(completed,field_evaluations=result.field_evaluations-1,base_steps=1,corrections=int(result.field_evaluations>1),head_steps=head_steps)
        if reliable is not None:reliable.commit(decision,completed,True)
        if legacy_curriculum is not None and sets is not None:
            event=legacy_curriculum.update(metrics,step=metrics['step'],encoder_version=metrics['encoder_version'],hold=not metrics.get('valid',False))
            append(output/'legacy_curriculum.jsonl',dict(step=completed,**event))
            if event['changed']:metrics={}
        window=stream.commit(ticket)
        if window:append(output/'ticket_windows.jsonl',window)
        action=decision['action'] if decision['action'] in actions else 'NORMAL'
        actions[action]+=1;head_total+=head_steps
        record=dict(step=completed,execution_epoch=live_epoch,origin_epoch=ticket.epoch,origin_batch_index=ticket.batch_index,
            ticket_id=ticket.ticket_id,weights=ticket.weights,source_ids=ctx.sample_ids,
            unlabeled_ids=list(ubatch[3]['sample_id']) if ubatch is not None else [],
            selected_mask=ctx.satellite_selected_mask.cpu().tolist(),scenario=ctx.satellite_scenario,channel_seed=ticket.channel_seed,
            coverage_driven=coverage,capability_confirmed=reliable.capability(completed) if reliable else None,
            action=decision['action'],reason=decision['reason'],head_steps=head_steps,field_evaluations=result.field_evaluations,
            field_components=ctx.field_components,loss=result.loss,loss_replay=result.loss_replay,grad_norm=result.grad_norm,
            terms=ctx.origin_terms,fusion=ctx.fusion_telemetry,accepted=True,elapsed_seconds=time.perf_counter()-started)
        append(output/'actions.jsonl',record);epoch_rows.append(record);completed+=1
        if completed%stream.steps==0:
            validation=evaluate_source(model,source,args)
            summary=dict(epoch=live_epoch,total_step=completed,accepted=len(epoch_rows),
                         mean_loss=float(np.mean([r['loss'] for r in epoch_rows])),source_validation=validation,
                         terms={k:float(np.mean([r['terms'].get(k,0.) for r in epoch_rows])) for k in epoch_rows[0]['terms']},
                         elapsed_seconds=time.perf_counter()-started,source_only=True)
            append(output/'logs.jsonl',summary);epoch_rows=[]
            payload=dict(schema='adv3b02_xuc_v1',model=model.state_dict(),ema=ema.state_dict() if ema else None,
                optimizer=optimizer.state_dict(),prototype=deepcopy(vars(proto)),args=vars(args),row=row,epoch=live_epoch,
                step=completed,source_info=source.info,initialization='scratch_only',from_scratch=True,target_contact=False,
                solver=solver.state_dict(),tickets=stream.state_dict(),cstar=reliable.state_dict() if reliable else None,
                legacy_controller=legacy_controller.state_dict() if legacy_controller else None,
                legacy_curriculum=legacy_curriculum.state_dict() if legacy_curriculum else None)
            temp=output/'latest_ssdg.tmp';torch.save(payload,temp);temp.replace(output/'latest_ssdg.pth')
            if live_epoch==args.epochs:torch.save(payload,output/'final_ssdg.pth')
            print(f'[EPOCH-END] E{live_epoch:03d}/{args.epochs} accepted={stream.steps} loss={summary["mean_loss"]:.5f}',flush=True)
    json_write(output/'completion.json',dict(status='TRAINING_COMPLETE',epochs=args.epochs,steps=completed,
        actions=actions,head_steps=head_total,cstar='CONTROL_NOT_ACTIVATED' if reliable and not reliable.actions else None,
        target_evaluated=False,elapsed_seconds=time.perf_counter()-started))
    return 0
