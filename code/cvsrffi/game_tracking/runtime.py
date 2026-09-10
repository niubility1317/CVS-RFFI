from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from cvsrffi.schedule import build_stage_state, build_aug_base_cfg, make_augmentor, configure_augmentor_for_epoch, configure_mixstyle_for_epoch
from cvsrffi.tensors import set_seed
from cvsrffi.eval import apply_sat_channel_for_scenario
from .legacy.model_dual_cvsincnet import build_dual_model
from .legacy.options import _loss_weights, _update_ema_model
from .legacy.losses import PrototypeMemoryBank
from .config import SCENES
from .data import build_source,audit_indices
from .state import RNGState,clone_buffers,restore_buffers
from .solvers import GameSolver
from .parameter_roles import role_manifest
from .step_context import prepare_context,Core90Objective
from .source_audit import SourceAuditor,AuditConfig,isolated_rng
from .capability import evaluate_capability,CapabilityConfig
from .gradient_audit import gradient_diagnostics
from .controller import GameController,ControllerConfig,signal_fresh
from .curriculum import CapabilityCurriculum,CurriculumConfig
from .budget import ComputeBudget,BudgetConfig

SCHEMA = 'core90_game_epoch_boundary_v1'


def make_grad_scaler(device,enabled):
    if hasattr(torch.amp,'GradScaler'):
        return torch.amp.GradScaler(device.type,enabled=enabled)
    if enabled and device.type!='cuda':
        raise ValueError('This PyTorch version supports GradScaler only on CUDA')
    return torch.cuda.amp.GradScaler(enabled=bool(enabled and device.type=='cuda'))


def update_pseudo_problem(solver,ctx,num_classes,threshold,window=1):
    if ctx.pseudo is None: return False
    mask=ctx.base_mask
    counts=torch.bincount(ctx.pseudo[mask],minlength=num_classes).float()
    current=counts.cpu().tolist()
    acc=solver.pseudo_accumulator or dict(batches=0,samples=0,selected=0,counts=[0.]*num_classes)
    acc['batches']+=1;acc['samples']+=len(mask);acc['selected']+=int(mask.sum())
    acc['counts']=[a+b for a,b in zip(acc['counts'],current)]
    solver.pseudo_accumulator=acc
    if acc['batches']<window: return False
    histogram=[v/max(1,acc['selected']) for v in acc['counts']]
    signature=dict(selected_fraction=acc['selected']/acc['samples'],histogram=histogram)
    solver.pseudo_accumulator=None
    previous=solver.pseudo_signature
    changed=previous is not None and max(abs(signature['selected_fraction']-previous['selected_fraction']),
                .5*sum(abs(a-b) for a,b in zip(histogram,previous['histogram'])))>threshold
    if changed: solver.reset_history('significant_teacher_pseudo_distribution_change')
    solver.pseudo_signature=signature
    return changed

def plain(value):
    if torch.is_tensor(value): return plain(value.detach().cpu().tolist())
    if isinstance(value,dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    if isinstance(value,np.generic): return plain(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def json_write(path,value):
    Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(plain(value),ensure_ascii=False,allow_nan=False)+'\n')

def build_model(args,num_domains,device):
    keys = ('model_size','model_variant','branch_ablation','domain_branch_ablation',
            'domain_enhancer','domain_enhancer_strength','mixstyle_p','mixstyle_alpha','mixstyle_eps',
            'mixstyle_layers','mixstyle_use_domain_label','mixstyle_mix','mixstyle_strength','mixstyle_fallback')
    kw = {k:getattr(args,k) for k in keys}
    kw.update(dataset='wisig',input_len=args.wisig_out_len,sample_rate_hz=25e6,mixstyle_on=args.use_mixstyle,
              id_feature_key='feat_joint',dom_feature_key='feat_imp',use_tx_adv_on_zdom=False)
    return build_dual_model(args.num_classes,num_domains,**kw).to(device)

def extract(model,dataset,indices,batch_size=128):
    device = next(model.parameters()).device
    zs,ys,ds,rxs,xs,groups = [],[],[],[],[],[]
    flags = {m:m.training for m in model.modules()}
    buffers = clone_buffers(model)
    try:
        model.eval()
        with torch.no_grad():
            for x,y,d,meta in DataLoader(Subset(dataset,indices),batch_size=batch_size,shuffle=False):
                x = x.to(device)
                out = model(x,return_aux=True)
                zs.append(out['z_id'].float().detach()); ys.append(y.to(device)); ds.append(d.to(device))
                rxs.append(meta['rx_i'].to(device)); xs.append(x); groups.extend(meta['capture_group'])
        return dict(z=torch.cat(zs),y=torch.cat(ys),d=torch.cat(ds),rx=torch.cat(rxs),x=torch.cat(xs),groups=groups)
    finally:
        restore_buffers(model,buffers)
        for m,flag in flags.items(): m.training=flag

def audit(model,source,indexes,auditor,args,step,version,epoch=1,proto=None):
    started = time.perf_counter()
    with isolated_rng(8103):
        sets = {k:extract(model,source.train,v,args.eval_batch_size) for k,v in indexes.items()}
        fit,mon = sets['domain_fit'],sets['domain_monitor']
        unique_rx = sorted(set(fit['rx'].tolist())|set(mon['rx'].tolist()))
        rx_map = {r:i for i,r in enumerate(unique_rx)}
        fit_rx = torch.tensor([rx_map[r] for r in fit['rx'].tolist()],device=fit['z'].device)
        mon_rx = torch.tensor([rx_map[r] for r in mon['rx'].tolist()],device=mon['z'].device)
        result = auditor.run(model.adv_head,fit['z'],fit['d'],mon['z'],mon['d'],
                             fit_groups=fit['groups'],monitor_groups=mon['groups'],step=step,encoder_version=version,
                             label_kind='domain',rx_fit_labels=fit_rx,rx_monitor_labels=mon_rx,
                             independence_verified=True,grouping_kind='whole_tx_rx_day_container_cross_tx_probe')
        cf,cm = sets['capability_fit'],sets['capability_monitor']
        # Fixed source development perturbation, separate generator from training.
        flags = {m:m.training for m in model.modules()}
        model.eval()
        try:
            with torch.no_grad():
                generator = torch.Generator(device=cm['x'].device).manual_seed(88103)
                aug,_ = apply_sat_channel_for_scenario(cm['x'],'leo_clear_weak',args,gen=generator,return_meta=False)
                augz = model(aug,return_aux=True)['z_id'].float()
            cap = evaluate_capability(cf['z'],cf['y'],cf['rx'],cm['z'],cm['y'],cm['rx'],
                                      fit_groups=cf['groups'],monitor_groups=cm['groups'],monitor_aug=augz,
                                      config=CapabilityConfig(steps=args.game_probe_steps),step=step,encoder_version=version,
                                      independence_verified=True)
            # Fixed diagnostic batch/augmentation; no GRL in these RX derivatives.
            x,y,d = mon['x'][:32],mon['y'][:32],mon['d'][:32]
            out = model(x,y_tx=y,return_aux=True)
            params = list(model.id_backbone.parameters())
            gi = torch.autograd.grad(F.cross_entropy(out['tx_logits'],y),params,retain_graph=True,allow_unused=True)
            model.adv_head.eval(); result.recovered_head.eval()
            current_weights=_loss_weights(args,build_stage_state(epoch,args))
            weight = current_weights['adv']
            ga = torch.autograd.grad(-weight*F.cross_entropy(model.adv_head(out['z_id']),d),params,retain_graph=True,allow_unused=True)
            gr = torch.autograd.grad(-weight*F.cross_entropy(result.recovered_head(out['z_id']),d),params,retain_graph=proto is not None,allow_unused=True)
            gi = [torch.zeros_like(p) if g is None else g for p,g in zip(params,gi)]
            ga = [torch.zeros_like(p) if g is None else g for p,g in zip(params,ga)]
            gr = [torch.zeros_like(p) if g is None else g for p,g in zip(params,gr)]
            grad = gradient_diagnostics(gi,ga,gr)
            grad['identity_scope']='ordinary_TX_CE'
            grad['effective_adversarial_weight']=weight
            if proto is not None:
                from .legacy.objective import labeled_terms
                nonadv_weights=dict(current_weights,adv=0.)
                nonadv,_=labeled_terms(out,y,d,args,epoch,1,nonadv_weights,proto)
                gn=torch.autograd.grad(nonadv,params,allow_unused=True)
                gn=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gn)]
                grad['labeled_nonadversarial']=gradient_diagnostics(gn,ga,gr)
                grad['labeled_nonadversarial_scope']='all_current_labeled_clean_terms; satellite_and_U_objectives_not_in_this_reference_batch'
        finally:
            for m,flag in flags.items(): m.training=flag
    metrics = dict(result.metrics)
    metrics.update(identity=cap['identity'],margin=cap['margin'],identity_valid=cap['valid'],collapsed=cap['collapsed'],
                   consistency=cap['consistency'],capability=cap,gradient=grad)
    metrics['valid'] = bool(metrics['valid'] and cap['valid'])
    # Prefer online/recovered direction disagreement, not random minibatch oscillation.
    cosine = grad.get('online_recovered',{}).get('cosine')
    metrics['direction_imbalance'] = 1.-cosine if isinstance(cosine,(float,int)) else None
    metrics['gradient_valid'] = bool(grad.get('valid',False))
    metrics['model_forwards'] = sum(math.ceil(len(v)/args.eval_batch_size) for v in indexes.values())+2
    metrics['gradient_backward_evaluations'] = 4 if proto is not None else 3
    metrics['elapsed_seconds'] = time.perf_counter()-started
    return metrics,sets

def calibrate(observations,args):
    valid = [m for m in observations if m.get('valid') and all(isinstance(m.get(k),(int,float)) and math.isfinite(m[k]) for k in ('G_lag','identity','margin','consistency'))]
    if len(valid)<3: return None,None
    def q(key,p): return float(np.quantile([m[key] for m in valid],p))
    lag = max(.01,q('G_lag',.75)+float(np.std([m['G_lag'] for m in valid])))
    identity = max(1./args.num_classes,min(.95,q('identity',.25)*.9))
    margin = q('margin',.25)-max(.01,float(np.std([m['margin'] for m in valid])))
    consistency = max(-1.,min(.98,q('consistency',.25)-.05))
    freshness = max(1,min(10,args.game_audit_interval//10))
    c = ControllerConfig(lag_enter=lag,lag_exit=.5*lag,identity_min=identity,margin_min=margin,
                         catchup_steps=args.game_max_extra_head,max_age_steps=freshness,max_version_lag=freshness,
                         cooldown_steps=args.game_audit_interval,audit_interval=args.game_audit_interval,
                         sparse_audit_interval=2*args.game_audit_interval,calibration_id='source_warmup_three_or_more_audits')
    cu = CurriculumConfig(identity_enter=min(.99,identity+.05),identity_exit=identity,
                          margin_enter=margin+.02,margin_exit=margin,lag_max=lag,
                          consistency_enter=min(1.,consistency+.05),consistency_exit=consistency,
                          cooldown_steps=args.game_audit_interval,max_age_steps=freshness,max_version_lag=freshness,
                          calibration_id=c.calibration_id)
    return GameController(c),CapabilityCurriculum(cu)

def head_catchup(model,optimizer,ctx,k,args):
    if k<=0: return 0,0
    flags = {m:m.training for m in model.modules()}
    try:
        model.eval()
        with torch.no_grad(): z = model(ctx.x,return_aux=True)['z_id'].detach()
    finally:
        for m,flag in flags.items(): m.training=flag
    solver = GameSolver(model,optimizer,nonfinite='skip',max_grad_norm=args.game_max_grad_norm)
    count = 0
    for _ in range(k):
        scale = ctx.weights['adv'] if args.game_head_scale=='legacy_weighted' else 1.
        if scale<=0: break
        result = solver.step(lambda: scale*F.cross_entropy(model.adv_head(z),ctx.domain))
        count += int(result.accepted)
    return count,1

def checkpoint(path,model,ema,optimizer,scaler,proto,solver,controller,curriculum,budget,auditor,gen,args,source,epoch,step,version,observations,metrics,elapsed,next_audit=0):
    payload = dict(schema=SCHEMA,model=model.state_dict(),ema=ema.state_dict() if ema is not None else None,
                   optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),prototype=deepcopy(vars(proto)),
                   solver=solver.state_dict(),controller=controller.state_dict() if controller else None,
                   curriculum=curriculum.state_dict() if curriculum else None,budget=budget.state_dict(),auditor=auditor.state_dict(),
                   rng=RNGState.capture(),satellite_generator=gen.get_state(),args=vars(args),source_info=source.info,
                   epoch=epoch,step=step,encoder_version=version,calibration_observations=observations,last_metrics=metrics,
                   elapsed_seconds=elapsed,next_audit=next_audit,initialization='scratch_only',checkpoint_selection='final_only',
                   resume_boundary='epoch_end; source sampler deterministically seeded by epoch',target_contact=False)
    tmp = Path(str(path)+'.tmp')
    torch.save(payload,tmp)
    tmp.replace(path)

def train(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.game_resume:
        raise FileExistsError('Output exists; refusing to overwrite '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        json_write(output/'resolved_config.json',vars(args)); return 0
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    source = build_source(args)
    model = build_model(args,len(source.domains),device)
    ema = deepcopy(model).eval() if args.use_ema_teacher else None
    if ema is not None:
        for p in ema.parameters(): p.requires_grad_(False)
    head_ids = {id(p) for p in model.adv_head.parameters()}
    groups = [dict(params=[p for p in model.parameters() if id(p) not in head_ids],lr=args.lr),
              dict(params=list(model.adv_head.parameters()),lr=args.lr*args.game_head_lr_ratio)]
    optimizer_cls = torch.optim.AdamW if args.game_optimizer=='adamw' else torch.optim.SGD
    optimizer = optimizer_cls(groups,lr=args.lr,weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device,args.amp)
    solver = GameSolver(model,optimizer,args.game_solver,nonfinite='skip',max_grad_norm=args.game_max_grad_norm,scaler=scaler)
    proto = PrototypeMemoryBank(args.num_classes,len(source.domains),momentum=args.proto_momentum,
                                margin=args.proto_margin,domain_align_weight=args.proto_domain_align_weight,
                                push_weight=args.proto_push_weight,min_count=args.proto_min_count)
    # Initialization happens outside any loss transaction.
    proto._lazy_init(160,device,torch.float32)
    objective = Core90Objective(model,args,proto)
    auditor = SourceAuditor(AuditConfig(steps=args.game_probe_steps,independent_interval=args.game_independent_probe_every))
    budget = ComputeBudget(BudgetConfig(window_steps=100,max_head_steps=20,max_field_evaluations=20,
                                       max_audit_seconds=120.,max_correction_fraction=args.game_correction_fraction))
    controller=curriculum=None
    observations=[]; metrics={}; step=version=0; first_epoch=1; prior_elapsed=0.
    gen = torch.Generator(device=device).manual_seed(args.seed+991)
    saved = None
    if args.game_resume:
        saved = torch.load(args.game_resume,map_location='cpu',weights_only=False)
        if saved.get('schema')!=SCHEMA or saved.get('initialization')!='scratch_only' or saved.get('target_contact'):
            raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: only this scratch lineage epoch-boundary resume is allowed')
        if saved['source_info'] != source.info:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        allowed = {'game_resume','output_dir','game_skip_final_eval'}
        changed = {k for k,v in vars(args).items() if k not in allowed and saved['args'].get(k)!=v}
        if changed: raise ValueError('Resume changes training contract: '+str(changed))
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        if ema is not None: ema.load_state_dict(saved['ema'])
        vars(proto).update({k:v.to(device) if torch.is_tensor(v) else v for k,v in saved['prototype'].items()}); solver.load_state_dict(saved['solver']); budget.load_state_dict(saved['budget']); auditor.load_state_dict(saved['auditor'])
        if saved['controller']:
            controller=GameController(ControllerConfig(**saved['controller']['config']));controller.load_state_dict(saved['controller'])
            curriculum=CapabilityCurriculum(CurriculumConfig(**saved['curriculum']['config']));curriculum.load_state_dict(saved['curriculum'])
        observations=saved['calibration_observations']; metrics=saved['last_metrics']; step=saved['step'];version=saved['encoder_version']
        first_epoch=saved['epoch']+1;prior_elapsed=saved['elapsed_seconds']
        gen.set_state(saved['satellite_generator'].cpu());saved['rng'].restore()
    indexes = audit_indices(source,args.game_audit_samples_per_capture) if not args.game_no_audit else None
    json_write(output/'resolved_config.json',vars(args)); json_write(output/'source_contract.json',source.info)
    json_write(output/'parameter_roles.json',role_manifest(model))
    from .coverage_audit import coverage_audit
    from .data import SourceRoleDataset
    if isinstance(source.train,SourceRoleDataset):
        records=[source.train.base.index[i] for i in source.train.indices]
        coverage=coverage_audit([r.rx_i for r in records],tx=[r.tx_i for r in records],day=[r.day_i for r in records],grouping_kind='whole_tx_rx_day_container')
    else:
        records=[source.train[i] for i in range(len(source.train))]
        coverage=coverage_audit([r[3]['rx_i'] for r in records],tx=[r[1] for r in records],day=[r[3]['day_i'] for r in records],grouping_kind='synthetic_only')
    json_write(output/'coverage_audit.json',coverage)
    aug_cfg = build_aug_base_cfg(args) if args.use_aug else None
    augmentor = make_augmentor(aug_cfg) if aug_cfg else None
    if augmentor is not None and hasattr(augmentor,'to'): augmentor=augmentor.to(device)
    # This augmentor uses global RNG and deterministic class signatures, not a private generator.
    if saved is not None: saved['rng'].restore()
    started=time.perf_counter(); next_audit=saved['next_audit'] if saved else step; consecutive_empty=0
    replay = {}
    if args.game_actions_replay:
        schedule_path=Path(args.game_actions_replay)
        replay_meta=json.loads((schedule_path.parent/'metadata.json').read_text(encoding='utf-8'))
        if (not replay_meta.get('source_only') or replay_meta.get('target_evaluated') or
            replay_meta.get('recipient_seed')!=args.seed or replay_meta.get('donor_seed')==args.seed or
            replay_meta.get('epochs')!=args.epochs or replay_meta.get('game_split_seed')!=args.game_split_seed or
            replay_meta.get('steps_per_epoch')!=len(source.train)//args.batch_size):
            raise ValueError('Replay donor/recipient source, seed or horizon contract mismatch')
        if args.game_solver!='simultaneous' or args.game_fixed_head_steps or args.game_response_tracking:
            raise ValueError('Count-matched replay requires ordinary base updates without other extra actions')
        for line in schedule_path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);replay[int(row['step'])]=row
        if sorted(replay)!=list(range(replay_meta['horizon'])):
            raise ValueError('Replay steps are incomplete')
    print(f'[CORE90-GAME-INIT] scratch_only=true domains={len(source.domains)} params={sum(p.numel() for p in model.parameters())} solver={solver.algorithm}',flush=True)
    last_epoch=first_epoch-1
    previous_weights=_loss_weights(args,build_stage_state(first_epoch-1,args)) if first_epoch>1 else None
    for epoch in range(first_epoch,args.epochs+1):
        epoch_start=time.perf_counter();model.train()
        configure_mixstyle_for_epoch(model,args,epoch)
        if augmentor is not None: configure_augmentor_for_epoch(augmentor,aug_cfg,min(epoch,args.label_epochs),args)
        weights=_loss_weights(args,build_stage_state(epoch,args))
        if previous_weights is not None and weights!=previous_weights: solver.reset_history('objective_weights_change')
        previous_weights=dict(weights)
        if epoch in (41,91,args.label_epochs+1,args.sat_cons_start_epoch): solver.reset_history('scheduled_problem_change')
        loader=source.loader('train',args.batch_size,seed=args.seed+epoch,shuffle=True,workers=args.num_workers,drop_last=True)
        uloader=source.unlabeled_epoch_loader(args.batch_size,epoch_index=max(0,epoch-args.label_epochs-1),
                         steps=len(loader),seed=args.seed,workers=args.num_workers)
        uit=iter(uloader) if epoch>args.label_epochs and args.use_unlabeled else None
        rows=[]; accepted=0;u_seen=set()
        for batch_idx,batch in enumerate(loader,1):
            if args.game_max_steps_per_epoch and batch_idx>args.game_max_steps_per_epoch: break
            sets=None
            audit_due = indexes is not None and step>=next_audit
            forecast = min(budget.config.max_audit_seconds,max(1.,metrics.get('elapsed_seconds',1.) or 1.))
            audit_allowed = budget.can_afford(step,audit_seconds=forecast)[0] if audit_due else False
            if audit_due and not audit_allowed:
                metrics=dict(valid=False,step=step,encoder_version=version,reason='audit_budget_exhausted')
                append(output/'game_audit.jsonl',metrics)
                next_audit=step+args.game_audit_interval
            if audit_due and audit_allowed:
                metrics,sets=audit(model,source,indexes,auditor,args,step,version,epoch,proto)
                metrics['training_elapsed_seconds']=prior_elapsed+time.perf_counter()-started
                metrics['epoch']=epoch
                budget.commit(step,audit_seconds=metrics['elapsed_seconds'])
                append(output/'game_audit.jsonl',metrics)
                if controller is None:
                    observations.append(metrics)
                    if step>=args.game_source_calibration_steps:
                        controller,curriculum=calibrate(observations,args)
                        if controller: json_write(output/'source_calibration.json',dict(controller=asdict(controller.config),curriculum=asdict(curriculum.config),source_observation_steps=[m['step'] for m in observations]))
                next_audit=step+args.game_audit_interval
            decision=dict(action='NORMAL',catchup_steps=0,reason='fixed_or_source_calibration',hold_curriculum=True)
            if controller is not None and args.game_control in ('catchup','correction','both'):
                c=controller.config
                if audit_due or signal_fresh(metrics,step,version,c.max_age_steps,c.max_version_lag):
                    decision=controller.decide(metrics,step=step,encoder_version=version,budget=budget)
                else:
                    # Natural expiry between scheduled audits is not a new
                    # contradictory observation. Keep confirmation history,
                    # but never execute an action from stale evidence.
                    decision.update(action='HOLD_CURRICULUM',reason='waiting_for_fresh_source_audit')
                if (decision['action']=='CATCHUP' and args.game_control=='correction') or (decision['action']=='CORRECT' and args.game_control=='catchup'):
                    decision.update(action='NORMAL',catchup_steps=0,reason='ablation_disabled_action')
                if sets is not None:
                    next_audit=step+decision['next_audit_interval']
            if args.game_control in ('random','fixed') and step>=args.game_source_calibration_steps:
                enabled = random.Random(args.seed*1000003+step).random()<args.game_correction_fraction if args.game_control=='random' else step%max(1,round(1/max(args.game_correction_fraction,1e-6)))==0
                if enabled and budget.can_afford(step,field_evaluations=1,corrections=1,base_steps=1)[0]: decision.update(action='CORRECT',reason=args.game_control+'_budget_schedule')
            if args.game_control=='replay' and step in replay:
                old=replay[step]
                decision.update(action=old['action'],catchup_steps=old.get('requested_head_steps',0),reason='other_seed_replay')
                costs = dict(field_evaluations=1,corrections=1,base_steps=1) if old['action']=='CORRECT' else dict(head_steps=decision['catchup_steps'])
                if not budget.can_afford(step,**costs)[0]:
                    decision.update(action='NORMAL',catchup_steps=0,reason='replay_budget_exhausted')
            ubatch=None
            if uit is not None:
                try: ubatch=next(uit)
                except StopIteration: uit=iter(uloader);ubatch=next(uit)
                u_seen.update(ubatch[3]['sample_id'])
            level=curriculum.level if curriculum is not None and args.game_curriculum=='capability' else (0. if args.game_curriculum=='capability' else None)
            ctx=prepare_context(batch,ubatch,model,ema,args,epoch,batch_idx,weights,gen,augmentor,level)
            pseudo_problem_changed=update_pseudo_problem(solver,ctx,args.num_classes,args.game_optimistic_pseudo_change,args.game_pseudo_change_window)
            if sets is not None and args.game_jacobian_interval and step%args.game_jacobian_interval==0:
                from .field import audit_core90_field
                if budget.can_afford(step,audit_seconds=1.)[0]:
                    field_metrics=audit_core90_field(model,ctx,args,proto,seed=args.seed)
                    budget.commit(step,audit_seconds=field_metrics['elapsed_seconds'])
                else: field_metrics=dict(valid=False,reason='audit_budget_exhausted',elapsed_seconds=0.)
                append(output/'jacobian_audit.jsonl',dict(step=step,**field_metrics))
            count=max(args.game_fixed_head_steps,decision.get('catchup_steps',0))
            if args.game_solver=='alternating': count=max(1,count)
            # Fixed-k is a solver baseline; selective extra steps obey rolling budget.
            if not args.game_fixed_head_steps and args.game_solver!='alternating' and count and not budget.can_afford(step,head_steps=count)[0]: count=0
            head_steps,head_enc=head_catchup(model,optimizer,ctx,count,args)
            budget.commit(step,head_steps=head_steps)
            mode='extragradient' if decision['action']=='CORRECT' else args.game_solver
            solver.mode=mode
            reference=deepcopy(model).eval() if args.game_response_tracking and sets is not None and budget.can_afford(step,audit_seconds=1.)[0] else None
            with torch.autocast(device_type=device.type,enabled=args.amp):
                optimistic_history_used=mode=='optimistic' and solver.previous is not None
                result=solver.step(lambda:objective(ctx),exclude_head=(mode=='alternating'))
            if result.accepted:
                version+=1;accepted+=1
            budget.commit(step,field_evaluations=max(0,result.field_evaluations-1),base_steps=int(result.accepted),corrections=int(result.accepted and result.field_evaluations>1))
            response=None
            if reference is not None and result.accepted:
                from .field import apply_response_tracking
                fit,mon=sets['domain_fit'],sets['domain_monitor']
                response=apply_response_tracking(reference,model,fit['x'][:32],fit['d'][:32],monitor_x=mon['x'][:32],monitor_domain=mon['d'][:32])
                budget.commit(step,audit_seconds=response['elapsed_seconds'])
                if response.get('accepted'): solver.reset_history('implicit_head_compensation')
                append(output/'response_tracking.jsonl',dict(step=step,**response))
            if result.accepted:
                if ema is not None: _update_ema_model(ema,model,args.ema_decay)
                proto.update(ctx.origin_features,ctx.y,ctx.domain)
            if curriculum is not None and args.game_curriculum=='capability' and sets is not None:
                # Audit belongs to the pre-step state. Evaluate curriculum against that state,
                # apply it only to the next batch, never in a predictor/corrector pair.
                event=curriculum.update(metrics,step=metrics['step'],encoder_version=metrics['encoder_version'],hold=not metrics.get('valid',False))
                append(output/'curriculum_events.jsonl',dict(step=step,**event))
                if event['changed']: solver.reset_history('capability_curriculum_change');metrics={}
            action=dict(step=step,epoch=epoch,encoder_version=version,action=decision['action'],reason=decision['reason'],
                        elapsed_seconds=prior_elapsed+time.perf_counter()-started,
                        optimistic_history_used=optimistic_history_used,pseudo_problem_changed=pseudo_problem_changed,
                        unlabeled_samples=0 if ubatch is None else len(ubatch[0]),
                        unlabeled_first_id=None if ubatch is None else ubatch[3]['sample_id'][0],
                        unlabeled_last_id=None if ubatch is None else ubatch[3]['sample_id'][-1],
                        requested_head_steps=count,committed_head_steps=head_steps,main_solver=mode,
                        accepted=result.accepted,field_evaluations=result.field_evaluations,algorithm=result.algorithm,
                        forward_calls=ctx.forward_calls+head_enc,head_feature_cache_version=version-int(result.accepted),
                        main_backward_evaluations=result.field_evaluations,extra_head_forward_backward_evaluations=head_steps,
                        sample_count=len(ctx.y),satellite_count=ctx.satellite_mask_count,satellite_scenario=ctx.satellite_scenario,
                        pseudo_selected=ctx.pseudo_selected,loss=result.loss,loss_replay=result.loss_replay,
                        grad_norm=result.grad_norm,failure_stage=result.failure_stage)
            append(output/'game_actions.jsonl',action)
            rows.append(dict(action,terms=ctx.origin_terms));step+=1
        last_epoch=epoch
        consecutive_empty=consecutive_empty+1 if accepted==0 else 0
        record=dict(epoch=epoch,steps=len(rows),accepted=accepted,total_step=step,epoch_seconds=time.perf_counter()-epoch_start,
                    unlabeled_unique_samples=len(u_seen),unlabeled_sampling='permuted_contiguous_windows_cross_epoch_rotation',
                    mean_loss=float(np.mean([r['loss'] for r in rows])) if rows else None,
                    terms={k:float(np.mean([r['terms'].get(k,0.) for r in rows])) for k in (rows[0]['terms'] if rows else [])},
                    source_only=True,peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                    elapsed_seconds=prior_elapsed+time.perf_counter()-started)
        append(output/'logs.jsonl',record)
        solver.mode=args.game_solver
        checkpoint(output/'latest_ssdg.pth',model,ema,optimizer,scaler,proto,solver,controller,curriculum,budget,auditor,gen,args,source,epoch,step,version,observations,metrics,record['elapsed_seconds'],next_audit)
        print(f'[EPOCH-END] E{epoch:03d}/{args.epochs} accepted={accepted}/{len(rows)} loss={record["mean_loss"]} seconds={record["epoch_seconds"]:.1f}',flush=True)
        if consecutive_empty>=2: raise RuntimeError('SYSTEMIC_TECHNICAL_FAILURE: two epochs without a finite main update')
        if args.game_time_budget_s and record['elapsed_seconds']>=args.game_time_budget_s: break
    final=output/'final_ssdg.pth'
    checkpoint(final,model,ema,optimizer,scaler,proto,solver,controller,curriculum,budget,auditor,gen,args,source,last_epoch,step,version,observations,metrics,prior_elapsed+time.perf_counter()-started,next_audit)
    torch.save(dict(schema='core90_game_deployment_v1',model=model.state_dict(),args=vars(args),
                    source_info=source.info,prototype=deepcopy(vars(proto)),source_checkpoint='final_ssdg.pth',
                    training_only_modules_excluded=['optimizer','scaler','controller','auditor','curriculum'],
                    initialization='scratch_only',target_contact=False),output/'deployment.pth')
    json_write(output/'resource_summary.json',dict(total_seconds=prior_elapsed+time.perf_counter()-started,budget=budget.state_dict(),epochs=last_epoch,
                                                  full_budget_completed=last_epoch==args.epochs,checkpoint=str(final),source_only=True))
    if not args.game_skip_final_eval:
        from scripts.core90_game_evaluate import evaluate_source
        evaluate_source(model,source,args,output/'source_final_eval')
    json_write(output/'completion.json',dict(status='SOURCE_ARTIFACTS_COMPLETE' if not args.game_skip_final_eval else 'TRAINING_COMPLETE',
                                            epoch=last_epoch,step=step,target_evaluated=False,scientific_verdict='SOURCE_ONLY_NO_PROMOTION'))
    return 0
