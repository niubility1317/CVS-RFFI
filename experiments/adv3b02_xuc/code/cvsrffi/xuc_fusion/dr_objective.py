"""Native A1 DAOT and RC4 objectives inside the XUC atomic field transaction."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from SSDG import train_ssdg as native
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.orbit_teacher import EMALossScaleNormalizer
from cvsrffi.game_tracking.source_audit import isolated_rng
from cvsrffi.game_tracking.step_context import slice_batch
from .native import native_argv
from .objective import labeled_forward_context


def options(reference, output):
    args=native.build_arg_parser().parse_args(native_argv(reference,'M09','unused',output))
    native._validate_daot_config(args)
    # Keep all A1 mechanism coefficients; this carrier has its own 49-step epoch.
    args.epochs=200
    if not args.fasttrust_rc4 or not args.use_adv3b02_daot_stn:
        raise ValueError('both DAOT and FastTrust-RC4 must be enabled')
    if args.rc4_use_anchor or args.daot_teacher_mode=='temporal_memory':
        raise ValueError('this scratch A1 contract has neither inherited anchor nor temporal memory')
    return args


def resolved_options(reference,args):
    result=options(reference,args.output_dir);result.seed=args.seed
    joint=getattr(args,'joint',None)
    if joint:
        from .joint_config import dr_overrides
        for key,value in dr_overrides(joint).items():setattr(result,key,value)
        result.sat_view_seed=joint['augmentation_seed']
        native._validate_daot_config(result)
    return result

class DROT:
    def __init__(self,model,ema,source,args,reference):
        self.model,self.ema,self.source=model,ema,source
        self.args=resolved_options(reference,args)
        self.joint=getattr(args,'joint',None)
        self.args.daot_diagnostic_epochs='' # expensive diagnostics do not change the objective
        self.scale=EMALossScaleNormalizer(momentum=self.args.daot_scale_momentum,batched_readback=True)
        self.calibration=None;self.calibration_epoch=None;self.commits=0
        self.route_history={}
        self.source_args=args

    def calibrate(self,epoch):
        boundary=max(e for e in (1,21,41,91,161) if e<=epoch)
        if self.calibration_epoch==boundary:return None
        device=next(self.model.parameters()).device
        cols=[[] for _ in range(7)]
        # Full single V, no fit/selector split, no backward or persistent model updates.
        flags={m:m.training for m in self.ema.modules()}
        self.ema.eval()
        try:
            with isolated_rng((self.joint['data_seed'] if self.joint else self.args.seed)+boundary*7919),torch.no_grad():
                for x,y,d,meta in self.source.loader('val',256,workers=0):
                    x,y,d=x.to(device),y.to(device),d.to(device)
                    weak=native._strong_augment(x,max(1e-5,self.args.strong_noise_std*.25))
                    o=self.ema(torch.cat((x,weak)),return_aux=True,domain_labels=torch.cat((d,d)))
                    a,b=slice_batch(o,0,len(x),2*len(x)),slice_batch(o,len(x),2*len(x),2*len(x))
                    values=[a['tx_logits'],a['tx_logits'],b['tx_logits'],y,d,a['z_id'].norm(dim=-1),meta['rx_i'].to(device)]
                    for col,value in zip(cols,values):col.append(value.detach())
                v=[torch.cat(c) for c in cols]
                self.calibration_audit={}
                self.calibration=native.build_rc4_calibration(*v[:6],receivers=v[6],
                    num_classes=self.source_args.num_classes,num_domains=len(self.source.domains),folds=5,
                    hard_precision_target=.98,partial_coverage_target=.95,partial_precision_target=.98,
                    partial_min_coverage=.01,partial_candidate_max_classes=self.args.muse_candidate_max_classes,
                    negative_false_exclusion_target=.01,decouple_partial_negative_aps=self.args.rc4_decouple_partial_negative_aps,
                    partial_threshold_scope=self.args.rc4_partial_threshold_scope,
                    audit_trace=self.calibration_audit if self.joint else None)
        finally:
            for m,flag in flags.items():m.training=flag
        self.calibration_epoch=boundary
        return dict(epoch=epoch,boundary=boundary,records=len(v[3]),source_V_read_only=True,
            calibration_metrics={k:value for k,value in vars(self.calibration).items() if isinstance(value,(int,float,bool,str))},
            risk_crossfit_audit=self.calibration_audit,
            metric_scope='source_V_calibration_crossfit_estimators_not_independent_target_guarantees')

    def prepare(self,ctx,ubatch):
        if ubatch is None:raise ValueError('DAOT/RC4 requires U from E1')
        x,hidden,d,meta=ubatch
        if not bool((hidden==-1).all()) or any('tx' in k.lower() for k in meta):raise ValueError('U labels exposed')
        device=ctx.x.device
        ctx.dr_x=x.to(device);ctx.dr_d=d.to(device);ctx.dr_rx=meta['rx_i'].to(device)
        ctx.dr_strong=native._strong_augment(ctx.dr_x,self.args.strong_noise_std).detach()
        with torch.no_grad():
            self.ema.eval()
            weak=native._strong_augment(ctx.dr_x,max(1e-5,self.args.strong_noise_std*.25))
            both=self.ema(torch.cat((ctx.dr_x,weak)),return_aux=True,domain_labels=torch.cat((ctx.dr_d,ctx.dr_d)))
            n=len(x)
            a,b=slice_batch(both,0,n,2*n),slice_batch(both,n,2*n,2*n)
            ctx.dr_teacher=a
            cfg=dict(hard_max_fraction=self.args.sat_anchor_hard_max_fraction,candidate_max_classes=self.args.muse_candidate_max_classes)
            ctx.dr_route_trace={}
            if self.joint:cfg['trace']=ctx.dr_route_trace
            for name in ['hard_effective_budget','partial_effective_budget','negative_effective_budget','total_identity_effective_budget',
                         'use_calibrated_partial_threshold','enable_hard','enable_partial','enable_negative','class_receiver_cap','class_receiver_effective_budget']:
                cfg[name]=getattr(self.args,'rc4_'+name)
            ctx.dr_route=native.route_fasttrust_rc4(a['tx_logits'],a['tx_logits'],b['tx_logits'],
                domains=ctx.dr_d,receivers=ctx.dr_rx,z_norm=a['z_id'].norm(dim=-1),calibration=self.calibration,
                batched_readback=True,reliability_weight_mode=self.args.a1_rc4_reliability_weight,
                use_calibrated_risk=self.args.rc4_use_correctness_calibration,**cfg)
        ctx.dr_scale_origin=deepcopy(self.scale.state_dict());ctx.dr_pending_scale=None
        ctx.dr_used_divisors=None;ctx.dr_field_divisors=[]
        ctx.dr_l_cache={};ctx.dr_u_cache={}
        ctx.dr_telemetry={}
        ctx.dr_ids=list(meta['sample_id'])

    def objective(self,ctx,out):
        # Each EG field gets an isolated scale estimator initialized at the same origin.
        local=deepcopy(self.scale);local.load_state_dict(ctx.dr_scale_origin)
        if self.joint:
            from .joint_normalization import FieldNormalizer
            replay=ctx.dr_used_divisors if self.joint['normalization_scales']=='reuse_origin_used_scales' else None
            local=FieldNormalizer(local,replay=replay)
        if self.joint and not self.joint['native_dr']:
            labeled={'loss':out['z_id'].sum()*0.,'components':{}}
        else:
            with labeled_forward_context(self.model,ctx.grid_plan,ctx.x.device,views=1):
                labeled=native._compute_daot_labeled_step(model=self.model,ema_model=self.ema,student_clean=out,
                    x_clean=ctx.x,y_clean=ctx.y,d_clean=ctx.domain,args=self.args,epoch=ctx.epoch,batch_idx=ctx.batch_index,
                    apply_sat_fn=apply_sat_channel_for_scenario,prototype_matrix=None,loss_normalizer=local,prepared_cache=ctx.dr_l_cache if self.joint else None)
        # Match the native RC4 U student: its domain head trains, but this CE
        # must not introduce a new adversarial gradient into the identity trunk.
        strong=self.model(ctx.dr_strong,return_aux=True,domain_labels=ctx.dr_d,grl_lambda=0.)
        if self.joint and not self.joint['native_dr']:
            unlabeled={'loss':strong['z_id'].sum()*0.,'components':{}}
        else:
            unlabeled=native._compute_daot_unlabeled_step(model=self.model,ema_model=self.ema,teacher_clean=ctx.dr_teacher,
                student_strong=strong,x_unlabeled=ctx.dr_x,d_unlabeled=ctx.dr_d,args=self.args,epoch=ctx.epoch,batch_idx=ctx.batch_index,
                apply_sat_fn=apply_sat_channel_for_scenario,prototype_matrix=None,loss_normalizer=local,prepared_cache=ctx.dr_u_cache if self.joint else None)
        rc4=native._compute_rc4_unlabeled_losses(route=ctx.dr_route,ema_outputs=ctx.dr_teacher,anchor_outputs=None,
            student_views=dict(clean=strong,satellite=None,satellite_indices=torch.empty(0,dtype=torch.long,device=ctx.x.device)),
            domains=ctx.dr_d,model=self.model,muse_state=dict(args=self.args,heads=self.model.xuc_rc4_heads),epoch=ctx.epoch)
        if self.joint:
            from .joint_config import hp_multiplier
            identity=sum(rc4['rc4_gradient_losses'][k] for k in ('hard','partial_set','partial_conditional'))
            multiplier=hp_multiplier(ctx.epoch,self.joint['hp_ramp']) if self.joint['native_dr'] else 0.
            rc4['total']=rc4['total']+(multiplier-1.)*identity
            rc4['weighted_identity']=multiplier*identity
            ctx.dr_weighted_identity=rc4['weighted_identity']
            if self.joint['disable_adversarial_head_loss']:
                rc4['total']=rc4['total']-rc4['rc4_gradient_losses']['adv']
            if not self.joint['native_dr']:
                labeled['loss']=labeled['loss']*0.;unlabeled['loss']=unlabeled['loss']*0.
            local.finish()
            ctx.dr_field_divisors.append(deepcopy(local.used))
            if ctx.dr_used_divisors is None:ctx.dr_used_divisors=deepcopy(local.used)
        terms=dict(daot_labeled=labeled['loss'],daot_unlabeled=unlabeled['loss'],rc4_total=rc4['total'],
                   rc4_identity=rc4['identity'],rc4_domain=rc4['domain'],rc4_self=rc4['self'])
        if ctx.dr_pending_scale is None:
            ctx.dr_pending_scale=deepcopy(local.state_dict())
            ctx.dr_telemetry=dict(configured=True,daot_executed=ctx.epoch>=21,rc4_executed=True,
                identity_active=ctx.epoch>=self.args.rc4_identity_start_epoch,u_samples=len(ctx.dr_x),
                hard_count=int(rc4['route'].high.sum()),partial_count=int(rc4['route'].mid.sum()),
                daot_labeled=float(labeled['loss'].detach()),daot_unlabeled=float(unlabeled['loss'].detach()),
                rc4_identity=float(rc4['identity'].detach()),rc4_total=float(rc4['total'].detach()),scale_commits_before=self.commits)
            if self.joint:
                from .joint_diagnostics import paired_teacher
                ctx.dr_telemetry.update(native_dr_enabled=self.joint['native_dr'],
                    daot_executed=self.joint['native_dr'] and ctx.epoch>=21,
                    identity_active=self.joint['native_dr'] and multiplier>0 and ctx.epoch>=self.args.rc4_identity_start_epoch,
                    hp_multiplier=multiplier,weighted_identity=float(rc4['weighted_identity'].detach()),
                    used_divisors=deepcopy(local.used),normalization_mode=self.joint['normalization_scales'],
                    negative_enabled=False,anchor_enabled=False,extra_daot_enabled=False,
                    teacher_student=paired_teacher(strong,ctx.dr_teacher),
                    route_funnel=ctx.dr_route_trace,
                    calibration_age_epochs=ctx.epoch-self.calibration_epoch,
                    components={side:{k:float(v.detach()) for k,v in obj.get('components',{}).items() if torch.is_tensor(v) and v.numel()==1}
                        for side,obj in [('L',labeled),('U',unlabeled)]})
            if ctx.audit_gradients and not self.joint:
                for key,loss in [('daot',labeled['loss']+unlabeled['loss']),('rc4',rc4['total'])]:
                    gs=torch.autograd.grad(loss,[p for p in self.model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
                    ctx.dr_telemetry[key+'_grad_norm']=sum(float(g.detach().square().sum()) for g in gs if g is not None)**.5
        return labeled['loss']+unlabeled['loss']+rc4['total'],terms

    def commit(self,ctx):
        self.scale.load_state_dict(ctx.dr_pending_scale);self.commits+=1
        if self.joint:
            states=torch.where(ctx.dr_route.hard,1,torch.where(ctx.dr_route.partial,2,0)).cpu().tolist()
            seen=changed=0;lengths=[]
            for identity,state in zip(ctx.dr_ids,states):
                old=self.route_history.get(identity)
                seen+=int(old is not None);changed+=int(old is not None and old[0]!=state)
                length=old[1]+1 if old is not None and old[0]==state else 1
                self.route_history[identity]=(state,length);lengths.append(length)
            ctx.dr_telemetry.update(route_reobserved=seen,route_changed=changed,
                route_switch_rate=changed/seen if seen else None,stable_observation_lengths=lengths,
                stability_unit='repeat_observations_of_same_U_ID_not_consecutive_global_steps',
                field_divisors=ctx.dr_field_divisors)

    def state_dict(self):
        return dict(scale=self.scale.state_dict(),calibration=self.calibration,calibration_epoch=self.calibration_epoch,
                    commits=self.commits,args=vars(self.args),route_history=self.route_history)
