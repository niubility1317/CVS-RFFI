"""All requested DAOT/RC4 extensions, with explicit source-only state lineage."""
from copy import deepcopy
from pathlib import Path
import torch
from SSDG import train_ssdg as native
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.game_tracking.source_audit import isolated_rng
from cvsrffi.game_tracking.step_context import slice_batch
from cvsrffi.game_tracking.runtime import json_write
from cvsrffi.orbit_teacher import adv3b02_daot_schedule
from .dr_objective import DROT
from .objective import labeled_forward_context


FULL_OPTIONS=dict(
    daot_ablation='',daot_loss_ablation='none',daot_teacher_mode='three_view',
    daot_teacher_view_count=3,daot_aggregation='robust_deployment',daot_skip_mean_metadata=False,
    daot_tangent_mode='branch_selective',daot_enable_relation=True,
    daot_lambda_orbit_z=.5,daot_lambda_orbit_logit=.2,daot_lambda_orbit_proto=.2,
    daot_lambda_orbit_relation=.05,daot_lambda_tangent=.05,daot_lambda_nuisance=.1,
    daot_lambda_fingerprint=.1,daot_tangent_sample_ratio=.25,
    rc4_use_anchor=True,rc4_enable_hard=True,rc4_enable_partial=True,
    rc4_enable_partial_set=True,rc4_enable_partial_conditional=True,rc4_enable_negative=True,
    # Native total-identity allocation only supports H/P. With N enabled use
    # its independent H/P/N quality budgets instead of an incompatible mode.
    rc4_total_identity_effective_budget=0.,rc4_hard_effective_budget=.1,
    rc4_partial_effective_budget=.1,rc4_negative_effective_budget=.1,
    rc4_satellite_hard_only=True,rc4_lambda_hard=.6,rc4_lambda_partial=.4,
    rc4_lambda_negative=.2,rc4_lambda_domain=.16,rc4_lambda_self=.1,
    rc4_lambda_satellite=.1,rc4_lambda_feature_anchor=.05,
    rc4_identity_tail_partial_conditional_final=.2,
)


class FullDROT(DROT):
    def __init__(self,model,ema,source,args,reference,proto):
        super().__init__(model,ema,source,args,reference)
        for key,value in FULL_OPTIONS.items():setattr(self.args,key,value)
        native._validate_daot_config(self.args)
        self.args.daot_diagnostic_epochs=''
        self.proto=proto;self.anchor=None;self.anchor_epoch=None
        self.anchor_record=None

    def ensure_anchor(self,epoch):
        if epoch<21 or self.anchor is not None:return
        # Frozen E20 EMA is derived solely from THIS scratch run's source data.
        # It is never loaded from any earlier experiment/checkpoint.
        self.anchor=deepcopy(self.ema).eval()
        for p in self.anchor.parameters():p.requires_grad_(False)
        self.anchor_epoch=20
        self.anchor_record=dict(origin='this_run_ema_after_E20',frozen_at_epoch=epoch,
            training_contract=self.source.info.get('counts'),external_checkpoint_sources=[],
            target_contact=False,source_V_used_for_training=False)
        if not self.source.info.get('synthetic'):
            out=Path(self.source_args.output_dir)
            path=out/'source_anchor_E20.pth'
            if path.exists():raise FileExistsError('anchor output already exists')
            torch.save(dict(model=self.anchor.state_dict(),provenance=self.anchor_record),path)
            json_write(out/'anchor_provenance.json',self.anchor_record)

    def calibrate(self,epoch):
        self.ensure_anchor(epoch)
        boundary=max(e for e in (1,21,41,91,161) if e<=epoch)
        if self.calibration_epoch==boundary:return None
        device=next(self.model.parameters()).device;cols=[[] for _ in range(7)]
        flags={m:m.training for m in self.ema.modules()};self.ema.eval()
        try:
            with isolated_rng(self.args.seed+boundary*7919),torch.no_grad():
                for x,y,d,meta in self.source.loader('val',256,workers=0):
                    x,y,d=x.to(device),y.to(device),d.to(device)
                    weak=native._strong_augment(x,max(1e-5,self.args.strong_noise_std*.25))
                    out=self.ema(torch.cat((x,weak)),return_aux=True,domain_labels=torch.cat((d,d)))
                    a,b=slice_batch(out,0,len(x),2*len(x)),slice_batch(out,len(x),2*len(x),2*len(x))
                    anchor=self.anchor(x,return_aux=True,domain_labels=d) if self.anchor else a
                    values=[anchor['tx_logits'],a['tx_logits'],b['tx_logits'],y,d,a['z_id'].norm(dim=-1),meta['rx_i'].to(device)]
                    for col,value in zip(cols,values):col.append(value.detach())
                v=[torch.cat(c) for c in cols]
                self.calibration=native.build_rc4_calibration(*v[:6],receivers=v[6],
                    num_classes=self.source_args.num_classes,num_domains=len(self.source.domains),folds=5,
                    hard_precision_target=.98,partial_coverage_target=.95,partial_precision_target=.98,
                    partial_min_coverage=.01,partial_candidate_max_classes=self.args.muse_candidate_max_classes,
                    negative_false_exclusion_target=.01,decouple_partial_negative_aps=self.args.rc4_decouple_partial_negative_aps,
                    partial_threshold_scope=self.args.rc4_partial_threshold_scope)
        finally:
            for m,flag in flags.items():m.training=flag
        self.calibration_epoch=boundary
        return dict(epoch=epoch,boundary=boundary,records=len(v[3]),source_V_read_only=True,
                    anchor_origin=self.anchor_record,anchor_active=self.anchor is not None)

    def prepare(self,ctx,ubatch):
        if ubatch is None:raise ValueError('full DR requires U from E1')
        x,hidden,d,meta=ubatch
        if not bool((hidden==-1).all()) or any('tx' in k.lower() for k in meta):raise ValueError('U labels exposed')
        device=ctx.x.device;ctx.dr_x=x.to(device);ctx.dr_d=d.to(device);ctx.dr_rx=meta['rx_i'].to(device)
        ctx.dr_strong=native._strong_augment(ctx.dr_x,self.args.strong_noise_std).detach()
        with torch.no_grad():
            self.ema.eval()
            weak=native._strong_augment(ctx.dr_x,max(1e-5,self.args.strong_noise_std*.25))
            both=self.ema(torch.cat((ctx.dr_x,weak)),return_aux=True,domain_labels=torch.cat((ctx.dr_d,ctx.dr_d)))
            n=len(x);a,b=slice_batch(both,0,n,2*n),slice_batch(both,n,2*n,2*n)
            ctx.dr_teacher=a
            ctx.dr_anchor=self.anchor(ctx.dr_x,return_aux=True,domain_labels=ctx.dr_d) if self.anchor else None
            cfg=dict(hard_max_fraction=self.args.sat_anchor_hard_max_fraction,candidate_max_classes=self.args.muse_candidate_max_classes)
            for name in ['hard_effective_budget','partial_effective_budget','negative_effective_budget','total_identity_effective_budget',
                         'use_calibrated_partial_threshold','enable_hard','enable_partial','enable_negative','class_receiver_cap','class_receiver_effective_budget']:
                cfg[name]=getattr(self.args,'rc4_'+name)
            ctx.dr_route=native.route_fasttrust_rc4((ctx.dr_anchor or a)['tx_logits'],a['tx_logits'],b['tx_logits'],
                domains=ctx.dr_d,receivers=ctx.dr_rx,z_norm=a['z_id'].norm(dim=-1),calibration=self.calibration,
                batched_readback=True,reliability_weight_mode=self.args.a1_rc4_reliability_weight,
                use_calibrated_risk=self.args.rc4_use_correctness_calibration,**cfg)
        ctx.dr_prototypes=None
        if ctx.epoch>=21:
            if self.anchor is None:raise RuntimeError('E21 full DR requires same-run frozen source anchor')
            active=self.proto.class_count>=self.proto.min_count
            if not bool(active.all()):raise RuntimeError('full DAOT prototype bank has missing source classes')
            ctx.dr_prototypes=self.proto.class_proto.detach().clone()
        self.prepare_satellite(ctx)
        ctx.dr_scale_origin=deepcopy(self.scale.state_dict());ctx.dr_pending_scale=None;ctx.dr_telemetry={}

    def prepare_satellite(self,ctx):
        """Freeze augmentation and routing outside both EG fields; testable with controlled routes."""
        hard=ctx.dr_route.hard if ctx.epoch>=21 else torch.zeros_like(ctx.dr_route.hard)
        ctx.dr_sat_indices=hard.nonzero(as_tuple=False).flatten()
        ctx.dr_sat=None
        if ctx.dr_sat_indices.numel():
            scene=native.select_adv3b02_u_satellite_scenario(ctx.epoch,ctx.batch_index,self.args.seed)
            gen=torch.Generator(device=ctx.x.device).manual_seed(self.args.seed+ctx.epoch*100003+ctx.batch_index*117)
            sat,_=apply_sat_channel_for_scenario(ctx.dr_x.index_select(0,ctx.dr_sat_indices),scene,self.args,gen=gen,return_meta=True)
            ctx.dr_sat=sat.detach()

    def objective(self,ctx,out):
        local=deepcopy(self.scale);local.load_state_dict(ctx.dr_scale_origin)
        with labeled_forward_context(self.model,ctx.grid_plan,ctx.x.device,views=1):
            labeled=native._compute_daot_labeled_step(model=self.model,ema_model=self.ema,student_clean=out,
                x_clean=ctx.x,y_clean=ctx.y,d_clean=ctx.domain,args=self.args,epoch=ctx.epoch,batch_idx=ctx.batch_index,
                apply_sat_fn=apply_sat_channel_for_scenario,prototype_matrix=ctx.dr_prototypes,loss_normalizer=local)
        combined=ctx.dr_strong if ctx.dr_sat is None else torch.cat((ctx.dr_strong,ctx.dr_sat))
        domains=ctx.dr_d if ctx.dr_sat is None else torch.cat((ctx.dr_d,ctx.dr_d[ctx.dr_sat_indices]))
        outputs=self.model(combined,return_aux=True,domain_labels=domains,grl_lambda=0.)
        n=len(ctx.dr_x);strong=slice_batch(outputs,0,n,len(combined))
        satellite=slice_batch(outputs,n,len(combined),len(combined)) if ctx.dr_sat is not None else None
        unlabeled=native._compute_daot_unlabeled_step(model=self.model,ema_model=self.ema,teacher_clean=ctx.dr_teacher,
            student_strong=strong,x_unlabeled=ctx.dr_x,d_unlabeled=ctx.dr_d,args=self.args,epoch=ctx.epoch,batch_idx=ctx.batch_index,
            apply_sat_fn=apply_sat_channel_for_scenario,prototype_matrix=ctx.dr_prototypes,loss_normalizer=local)
        rc4=native._compute_rc4_unlabeled_losses(route=ctx.dr_route,ema_outputs=ctx.dr_teacher,anchor_outputs=ctx.dr_anchor,
            student_views=dict(clean=strong,satellite=satellite,satellite_indices=ctx.dr_sat_indices),
            domains=ctx.dr_d,model=self.model,muse_state=dict(args=self.args,heads=self.model.xuc_rc4_heads),epoch=ctx.epoch)
        zero=strong['z_id'].sum()*0
        terms=dict(daot_labeled=labeled['loss'],daot_unlabeled=unlabeled['loss'],rc4_total=rc4['total'],
                   rc4_identity=rc4['identity'],rc4_domain=rc4['domain'],rc4_self=rc4['self'])
        schedule=adv3b02_daot_schedule(ctx.epoch)
        contributions={}
        for key in ('orbit_z','orbit_logit','orbit_proto','orbit_relation','tangent','nuisance','fingerprint'):
            for side,obj in [('L',labeled),('U',unlabeled)]:terms[f'daot_{side}_{key}']=obj.get('components',{}).get(key,zero)
            value=sum(obj.get('normalized_components',{}).get(key,zero) for obj in [labeled,unlabeled])
            scale=schedule.tangent_scale if key=='tangent' else schedule.orbit_scale
            contributions['daot_'+key]=scale*getattr(self.args,'daot_lambda_'+key)*value
        for key in ('hard','rc4_partial_set','rc4_partial_conditional','rc4_negative_set','adv','satellite','feature_anchor'):
            terms['rc4_'+key.removeprefix('rc4_')]=rc4[key]
        contributions.update({'rc4_'+k:v for k,v in rc4['rc4_gradient_losses'].items()})
        contributions.update(rc4_negative=self.args.rc4_lambda_negative*rc4['rc4_negative_set'],
            rc4_domain=rc4['rc4_tail_scale']*self.args.rc4_lambda_domain*rc4['domain'],
            rc4_self=self.args.rc4_lambda_self*rc4['self'],rc4_satellite=self.args.rc4_lambda_satellite*rc4['satellite'],
            rc4_anchor=self.args.rc4_lambda_feature_anchor*rc4['feature_anchor'])
        if ctx.dr_pending_scale is None:
            ctx.dr_pending_scale=deepcopy(local.state_dict())
            ctx.dr_telemetry=dict(configured=True,full_extensions=True,daot_executed=ctx.epoch>=21,
                rc4_executed=True,identity_active=ctx.epoch>=21,tangent_scheduled=ctx.epoch>=61,
                u_samples=n,hard_count=int(rc4['route'].high.sum()),
                partial_count=int(ctx.dr_route.partial.sum()) if ctx.epoch>=21 else 0,
                negative_count=int(ctx.dr_route.negative.sum()) if ctx.epoch>=21 else 0,
                satellite_samples=int(ctx.dr_sat_indices.numel()),prototype_rows=0 if ctx.dr_prototypes is None else len(ctx.dr_prototypes),
                anchor_active=ctx.dr_anchor is not None,anchor_origin=self.anchor_record,rc4_u_grl_lambda=0.,
                scale_commits_before=self.commits,
                weighted_components={k:float(v.detach()) for k,v in contributions.items()},
                **{k:float(terms[k].detach()) for k in ('daot_labeled','daot_unlabeled','rc4_identity','rc4_total')})
            if ctx.audit_gradients:
                named=[(k,p) for k,p in self.model.named_parameters() if p.requires_grad]
                for key,loss in [('daot',labeled['loss']+unlabeled['loss']),('rc4',rc4['total']),*contributions.items()]:
                    grads=torch.autograd.grad(loss,[p for _,p in named],retain_graph=True,allow_unused=True) if loss.requires_grad else []
                    ctx.dr_telemetry[key+'_grad_norm']=sum(float(g.detach().square().sum()) for g in grads if g is not None)**.5
                nuisance=contributions['daot_nuisance']
                gs=torch.autograd.grad(nuisance,list(self.model.daot_nuisance_head.parameters()),retain_graph=True,allow_unused=True) if nuisance.requires_grad else []
                ctx.dr_telemetry['nuisance_head_grad_norm']=sum(float(g.detach().square().sum()) for g in gs if g is not None)**.5
        return labeled['loss']+unlabeled['loss']+rc4['total'],terms

    def state_dict(self):
        result=super().state_dict()
        result.update(anchor=self.anchor.state_dict() if self.anchor else None,anchor_provenance=self.anchor_record,
                      prototype_origin='same_run_accepted_labeled_updates',full_extensions=True)
        return result
