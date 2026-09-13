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


class DROT:
    def __init__(self,model,ema,source,args,reference):
        self.model,self.ema,self.source=model,ema,source
        self.args=options(reference,args.output_dir)
        self.args.seed=args.seed
        self.args.daot_diagnostic_epochs='' # expensive diagnostics do not change the objective
        self.scale=EMALossScaleNormalizer(momentum=self.args.daot_scale_momentum,batched_readback=True)
        self.calibration=None;self.calibration_epoch=None;self.commits=0
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
            with isolated_rng(self.args.seed+boundary*7919),torch.no_grad():
                for x,y,d,meta in self.source.loader('val',256,workers=0):
                    x,y,d=x.to(device),y.to(device),d.to(device)
                    weak=native._strong_augment(x,max(1e-5,self.args.strong_noise_std*.25))
                    o=self.ema(torch.cat((x,weak)),return_aux=True,domain_labels=torch.cat((d,d)))
                    a,b=slice_batch(o,0,len(x),2*len(x)),slice_batch(o,len(x),2*len(x),2*len(x))
                    values=[a['tx_logits'],a['tx_logits'],b['tx_logits'],y,d,a['z_id'].norm(dim=-1),meta['rx_i'].to(device)]
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
        return dict(epoch=epoch,boundary=boundary,records=len(v[3]),source_V_read_only=True)

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
            for name in ['hard_effective_budget','partial_effective_budget','negative_effective_budget','total_identity_effective_budget',
                         'use_calibrated_partial_threshold','enable_hard','enable_partial','enable_negative','class_receiver_cap','class_receiver_effective_budget']:
                cfg[name]=getattr(self.args,'rc4_'+name)
            ctx.dr_route=native.route_fasttrust_rc4(a['tx_logits'],a['tx_logits'],b['tx_logits'],
                domains=ctx.dr_d,receivers=ctx.dr_rx,z_norm=a['z_id'].norm(dim=-1),calibration=self.calibration,
                batched_readback=True,reliability_weight_mode=self.args.a1_rc4_reliability_weight,
                use_calibrated_risk=self.args.rc4_use_correctness_calibration,**cfg)
        ctx.dr_scale_origin=deepcopy(self.scale.state_dict());ctx.dr_pending_scale=None
        ctx.dr_telemetry={}

    def objective(self,ctx,out):
        # Each EG field gets an isolated scale estimator initialized at the same origin.
        local=deepcopy(self.scale);local.load_state_dict(ctx.dr_scale_origin)
        with labeled_forward_context(self.model,ctx.grid_plan,ctx.x.device,views=1):
            labeled=native._compute_daot_labeled_step(model=self.model,ema_model=self.ema,student_clean=out,
                x_clean=ctx.x,y_clean=ctx.y,d_clean=ctx.domain,args=self.args,epoch=ctx.epoch,batch_idx=ctx.batch_index,
                apply_sat_fn=apply_sat_channel_for_scenario,prototype_matrix=None,loss_normalizer=local)
        strong=self.model(ctx.dr_strong,return_aux=True,domain_labels=ctx.dr_d,grl_lambda=1.)
        unlabeled=native._compute_daot_unlabeled_step(model=self.model,ema_model=self.ema,teacher_clean=ctx.dr_teacher,
            student_strong=strong,x_unlabeled=ctx.dr_x,d_unlabeled=ctx.dr_d,args=self.args,epoch=ctx.epoch,batch_idx=ctx.batch_index,
            apply_sat_fn=apply_sat_channel_for_scenario,prototype_matrix=None,loss_normalizer=local)
        rc4=native._compute_rc4_unlabeled_losses(route=ctx.dr_route,ema_outputs=ctx.dr_teacher,anchor_outputs=None,
            student_views=dict(clean=strong,satellite=None,satellite_indices=torch.empty(0,dtype=torch.long,device=ctx.x.device)),
            domains=ctx.dr_d,model=self.model,muse_state=dict(args=self.args,heads=self.model.xuc_rc4_heads),epoch=ctx.epoch)
        terms=dict(daot_labeled=labeled['loss'],daot_unlabeled=unlabeled['loss'],rc4_total=rc4['total'],
                   rc4_identity=rc4['identity'],rc4_domain=rc4['domain'],rc4_self=rc4['self'])
        if ctx.dr_pending_scale is None:
            ctx.dr_pending_scale=deepcopy(local.state_dict())
            ctx.dr_telemetry=dict(configured=True,daot_executed=ctx.epoch>=21,rc4_executed=True,
                identity_active=ctx.epoch>=self.args.rc4_identity_start_epoch,u_samples=len(ctx.dr_x),
                hard_count=int(rc4['route'].high.sum()),partial_count=int(rc4['route'].mid.sum()),
                daot_labeled=float(labeled['loss'].detach()),daot_unlabeled=float(unlabeled['loss'].detach()),
                rc4_identity=float(rc4['identity'].detach()),rc4_total=float(rc4['total'].detach()),scale_commits_before=self.commits)
            if ctx.audit_gradients:
                for key,loss in [('daot',labeled['loss']+unlabeled['loss']),('rc4',rc4['total'])]:
                    gs=torch.autograd.grad(loss,[p for p in self.model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
                    ctx.dr_telemetry[key+'_grad_norm']=sum(float(g.detach().square().sum()) for g in gs if g is not None)**.5
        return labeled['loss']+unlabeled['loss']+rc4['total'],terms

    def commit(self,ctx):
        self.scale.load_state_dict(ctx.dr_pending_scale);self.commits+=1

    def state_dict(self):
        return dict(scale=self.scale.state_dict(),calibration=self.calibration,calibration_epoch=self.calibration_epoch,
                    commits=self.commits,args=vars(self.args))
