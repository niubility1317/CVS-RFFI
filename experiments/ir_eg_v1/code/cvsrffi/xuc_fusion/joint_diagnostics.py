"""Read-only source diagnostics; no target labels, no model-state commits."""
from copy import deepcopy
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from cvsrffi.game_tracking.source_audit import isolated_rng

def mass(weights):
    w=weights.detach().float()
    total=float(w.sum());sq=float(w.square().sum())
    return dict(count=len(w),weight_sum=total,ess=total*total/sq if sq else 0.)

def scoped_gradients(model,components):
    # Explicit identity backbone excludes domain backbone, classification and auxiliary heads.
    named=[(n,p) for n,p in model.named_parameters() if n.startswith('id_backbone.') and p.requires_grad]
    if not named:raise ValueError('identity backbone parameter scope absent')
    vectors={}
    for key,loss in components.items():
        grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True,retain_graph=True) if loss.requires_grad else [None]*len(named)
        vectors[key]=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for (_,p),g in zip(named,grads)])
    norms={k:float(v.double().norm()) for k,v in vectors.items()}
    cosine={}
    keys=list(vectors)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            denominator=norms[a]*norms[b]
            cosine[a+'__'+b]=float(torch.dot(vectors[a].double(),vectors[b].double()))/denominator if denominator else None
    return dict(scope=[n for n,_ in named],norms=norms,cosines=cosine,
        backward_calls=len(components),nonzero={k:v>0 for k,v in norms.items()})

def representation(z,other,y):
    z=F.normalize(z.detach().float(),dim=-1);other=F.normalize(other.detach().float(),dim=-1)
    distance=1-z@z.T;different=y[:,None]!=y[None,:]
    numerator=float((1-(z*other).sum(-1)).mean())
    denominator=float(distance[different].mean()) if different.any() else None
    singular=torch.linalg.svdvals(z-z.mean(0));p=singular/singular.sum().clamp_min(1e-12)
    rank=float(torch.exp(-(p*p.clamp_min(1e-12).log()).sum()))
    margins={}
    for cls in y.unique():
        mask=y==cls
        inside=distance[mask][:,mask];outside=distance[mask][:,~mask]
        margins[str(int(cls))]=float(outside.min()-inside.mean()) if outside.numel() else None
    return dict(orbit_distance=numerator,inter_tx_distance=denominator,
        invariance_ratio=numerator/(denominator+1e-8) if denominator is not None else None,
        effective_rank=rank,tx_margins=margins)

def paired_teacher(student,teacher):
    p=teacher['tx_logits'].detach().float().softmax(-1)
    kl=(p*(p.clamp_min(1e-8).log()-student['tx_logits'].detach().float().log_softmax(-1))).sum(-1)
    angle=(F.normalize(student['z_id'].detach().float(),dim=-1)*F.normalize(teacher['z_id'].detach().float(),dim=-1)).sum(-1).clamp(-1,1).acos()
    return dict(mean_kl=float(kl.mean()),mean_angle_rad=float(angle.mean()))

def empirical_gap(head,z_l,d_l,z_u,d_u,l_weight,u_weight,*,lr,weight_decay,steps=40):
    if l_weight+u_weight<=0:return dict(valid=False,gap=None,reason='adversarial_objective_disabled')
    copy=deepcopy(head).eval()
    def risk():return l_weight*F.cross_entropy(copy(z_l.detach()),d_l)+u_weight*F.cross_entropy(copy(z_u.detach()),d_u)
    with torch.no_grad():initial=float(risk())
    opt=torch.optim.AdamW(copy.parameters(),lr=lr,weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True);loss=risk()
        if not torch.isfinite(loss):return dict(valid=False,gap=None,reason='nonfinite_recovery')
        loss.backward();opt.step()
    with torch.no_grad():final=float(risk())
    valid=math.isfinite(final) and final<=initial
    return dict(valid=valid,gap=initial-final if valid else None,initial_risk=initial,final_risk=final,
        reason='ok' if valid else 'recovery_did_not_improve',estimand='empirical_optimization_gap',
        steps=steps,l_weight=l_weight,u_weight=u_weight,independent_generalization=False)

def cross_tx_readout(z,y,d,groups,*,fold,seed,steps=40):
    classes=sorted(int(v) for v in y.unique())
    if len(classes)<2:return dict(valid=False,reason='insufficient_TX')
    held=classes[fold%len(classes)];monitor=y==held;fit=~monitor
    fit_groups={groups[i] for i in range(len(groups)) if fit[i]}
    monitor_groups={groups[i] for i in range(len(groups)) if monitor[i]}
    if fit_groups&monitor_groups:raise ValueError('probe B acquisition groups overlap')
    if not set(d[monitor].tolist())<=set(d[fit].tolist()):return dict(valid=False,reason='domain_coverage',held_out_tx=held)
    result={}
    with isolated_rng(seed):
        for name in ('linear','mlp'):
            head=(nn.Linear(z.shape[1],int(d.max())+1) if name=='linear' else
                nn.Sequential(nn.Linear(z.shape[1],32),nn.ReLU(),nn.Linear(32,int(d.max())+1))).to(z.device)
            opt=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=0.)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True);F.cross_entropy(head(z[fit].detach()),d[fit]).backward();opt.step()
            with torch.no_grad():
                result[name]=dict(fit_ce=float(F.cross_entropy(head(z[fit]),d[fit])),
                    monitor_ce=float(F.cross_entropy(head(z[monitor]),d[monitor])),
                    monitor_accuracy=float((head(z[monitor]).argmax(1)==d[monitor]).float().mean()))
    return dict(valid=True,held_out_tx=held,fit_groups=sorted(fit_groups),monitor_groups=sorted(monitor_groups),
        readouts=result,scope='source_L_batch_cross_TX_capture_disjoint',independent_acquisition_claim=False)

def run_probes(model,ctx,args,dr,optimizer,step):
    start=time.perf_counter();flags={m:m.training for m in model.modules()}
    try:
        with isolated_rng(args.joint['data_seed']+step):
            model.eval()
            with torch.no_grad():
                left=model(ctx.x,return_aux=True,domain_labels=ctx.domain)
                right=model(ctx.dr_strong,return_aux=True,domain_labels=ctx.dr_d,grl_lambda=0.)
                satellite=model(ctx.satellite,return_aux=True,domain_labels=ctx.domain)
            from cvsrffi.muse_ssdg import rc4_tail_transition_scale
            u_weight=dr.args.rc4_lambda_domain*rc4_tail_transition_scale(ctx.epoch,
                start_epoch=dr.args.rc4_tail_transition_start_epoch,ramp_epochs=dr.args.rc4_tail_transition_epochs,
                floor=dr.args.rc4_tail_transition_floor)
            if args.joint['disable_adversarial_head_loss']:u_weight=0.
            a=empirical_gap(model.adv_head,left['z_id'],ctx.domain,right['z_id'],ctx.dr_d,
                ctx.weights['adv'],u_weight,lr=optimizer.param_groups[-1]['lr'],weight_decay=args.weight_decay)
            groups=[f'{int(y)}:{int(r)}:{int(d)}' for y,r,d in zip(ctx.y,ctx.rx,ctx.day)]
            b=cross_tx_readout(left['z_id'],ctx.y,ctx.domain,groups,fold=step//max(1,args.joint['probe_interval']),
                seed=args.joint['data_seed']+step)
        return dict(step=step,probe_A=a,probe_B=b,representation=representation(left['z_id'],satellite['z_id'],ctx.y),
            elapsed_seconds=time.perf_counter()-start,
            controls_training=False,gate_states=dict(recovery_valid=a['valid'],geometry_ready=False,
                observation_fresh=True,action_confirmed=False,cooldown_passed=None,budget_available=None),
            action='OFF',reason='fixed_solver_controller_disabled')
    finally:
        for m,flag in flags.items():m.training=flag

class ForwardCounter:
    def __init__(self,model,teacher):
        self.counts={};self.handles=[]
        for prefix,parent in [('student',model),('teacher',teacher)]:
            if parent is None:continue
            for name in ('id_backbone','dom_backbone'):
                key=prefix+'_'+name;self.counts[key]=0
                def hook(module,inputs,output,k=key):self.counts[k]+=1
                self.handles.append(getattr(parent,name).register_forward_hook(hook))
    def snapshot(self):return dict(self.counts)
    def close(self):
        for h in self.handles:h.remove()
