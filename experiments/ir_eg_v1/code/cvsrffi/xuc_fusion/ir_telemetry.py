"""Passive source-only IR diagnostics. No controller consumes these measurements."""
from contextlib import contextmanager
from copy import deepcopy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from cvsrffi.game_tracking.state import RNGState


@contextmanager
def isolated_diagnostics(model, optimizer=None, stateful=()):
    rng = RNGState.capture()
    values = deepcopy(model.state_dict())
    parameters = [(p, p.requires_grad, None if p.grad is None else p.grad.clone()) for p in model.parameters()]
    flags = [(m, m.training) for m in model.modules()]
    optim = deepcopy(optimizer.state_dict()) if optimizer is not None else None
    states = [(obj, deepcopy(obj.state_dict())) for obj in stateful]
    try:
        yield
    finally:
        model.load_state_dict(values)
        if optimizer is not None:
            optimizer.load_state_dict(optim)
        for obj, state in states:
            obj.load_state_dict(state)
        for p, flag, grad in parameters:
            p.requires_grad_(flag); p.grad = grad
        for m, flag in flags:
            m.training = flag
        rng.restore()


def diagnostics_due(accepted_index, stage_boundary, interval=1000):
    return bool(stage_boundary or (accepted_index > 0 and accepted_index % interval == 0))


def recovery_summary(ce_before, ce_after, accuracy_before, accuracy_after):
    return dict(ce_before=ce_before, ce_after=ce_after,
                ce_reduction=ce_before-ce_after, accuracy_before=accuracy_before,
                accuracy_after=accuracy_after, accuracy_gain=accuracy_after-accuracy_before,
                controls_training=False)


def _unflatten(vector, parameters):
    result=[]; offset=0
    for p in parameters:
        result.append(vector[offset:offset+p.numel()].reshape_as(p)); offset += p.numel()
    if offset != vector.numel():
        raise ValueError('parameter vector length mismatch')
    return tuple(result)


def nonlinear_residual(phi, batch, h0, delta, metric):
    from .ir_head import head_gradient
    flat=torch.cat([p.detach().reshape(-1) for p in phi])
    gradient=head_gradient(_unflatten(flat+delta,phi),batch)
    if isinstance(gradient,(tuple,list)):
        gradient=torch.cat([g.reshape(-1) for g in gradient])
    if isinstance(h0,(tuple,list)):
        h0=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(phi,h0)])
    ids=metric.active_indices
    residual=delta[ids]/metric.sqrt_diagonal+metric.sqrt_diagonal*(gradient[ids]-h0[ids])
    return float(residual.double().norm())


def head_recovery_probe(head, train_batch, monitor_batch, *, steps=10, lr=2e-4):
    """Fit a disposable head on detached source training records; monitor disjoint V.

    The registered fixed dropout masks are replayed throughout this probe, both
    for training and monitoring. This convention is recorded in the result.
    """
    from .ir_head import head_logits
    if steps != 10:
        raise ValueError('the registered recovery probe uses exactly ten updates')
    train_ids={i for c in train_batch.calls for i in c.physical_ids}
    monitor_ids={i for c in monitor_batch.calls for i in c.physical_ids}
    if train_ids & monitor_ids or not train_ids or not monitor_ids:
        raise ValueError('training and V monitor physical records must be disjoint and nonempty')
    rng=RNGState.capture(); started=time.perf_counter()
    try:
        phi=tuple(p.detach().clone().requires_grad_(True) for p in head.parameters())
        optimizer=torch.optim.AdamW(phi, lr=lr)
        def loss(batch):
            return sum((F.cross_entropy(head_logits(phi,c),c.domain,reduction='none')*c.sample_weight).sum() for c in batch.calls)
        def metrics():
            with torch.no_grad():
                count=sum(len(c.domain) for c in monitor_batch.calls)
                ce=sum(float(F.cross_entropy(head_logits(phi,c),c.domain,reduction='sum')) for c in monitor_batch.calls)/count
                correct=sum(int((head_logits(phi,c).argmax(-1)==c.domain).sum()) for c in monitor_batch.calls)
                return ce, correct/count
        before=metrics()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True); loss(train_batch).backward(); optimizer.step()
        after=metrics()
        return dict(recovery_summary(before[0],after[0],before[1],after[1]),
                    steps=steps, dropout_policy='replay_registered_fixed_masks',
                    fit_records=len(train_ids), monitor_records=len(monitor_ids),
                    elapsed_seconds=time.perf_counter()-started)
    finally:
        rng.restore()


def source_identity_summary(logits, features, tx, rx):
    """Descriptive source slices and two-way statistical interaction, never physics."""
    if not len(tx) or logits.shape[0]!=len(tx) or features.shape[0]!=len(tx):
        raise ValueError('nonempty aligned source panel required')
    with torch.no_grad():
        pred=logits.argmax(-1); correct=pred==tx
        masked=logits.clone(); masked.scatter_(1,tx[:,None],-torch.inf)
        margin=logits.gather(1,tx[:,None]).squeeze(1)-masked.max(1).values
        rows=[]; tx_scores=[]; rx_scores=[]; interactions=[]; deltas=[]
        grand=features.mean(0)
        for y in tx.unique():
            ym=tx==y; tx_scores.append(float(correct[ym].float().mean()))
            for r in rx.unique():
                mask=ym & (rx==r)
                if not mask.any():continue
                centroid=features[mask].mean(0)
                interaction=centroid-features[ym].mean(0)-features[rx==r].mean(0)+grand
                interactions.append(float(interaction.norm()))
                rows.append(dict(tx=int(y),rx=int(r),n=int(mask.sum()),accuracy=float(correct[mask].float().mean()),margin=float(margin[mask].mean()),interaction_norm=float(interaction.norm())))
        for r in rx.unique():
            rx_scores.append(float(correct[rx==r].float().mean()))
            labels=tx[rx==r].unique().tolist()
            for i,a in enumerate(labels):
                for b in labels[i+1:]:
                    global_difference=features[tx==a].mean(0)-features[tx==b].mean(0)
                    local_difference=features[(tx==a)&(rx==r)].mean(0)-features[(tx==b)&(rx==r)].mean(0)
                    deltas.append(dict(a=a,b=b,rx=int(r),delta_ab_r=(local_difference-global_difference).cpu().tolist()))
        return dict(tx_margin=float(margin.mean()),worst_tx_accuracy=min(tx_scores),worst_rx_accuracy=min(rx_scores),
                    rx_tx=rows,delta_ab_r=deltas,statistical_interaction_norm=sum(interactions)/len(interactions),
                    physical_parameter_recovery_claim=False,controls_training=False)


def clip_attribution(model, optimizer, parameters, eg_raw, ir_raw, *, max_grad_norm, common_coefficient):
    """Measure disposable AdamW steps from one origin using native and common clip.

    Both gradient fields must come from the same materialized step/context. The
    common coefficient is provided by the native EG clipping calculation.
    """
    if not 0 <= common_coefficient <= 1:
        raise ValueError('clip coefficient must be in [0,1]')
    if len(parameters)!=len(eg_raw) or len(parameters)!=len(ir_raw):
        raise ValueError('gradient layout mismatch')
    result={};vectors={}
    for label, raw in [('EG',eg_raw),('IR',ir_raw)]:
        for common in (False,True):
            with isolated_diagnostics(model,optimizer):
                origin=[p.detach().clone() for p in parameters]
                optimizer.zero_grad(set_to_none=True)
                for p,g in zip(parameters,raw):
                    p.grad=None if g is None else g.detach().clone()
                if common:
                    for p in parameters:
                        if p.grad is not None:p.grad.mul_(common_coefficient)
                else:
                    torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
                optimizer.step()
                delta=torch.cat([(p.detach()-v).reshape(-1) for p,v in zip(parameters,origin)])
                key=label+('_common_clip' if common else '_native_clip')
                vectors[key]=delta
                result[key]=dict(displacement_norm=float(delta.double().norm()))
    for suffix in ('native_clip','common_clip'):
        a,b=vectors['EG_'+suffix].double(),vectors['IR_'+suffix].double()
        denom=float(a.norm()*b.norm())
        result[suffix+'_comparison']=dict(difference_norm=float((a-b).norm()),cosine=float(torch.dot(a,b))/denom if denom else None)
    result.update(controls_training=False,common_coefficient=common_coefficient)
    return result


def native_heavy_diagnostics(solver, ctx, dr):
    """Read-only current source train fitting and fixed, disjoint V monitoring."""
    from .ir_head import capture_head_calls
    trace=solver.last_trace
    started=time.perf_counter()
    with isolated_diagnostics(solver.model,solver.optimizer):
        model=solver.model
        model.eval();model.adv_head.train()
        device=next(model.parameters()).device
        # A deterministic fixed panel, never a rotating or target-dependent selection.
        vx,vy,vd,meta=next(iter(dr.source.loader('val',256,seed=0,shuffle=False,workers=0)))
        vx,vy,vd=vx.to(device),vy.to(device),vd.to(device)
        val_ids=tuple(map(str,meta['sample_id']))
        train_ids=tuple(map(str,ctx.sample_ids))+tuple(map(str,ctx.dr_ids))
        if set(val_ids)&set(train_ids):raise ValueError('source V overlaps source training')
        def capture(parts,label):
            with capture_head_calls(model.adv_head,solver.layout.signature,'diagnostic/'+label) as tape:
                outs=[]
                with torch.no_grad():
                    for key,x,d,ids,weight in parts:
                        outs.append(model(x,return_aux=True,domain_labels=d,grl_lambda=0.))
                        tape.register(key,ids,d,torch.full((len(d),),weight/len(d),device=device),0.)
                return tape.batch(),outs
        weights={c.key.split('/')[0]:float(c.sample_weight.sum()) for c in trace['head_batch'].calls}
        train_batch,_=capture([('L',ctx.x,ctx.domain,ctx.sample_ids,weights.get('L',1.)),
                               ('U',ctx.dr_strong,ctx.dr_d,ctx.dr_ids,weights.get('U',1.))],'train')
        monitor_batch,outputs=capture([('V',vx,vd,val_ids,1.)],'monitor')
        result=dict(recovery=head_recovery_probe(model.adv_head,train_batch,monitor_batch),
                    source_identity=source_identity_summary(outputs[0]['tx_logits'],outputs[0]['z_id'],vy,meta['rx_i'].to(device)),
                    source_panel=dict(role='V',records=len(val_ids),physical_ids=val_ids,policy='fixed_first_256_native_nonshuffled_V'),
                    controls_training=False)
        result['nonlinear_residual_full']=nonlinear_residual(trace['phip'],trace['head_batch'],trace['h0'],trace['delta'],trace['metric'])
        result['nonlinear_residual_applied']=nonlinear_residual(trace['phip'],trace['head_batch'],trace['h0'],trace['delta']*solver.config['ir_kappa'],trace['metric'])
    result['elapsed_seconds']=time.perf_counter()-started
    return result


METRICS=('accepted_index','physical_exposure','effective_kappa','h0_norm','hp_norm','e_norm',
         'response_norm','cg_iterations','cg_status','linear_residual','origin_clip_coefficient',
         'formal_clip_coefficient','parameter_displacement_norm','optimizer_commits','model_calls',
         'backward_calls','head_only_calls','jvp_calls','vjp_calls','diagnostic_seconds')


def step_record(result, **fields):
    telemetry=result.telemetry or {}
    row=dict(telemetry)
    row.update(method=result.algorithm,accepted=result.accepted,loss=result.loss,
               loss_replay=result.loss_replay,field_evaluations=result.field_evaluations,
               fallback_reason=result.failure_stage)
    row.update(fields)
    aliases={'predictor_clip_coefficient':'origin_clip_coefficient',
             'linear_relative_residual':'linear_residual','model_forward_calls':'model_calls',
             'full_backward_calls':'backward_calls','head_only_backward_calls':'head_only_calls',
             'formal_optimizer_commits':'optimizer_commits'}
    for original,canonical in aliases.items():
        if canonical not in row and original in row:row[canonical]=row[original]
    for name in ('configured','eligible','executed','applied','nonzero','fallback'):
        row.setdefault(name,None)
    for name in METRICS:
        row.setdefault(name,None)
    row['missing_reasons']={k:'not measured by this execution path' for k,v in row.items() if v is None}
    return row


class TelemetryWriter:
    def __init__(self,path):
        self.path=Path(path)

    def write(self,record):
        encoded=json.dumps(record,ensure_ascii=False,allow_nan=False,sort_keys=True)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(encoded+'\n')
