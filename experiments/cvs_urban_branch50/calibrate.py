"""One common warmup snapshot, actual global feature-gradient strength, no updates."""
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import branches as b,objective,gradient as g,design as d
from .channel import PairedChannel
from experiments.cvs_urban_sigtime50.panel import metadata,apply_fixed_scene


def publish(result,path):
    """Make a complete recipe visible to the queue in one same-directory rename."""
    path=Path(path);temporary=path.with_suffix(path.suffix+'.writing')
    if path.exists() or temporary.exists():raise FileExistsError('Reconcile existing calibration output; never overwrite')
    d.write(temporary,result)
    temporary.rename(path)


def run(model,teacher,panel):
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    tbefore={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    modes={m:m.training for m in model.modules()}
    teacher_modes={m:m.training for m in teacher.modules()}
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError('EMA has accumulated gradients')
    oldgrads={p:None if p.grad is None else p.grad.detach().clone() for p in model.parameters()}
    device=next(model.parameters()).device;started=time.perf_counter();model.eval()
    channel=PairedChannel(d.config(d.rows()[0]));keys=(*b.BRANCHES,'J')
    norms={k:[] for k in ('CE',*keys,'ALL')};records=[];cache=[];forwards=0;theta=None
    try:
        for start in range(0,720,90):
            stop=start+90;x=panel['x'][start:stop].to(device);y=panel['y'][start:stop].to(device)
            logits,target=b.frozen_teacher(teacher,x);forwards+=len(x)
            reliability=objective.labeled_reliability(logits,y)
            for scene in ('practical_mid_urban','practical_low_urban'):
                sat,_=apply_fixed_scene(channel,x,metadata(panel,start,stop),scene)
                out,values=b.forward(model,sat);forwards+=len(sat)
                if theta is None:
                    theta=b.feature_parameters(model,out['id_feat_joint'])
                    if sum(p.numel() for _,p in theta)!=195041:raise ValueError('Actual feature extractor scope changed')
                ce=.5*F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
                cg=g.vectors(ce,theta);cn=g.norm(cg);norms['CE'].append(cn)
                losses=objective.losses(values,target,reliability)
                for key,loss in losses.items():
                    ag=g.vectors(loss,theta);norms[key].append(g.norm(ag))
                    records.append(dict(batch=start//90,scene=scene,position=key,
                        teacher_eligible=int(reliability.sum()),**g.compare(cg,ag),
                        layer_gradients={name:dict(CE=float(cg[name].norm()),aux=float(ag[name].norm())) for name,_ in theta}))
                cache.append((sat.detach(),y,target,reliability.detach()))
        coefficients={key:g.coefficient(norms['CE'],norms[key]) for key in keys}
        weights={key:.2 for key in b.BRANCHES}
        branch_scales={key:coefficients[key]['coefficient'] for key in b.BRANCHES}
        # Re-run only the student on the identical cached IQ/teacher targets;
        # first normalize branch scales, then calibrate the combined total.
        for sat,y,target,reliability in cache:
            out,values=b.forward(model,sat);forwards+=len(sat)
            values=objective.losses(values,target,reliability)
            loss=objective.mixture(values,branch_scales,weights)
            norms['ALL'].append(g.norm(g.vectors(loss,theta)))
        coefficients['ALL']=g.coefficient(norms['CE'],norms['ALL'])
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(
                not torch.equal(v,teacher.state_dict()[k]) for k,v in tbefore.items()):
            raise ValueError('Calibration changed model or teacher state')
        for p,old in oldgrads.items():
            if (p.grad is None)!=(old is None) or (old is not None and not torch.equal(p.grad,old)):
                raise ValueError('Calibration changed pending training gradients')
        if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Calibration updated teacher gradients')
    finally:
        for module,mode in modes.items():module.training=mode
        for module,mode in teacher_modes.items():module.training=mode
    if forwards!=3600:raise ValueError('Unexpected once-only source calibration forward budget')
    return dict(status='SOURCE_BRANCH_CALIBRATED',source_only=True,query_read=False,U_truth_read=False,
        optimizer_updates=0,head_fit_updates=0,warmup_epoch=5,source_role='L_dev',source_count=720,
        checkpoint_selection='fixed epoch5; no source/target checkpoint ranking',model_state_unchanged=True,
        teacher_state_unchanged=True,training_gradients_unchanged=True,feature_parameters=195041,
        feature_tensors=len(theta),feature_names=[k for k,_ in theta],default_W_excluded=True,
        global_norms=norms,coefficients=coefficients,ALL_weights=weights,records=records,
        forward_packets=forwards,seconds=time.perf_counter()-started,
        once_for_all_positions_and_seeds=True,dropout='eval for one static source calibration; native train dropout unchanged')
