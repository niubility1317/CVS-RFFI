"""Own epoch5 calibration only; source labels, zero updates, unchanged pending grads."""
import time
import torch
from torch.nn import functional as F
from . import design as d,features as f,objective as objective
from .channel import PairedChannel
from experiments.cvs_urban_branch50 import branches as b,gradient as g
from experiments.cvs_urban_bridge.signature import teacher_scales
from experiments.cvs_urban_sigtime50.panel import metadata


def run(model,teacher,panel,c):
    d.validate(c)
    if (len(panel['y'])!=720 or model._feature_l_steps!=1110
            or any(p.requires_grad or p.grad is not None for p in teacher.parameters())):
        raise ValueError('Own frozen EMA and complete legal L_dev720 required')
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    tbefore={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    grads={p:None if p.grad is None else p.grad.detach().clone() for p in model.parameters()}
    modes={m:m.training for m in model.modules()};device=next(model.parameters()).device
    started=time.perf_counter();cache=[];descriptors=[];norms={k:[] for k in ('CE','joint','local')};records=[];forwards=0
    channel=PairedChannel(c);theta=None;model.eval()
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            for start in range(0,720,90):
                stop=start+90;x=panel['x'][start:stop].to(device);y=panel['y'][start:stop].to(device)
                for scene in ('practical_mid_urban','practical_low_urban'):
                    full,hq,meta=channel.apply(x,metadata(panel,start,stop),5,0,view='source_validation_bridge_cal',scene=scene)
                    logits,target=f.frozen_teacher(teacher,torch.cat((x,hq)));forwards+=180
                    descriptors.append(target['D'][90:]);jw,sw=objective.weights(logits[:90],y,meta)
                    cache.append((full,hq,y,target['J'][:90],target['D'][90:],jw,sw,scene,start//90))
            scales=teacher_scales(torch.cat(descriptors))
            for full,hq,y,clean_target,hq_target,jw,sw,scene,batch in cache:
                out,values=f.forward(model,torch.cat((full,hq)));forwards+=180
                if theta is None:
                    theta=b.feature_parameters(model,out['id_feat_joint'])
                    if sum(p.numel() for _,p in theta)!=195041:raise ValueError('Original feature parameter scope changed')
                ce=.5*F.cross_entropy(out['tx_logits'][:90],y,label_smoothing=.01)
                joint=objective.joint(values['J'][:90],values['J'][90:],clean_target,jw)
                local,_=objective.local(values['D'][:90],hq_target,scales,sw)
                cg=g.vectors(ce,theta);norms['CE'].append(g.norm(cg))
                for name,loss in (('joint',joint),('local',local)):
                    ag=g.vectors(loss,theta);norms[name].append(g.norm(ag))
                    records.append(dict(scene=scene,batch=batch,loss=name,**g.compare(cg,ag)))
            coefficients={}
            for name in ('joint','local'):
                try:coefficients[name]=g.coefficient(norms['CE'],norms[name])
                except ValueError as exc:raise ValueError(f'{name} source calibration: {exc}') from exc
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in tbefore.items()):
            raise ValueError('Own warmup calibration changed student/EMA state')
        if any((p.grad is None)!=(v is None) or (v is not None and not torch.equal(p.grad,v)) for p,v in grads.items()):
            raise ValueError('Source calibration changed pending training gradients')
        if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Source calibration fitted own EMA')
    finally:
        for module,mode in modes.items():module.training=mode
    if forwards!=5760:raise ValueError('Own epoch5 calibration budget differs')
    return dict(status='SOURCE_BRIDGE_CALIBRATED',source_only=True,query_read=False,U_truth_read=False,
        optimizer_updates=0,EMA_updates=0,head_fit_updates=0,warmup_epoch=5,source_role='L_dev720',
        forward_packets=forwards,feature_parameters=195041,coefficients=coefficients,scales=scales,
        global_norms=norms,records=records,model_state_unchanged=True,EMA_state_unchanged=True,
        pending_gradients_unchanged=True,seconds=time.perf_counter()-started,
        source_stream=c['bridge']['calibration_stream'],scope='Own scratch epoch5 only; both rows compute both calibrations; local coefficient used only in BRIDGE_SIG')
