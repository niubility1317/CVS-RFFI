"""Source-only static auxiliary coefficients and common source curriculum."""
import copy,json,math,statistics,time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.cvs_urban_sigtime50 import design as reference
from experiments.cvs_urban_sigtime50.model import own_model
from experiments.cvs_urban_sigtime50.panel import load,metadata,apply_fixed_scene
from experiments.cvs_urban_sigtime50.channel import PairedChannel
from . import features as f
from . import objective as objective

RUN='20261015-phase1-sigtime-joint160-calibration-manysig-s2026092701-r01'
BASE=Path(reference.PROJECT)/'runs'/RUN
TARGET_RATIO=.1


def group_gradients(loss,groups):
    params=[p for values in groups.values() for p in values]
    values=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
    result={};offset=0
    for name,ps in groups.items():
        gs=values[offset:offset+len(ps)];offset+=len(ps)
        result[name]=torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten() for p,g in zip(ps,gs)])
    return result


def coefficient(ratios,target=TARGET_RATIO):
    if not ratios or any(not math.isfinite(v) or v<0 for v in ratios):raise ValueError('Invalid source-only gradient ratios')
    raw=statistics.median(ratios)
    if raw<=1e-12:raise ValueError('No effective backbone auxiliary gradient; no training launch')
    value=max(1e-4,min(10.,target/raw))
    return dict(coefficient=value,raw_median_gradient_ratio=raw,weighted_median_gradient_ratio=value*raw,
        target_gradient_ratio=target,bounds=[1e-4,10.],clipped=value!=target/raw)


def run():
    # All readiness comes from source metadata. Never inspect the row's target scores.
    config=reference.config(next(r for r in reference.rows() if r['arm']=='C3'))
    source=Path(config['output_root']);history=reference.read(source/'source_learning_progress.json')
    if not history['source_only'] or history['target_feedback'] or len(history['records'])!=50:
        raise ValueError('Need the fixed 50-epoch source-only A curriculum')
    model=own_model(config,'cpu');checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if checkpoint['ema_model'] is None:raise ValueError('Own scratch EMA source unavailable')
    teacher=copy.deepcopy(model);teacher.load_state_dict(checkpoint['ema_model'],strict=True);teacher.eval().requires_grad_(False)
    if BASE.exists():raise FileExistsError('Reconcile existing source calibration; do not overwrite')
    BASE.mkdir();torch.set_num_threads(2);started=time.perf_counter();panel=load()
    root=Path(__file__).resolve().parents[2]
    spec=reference.read(root/'experiments/cvs_urban_identity50/registration_calibration.json')
    spec['code']['commit']=(root/'release_commit.txt').read_text().strip()
    reference.write(BASE/'experiment.json',spec)
    before={k:v.clone() for k,v in model.state_dict().items()};teacher_before={k:v.clone() for k,v in teacher.state_dict().items()}
    model.requires_grad_(True).eval();groups=f.parameter_groups(model);boundary=f.boundary_parameters(model)
    channel=PairedChannel(config);records=[];ratios={kind:[] for kind in ('output_kl','joint_direction')};forwards=0
    try:
        for start in range(0,720,90):
            stop=start+90;x=panel['x'][start:stop];y=panel['y'][start:stop]
            target=f.frozen_teacher(teacher,x);forwards+=len(x)
            reliability=objective.labeled_reliability(target['tx_logits'],y)
            if start==0:
                with torch.no_grad():direct=teacher.forward_identity_only(x)['id_feat_joint']
                forwards+=len(x)
                if not torch.equal(direct,target['id_feat_joint']):raise ValueError('Teacher target is not deployed joint160')
            for scene in ('practical_mid_urban','practical_low_urban'):
                satellite,details=apply_fixed_scene(channel,x,metadata(panel,start,stop),scene)
                out,z=f.satellite_forward(model,satellite);forwards+=len(x)
                satellite_CE=.5*F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
                losses=dict(output_kl=objective.output_kl(z,target['tx_logits'],reliability,model.id_backbone.encoder.core.id_backbone.cls_head),
                    joint_direction=objective.direction(z,target['id_feat_joint'],reliability))
                ce=group_gradients(satellite_CE,groups)
                for kind,loss in losses.items():
                    gs=group_gradients(loss,groups)
                    bg=torch.autograd.grad(loss,list(boundary.values()),allow_unused=True,retain_graph=True)
                    if any(g is not None and float(g.abs().max())>1e-8 for g in bg):raise ValueError('Auxiliary gradient leaked to W or fusion gains')
                    for name,a in ce.items():
                        b=gs[name];an=float(a.norm());bn=float(b.norm());ratio=bn/(an+1e-12)
                        if not math.isfinite(ratio) or not torch.isfinite(b).all():raise ValueError('Nonfinite source auxiliary gradient')
                        ratios[kind].append(ratio)
                        records.append(dict(kind=kind,scene=scene,batch=start//90,count=len(x),teacher_eligible=int(reliability.sum()),
                            group=name,satellite_CE_gradient_norm=an,aux_gradient_norm=bn,raw_ratio=ratio,
                            cosine=float(a@b)/(an*bn) if an*bn else None))
                if start==0 and scene=='practical_mid_urban':print('SMOKE PASS real source C3/own EMA checkpoint; actual joint160; query_read=false; updates=0',flush=True)
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in teacher_before.items()):
            raise ValueError('Source calibration changed model or EMA state')
        if any(p.grad is not None for p in model.parameters()) or any(p.grad is not None for p in teacher.parameters()):
            raise ValueError('Read-only calibration accumulated training gradients')
    finally:model.requires_grad_(False)
    eta=[0.]+[float(r['eta_next']) for r in history['records'][:-1]]
    if len(eta)!=50 or any(v<0 or v>.5 for v in eta):raise ValueError('Invalid source A curriculum')
    result=dict(status='SOURCE_CALIBRATED',source_only=True,source_role='L_s',source_count=720,
        source_checkpoint=str(source/'final_ssdg.pth'),initialization_used=False,added_training_updates=0,head_fit_updates=0,
        U_truth_read=False,query_read=False,model_state_unchanged=True,teacher_state_unchanged=True,
        dropout='eval only for source coefficient diagnosis; training recipe unchanged',
        coefficient={kind:coefficient(values) for kind,values in ratios.items()},
        gradient_denominator='weighted satellite CE = .5 * CE',groups=list(groups),records=records,
        common_training_eta=eta,eta_basis='fixed actual source A schedule; no target score read',
        forward_packets=forwards,seconds=time.perf_counter()-started,
        commit=(Path(__file__).resolve().parents[2]/'release_commit.txt').read_text().strip())
    reference.write(BASE/'results.json',result)
    print(json.dumps({k:result[k] for k in ('status','coefficient','forward_packets','seconds','query_read')}),flush=True)
    return result


if __name__=='__main__':run()
