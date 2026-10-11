"""Peer soft objective with optional detached joint Fishr gradient cap."""
import torch
from torch.nn import functional as F
from experiments.cvs_early_geometry.objective import output,weak
from .multilevel import fixed_fishr as budget_fishr
from .style import styled_forward,capture_rng
from .joint_cap import apply_joint_cap,inactive_detail

BASE_WEIGHTED_KEYS=('ce','KL','feature','margin','style_ce','aux_ce','fishr')
AUDIT_ONLY_KEYS=('KL_same','KL_disagree')


def objective(c,model,bank,L,li,U,ui,step,device):
    if U is not None and 'y' in U:raise ValueError('U truth forbidden')
    teacher=c['teacher'];mc=c['multilevel']
    x=L['x'][li].to(device);y=L['y'][li].to(device)
    clean_rng=capture_rng(model) if mc['mixstyle_l'] else None
    logits=model(x);zero=logits.sum()*0
    parts=dict(ce=F.cross_entropy(logits,y),KL=zero,feature=zero,margin=zero,style_ce=zero)
    ramp=max(0.,min(1.,(step-c['semi']['warmup_updates'])/c['semi']['ramp_updates']))
    active=step>c['semi']['warmup_updates'];same_loss=disagree_loss=zero
    detail=dict(L_accuracy=float((logits.argmax(1)==y).float().mean()),U_packets=0,accepted_U=0,physical_L=0,physical_U=0,feature_U=0,margin_U=0,ramp=ramp,weak_forwards=0,
        auxiliary_steps=0,auxiliary_L=0,auxiliary_forwards=0,fishr_steps=0,mixstyle_steps=0,mixstyle_U_steps=0,mixed_U_packets=0,
        aux_ce_to_backbone=mc['aux_ce_to_backbone'],fishr_active_levels=list(mc['fishr_levels']),fishr_fixed_denominator=2,
        teacher_source=teacher['source'],teacher_selection=teacher['selection'],teacher_forwards=0,
        style_L_packets=0,mixed_L_packets=0,style_search_forwards=0,style_search_L=0,audit_only_weighted_keys=list(AUDIT_ONLY_KEYS),U_accuracy=None,U_accuracy_reason='U TX truth hidden')
    if active and mc['mixstyle_l']:
        ids=[L['ids'][i] for i in li.tolist()]
        styled_logits,style=styled_forward(model,x,y,L['rx'][li].to(device),L['day'][li].to(device),ids,step,dict(mc,model_seed=c['model_seed']),ramp,clean_rng=clean_rng)
        parts['style_ce']=F.cross_entropy(styled_logits,y)
        clean_margin=logits.detach().gather(1,y[:,None]).squeeze(1)-logits.detach().masked_fill(F.one_hot(y,6).bool(),float('-inf')).max(1).values
        styled_margin=styled_logits.detach().gather(1,y[:,None]).squeeze(1)-styled_logits.detach().masked_fill(F.one_hot(y,6).bool(),float('-inf')).max(1).values
        detail.update(L_style=style,mixstyle_steps=1,style_L_packets=len(y),mixed_L_packets=sum(s.get('changed_packets',0) for s in style['levels'].values()),
            style_L_accuracy=float((styled_logits.argmax(1)==y).float().mean()),style_prediction_change=float((styled_logits.argmax(1)!=logits.argmax(1)).float().mean()),
            clean_identity_margin=float(clean_margin.mean()),styled_identity_margin=float(styled_margin.mean()),
            style_search_forwards=style['eval_forwards'],
            style_search_L=style['eval_forwards']*len(y),
            style_prediction_diagnostic_scope='same clean dropout RNG replay; eval candidate selection uses labeled source batch mean CE only')
    if ui is not None:
        if not active or not c['unlabeled'] or teacher['source']=='none':raise ValueError('Unexpected U forward')
        if teacher['temperature']!=1. or teacher['confidence']!=.95 or teacher['selection'] not in ('gate','all'):raise ValueError('Teacher plan differs')
        ux=U['x'][ui].to(device)
        if teacher['source']=='self':
            if bank is not None or teacher['seeds']:raise ValueError('Self teacher cannot use a peer bank')
            a,_=weak(model,ux);q=a.softmax(1).detach();live=q
        elif teacher['source']=='peer':
            if bank is None or bank['teacher_seeds']!=teacher['seeds'] or c['model_seed'] in bank['teacher_seeds']:raise ValueError('Wrong peer teacher bank')
            selected=ui.tolist()
            if any(bank['ids'][i]!=U['ids'][i] for i in selected):raise ValueError('Peer/U PID alignment differs')
            q=bank['q'][ui].to(device).detach()
            devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):a,_=weak(model,ux)
            live=a.softmax(1).detach()
            detail.update(peer_disagreement=float(bank['peer_disagreement'][ui].float().mean()),peer_JS=float(bank['peer_JS'][ui].mean()))
        else:raise ValueError('Unknown teacher source')
        noisegen=torch.Generator().manual_seed(c['model_seed']+step*9901)
        noise=torch.randn(ux.shape,generator=noisegen,dtype=ux.dtype).to(device)*c['semi']['noise_std']
        if mc['mixstyle_u']:
            if tuple(mc['style_levels'])!=('time_down','t2.pool'):raise ValueError('Both U style layers required')
            ids=[U['ids'][i] for i in ui.tolist()]
            with model.multilevel.main_style(q,U['rx'][ui].to(device),U['day'][ui].to(device),ids,step,mc) as style:b,g=output(model,ux+noise)
            detail.update(U_style=style,mixstyle_U_steps=1)
            detail['mixed_U_packets']=sum(v.get('changed_packets',0) for v in style['levels'].values())
        else:b,g=output(model,ux+noise)
        mask=q.max(1).values>=teacher['confidence'] if teacher['selection']=='gate' else torch.ones(len(q),dtype=torch.bool,device=q.device)
        disagreement=q.argmax(1)!=live.argmax(1)
        per_sample=F.kl_div(b.log_softmax(1),q,reduction='none').sum(1)*mask
        parts['KL']=per_sample.mean();same_loss=(per_sample*(~disagreement)).mean();disagree_loss=(per_sample*disagreement).mean()
        detail.update(U_packets=len(ux),accepted_U=int(mask.sum()),weak_forwards=1,U_confidence=float(q.max(1).values.mean()),
            unit_feature_std=float(F.normalize(g.detach(),dim=1,eps=1e-4).std(0,unbiased=False).mean()),
            teacher_entropy=float(-(q*q.clamp_min(1e-12).log()).sum(1).mean()),teacher_live_disagreement=float(disagreement.float().mean()),
            teacher_live_disagreement_count=int(disagreement.sum()),teacher_live_agreement_count=int((~disagreement).sum()),
            accepted_disagreement=int((mask&disagreement).sum()),accepted_agreement=int((mask&~disagreement).sum()),
            KL_same=float(same_loss.detach()),KL_disagree=float(disagree_loss.detach()),KL_full_batch_denominator=len(q),
            weak_forward_role='self_teacher' if teacher['source']=='self' else 'live_student_diagnostic_only')
    elif teacher['source']=='none' and bank is not None:raise ValueError('CE must not load a teacher')
    weighted={k:v*(mc['style_ce_weight'] if k=='style_ce' else c['losses'][k])*(1. if k=='ce' else ramp) for k,v in parts.items()}
    if mc['enabled'] and active and step%mc['interval']==0:
        aux_ce,terms,info=model.multilevel.loss(model,L,step,device,mc)
        fishr=torch.stack([terms.get(level,zero) for level in ('time_down','t2.pool')]).mean() if terms else zero
        wf,budget=budget_fishr(model,parts['ce'],terms,mc,ramp)
        parts.update(aux_ce=aux_ce,fishr=fishr);weighted.update(aux_ce=aux_ce*mc['aux_ce_weight']*ramp,fishr=wf)
        detail.update(multilevel=info,fishr_budget=budget,auxiliary_steps=1,auxiliary_L=info['auxiliary_packets'],auxiliary_forwards=info['aux_forwards'],fishr_steps=int(bool(mc['fishr_levels'])))
    else:parts.update(aux_ce=zero,fishr=zero);weighted.update(aux_ce=zero,fishr=zero)
    cap_config=mc['joint_gradient_cap']
    if detail['auxiliary_steps'] and cap_config['enabled']:
        identity_loss=sum(weighted[key] for key in BASE_WEIGHTED_KEYS if key!='fishr')
        weighted['fishr'],cap=apply_joint_cap(model,identity_loss,weighted['fishr'],cap_config)
        detail['joint_gradient_cap']=cap
    else:
        detail['joint_gradient_cap']=inactive_detail('disabled_by_policy' if not cap_config['enabled'] else 'no_active_auxiliary_step')
    if 'fishr_budget' in detail:
        detail['fishr_budget']['scope']='raw_pre_joint_cap'
        detail['fishr_budget']['joint_cap_scalar']=detail['joint_gradient_cap']['scalar']
        detail['fishr_budget']['effective_weighted_loss']=float(weighted['fishr'].detach())
    total=sum(weighted[key] for key in BASE_WEIGHTED_KEYS)
    weighted.update(KL_same=same_loss*c['losses']['KL']*ramp,KL_disagree=disagree_loss*c['losses']['KL']*ramp)
    return total,parts,weighted,detail
