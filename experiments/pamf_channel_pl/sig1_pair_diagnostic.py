"""Cached-source study: rejected packets, raw-soft targets and same-packet pairs."""
import argparse
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write
from .restored_source import SCENES

RUN='20261026-analysis-PAMF-SIG1-rejected-pairs-m1-r01'
RELEASE='20261026-analysis-PAMF-SIG1-rejected-pairs-release-r01'
INPUT_RUN='20261026-diagnostic-PAMF-SIG1-LU-restored-views-m1-r01'
PREFIX_RUN='20261024-phase1-PAMF-REPL-completion-m201-r01'
SEED=2026092701
RULES=('raw_soft','G_peer_probability','G_peer_cosine_scores')
NAMES=('raw','raw_noise40')+tuple(s+'::'+v for s in SCENES for v in ('G_a','G_b','pre40','post40'))


def probability(logits):
    a=np.asarray(logits,np.float64)
    if a.ndim<2 or a.shape[-1]!=6 or not np.isfinite(a).all():raise ValueError('Finite six-class archived logits required')
    a=a-a.max(-1,keepdims=True);p=np.exp(a);return p/p.sum(-1,keepdims=True)


def gradients(rule,p0,pa,pb,za,zb):
    if rule not in RULES or any(v.shape!=p0.shape for v in (pa,pb,za,zb)) or p0.ndim!=2 or p0.shape[1]!=6:
        raise ValueError('Aligned six-class paired source views required')
    if not all(np.isfinite(v).all() for v in (p0,pa,pb,za,zb)):raise ValueError('Finite arrays only')
    if rule=='raw_soft':return pa-p0,pb-p0
    if rule=='G_peer_probability':return pa-pb,pb-pa
    # Mean squared difference between the original cosine scores (logits/30).
    delta=(za-zb)/(6*30.**2)
    return delta,-delta


def feature_gradient(h,logit_gradient,weight):
    if h.ndim!=2 or h.shape[1]!=160 or logit_gradient.shape!=(len(h),6) or weight.shape!=(6,160):raise ValueError('Default deployed head dimensions required')
    z=h/np.linalg.norm(h,axis=1,keepdims=True).clip(1e-12);w=weight/np.linalg.norm(weight,axis=1,keepdims=True).clip(1e-12)
    value=30.*(logit_gradient@w)
    return value-z*(z*value).sum(1,keepdims=True)


def summary(a):
    a=np.asarray(a,np.float64)
    if not a.size:return None
    if not np.isfinite(a).all():raise ValueError('Finite diagnostic metrics required')
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)))


def validate_runtime(rt):
    expected={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
    if (set(rt)!=expected or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN
            or rt['release_root']!=rt['project']+'/releases/'+RELEASE or rt['context']!=rt['release_root']+'/context.json'
            or rt['runtime_file']!=rt['release_root']+'/runtime.json' or rt['allowed_gpus']!=[]
            or rt['launch_owner']!='codex-pamf-sig1-rejected-pairs-20261026'):
        raise ValueError('One immutable source-only CPU analysis required')


def run(rt):
    import torch
    from .restored_source import validate_context
    from .sig1_repl_base import prefix,check_state
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    context=read(rt['context']);validate_context(context);parent=Path(prefix(rt['project']));done=read(parent/'complete.json');cfg=done['config'];contract=read(cfg['source_contract'])
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=20 or done['updates']!=4440 or done['initialization']!='scratch' or done['ancestors']!=[]
            or done['target_access'] or done['U_truth_read'] or done['head_fit_updates'] or cfg['model_seed']!=SEED
            or contract!=read(next(v['expected_source_contract'] for v in context['cases'] if v['model_seed']==SEED))):
        raise ValueError('Known source-only scratch single-view E20 required')
    cp=torch.load(parent/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp)
    if cp['run_id']!=cfg['run_id'] or cp['candidate_id']!=cfg['row_id'] or cp['epoch']!=20 or cp['checkpoint_selection']!='final_only':raise ValueError('Original E20 payload differs')
    weight=np.asarray(cp['model']['id_backbone.encoder.core.id_backbone.cls_head.weight'].double().tolist(),np.float64);del cp
    wn=weight/np.linalg.norm(weight,axis=1,keepdims=True)
    input_root=Path(rt['project'])/'runs'/INPUT_RUN;header=read(input_root/'complete.json')
    if (header['status']!='SIG1_LU_RESTORED_VIEWS_ANALYZED' or header['teacher_epoch']!=20 or header['model_seed']!=SEED
            or not header['parameters_and_all_registered_buffers_unchanged'] or any(header[k] for k in ('optimizer_updates','U_truth_reads','V_reads','target_reads'))):
        raise ValueError('Completed source role/view arrays required')
    rows=[];packet_rows=[];identity_max_error=0.;head_max_error=0.;cases=0
    for role in ('L_s','U_s'):
        with np.load(input_root/role/'source_predictions_features.npz',allow_pickle=False) as a:
            if ('y' in a.files)!=(role=='L_s'):raise ValueError('Source label visibility differs')
            logits=a['logits'].astype(np.float64);features=a['features'].astype(np.float64);ids=a['ids'];rx=a['rx_i'];y=a['y'] if role=='L_s' else None
        if logits.shape!=(14,6300,6) or features.shape!=(14,6300,160) or len(set(ids.tolist()))!=6300 or not np.isfinite(features).all():raise ValueError('Complete cached source role required')
        if set(ids.tolist())-set(contract['role_ids'][role]):raise ValueError('Foreign physical source IDs')
        h=features/np.linalg.norm(features,axis=-1,keepdims=True).clip(1e-12)
        reconstructed=30.*(h@wn.T);error=float(np.max(np.abs(reconstructed-logits)));head_max_error=max(head_max_error,error)
        if error>1e-4:raise ValueError('Frozen default-head algebra differs from archived predictions')
        if role=='L_s':
            if not np.array_equal(ids,np.asarray(contract['role_ids']['L_s'])):raise ValueError('Complete original L order required')
            if not np.array_equal(reconstructed[0,:2].argmax(1),logits[0,:2].argmax(1)):raise ValueError('Real E20 head/cached-L2 smoke failed')
            write(root/'smoke.json',dict(status='PASS',actual_E20_default_head=True,cached_source_L_packets=2,head_matrix_algebra_only=True,
                fitted_parameters=0,optimizer_updates=0,new_IQ_reads=0,U_truth_reads=0,target_reads=0,max_abs_logit_difference=error))
        p=probability(logits);accepted=p[0].max(1)>=.95;raw_pred=p[0].argmax(1)
        groups=[('ALL',np.ones(6300,bool)),('raw_accepted',accepted),('raw_rejected',~accepted)]+[('RX'+str(v),rx==v) for v in np.unique(rx)]
        if y is not None:groups += [('rejected_raw_correct',(~accepted)&(raw_pred==y)),('rejected_raw_wrong',(~accepted)&(raw_pred!=y))]
        for scene_index,scene in enumerate(SCENES):
            for pairing in ('same_scene','cross_scene'):
                other=scene if pairing=='same_scene' else SCENES[(scene_index+1)%3]
                ai=NAMES.index(scene+'::G_a');bi=NAMES.index(other+'::G_b');pa,pb=p[ai],p[bi];za,zb=logits[ai],logits[bi];ha,hb=h[ai],h[bi]
                agree=pa.argmax(1)==pb.argmax(1);gg=(pa+pb)/2;allmean=(p[0]+pa+pb)/3
                for name,mask in groups:
                    if not mask.any():continue
                    record=dict(role=role,pairing=pairing,first_scene=scene,second_scene=other,group=name,n=int(mask.sum()),
                        G_argmax_agreement=float(agree[mask].mean()),raw_confidence=summary(p[0,mask].max(1)),
                        raw_entropy=summary(-(p[0,mask]*np.log(p[0,mask].clip(1e-30))).sum(1)),
                        new95_G_consensus=int((mask&agree&(pa.max(1)>=.95)&(pb.max(1)>=.95)).sum()))
                    if y is not None:
                        gate=mask&agree&(pa.max(1)>=.95)&(pb.max(1)>=.95)
                        record.update(raw_accuracy=float((raw_pred[mask]==y[mask]).mean()),G_consensus95_correct=int((gate&(pa.argmax(1)==y)).sum()),
                            G_agree_but_wrong=float((agree[mask]&(pa[mask].argmax(1)!=y[mask])).mean()),
                            GG_mean_repairs=int(((raw_pred!=y)&(gg.argmax(1)==y)&mask).sum()),GG_mean_damage=int(((raw_pred==y)&(gg.argmax(1)!=y)&mask).sum()),
                            all3_mean_repairs=int(((raw_pred!=y)&(allmean.argmax(1)==y)&mask).sum()),all3_mean_damage=int(((raw_pred==y)&(allmean.argmax(1)!=y)&mask).sum()))
                    packet_rows.append(record)
                true=None
                if y is not None:
                    t=np.full((6300,6),.01/6);t[np.arange(6300),y]+=.99;true=(pa-t,pb-t)
                for rule in RULES:
                    ga,gb=gradients(rule,p[0],pa,pb,za,zb);fa,fb=feature_gradient(ha,ga,weight),feature_gradient(hb,gb,weight)
                    norm=np.sqrt((fa*fa).sum(1)+(fb*fb).sum(1))
                    if true is not None:
                        ta,tb=true;fta,ftb=feature_gradient(ha,ta,weight),feature_gradient(hb,tb,weight)
                        norm_true=np.sqrt((fta*fta).sum(1)+(ftb*ftb).sum(1));valid=(norm>1e-10)&(norm_true>1e-10)
                        cos=((fa*fta).sum(1)+(fb*ftb).sum(1))/(norm*norm_true).clip(1e-30)
                        dot=(ga*ta).sum(1)+(gb*tb).sum(1)
                        if rule=='G_peer_probability':
                            identity=np.square(pa-pb).sum(1);identity_max_error=max(identity_max_error,float(np.max(np.abs(identity-dot))))
                            if identity_max_error>1e-12:raise ValueError('Same-target logit-gradient identity failed')
                        if rule=='G_peer_cosine_scores' and dot.min() < -1e-12:raise ValueError('Softmax monotonicity identity failed')
                    for name,mask in groups:
                        if not mask.any():continue
                        record=dict(role=role,pairing=pairing,first_scene=scene,second_scene=other,group=name,rule=rule,n=int(mask.sum()),
                            normalized_feature_leaf_gradient_norm=summary(norm[mask]))
                        if true is not None:
                            use=mask&valid;record.update(valid_comparisons=int(use.sum()),normalized_feature_leaf_gradient_cosine=summary(cos[use]),
                                negative_feature_gradient_fraction=float((cos[use]<0).mean()) if use.any() else None,
                                logit_gradient_dot_true_CE=summary(dot[mask]))
                        rows.append(record)
                cases+=1
        write(root/(role+'_complete.json'),dict(status='CACHED_ROLE_ANALYZED',role=role,records=6300,U_truth_read=False,optimizer_updates=0))
    write(root/'complete.json',dict(status='SIG1_REJECTED_PAIR_ANALYSIS_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],input_run=INPUT_RUN,
        model_seed=SEED,teacher_epoch=20,source_cases=cases,packet_results=packet_rows,gradient_results=rows,
        head_max_abs_logit_difference=head_max_error,logit_identity_max_abs_error=identity_max_error,
        new_backbone_forwards=0,new_G_forwards=0,new_IQ_reads=0,U_truth_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,fitted_head_updates=0,seconds=time.perf_counter()-start,
        scope='Source L is already-seen training proxy; U true precision unknown; gradients are joint logits or unit-feature leaves,not backbone parameters/optimizer updates; no student is selected or trained'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();rt=read(a.config)
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    torch.set_num_threads(2);run(rt)
