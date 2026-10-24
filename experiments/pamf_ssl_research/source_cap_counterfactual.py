"""Fixed cap1/cap2 arithmetic on complete logged source gradients; no model replay."""
import math
import statistics
from .source_gradient_trajectory import GROUPS,KEYS,metric,group_observation


def group_counterfactual(v):
    group_observation(v)
    L,U,A=(v[k] for k in ('L_norm','U_norm','applied_U_norm'))
    if bool(L*L<=1e-12)!=bool(v['tiny_L']):raise ValueError('Actual tinyL boundary differs')
    if v['tiny_L']:
        projected=0.
    elif not v['capped']:
        # The measured uncapped postprojection norm is more precise near cancellation.
        projected=A
    else:
        projected=math.sqrt(max(0.,U*U-min(v['dot_before'],0.)**2/(L*L)))
    reconstructed=min(projected,L)
    if not math.isclose(reconstructed,A,rel_tol=2e-5,abs_tol=2e-7):
        raise ValueError('Logged cap1 norm does not match the fixed projection arithmetic')
    if not v['tiny_L'] and not v['capped']:
        expected=math.sqrt(max(0.,U*U-min(v['dot_before'],0.)**2/(L*L)))
        if not math.isclose(expected,A,rel_tol=2e-5,abs_tol=2e-7):raise ValueError('Uncapped source projection differs')
    A2=min(projected,2*L);dot2=v['dot_after']*(A2/A) if A>0 else 0.
    if (A2<A-2e-7 or A2>2*L*(1+1e-6)+1e-8
        or dot2 < -1e-5*(L*A2+1e-12)):
        raise ValueError('Fixed cap2 must retain the projected direction and its declared norm bound')
    combined1=max(0.,L*L+A*A+2*v['dot_after'])
    combined2=max(0.,L*L+A2*A2+2*dot2)
    return dict(L_norm=L,cap1_applied_U_norm=A,cap2_applied_U_norm=A2,
        cap1_applied_U_to_L=A/L if L>0 else None,cap2_applied_U_to_L=A2/L if L>0 else None,
        cap1_capped=int(v['capped']),cap2_capped=int(projected>2*L),
        cap1_dot_after=v['dot_after'],cap2_dot_after=dot2,
        cap1_combined_gradient_norm_squared=combined1,cap2_combined_gradient_norm_squared=combined2)


def analyze_segment(records,expected_steps,start_epoch):
    if (start_epoch,expected_steps) not in ((20,2220),(30,4440)):
        raise ValueError('Complete originalE20 own10/own20 source segments required')
    groups={g:[] for g in GROUPS};whole=[];accepted=seen=0;count=0
    conflicts={g:0 for g in GROUPS};caps=dict(conflicts)
    for count,row in enumerate(records,1):
        if count>expected_steps or row['step']!=count:raise ValueError('No missing,duplicate or extra source records allowed')
        m=row['metrics'];ep,rem=divmod(count,222);expected_seen=ep*56700+rem*256
        current_accepted=metric(m,'HPL_accepted')
        if (m.get('train/pseudo_truth_available',0) or metric(m,'HPL_weight')!=6
            or metric(m,'HPL_GP_calls')!=count or metric(m,'HPL_calls')!=count
            or metric(m,'HPL_seen')!=expected_seen or metric(m,'HPL_student_views')!=6*expected_seen
            or metric(m,'HPL_EMA_updates') or metric(m,'T1_L_packets')!=768
            or int(current_accepted)!=current_accepted or not 0<=current_accepted-accepted<=expected_seen-seen):
            raise ValueError('Actual full source exposure,weight6,guard or hidden-U permissions differ')
        now={}
        for g in GROUPS:
            v={k:metric(m,'HPL_GP_'+g+'_'+k) for k in KEYS};q=group_counterfactual(v)
            conflicts[g]+=v['conflict'];caps[g]+=v['capped']
            if v['conflicts']!=conflicts[g] or v['caps']!=caps[g]:raise ValueError('Actual source guard counters differ')
            groups[g].append(q);now[g]=q
        L=math.sqrt(sum(q['L_norm']**2 for q in now.values()))
        n1=math.sqrt(sum(q['cap1_applied_U_norm']**2 for q in now.values()))
        n2=math.sqrt(sum(q['cap2_applied_U_norm']**2 for q in now.values()))
        combined1=math.sqrt(sum(q['cap1_combined_gradient_norm_squared'] for q in now.values()))
        combined2=math.sqrt(sum(q['cap2_combined_gradient_norm_squared'] for q in now.values()))
        whole.append(dict(cap1_applied_U_to_L=n1/L if L>0 else None,cap2_applied_U_to_L=n2/L if L>0 else None,
            cap1_combined_gradient_norm=combined1,cap2_combined_gradient_norm=combined2,
            cap1_combined_norm_gt5=int(combined1>5.),cap2_combined_norm_gt5=int(combined2>5.)))
        accepted=int(current_accepted);seen=expected_seen
    if count!=expected_steps:raise ValueError('Complete closed source segment required')
    def summarize(rows):
        return {key:dict(count=len(vals),mean=statistics.fmean(vals) if vals else None)
                for key in rows[0] for vals in [[r[key] for r in rows if r[key] is not None]]}
    return dict(records=count,seen=seen,accepted=accepted,groups={g:summarize(v) for g,v in groups.items()},
        whole_parameter_gradient=summarize(whole),final_guard_counters=dict(calls=count,conflicts=conflicts,caps=caps),
        measured_stage='Same logged gradients;fixed cap2 arithmetic before commonclip5/AdamW,not new trajectory or optimizer steps',
        U_truth_reads=0,target_reads=0,model_updates=0)


def pool(segments):
    if len(segments)!=2 or [s['records'] for s in segments]!=[2220,4440]:raise ValueError('Both complete native source segments required')
    def merge(a,b):
        n=a['count']+b['count']
        return dict(count=n,mean=sum(v['count']*v['mean'] for v in (a,b) if v['count'])/n if n else None)
    result=dict(records=6660,seen=sum(s['seen'] for s in segments),accepted=sum(s['accepted'] for s in segments),
        groups={g:{k:merge(segments[0]['groups'][g][k],segments[1]['groups'][g][k]) for k in segments[0]['groups'][g]} for g in GROUPS},
        whole_parameter_gradient={k:merge(segments[0]['whole_parameter_gradient'][k],segments[1]['whole_parameter_gradient'][k]) for k in segments[0]['whole_parameter_gradient']})
    for values in [*result['groups'].values(),result['whole_parameter_gradient']]:
        before=values['cap1_applied_U_to_L']['mean'];after=values['cap2_applied_U_to_L']['mean']
        values['relative_applied_U_to_L_increase']=(after-before)/before if before and after is not None else None
    g=result['groups']['t2'];w=result['whole_parameter_gradient']
    result['scientific_rule_pass']=bool(g['cap1_capped']['mean']>=.5 and g['relative_applied_U_to_L_increase'] is not None
        and g['relative_applied_U_to_L_increase']>=.15 and w['relative_applied_U_to_L_increase'] is not None
        and w['relative_applied_U_to_L_increase']>=.10
        and w['cap2_combined_norm_gt5']['mean']-w['cap1_combined_norm_gt5']['mean']<=.05)
    return result
