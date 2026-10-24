"""Read complete source guard records; quantify weighted-U gradients before global clip/AdamW."""
import math
import statistics

GROUPS = ('t2', 'cls_head', 'other')
KEYS = ('L_norm', 'U_norm', 'applied_U_norm', 'dot_before', 'dot_after',
        'conflict', 'capped', 'tiny_L', 'conflicts', 'caps')


def metric(metrics, name):
    found = [v for k, v in metrics.items() if k.endswith(name)]
    if len(found) != 1 or not isinstance(found[0], (int, float)) or not math.isfinite(found[0]):
        raise ValueError('One finite logged source metric required: '+name)
    return found[0]


def group_observation(values):
    """Ratios are of logged parameter gradients, not optimizer steps or sample accuracy."""
    v = values
    if any(not math.isfinite(v[k]) for k in KEYS):
        raise ValueError('Finite source gradient metrics required')
    L, U, A = (v[k] for k in ('L_norm', 'U_norm', 'applied_U_norm'))
    if (min(L, U, A) < 0 or any(v[k] not in (0, 1) for k in ('conflict', 'capped', 'tiny_L'))
            or v['conflict'] != int(v['dot_before'] < 0)
            or A > L*(1+1e-6)+1e-8
            or v['dot_after'] < -1e-5*(L*A+1e-12)
            or abs(v['dot_before']) > L*U*(1+1e-5)+1e-10
            or abs(v['dot_after']) > L*A*(1+1e-5)+1e-10):
        raise ValueError('Source gradient norm/cosine/guard facts differ')
    if v['tiny_L'] and A > 1e-12:
        raise ValueError('Tiny-L group must contribute zero U gradient')
    ratio = lambda a, b: a/b if b > 0 else None
    return dict(L_norm=L, weighted_U_norm=U, applied_U_norm=A,
                U_to_L=ratio(U,L), applied_U_to_L=ratio(A,L), retained_U_norm=ratio(A,U),
                cosine_before=ratio(v['dot_before'],L*U), cosine_after=ratio(v['dot_after'],L*A),
                conflict=int(v['conflict']), capped=int(v['capped']), tiny_L=int(v['tiny_L']))


def distribution(values):
    x = sorted(v for v in values if v is not None)
    if not x:
        return dict(count=0, mean=None, min=None, p10=None, median=None, p90=None, max=None)
    def quantile(q):
        position = (len(x)-1)*q; lower = int(position); upper = min(lower+1,len(x)-1)
        return x[lower]+(position-lower)*(x[upper]-x[lower])
    return dict(count=len(x), mean=statistics.fmean(x), min=x[0], p10=quantile(.1),
                median=quantile(.5), p90=quantile(.9), max=x[-1])


def summarize(observations):
    if not observations:
        raise ValueError('Complete nonempty source records required')
    flags=('conflict','capped','tiny_L')
    return dict(records=len(observations),
                counts={k:sum(v[k] for v in observations) for k in flags},
                fractions={k:sum(v[k] for v in observations)/len(observations) for k in flags},
                distributions={k:distribution(v[k] for v in observations) for k in observations[0] if k not in flags})


def analyze_segment(records, expected_steps, start_epoch):
    """Require all native rows in order; validate real cumulative counters independently."""
    if expected_steps not in (2220,4440) or (start_epoch,expected_steps) not in ((20,2220),(30,4440)):
        raise ValueError('Fixed own E20+10+20 native source segments required')
    observations={g:[] for g in GROUPS}; epochs={}; totals=[]
    conflicts={g:0 for g in GROUPS}; caps=dict(conflicts); accepted=seen=0; count=0
    for count,record in enumerate(records,1):
        if count>expected_steps or record['step']!=count:
            raise ValueError('No missing/duplicate/extra source steps allowed')
        m=record['metrics']
        if m.get('train/pseudo_truth_available',0)!=0:
            raise ValueError('Source U truth must remain hidden')
        if metric(m,'HPL_weight')!=6 or metric(m,'HPL_GP_calls')!=count or metric(m,'HPL_calls')!=count:
            raise ValueError('Actual weight6 and one complete-L guard per native U loss required')
        ep,rem=divmod(count,222);seen_expected=ep*56700+rem*256
        new_seen=seen_expected-seen;new_accepted=metric(m,'HPL_accepted')-accepted
        if (metric(m,'HPL_seen')!=seen_expected or int(new_accepted)!=new_accepted or not 0<=new_accepted<=new_seen
                or metric(m,'HPL_student_views')!=6*seen_expected
                or metric(m,'HPL_EMA_updates')!=0 or metric(m,'T1_L_packets')!=768):
            raise ValueError('Actual complete source exposure differs')
        epoch=start_epoch+(count-1)//222+1
        stage=epochs.setdefault(epoch,{g:[] for g in GROUPS})
        current={}
        for group in GROUPS:
            values={k:metric(m,'HPL_GP_'+group+'_'+k) for k in KEYS}
            v=group_observation(values)
            conflicts[group]+=v['conflict'];caps[group]+=v['capped']
            if values['conflicts']!=conflicts[group] or values['caps']!=caps[group]:
                raise ValueError('Actual source conflict/cap counters differ')
            observations[group].append(v);stage[group].append(v);current[group]=v
        L=math.sqrt(sum(v['L_norm']**2 for v in current.values()))
        U=math.sqrt(sum(v['weighted_U_norm']**2 for v in current.values()))
        A=math.sqrt(sum(v['applied_U_norm']**2 for v in current.values()))
        totals.append(dict(weighted_U_to_L=U/L if L>0 else None,
                           applied_U_to_L=A/L if L>0 else None,retained_weighted_U_norm=A/U if U>0 else None))
        accepted+=int(new_accepted);seen=seen_expected
    if count!=expected_steps:
        raise ValueError('Only complete closed native source segments accepted')
    return dict(records=count,start_epoch=start_epoch,final_epoch=start_epoch+count//222,
                seen=seen,accepted=accepted,groups={g:summarize(observations[g]) for g in GROUPS},
                whole_parameter_gradient={k:distribution(v[k] for v in totals) for k in totals[0]},
                epochs=[dict(epoch=e,records=222,groups={g:summarize(v[g]) for g in GROUPS}) for e,v in sorted(epochs.items())],
                final_guard_counters=dict(calls=count,conflicts=conflicts,caps=caps),
                gradient_stage='Logged weighted-U group gradients after projection/cap;before native global clip5 and AdamW/decay',
                U_truth_reads=0,target_reads=0,model_updates=0)
