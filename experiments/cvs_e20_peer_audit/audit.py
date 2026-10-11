"""Independent-seed P0 source teacher complementarity; aggregation only."""
from itertools import combinations
import torch

SEEDS=(2026092701,2026092702,2026092703)
THRESHOLD=.95
ECE_EDGES=tuple(i/15 for i in range(16))


def select_p0(source_predictions):
    """Select only saved parent records; never inspect the endpoints mapping."""
    if set(source_predictions)!= {'L','V'}:raise ValueError('Only source L/V roles allowed')
    return {role:source_predictions[role]['parent'] for role in ('L','V')}


def _record(record,role):
    if record.get('role')!=role or role not in ('L','V'):raise ValueError('Source role contamination')
    ids=list(record['ids']);n=len(ids)
    if not n or len(set(ids))!=n:raise ValueError('Nonempty unique physical IDs required')
    q=record['probabilities']
    if not torch.is_tensor(q) or not q.is_floating_point() or q.shape!=(n,6):raise ValueError('Expected six-class probabilities')
    q=q.detach().cpu().double()
    if not torch.isfinite(q).all() or (q<0).any() or (q>1).any() or not torch.allclose(q.sum(1),torch.ones(n,dtype=torch.float64),atol=1e-6,rtol=1e-6):
        raise ValueError('Invalid probability simplex')
    metadata={}
    for key in ('y','rx','day'):
        t=record[key]
        if not torch.is_tensor(t) or t.shape!=(n,) or t.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):raise ValueError('Invalid source metadata')
        metadata[key]=t.detach().cpu().long()
    if not bool(((metadata['y']>=0)&(metadata['y']<6)).all()):raise ValueError('Labels outside registered six classes')
    return dict(role=role,ids=ids,probabilities=q,**metadata)


def _metrics(q,y):
    n=len(y)
    conf,pred=q.max(1);yes=pred==y
    bins=[];ece=0.
    for i,(lo,hi) in enumerate(zip(ECE_EDGES[:-1],ECE_EDGES[1:])):
        mask=(conf>=lo)&((conf<=hi) if i==14 else (conf<hi));count=int(mask.sum())
        accuracy=float(yes[mask].double().mean()) if count else None
        confidence=float(conf[mask].mean()) if count else None
        if count:ece+=count/n*abs(accuracy-confidence)
        bins.append(dict(lower=lo,upper=hi,upper_inclusive=i==14,count=count,accuracy=accuracy,confidence=confidence))
    return dict(count=n,accuracy=float(yes.double().mean()) if n else None,
        NLL=float(-q.gather(1,y[:,None]).clamp_min(1e-12).log().mean()) if n else None,
        mean_confidence=float(conf.mean()) if n else None,ECE15=ece if n else None,ECE15_bins=bins,
        NLL_probability_floor=1e-12)


def _strata(record):
    result={}
    for label,key in (('RX','rx'),('day','day'),('TX','y')):
        result[label]={str(v):record[key]==v for v in record[key].unique(sorted=True).tolist()}
    return result


def readout(q,record):
    y=record['y']
    return dict(overall=_metrics(q,y),by_stratum={key:{label:_metrics(q[mask],y[mask]) for label,mask in values.items()} for key,values in _strata(record).items()})


def _pair(student,teacher,y,consensus=None):
    n=len(y);sc,sp=student.max(1);tc,tp=teacher.max(1)
    old=sp==y;new=tp==y;changed=sp!=tp;sh=sc>=THRESHOLD;th=tc>=THRESHOLD
    def subset(mask):
        count=int(mask.sum());disagree=mask&changed;changed_count=int(disagree.sum())
        accepted_error=int((mask&~new).sum())
        return dict(count=count,coverage=count/n if n else None,student=_metrics(student[mask],y[mask]),teacher=_metrics(teacher[mask],y[mask]),
            fixed=int((mask&~old&new).sum()),broken=int((mask&old&~new).sum()),net_corrections=int((mask&new).sum())-int((mask&old).sum()),
            persistent_error=int((mask&~old&~new).sum()),persistent_same_wrong_class=int((mask&~old&~new&~changed).sum()),
            high_confidence_original_errors=int((mask&sh&~old).sum()),high_confidence_original_errors_fixed=int((mask&sh&~old&new).sum()),
            changed_count=changed_count,changed_coverage=changed_count/n if n else None,
            teacher_accuracy_when_changed=float(new[disagree].double().mean()) if changed_count else None,
            student_accuracy_when_changed=float(old[disagree].double().mean()) if changed_count else None,
            selection_error_count=accepted_error,selection_risk=accepted_error/count if count else None)
    allmask=torch.ones(n,dtype=torch.bool)
    masks=dict(all=allmask,teacher_q095=th,teacher_changed=changed,teacher_q095_and_changed=th&changed)
    if consensus is not None:
        if consensus.shape!=(n,) or consensus.dtype!=torch.bool:raise ValueError('Bad consensus mask')
        masks.update(peer_consensus=consensus,peer_consensus_and_changed=consensus&changed)
    return dict(overall=subset(allmask),selections={name:subset(mask) for name,mask in masks.items()},
        student_initial_cohorts={name:dict(summary=subset(mask),selections={key:subset(mask&selection) for key,selection in masks.items()})
            for name,mask in dict(high_correct=sh&old,high_error=sh&~old,low_correct=~sh&old,low_error=~sh&~old).items()},
        coverage_denominator='all samples in this role/stratum; cohort count reported separately',threshold=THRESHOLD)


def paired(student,teacher,record,consensus=None):
    y=record['y']
    return dict(overall=_pair(student,teacher,y,consensus),
        by_stratum={key:{label:_pair(student[mask],teacher[mask],y[mask],consensus[mask] if consensus is not None else None)
            for label,mask in values.items()} for key,values in _strata(record).items()})


def _overlap(probabilities,y):
    errors={s:probabilities[s].argmax(1)!=y for s in SEEDS};n=len(y);rows=[]
    for left,right in combinations(SEEDS,2):
        a,b=errors[left],errors[right];both=int((a&b).sum());union=int((a|b).sum())
        rows.append(dict(left_seed=left,right_seed=right,left_errors=int(a.sum()),right_errors=int(b.sum()),
            both_wrong=both,either_wrong=union,error_jaccard=both/union if union else None,
            right_wrong_given_left_wrong=both/int(a.sum()) if a.any() else None,
            left_wrong_given_right_wrong=both/int(b.sum()) if b.any() else None,
            left_wrong_right_correct=int((a&~b).sum()),right_wrong_left_correct=int((b&~a).sum())))
    all_wrong=int(torch.stack(list(errors.values())).all(0).sum())
    return dict(count=n,pairwise=rows,all_three_wrong=all_wrong,at_least_one_correct=n-all_wrong,
        joint_oracle_accuracy_upper_bound=(n-all_wrong)/n if n else None,
        oracle_diagnostic_only=True,oracle_deployable=False)


@torch.no_grad()
def analyze(p0_by_seed):
    """{integer model seed: {'L': parent_record, 'V': parent_record}} -> JSON."""
    if set(p0_by_seed)!=set(SEEDS):raise ValueError('Exactly the three registered independent model seeds required')
    records={}
    for seed in SEEDS:
        roles=p0_by_seed[seed]
        if set(roles)!= {'L','V'}:raise ValueError('P0 source roles must be exactly L/V')
        records[seed]={role:_record(roles[role],role) for role in ('L','V')}
        if set(records[seed]['L']['ids'])&set(records[seed]['V']['ids']):raise ValueError('L/V physical IDs overlap')
    for role in ('L','V'):
        reference=records[SEEDS[0]][role]
        for seed in SEEDS[1:]:
            record=records[seed][role]
            if record['ids']!=reference['ids']:raise ValueError('Cross-seed ordered PID mismatch')
            if any(not torch.equal(record[k],reference[k]) for k in ('y','rx','day')):raise ValueError('Cross-seed truth/RX/day mismatch')
    result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',model_seeds=list(SEEDS),source_only=True,
        target_access=False,endpoint_variants_accessed=False,identity_updates=0,fitted_parameters=0,
        threshold=THRESHOLD,ECE15_edges=list(ECE_EDGES),ECE15_intervals='[lo,hi), final bin includes 1',
        consensus='the two other model seeds have identical argmax and EACH maximum probability >= 0.95',
        ALL3_interpretation='one probability-mean teacher; three student comparisons are paired, not independent replicates',roles={})
    for role in ('L','V'):
        record=records[SEEDS[0]][role];y=record['y'];q={s:records[s][role]['probabilities'] for s in SEEDS}
        all3=torch.stack([q[s] for s in SEEDS]).mean(0);students={}
        for student in SEEDS:
            peers=[s for s in SEEDS if s!=student];left,right=[q[s] for s in peers]
            lc,lp=left.max(1);rc,rp=right.max(1);consensus=(lp==rp)&(lc>=THRESHOLD)&(rc>=THRESHOLD)
            teacher=(left+right)/2
            students[str(student)]=dict(student_seed=student,own_P0=readout(q[student],record),
                LOO2=dict(teacher_seeds=peers,readout=readout(teacher,record),paired=paired(q[student],teacher,record,consensus)))
        result['roles'][role]=dict(count=len(y),students=students,
            ALL3=dict(teacher_count=1,teacher_seeds=list(SEEDS),readout=readout(all3,record),
                versus_students={str(s):paired(q[s],all3,record) for s in SEEDS}),
            error_overlap=dict(overall=_overlap(q,y),by_stratum={key:{label:_overlap({s:v[mask] for s,v in q.items()},y[mask])
                for label,mask in values.items()} for key,values in _strata(record).items()}))
    return result
