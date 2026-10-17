"""Separate reader of every saved scalar; does not import observer/core/qualifier."""
from collections import Counter
from itertools import permutations
import math
import statistics
import time
from pathlib import Path
import numpy as np
from . import design as d


def run():
    started = time.perf_counter()
    base = Path(d.BASE)
    completion = d.read(base/'completion.json')
    saved = d.read(base/'result.json')
    metadata = d.read(base/'source_ids.json')
    contract = d.read(d.SOURCE)
    config = d.read(base/'resolved_config.json')['config']
    if config != d.config() or completion['status'] != 'ARTIFACTS_COMPLETE_SOURCE_ONLY':
        raise ValueError('Complete fixed source row required')
    if any(completion[k] for k in ('CP', 'NN', 'updates', 'U_used', 'U_truth_read', 'target_access', 'automatic_training')):
        raise ValueError('No model/U/target activity allowed')
    means, ranges, independent_cells = {}, {}, {}
    count = 0
    with np.load(base/'observations.npz', allow_pickle=False) as data:
        if set(data.files) != {role+'_'+key for role in ('L_s','V') for key in ('primary','raw','odd','even','coherence')}:
            raise ValueError('Exact complete saved scalar observations required')
        for role, n in (('L_s', 70), ('V', 300)):
            m = metadata[role]
            size = 90*n
            if any(len(m[k]) != size for k in ('ids','y','rx','day')) or len(set(m['ids'])) != size or set(m['ids']) != set(contract['role_ids'][role]):
                raise ValueError('Original complete role IDs required')
            if any(set(m['ids']) & set(contract['role_ids'][other]) for other in ('L_s','U_s','V') if other != role):
                raise ValueError('Disjoint physical roles required')
            expected = {(t,r,day):n for t in range(6) for r in d.RX for day in (1,2,3)}
            if Counter(zip(m['y'],m['rx'],m['day'])) != Counter(expected):
                raise ValueError('All90 cell metadata counts required')
            observations = {}
            for key in ('primary','raw','odd','even','coherence'):
                array = data[role+'_'+key]
                if array.shape != (size,) or not np.isfinite(array).all() or (array<0).any() or (array>1+1e-12).any():
                    raise ValueError('Complete finite bounded scalars required')
                observations[key] = array.tolist()
                ranges[role+'_'+key] = dict(minimum=float(array.min()), maximum=float(array.max()), mean=statistics.fmean(observations[key]))
            grouped = {k:[] for k in expected}
            for value, key in zip(observations['primary'],zip(m['y'],m['rx'],m['day'])):
                grouped[key].append(value)
            means[role] = {k:statistics.fmean(v) for k,v in grouped.items()}
            independent_cells[role] = [dict(TX=t,RX=r,day=day,count=len(grouped[(t,r,day)]),
                    mean=means[role][(t,r,day)],std=statistics.stdev(grouped[(t,r,day)]))
                    for r in d.RX for day in (1,2,3) for t in range(6)]
            count += size
    contrasts = {role:{(t,r,day):value-statistics.fmean(means[role][(t,rr,day)] for rr in d.RX)
                           for (t,r,day),value in means[role].items()} for role in ('L_s','V')}
    errors = []
    def close(a,b):
        if a is None or b is None:
            if a is not None or b is not None:raise ValueError('Undefined baseline mismatch')
        else:
            errors.append(abs(a-b))
            if abs(a-b)>1e-12:raise ValueError('Independent scalar recount mismatch')
    for role in ('L_s','V'):
        for actual, expected in zip(independent_cells[role],saved['cells'][role]):
            if any(actual[k] != expected[k] for k in ('TX','RX','day','count')):raise ValueError('Cell metadata differs')
            close(actual['mean'],expected['mean']);close(actual['std'],expected['std'])
    deranged = [p for p in permutations(range(5)) if all(i != j for i,j in enumerate(p))]
    wrong_sse = [0.]*44
    per_rx = {r:[0.,0.] for r in d.RX}
    folds = []
    for day in (1,2,3):
        for tx in range(6):
            prediction = [statistics.fmean(contrasts['L_s'][(t,r,date)] for t in range(6) for date in (1,2,3) if t!=tx and date!=day) for r in d.RX]
            actual = [contrasts['V'][(tx,r,day)] for r in d.RX]
            b = [x*x for x in actual];e=[(a-p)**2 for a,p in zip(actual,prediction)]
            bs,es = math.fsum(b),math.fsum(e)
            gain = 1-es/bs if bs>1e-16 else None
            expected=saved['folds'][len(folds)]
            if (expected['TX'],expected['day'],expected['fit_L_cells'],expected['evaluation_V_cells'])!=(tx,day,50,5):raise ValueError('Fold scope differs')
            for a,p,aa,pp in zip(actual,prediction,expected['V_RX_contrast'],expected['prediction']):close(a,aa);close(p,pp)
            close(bs,expected['baseline_SSE']);close(es,expected['RX_SSE']);close(gain,expected['risk_gain'])
            folds.append(dict(TX=tx,day=day,risk_gain=gain))
            for i,r in enumerate(d.RX):per_rx[r][0]+=b[i];per_rx[r][1]+=e[i]
            for i,order in enumerate(deranged):wrong_sse[i]+=math.fsum((actual[j]-prediction[order[j]])**2 for j in range(5))
    baseline=math.fsum(x[0] for x in per_rx.values());error=math.fsum(x[1] for x in per_rx.values())
    gain=1-error/baseline if baseline>1e-16 else None
    positive=sum(f['risk_gain'] is not None and f['risk_gain']>0 for f in folds)
    conditions=dict(nonzero_baseline=baseline>1e-16,relative_risk_gain=gain is not None and gain>=.20,
                    positive_folds=positive>=15,each_RX_nonnegative=all(e<=b+1e-20 for b,e in per_rx.values()))
    decision='SOURCE_ASSOCIATION_CONFIRMED_ONLY' if all(conditions.values()) else 'CLOSED_FIXED_L_TO_V_SOURCE_BASIS'
    close(baseline,saved['baseline_SSE']);close(error,saved['RX_SSE']);close(gain,saved['risk_gain'])
    for r,(b,e) in per_rx.items():
        close(b,saved['per_RX'][str(r)]['baseline_SSE']);close(e,saved['per_RX'][str(r)]['RX_SSE'])
        close(1-e/b if b>1e-16 else None,saved['per_RX'][str(r)]['risk_gain'])
    for order,sse,row in zip(deranged,wrong_sse,saved['wrong_RX']):
        if list(order)!=row['permutation']:raise ValueError('Derangement mismatch')
        close(sse,row['SSE']);close(1-sse/baseline if baseline>1e-16 else None,row['risk_gain'])
    if conditions!=saved['conditions'] or decision!=saved['decision'] or decision!=completion['decision'] or positive!=saved['positive_folds'] or len(saved['folds'])!=18 or len(saved['wrong_RX'])!=44:
        raise ValueError('Original source decision mismatch')
    result=dict(status='VERIFIED_COMPLETE_SCALAR_RECOUNT',run_id=d.RUN,decision=decision,conditions=conditions,
            IQ_descriptors=33300,scalar_values=count*5,cell_means=180,folds=18,wrong_RX_controls=44,
            risk_gain=gain,per_RX_risk_gain={str(r):1-e/b if b>1e-16 else None for r,(b,e) in per_rx.items()},
            max_error=max(errors),ranges=ranges,real_IQ=0,CP=0,NN=0,updates=0,U_truth_read=False,target_access=False,
            observer_recomputed_from_IQ=False,historical_verdict_changed=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-started)
    if (base/'recount.json').exists():raise FileExistsError(base/'recount.json')
    d.write(base/'recount.json',result)
    return result


if __name__=='__main__':
    import json
    print(json.dumps(run(),allow_nan=False))
