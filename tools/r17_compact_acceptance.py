"""User-confirmed R17 target accuracy allowance: one percentage point.

Consumes recomputed target aggregates only. It does not select models or open data.
"""
import math
from numbers import Integral

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')
ALLOWANCE_PP=1


def target_acceptance(candidate,original):
    n=candidate.get('count')
    if (isinstance(n,bool) or not isinstance(n,Integral) or n<=0 or original.get('count')!=n
            or candidate.get('role')!='target' or original.get('role')!='target'
            or set(candidate.get('views',{}))!=set(VIEWS) or set(original.get('views',{}))!=set(VIEWS)):
        raise ValueError('Matching complete target aggregates required; source is not target performance evidence')
    per_view={}
    for view in VIEWS:
        counts=[]
        for summary in (candidate,original):
            row=summary['views'][view];correct=row.get('correct')
            if (row.get('count')!=n or isinstance(correct,bool) or not isinstance(correct,Integral)
                    or not 0<=correct<=n or not math.isfinite(row.get('accuracy',float('nan')))
                    or not math.isclose(row['accuracy'],100*correct/n,rel_tol=0,abs_tol=1e-9)):
                raise ValueError('Exact integer correct counts and consistent target accuracy required')
            counts.append(int(correct))
        current,baseline=counts;delta=current-baseline
        # Integer cross-multiplication avoids rounded accuracy accepting one excess error.
        accepted=100*(baseline-current)<=ALLOWANCE_PP*n
        per_view[view]=dict(count=int(n),original_correct=baseline,candidate_correct=current,correct_delta=delta,
            accuracy_delta_pp=100*delta/n,accepted=accepted,zero_drop=delta>=0)
    return dict(status='TARGET_ACCEPTANCE_EVALUATED',metric='accuracy',reference='original_R17',
        allowance_pp=float(ALLOWANCE_PP),unit='percentage_points',scope='each_of_seven_target_views',
        maximum_correct_loss=int(n*ALLOWANCE_PP//100),accepted=all(r['accepted'] for r in per_view.values()),
        zero_drop=all(r['zero_drop'] for r in per_view.values()),views=per_view)
