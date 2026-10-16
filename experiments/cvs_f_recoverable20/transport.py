"""Describe saved default-head error movement; does not select or train models."""
import numpy as np
NAMES={(True,True):'clean_both_correct',(True,False):'clean_lost',(False,True):'clean_repaired',(False,False):'clean_both_incorrect'}
def transport(predictions,candidate='C2',references=('C0','C1')):
    rows=[];c=predictions[candidate]
    for arm in references:
        b=predictions[arm];keys=set(b['clean'])
        if keys!=set(c['clean']):raise ValueError('Same physical clean observations required')
        subsets={name:[] for name in NAMES.values()}
        for k in sorted(keys):
            x,y=b['clean'][k],c['clean'][k]
            if x['label']!=y['label']:raise ValueError('Same source TX truth required')
            subsets[NAMES[x['prediction']==x['label'],y['prediction']==y['label']]].append(k)
        for scene in b:
            if keys!=set(b[scene]) or keys!=set(c[scene]):raise ValueError('Same physical seven-view observations required')
            for name,kk in subsets.items():
                n=len(kk);x,y=b[scene],c[scene]
                if any(x[k]['label']!=y[k]['label'] for k in kk):raise ValueError('Same source truth required')
                bc=sum(x[k]['label']==x[k]['prediction'] for k in kk);cc=sum(y[k]['label']==y[k]['prediction'] for k in kk)
                repaired=sum(x[k]['label']!=x[k]['prediction'] and y[k]['label']==y[k]['prediction'] for k in kk);lost=sum(x[k]['label']==x[k]['prediction'] and y[k]['label']!=y[k]['prediction'] for k in kk)
                if cc-bc!=repaired-lost:raise ValueError('Error transport must conserve correct counts')
                rows.append(dict(comparison=candidate+'-'+arm,scene=scene,clean_group=name,count=n,reference_correct=bc,candidate_correct=cc,repaired=repaired,broken=lost,net_correct=cc-bc,accuracy_gain_pp=100*(cc-bc)/n if n else None,reference_margin_over30=float(np.mean([x[k]['margin'] for k in kk])) if n else None,candidate_margin_over30=float(np.mean([y[k]['margin'] for k in kk])) if n else None))
    return rows

def checks():
    fixtures={a:{s:{} for s in ('clean','low')} for a in ('C0','C1','C2')}
    for i,(bc,cc,bs,cs) in enumerate((a,b,c,d) for a in (False,True) for b in (False,True) for c in (False,True) for d in (False,True)):
        for arm in fixtures:
            for s,correct in (('clean',cc if arm=='C2' else bc),('low',cs if arm=='C2' else bs)):
                fixtures[arm][s][i]=dict(label=0,prediction=0 if correct else 1,margin=1. if correct else -1.)
    rows=transport(fixtures)
    assert len(rows)==16 and all(r['count']==4 for r in rows)
    for scene in ('clean','low'):
        rr=[r for r in rows if r['scene']==scene and r['comparison']=='C2-C0'];assert sum(r['net_correct'] for r in rr)==0 and sum(r['count'] for r in rr)==16
    clean={r['clean_group']:r for r in rows if r['scene']=='clean' and r['comparison']=='C2-C0'}
    assert clean['clean_lost']['broken']==4 and clean['clean_repaired']['repaired']==4
    return dict(status='PASS',exhaustive_boolean_cases=16,IQ_checkpoint_network_access=0)
if __name__=='__main__':
    import json
    print(json.dumps(checks()))
