"""Synthetic complete-matrix checks; no formal data/checkpoint/query access."""
import copy
import statistics
import numpy as np
import pytest
from experiments.cvs_phase_curvature_clean.analyze import BASELINES,SEEDS,validate_results

CANDIDATE='phase_curvature_lag24'
RX=('1-1','14-7','2-1','20-1','7-14','7-7','8-8')

def metrics(cm):
    cm=np.asarray(cm);tp=np.diag(cm);n=int(cm.sum());den=cm.sum(0)+cm.sum(1)
    return dict(confusion=cm.tolist(),query_count=n,accuracy=float(tp.sum()/n),
        macro_accuracy=float((tp/cm.sum(1)).mean()),macro_f1=float((2*tp/den).mean()))

def fixture():
    records=[]
    for index,method in enumerate((*BASELINES,CANDIDATE)):
        for j,seed in enumerate(SEEDS):
            overall=np.zeros((6,6),dtype=np.int64)
            for k,rx in enumerate(RX):
                cm=np.eye(6,dtype=np.int64)*4000
                error=10*(index+j+k)
                for c in range(6):cm[c,c]-=error;cm[c,(c+1)%6]+=error
                overall+=cm
                records.append(dict(method=method,receiver=rx,model_seed=seed,view='clean',**metrics(cm)))
            records.append(dict(method=method,receiver='ALL',model_seed=seed,view='clean',**metrics(overall)))
    lookup={(r['method'],r['receiver'],r['model_seed']):r for r in records}
    summary=[];pairs=[]
    for method in (*BASELINES,CANDIDATE):
        for rx in ('ALL',*RX):
            row=dict(method=method,receiver=rx)
            for key in ('accuracy','macro_accuracy','macro_f1'):
                values=[lookup[(method,rx,seed)][key] for seed in SEEDS]
                row.update({key+'_mean':statistics.mean(values),key+'_seed_sd':statistics.stdev(values)})
            summary.append(row)
    for method in BASELINES:
        for rx in ('ALL',*RX):
            values=[100*(lookup[(CANDIDATE,rx,seed)]['accuracy']-lookup[(method,rx,seed)]['accuracy']) for seed in SEEDS]
            pairs.append(dict(candidate=CANDIDATE,baseline=method,receiver=rx,model_seeds=list(SEEDS),
                accuracy_delta_pp_by_seed=values,accuracy_delta_pp_mean=statistics.mean(values),
                accuracy_delta_pp_seed_sd=statistics.stdev(values),positive_seeds=sum(v>0 for v in values)))
    data=dict(marker=dict(status='SCORED_COMPLETE',rows=36,query_count=168000),results=dict(results=records),summary=dict(summary=summary,paired=pairs))
    selection=dict(selected_variant=CANDIDATE,new_candidate_selected=True,target_access=False,target_score_used=False)
    previous=dict(results=dict(results=copy.deepcopy([r for r in records if r['method'] in BASELINES])))
    return data,selection,previous

def test_complete288_matrix_recomputed_with_all32_controls():
    data,selection,previous=fixture();lookup,receivers=validate_results(data,selection,previous)
    assert len(lookup)==288 and set(receivers)==set(RX)

@pytest.mark.parametrize('case',('missing','duplicate','fractional_cm','metric','RX_partition','old_control','summary_mean','summary_sd','missing_summary','paired_seed','paired_sign','target_access','unselected','non_clean'))
def test_complete_analysis_rejects_corrupt_or_out_of_scope_results(case):
    data,selection,previous=fixture()
    if case=='missing':data['results']['results'].pop()
    elif case=='duplicate':data['results']['results'][-1]=copy.deepcopy(data['results']['results'][0])
    elif case=='fractional_cm':data['results']['results'][0]['confusion']=np.asarray(data['results']['results'][0]['confusion'],dtype=float).tolist()
    elif case=='metric':data['results']['results'][0]['macro_f1']+=.001
    elif case=='RX_partition':
        row=data['results']['results'][0];cm=np.asarray(row['confusion']);cm[0,0]-=1;cm[0,1]+=1;row.update(metrics(cm))
    elif case=='old_control':previous['results']['results'][0]['accuracy']+=.001
    elif case=='summary_mean':data['summary']['summary'][0]['accuracy_mean']+=.001
    elif case=='summary_sd':data['summary']['summary'][0]['macro_f1_seed_sd']+=.001
    elif case=='missing_summary':data['summary']['summary'].pop()
    elif case=='paired_seed':data['summary']['paired'][0]['model_seeds']=list(reversed(SEEDS))
    elif case=='paired_sign':data['summary']['paired'][0]['positive_seeds']=4
    elif case=='target_access':selection['target_access']=True
    elif case=='unselected':selection['new_candidate_selected']=False
    elif case=='non_clean':data['results']['results'][0]['view']='practical_mid'
    with pytest.raises(ValueError):validate_results(data,selection,previous)
