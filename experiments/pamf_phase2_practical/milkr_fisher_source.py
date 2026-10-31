"""Source-only strength selection for support between/within descriptor weights."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import subset
from .milkr import fisher_signature as method,support_calibrated as base


def calibrate(rt):
    from .milkr_cv_screen import source_episodes
    started=time.perf_counter();source=source_episodes(rt);prior=read(rt['prior_source_selection'])
    require(prior['status']=='SOURCE_FROZEN' and prior['target_reads']==0 and prior['query_reads']==0,'Source-only prior required')
    kernel=dict(prior['K1_rule']);strengths=rt['fisher_strengths'];require(tuple(strengths)==method.STRENGTHS,'Wrong strength grid')
    objectives={};byK={};validation=[]
    for strength in strengths:
        rule=dict(kernel,fisher_strength=strength);rows=[]
        for rx,episode,features in source:
            held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
            for k in (1,5,10,20):
                sup=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);y=np.repeat(np.arange(6),k)
                for sp in (0,1):
                    state=method.fit(subset(features[sp],sup),y,list(map(str,sup)),old_count=3,fixed=rule)
                    for qp in (0,1):
                        h=base.target_and_score(qy,state.score(subset(features[qp],held)).argmax(1),3)
                        rows.append(dict(strength=strength,rx=rx,episode=episode,k=k,S=sp,Q=qp,H=h))
        validation.extend(rows);byK[str(strength)]={str(k):float(np.mean([r['H'] for r in rows if r['k']==k])) for k in (1,5,10,20)}
        objectives[str(strength)]=float(np.mean(list(byK[str(strength)].values())))
        print('SOURCE_FISHER_STRENGTH',strength,objectives[str(strength)],flush=True)
    selected=max(strengths,key=lambda a:objectives[str(a)])
    gains={k:byK[str(selected)][k]-byK['0.0'][k] for k in ('1','5','10','20')};gain=objectives[str(selected)]-objectives['0.0']
    eligible=selected>0 and gain>=rt['source_min_H_gain'] and min(gains.values())>=-rt['source_max_perK_loss']
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=dict(kernel,fisher_strength=selected),control_K1_rule=dict(kernel,fisher_strength=0.),
        selected_fisher_strength=selected,strength_objectives=objectives,strength_H_byK=byK,support_policy_validation=validation,
        source_mean_H_gain=gain,source_byK_H_gain=gains,target_eligible=eligible,
        scientific_verdict='SOURCE_PROXY_PASS' if eligible else 'SOURCE_PROXY_GAIN_NOT_ESTABLISHED',
        seconds=time.perf_counter()-started,target_reads=0,query_reads=0,U_truth_reads=0,
        scope='known-six source proxy; support-label-derived weights; K1 exactly original R9'))
