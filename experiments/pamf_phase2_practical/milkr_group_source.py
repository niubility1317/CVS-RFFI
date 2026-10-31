"""Source-L-only calibration of the R8 group offset before target access."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import subset
from .milkr import support_calibrated as backend,group_calibrated as method


def calibrate(rt):
    from .milkr_cv_screen import source_episodes
    started=time.perf_counter();source=source_episodes(rt)
    prior=read(rt['prior_source_selection'])
    require(prior['status']=='SOURCE_FROZEN' and prior['target_reads']==0 and prior['query_reads']==0,
            'Prior kernel rule must be source-only frozen')
    rule=dict(prior['K1_rule'])
    betas=rt['old_new_beta_candidates'];quality={str(b):[] for b in betas}
    for rx,episode,features in source:
        sup=np.asarray([c*52 for c in range(6)]);y=np.arange(6)
        held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
        for old_count in (2,3,4,5):
            for sp in (0,1):
                sx=subset(features[sp],sup);state=backend.fit(sx,y,list(map(str,sup)),old_count=old_count,fixed=rule)
                scale=method.score_scale(state.score(sx))
                for qp in (0,1):
                    scores=state.score(subset(features[qp],held))
                    for beta in betas:
                        delta=beta*np.log((6-old_count)/old_count)*scale
                        h=backend.target_and_score(qy,method.add_offset(scores,delta,old_count).argmax(1),old_count)
                        quality[str(beta)].append(h)
    objective={b:float(np.mean(values)) for b,values in quality.items()}
    beta=max(betas,key=lambda b:(objective[str(b)],-abs(b),-b));rule['old_new_beta']=beta
    validation=[]
    for rx,episode,features in source:
        held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
        for k in (5,10,20):
            sup=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);y=np.repeat(np.arange(6),k)
            for sp in (0,1):
                state=method.fit(subset(features[sp],sup),y,list(map(str,sup)),old_count=3,fixed=rule)
                delta=state.selection['group_calibration']['delta']
                for qp in (0,1):
                    scores=state.score(subset(features[qp],held));before=method.add_offset(scores,-delta,3)
                    h=backend.target_and_score(qy,scores.argmax(1),3);ref=backend.target_and_score(qy,before.argmax(1),3)
                    validation.append(dict(rx=rx,episode=episode,k=k,S=sp,Q=qp,H=h,uncalibrated_H=ref,delta_H=h-ref,offset=delta))
        print('SOURCE_GROUP_CALIBRATION',rx,episode,flush=True)
    gains={str(k):float(np.mean([r['delta_H'] for r in validation if r['k']==k])) for k in (5,10,20)}
    gain=float(np.mean([r['delta_H'] for r in validation]))
    eligible=gain>=rt['source_min_H_gain'] and min(gains.values())>=-rt['source_max_perK_loss']
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=rule,K1_beta_objective=objective,
        support_policy_validation=validation,source_mean_H_gain=gain,source_byK_H_gain=gains,target_eligible=eligible,
        scientific_verdict='SOURCE_PROXY_PASS' if eligible else 'SOURCE_PROXY_GAIN_NOT_ESTABLISHED',
        target_reads=0,query_reads=0,U_truth_reads=0,seconds=time.perf_counter()-started,
        scope='known six source classes only; source scientific screen, not genuine novel identity confirmation'))
