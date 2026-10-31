"""Select one descriptor task-normalization mode using only source L held IQ."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import subset
from .milkr import task_signature as method,signature_kernel as sig,support_calibrated as base


def calibrate(rt):
    from .milkr_cv_screen import source_episodes
    started=time.perf_counter();source=source_episodes(rt)
    modes=rt['task_modes'];require(tuple(modes)==method.MODES,'Wrong preregistered task-normalization grid')
    rules={};all_validation=[];mode_objective={}
    for mode in modes:
        choices=[]
        for config in sig.CONFIGS:
            for ridge in base.LAMBDAS:
                rule=dict(config=list(config),ridge=ridge,task_mode=mode);values=[]
                for rx,episode,features in source:
                    sup=np.asarray([c*52 for c in range(6)]);held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)])
                    qy=np.repeat(np.arange(6),32)
                    for sp in (0,1):
                        state=method.fit(subset(features[sp],sup),np.arange(6),list(map(str,sup)),old_count=3,fixed=rule)
                        for qp in (0,1):values.append(base.target_and_score(qy,state.score(subset(features[qp],held)).argmax(1),3))
                choices.append(dict(rule=rule,K1_H=float(np.mean(values))))
        best=max(range(len(choices)),key=lambda i:(choices[i]['K1_H'],choices[i]['rule']['ridge'],-i))
        rule=choices[best]['rule'];rules[mode]=dict(rule=rule,K1_H=choices[best]['K1_H']);validation=[]
        for rx,episode,features in source:
            held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
            for k in (5,10,20):
                sup=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);y=np.repeat(np.arange(6),k)
                for sp in (0,1):
                    state=method.fit(subset(features[sp],sup),y,list(map(str,sup)),old_count=3,fixed=rule)
                    for qp in (0,1):
                        h=base.target_and_score(qy,state.score(subset(features[qp],held)).argmax(1),3)
                        validation.append(dict(mode=mode,rx=rx,episode=episode,k=k,S=sp,Q=qp,H=h))
        all_validation.extend(validation)
        byK={str(k):float(np.mean([r['H'] for r in validation if r['k']==k])) for k in (5,10,20)}
        byK['1']=rules[mode]['K1_H'];rules[mode]['H_byK']=byK
        mode_objective[mode]=float(np.mean(list(byK.values())))
        print('SOURCE_TASK_MODE',mode,mode_objective[mode],flush=True)
    # Original R9 is first, so it wins an exact objective tie.
    selected=max(modes,key=lambda mode:mode_objective[mode])
    gain=mode_objective[selected]-mode_objective['none']
    gains={k:rules[selected]['H_byK'][k]-rules['none']['H_byK'][k] for k in ('1','5','10','20')}
    eligible=selected!='none' and gain>=rt['source_min_H_gain'] and min(gains.values())>=-rt['source_max_perK_loss']
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=rules[selected]['rule'],
        selected_task_mode=selected,mode_objectives=mode_objective,source_rules=rules,support_policy_validation=all_validation,
        source_mean_H_gain=gain,source_byK_H_gain=gains,target_eligible=eligible,
        scientific_verdict='SOURCE_PROXY_PASS' if eligible else 'SOURCE_PROXY_GAIN_NOT_ESTABLISHED',
        seconds=time.perf_counter()-started,target_reads=0,query_reads=0,U_truth_reads=0,
        scope='known-six source proxy; normalization fitted only on each current support, never held/query'))
