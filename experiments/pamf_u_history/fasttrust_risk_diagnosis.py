"""One fixed candidate-set source falsification, before any new F training."""
import argparse
import json
from pathlib import Path
import numpy as np
from .fasttrust_diagnosis import probability,mass_set,set_quality
from .fasttrust_risk import risk_threshold,leave_receiver_out


def analyze(source,output):
    rows=[]
    with np.load(source,allow_pickle=False) as a:
        y=a['y'];rx=a['rx'];views=a['views'].tolist()
        if len(y)!=27000 or len(np.unique(a['ids']))!=27000 or set(rx.tolist())!={1,3,4,6,8}:
            raise ValueError('Full original source V only')
        indices=np.flatnonzero(a['methods']=='PLControl')
        if a['seeds'][indices].tolist()!=[2026092701,2026092702,2026092703]:raise ValueError('All three completed PL seeds required')
        for j in indices:
            p=probability(a['logits'][j,views.index('raw')]);sets=mass_set(p)
            size=sets.sum(1);eligible=(p.max(1)<.95)&(size>=2)&(size<=3)
            safe=sets[np.arange(len(y)),y];score=(p*sets).sum(1)
            selected,folds=leave_receiver_out(score,safe,eligible,rx,precision=.995,min_per_rx=20)
            threshold=risk_threshold(score,safe,eligible,rx,precision=.995,min_per_rx=20)
            quality=set_quality(sets,selected,y)
            passed=(quality['n']>=200 and all(r['selected']>=20 and r['precision_percent']>=99.5 for r in folds))
            rows.append(dict(seed=int(a['seeds'][j]),OOF=quality,folds=folds,full_source_frozen_threshold=threshold,
                             necessary_rule_passed=passed))
    result=dict(status='ANALYZED_SOURCE_CALIBRATION_ONLY',recipe=dict(mass=.95,raw_rejected_only=True,
        candidate_size=[2,3],score='probability mass inside proposed set',risk_target=.995,
        calibration='leave one source RX out; worst calibration RX must pass; threshold includes all tied scores',
        minimum_calibration_per_RX=20,minimum_OOF_total=200,minimum_OOF_per_RX=20),
        rows=rows,all_three_passed=all(r['necessary_rule_passed'] for r in rows),
        new_F_G_training=0,new_forward=0,U_truth_reads=0,target_reads=0,ordinary_PL_changed=False,
        interpretation='source threshold calibration allowed by PROJECT_PROTOCOL; no classifier/EMA/prototype/normalization update; internal cross-fit does not create new V roles',
        limitation='completed X3 PL source outputs; source empirical safety is not target coverage or accuracy guarantee')
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.source,a.output)
