"""Independent post-score clean correction/damage analysis by opaque ID."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prediction(folder, task):
    folder = Path(folder); done = read(folder/'complete.json')
    if done['status'] != 'ALL_PREDICTIONS_COMPLETE' or done['truth_read']:
        raise ValueError('Predictions not fixed before truth')
    with np.load(folder/task/'predictions.npz', allow_pickle=False) as f:
        return f['ids'].copy(), f['raw__clean'].copy()


def compare(ids, candidate, baseline, truths):
    if len(ids) != len(candidate) or len(ids) != len(baseline) or len(set(map(str, ids))) != len(ids):
        raise ValueError('Invalid prediction IDs or lengths')
    labels=np.asarray([truths[str(i)]['label'] for i in ids],dtype=np.int64)
    rx=np.asarray([str(truths[str(i)]['receiver']) for i in ids])
    day=np.asarray([str(truths[str(i)]['day']) for i in ids])
    right_new=candidate==labels;right_old=baseline==labels
    def stat(mask):
        gain=int((mask & ~right_old & right_new).sum())
        damage=int((mask & right_old & ~right_new).sum())
        return dict(count=int(mask.sum()),baseline_correct=int((mask & right_old).sum()),
                    candidate_correct=int((mask & right_new).sum()),gain=gain,damage=damage,net=gain-damage)
    out={'overall':stat(np.ones(len(ids),dtype=bool))}
    for name,values in (('label',labels.astype(str)),('receiver',rx),('day',day)):
        out[name]={str(v):stat(values==v) for v in sorted(set(values))}
    return out


def analyze(root):
    root=Path(root); state=read(root/'pipeline_state.json')
    if state['status']!='ANALYZED':
        raise ValueError('Wait for all registered predictions and independent scores')
    sources,controls=read(root/'sources.json'),read(root/'controls.json')
    runtime=read(root.parents[1]/'releases'/(root.name+'-release')/'runtime.json')
    context=read(runtime['context']); records=[]
    for rid,src in sources.items():
        if read(Path(src['scores'])/'complete.json')['status']!='SCORED_COMPLETE':
            raise ValueError('Candidate independently scored output missing')
        cfg=read(root/'configs'/('predict__'+rid+'.json'))
        setting=next(k for k in controls if rid.startswith(k+'-'))
        out={'row_id':rid,'setting':setting,'controls':{}}
        for control in ('pl','tri'):
            other=controls[setting][control]
            if read(Path(other['scores'])/'complete.json')['status']!='SCORED_COMPLETE':
                raise ValueError('Control score missing')
            per_task={}
            for task in ('cross_rx','cross_both'):
                ids,pred=prediction(src['predictions'],task)
                old_ids,old_pred=prediction(other['predictions'],task)
                if not np.array_equal(ids,old_ids):
                    raise ValueError('Exact physical ID ordering differs')
                truth=read(context['handles'][cfg['protocol_id']]['tests'][task]['truth'])
                per_task[task]=compare(ids,pred,old_pred,truth)
            total=Counter()
            for data in per_task.values():total.update(data['overall'])
            out['controls'][control]={'combined':dict(total),'per_test':per_task}
        records.append(out)
    return dict(status='ANALYZED_POST_SCORE_PAIRS',rows=records,
        claim_scope='selected seed and previously exposed test; exploratory; no training feedback')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.run_root),ensure_ascii=False,allow_nan=False))
