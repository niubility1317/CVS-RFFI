"""Independent scorer. Run only after acceptance and fixed predictions exist."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from runtime import validate_split

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); out=Path(args.output)
    if not (out/'acceptance.json').exists(): raise ValueError('No completed acceptance artifact')
    if (out/'metrics.json').exists(): raise FileExistsError('Scoring already exists')
    metadata=json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    if metadata.get('status')!='VERIFIED' or metadata.get('method') not in ('issl','lora'):
        raise ValueError('No completed registered method')
    with np.load(args.data,allow_pickle=False) as d, np.load(out/'predictions.npz',allow_pickle=False) as pred:
        roles=d['train']
        if roles.dtype!=np.bool_: raise ValueError('Role mask must be boolean')
        mask=~roles; truth=d['y'][mask]; ids=d['ids'][mask]
        validate_split(d['ids'][roles],ids)
        expected={'ids','A','C'} | ({'C_incremental_baseline'} if metadata['method']=='issl' else set())
        if set(pred.files)!=expected: raise ValueError('Incomplete or unexpected prediction stage set')
        if not np.array_equal(ids,pred['ids']): raise ValueError('Opaque ID order mismatch')
        if truth.dtype.kind not in 'iu' or set(truth.tolist())!=set(range(6)):
            raise ValueError('Expected query truth for all six registered classes')
        old=truth<3; new=~old
        result={}
        for stage in pred.files:
            if stage=='ids': continue
            y=pred[stage]
            if y.shape!=truth.shape: raise ValueError('Prediction coverage mismatch')
            registered=range(3) if stage=='A' else range(6)
            if y.dtype.kind not in 'iu' or not set(y.tolist())<=set(registered):
                raise ValueError('Invalid registered predicted class')
            if stage=='A':
                result[stage]={'old_accuracy':float(np.mean(y[old]==truth[old])),'new_accuracy':None,'H':None}
            else:
                if not set(y.tolist())<=set(range(6)): raise ValueError('Unregistered predicted class')
                oa=float(np.mean(y[old]==truth[old])); na=float(np.mean(y[new]==truth[new]))
                result[stage]={'old_accuracy':oa,'new_accuracy':na,'H':2*oa*na/(oa+na) if oa+na else 0.,
                               'macro_f1':float(f1_score(truth,y,labels=list(range(6)),average='macro',zero_division=0))}
        if metadata['method']=='lora': result['B']=result['A'].copy()
        else: result['B']={'old_accuracy':None,'new_accuracy':None,'H':None}
        train_counts=np.bincount(d['y'][roles],minlength=6)
        query_counts=np.bincount(truth,minlength=6)
        result.update(old_classes=3,new_classes=3,
                      K=int(train_counts[0]) if np.all(train_counts==train_counts[0]) else None,
                      query_per_class=int(query_counts[0]) if np.all(query_counts==query_counts[0]) else None,
                      train_counts=train_counts.tolist(),query_counts=query_counts.tolist(),
                      claim_scope='Real WiSig execution diagnostic only; not accuracy reproduction',
                      adaptation_gain_pp=None if result['B']['old_accuracy'] is None else 100*(result['B']['old_accuracy']-result['A']['old_accuracy']),
                      registration_old_drop_pp=None if result['B']['old_accuracy'] is None else 100*(result['B']['old_accuracy']-result['C']['old_accuracy']),
                      final_old_new_gap_pp=100*abs(result['C']['old_accuracy']-result['C']['new_accuracy']))
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
