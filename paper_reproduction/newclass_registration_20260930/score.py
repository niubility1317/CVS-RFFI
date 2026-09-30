"""Independent scorer. Run only after acceptance and fixed predictions exist."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); out=Path(args.output)
    if not (out/'acceptance.json').exists(): raise ValueError('No completed acceptance artifact')
    if (out/'metrics.json').exists(): raise FileExistsError('Scoring already exists')
    with np.load(args.data,allow_pickle=False) as d, np.load(out/'predictions.npz',allow_pickle=False) as pred:
        mask=~d['train']; truth=d['y'][mask]; ids=d['ids'][mask]
        if not np.array_equal(ids,pred['ids']): raise ValueError('Opaque ID order mismatch')
        old=truth<3; new=~old
        result={}
        for stage in pred.files:
            if stage=='ids': continue
            y=pred[stage]
            if y.shape!=truth.shape: raise ValueError('Prediction coverage mismatch')
            if stage=='A':
                result[stage]={'old_accuracy':float(np.mean(y[old]==truth[old])),'new_accuracy':None,'H':None}
            else:
                if not set(y.tolist())<=set(range(6)): raise ValueError('Unregistered predicted class')
                oa=float(np.mean(y[old]==truth[old])); na=float(np.mean(y[new]==truth[new]))
                result[stage]={'old_accuracy':oa,'new_accuracy':na,'H':2*oa*na/(oa+na) if oa+na else 0.,
                               'macro_f1':float(f1_score(truth,y,labels=list(range(6)),average='macro',zero_division=0))}
        metadata=json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
        if metadata['method']=='lora': result['B']=result['A'].copy()
        else: result['B']={'old_accuracy':None,'new_accuracy':None,'H':None}
        result.update(old_classes=3,new_classes=3,K=128,query_per_class=64,claim_scope='Real WiSig execution diagnostic only; not accuracy reproduction',
                      adaptation_gain_pp=None if result['B']['old_accuracy'] is None else 100*(result['B']['old_accuracy']-result['A']['old_accuracy']),
                      registration_old_drop_pp=None if result['B']['old_accuracy'] is None else 100*(result['B']['old_accuracy']-result['C']['old_accuracy']),
                      final_old_new_gap_pp=100*abs(result['C']['old_accuracy']-result['C']['new_accuracy']))
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
