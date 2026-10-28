"""Independent truth side. No model, source cache, feature fitting or feedback."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .io import read,write,require

def metrics(truth,predicted,classes):
    if not len(truth):return None
    cm=np.zeros((len(classes),len(classes)),np.int64)
    mapping={c:i for i,c in enumerate(classes)}
    for y,p in zip(truth,predicted):cm[mapping[y],p]+=1
    total=cm.sum(1);den=total+cm.sum(0)
    recalls=np.divide(cm.diagonal(),total,out=np.zeros(len(classes)),where=total!=0)
    f1=np.divide(2*cm.diagonal(),den,out=np.zeros(len(classes)),where=den!=0)
    return dict(count=int(cm.sum()),correct=int(cm.trace()),accuracy=float(cm.trace()/cm.sum()),macro_f1=float(f1[total>0].mean()),class_floor=float(recalls[total>0].min()),confusion=cm.tolist())

def run(capsule,truth_file,prediction,output,old_classes):
    prediction=Path(prediction);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    complete=read(prediction/'complete.json');m=read(Path(capsule)/'manifest.json')
    require(complete['status']=='PREDICTIONS_COMPLETE' and complete['capsule_id']==m['capsule_id'] and complete['truth_read'] is False,'Predictions must be fixed before truth')
    truth=read(truth_file);rows=[];seen=set();A={};B={}
    with (prediction/'predictions.jsonl').open(encoding='utf-8') as src:
        for line in src:
            r=json.loads(line);classes=r['registered_classes'];ids=r['query_ids'];pr=r['predicted_indices']
            require(len(ids)==len(pr) and len(ids)==len(set(ids)),'Prediction count mismatch')
            require((r['split_id'],r['stage']) not in seen,'Repeated prediction');seen.add((r['split_id'],r['stage']))
            labels=[truth[i]['transmitter'] for i in ids]
            require(all(y in classes for y in labels) and all(type(p) is int and 0<=p<len(classes) for p in pr),'Invalid prediction class coverage')
            row={k:r[k] for k in ('split_id','stage','receiver','scenario','k','support_seed')};row['classes']=len(classes)
            row['all']=metrics(labels,pr,classes)
            ix=[i for i,y in enumerate(labels) if y in old_classes];ni=[i for i,y in enumerate(labels) if y not in old_classes]
            row['old']=metrics([labels[i] for i in ix],[pr[i] for i in ix],classes)
            row['new']=metrics([labels[i] for i in ni],[pr[i] for i in ni],classes)
            if ni:
                a,b=row['old']['accuracy'],row['new']['accuracy'];row['H']=2*a*b/(a+b) if a+b else 0.
                row['old_restricted']=metrics([labels[i] for i in ix],[r['old_restricted_indices'][i] for i in ix],old_classes)
            rows.append(row)
    require(len(rows)==complete['predictions'],'Incomplete prediction artifact')
    for row in rows:
        if row['stage']=='A':A[(row['receiver'],row['scenario'])]=row['old']['accuracy']
        if row['stage']=='B':B[(row['receiver'],row['scenario'],row['k'],row['support_seed'])]=row['old']['accuracy']
    for row in rows:
        a=A.get((row['receiver'],row['scenario']));b=B.get((row['receiver'],row['scenario'],row['k'],row['support_seed']))
        if row['stage']=='B' and a is not None:row['DA_gain']=row['old']['accuracy']-a
        if row['stage']=='C' and a is not None and b is not None:
            row.update(DA_gain=b-a,registration_old_drop=b-row['old']['accuracy'],retained_gain=row['old']['accuracy']-a,
                       parameter_drift=b-row['old_restricted']['accuracy'],new_class_competition=row['old_restricted']['accuracy']-row['old']['accuracy'])
    write(root/'metrics.json',dict(binding=complete,rows=rows,truth_last=True,target_feedback=False))
    write(root/'complete.json',dict(status='SCORED_COMPLETE',predictions=len(rows),truth_last=True))
