"""Support-only A/B/C runner; C independently refits MILKR on current support."""
from pathlib import Path
import json
import time
import numpy as np
from .io import read,write,require,validate_split

def load_capsule(root):
    root=Path(root);m=read(root/'manifest.json');ids=np.load(root/'ids.npy',allow_pickle=False)
    require(len(ids)==m['received_count'] and len(set(ids.tolist()))==len(ids),'Invalid received ID set')
    paths=sorted((root/'splits').glob('*.json'))
    require(len(paths)==m['split_count'],'Incomplete split matrix')
    splits=[validate_split(read(f),m,len(ids)) for f in paths]
    supports={i for s in splits for i in s['support_indices']};queries={i for s in splits for i in s['query_indices']}
    require(not supports&queries,'Physical query reused as support across tasks')
    return m,ids,splits

def groups(splits,old_classes):
    result={}
    for s in splits:
        require(s['registered_classes'][:len(old_classes)]==old_classes,'Old class order mismatch')
        key=(s['receiver'],s['scenario'],s['k'],s['support_seed']);n=len(s['registered_classes'])-len(old_classes)
        require(n not in result.setdefault(key,{}),'Duplicate task')
        result[key][n]=s
    for task in result.values():
        require(0 in task,'Missing B task')
        old=task[0]
        for n,s in task.items():
            sx=[i for i,y in zip(s['support_indices'],s['support_labels']) if y<len(old_classes)]
            require(sx==old['support_indices'] and set(old['query_indices']).issubset(s['query_indices']),'B/C physical IDs differ')
    return result

def run(rt,seed,input_path,arm,output,technical_only=False):
    from .milkr.d92_branch_local_ridge import fit_branch_local_ridge
    from .features import NAMES
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    m,ids,splits=load_capsule(Path(rt['data_root'])/'capsule')
    f=Path(rt['run_root'])/'features'/f's{seed}-{input_path}';marker=read(f/'complete.json')
    require(marker['capsule_id']==m['capsule_id'] and marker['model_seed']==seed and marker['input_path']==input_path,'Feature binding mismatch')
    require(np.array_equal(ids,np.load(f/'ids.npy',allow_pickle=False)),'Feature ID order mismatch')
    arrays={k:np.load(f/(k+'.npy'),mmap_mode='r') for k in NAMES};default=np.load(f/'default_prediction.npy',mmap_mode='r')
    tasks=groups(splits,rt['old_classes']);seen_A=set();count=0;started=time.monotonic()
    selected=list(sorted(tasks.items()))
    if technical_only:
        # Fixed before target access: first RX, MIX01, first draw, K=1 and max K.
        selected=[(k,v) for k,v in selected if k[0]==rt['target_receivers'][0] and k[1]=='MIX01' and k[3]==rt['support_seeds'][0] and k[2] in (1,max(rt['shots']))]
    write(root/'resolved_config.json',dict(model_seed=seed,input_path=input_path,arm=arm,technical_only=technical_only,
        capsule_id=m['capsule_id'],source_runtime_fit=False,query_fit=False,C_inherits_B_kernel=False))
    with (root/'predictions.jsonl').open('x',encoding='utf-8') as dst,(root/'fit.jsonl').open('x',encoding='utf-8') as fitlog:
        def save(s,stage,predicted,restricted=None):
            nonlocal count
            row={k:s[k] for k in ('split_id','receiver','scenario','k','support_seed','registered_classes')}
            row.update(stage=stage,query_ids=ids[s['query_indices']].tolist(),predicted_indices=np.asarray(predicted,dtype=int).tolist())
            if restricted is not None:row['old_restricted_indices']=np.asarray(restricted,dtype=int).tolist()
            dst.write(json.dumps(row)+'\n');dst.flush();count+=1
        for key,rows in selected:
            old=rows[0];akey=key[:2]
            if akey not in seen_A:
                save(old,'A',default[old['query_indices']]);seen_A.add(akey)
            for n,s in sorted(rows.items()):
                if technical_only and n not in (0,20):continue
                ix=s['support_indices'];query=s['query_indices'];classes=s['registered_classes']
                support={k:np.array(v[ix],copy=True) for k,v in arrays.items()}
                state=fit_branch_local_ridge(**support,support_labels=np.asarray(s['support_labels'],dtype=np.int64),support_ids=ids[ix].tolist(),classes=classes,old_classes=rt['old_classes'],arm=arm)
                require(list(state.classes)==classes,'Wrong output class order')
                audit=state.audit_dict()
                require(audit['optimizer_steps']==0 and audit.get('query_rows_used',0)==0,'Illegal fitting state')
                # Core score is strictly per-record, including all registered outputs.
                scores=state.score(**{k:np.asarray(v[query]) for k,v in arrays.items()})
                order=np.asarray(sorted(range(len(classes)),key=lambda j:classes[j]))
                predicted=order[np.argmax(scores[:,order],axis=1)]
                old_order=np.asarray(sorted(range(len(rt['old_classes'])),key=lambda j:classes[j]))
                restricted=old_order[np.argmax(scores[:,old_order],axis=1)] if n else None
                save(s,'B' if n==0 else 'C',predicted,restricted)
                fitlog.write(json.dumps(dict(split_id=s['split_id'],stage='B' if n==0 else 'C',support_count=len(ix),query_rows_used_for_fit=0,source_rows_used_for_fit=0,audit=audit))+'\n');fitlog.flush()
            print('TASK_COMPLETE',seed,input_path,arm,key,count,flush=True)
    expected=len(seen_A)+sum(sum(n in (0,20) for n in rows) if technical_only else len(rows) for _,rows in selected)
    require(count==expected and count>0,'Incomplete prediction matrix')
    write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',model_seed=seed,input_path=input_path,arm=arm,
          capsule_id=m['capsule_id'],predictions=count,tasks=len(selected),truth_read=False,query_fit=False,seconds=time.monotonic()-started))
