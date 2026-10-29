"""Complete native external methods and explicitly named B-to-C extensions.

No method receives a query tensor while fitting. Source jobs receive a runtime
without target data handles. Both input paths close before independent scoring.
"""
import copy
import json
from pathlib import Path
import time
import numpy as np
from .io import read,write,require,activate
from .episodic_io import load_task
from .episodic_experiment import split_rows
from .features import tensor


def bridge(name):
    from . import external_native,external_additional
    if name in external_native.profiles():return external_native
    if name in external_additional.profiles():return external_additional
    raise ValueError('Unknown external method '+name)


def source_runtime(rt):
    keys=('project','python','run_root','release_root','code_commit','source_contract','source_cache',
          'native_source_root','native_sources','additional_sources','native_source_audit','phase1_release','frozen_sources')
    return {k:rt[k] for k in keys if k in rt}


def plain(value):
    import torch
    if value is None or isinstance(value,(bool,int,float,str)):return value
    if isinstance(value,Path):return str(value)
    if torch.is_tensor(value):
        if value.numel()==1:return value.detach().cpu().item()
        return dict(shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist() if value.size<=32 else dict(shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items() if k not in ('model','source_contract','source_provenance','source_replay_x','source_replay_y','native_state')}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return type(value).__name__


def logger(path,context):
    f=Path(path).open('x',encoding='utf-8')
    def emit(row):
        value={**context,**plain(row)};f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()
        epoch=value.get('epoch',value.get('iteration',value.get('step',0)))
        if isinstance(epoch,int) and epoch%10==0:print('FIT',json.dumps(value),flush=True)
    return emit,f


def smoke_bundle(rt,bundle,device):
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));x=data.x[:2].to(device)
    model=bundle['model'];model.eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.inference_mode():z=bundle['score_fn'](model,x)
    require(z.shape==(2,6) and bool(torch.isfinite(z).all()),'Actual source checkpoint smoke failed')
    require(all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'Source-only smoke changed model')
    return dict(status='PASS',real_checkpoint=True,source_rows=2,target_rows=0,query_rows=0,
                model_parameters=sum(p.numel() for p in model.parameters()),class_count=6)


def source_job(rt,job):
    import torch
    cfg=source_runtime(rt);activate(cfg);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    module=bridge(job['profile']);out=Path(job['output']);out.mkdir(parents=True,exist_ok=False)
    log,f=logger(out/'source_metrics.jsonl',dict(profile=job['profile'],model_seed=job['seed'],stage='SOURCE'))
    try:
        if job['action']=='train':
            module.train_source(cfg,job['profile'],job['seed'],job['source_output'],'cuda:0',logger=log)
        bundle=module.load_source(cfg,job['profile'],job['seed'],'cuda:0')
        check=smoke_bundle(cfg,bundle,'cuda:0');write(out/'source_smoke.json',check)
        write(out/'provenance.json',plain(bundle.get('provenance',bundle.get('resources',{}))))
        write(out/'complete.json',dict(status='SOURCE_READY',profile=job['profile'],model_seed=job['seed'],action=job['action'],
              source_path=bundle.get('source_path',job['source_output']),source_only=True,target_reads=0,query_reads=0,
              source_smoke='PASS',source_contract=rt['source_contract']))
    finally:f.close()


def path_task(rt,cap,s,stage,path):
    task=load_task(cap,s['split_id'],stage=stage)
    query=split_rows(cap,s);task['query_indices']=query
    if path=='P1':
        base=Path(rt['shared_dura_root'])/str(Path(cap).relative_to(rt['data_root']))
        marker=read(base/'complete.json');require(marker['status']=='RESTORED_COMPLETE','Shared DURA incomplete')
        restored=np.load(base/'received.npy',mmap_mode='r',allow_pickle=False)
        require(restored.shape==(task['manifest']['received_count'],2,256) and marker['count']==len(restored),'Wrong shared DURA binding')
        task['query_x']=restored[query]
        if stage!='A':task['support_x']=restored[np.asarray(s['support_indices'])]
    else:require(path=='P0','Unknown input path')
    return task


def infer(state,iq,count,batch=128):
    import torch
    model=state['model'];model.eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
    require(tuple(state['classes'])==tuple(range(count)),'Wrong all-registered class columns')
    device=next(model.parameters()).device;pred=np.empty(len(iq),np.int16);restricted=np.empty(len(iq),np.int16)
    with torch.inference_mode():
        for start in range(0,len(iq),batch):
            values=state['score_fn'](model,tensor(iq[start:start+batch],device))
            require(values.shape==(min(batch,len(iq)-start),count) and bool(torch.isfinite(values).all()),'Invalid native query scores')
            pred[start:start+len(values)]=values.argmax(1).cpu().tolist()
            restricted[start:start+len(values)]=values[:,:6].argmax(1).cpu().tolist()
    require(all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'Query inference updated model/BN')
    return pred,restricted


def seed_for(*parts):
    from .data import identity
    return int(identity(list(parts))[:16],16)%(2**63-1)


def predict(rt,job):
    import torch
    from .external_controls import build_control
    activate(rt);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    profile,seed,path=job['profile'],job['seed'],job['input_path'];module=bridge(profile)
    bundle=module.load_source(source_runtime(rt),profile,seed,'cuda:0');bundle['model'].eval()
    device=next(bundle['model'].parameters()).device
    source_before={k:v.detach().clone() for k,v in bundle['model'].state_dict().items()}
    cap=Path(rt['data_root'])/job['capsule'];m=read(cap/'manifest.json');specs=[read(p) for p in (cap/'splits').glob('*.json')]
    lookup={(s['k'],s['new_count']):s for s in specs};root=Path(job['output']);root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[];p=rt['profile_matrix'][profile];chains=p['chains']
    emit,f=logger(root/'fit_metrics.jsonl',dict(profile=profile,model_seed=seed,input_path=path,capsule_id=m['capsule_id']))
    def save(s,stage,arm,task,state,chain_ids=(),nature='NATIVE_METHOD',resources=None):
        tic=time.monotonic();pred,restricted=infer(state,task['query_x'],len(s['registered_classes']),rt.get('batch_size',128))
        name=f'{stage}-K{s["k"]}-N{s["new_count"]}-{arm}.npz';payload=dict(query_indices=task['query_indices'],predicted_indices=pred)
        if s['new_count']:payload['old_restricted_indices']=restricted
        np.savez_compressed(root/name,**payload)
        rows.append(dict(split_id=s['split_id'],stage=stage,arm=arm,input_path=path,prediction=name,k=s['k'],new_count=s['new_count'],
            registered_classes=s['registered_classes'],query_count=len(pred),profile=profile,chain_ids=list(chain_ids),nature=nature,
            inference_seconds=time.monotonic()-tic,resources=plain(resources if resources is not None else state.get('resources',{}))))
        print('PREDICTED',profile,path,s['k'],s['new_count'],arm,len(pred),flush=True)
    try:
        for k in rt['shots']:
            old=lookup[k,0];a=path_task(rt,cap,old,'A',path)
            save(old,'A',profile,a,bundle,[c['id'] for c in chains],nature=p['source_nature']);del a
            oldtask=path_task(rt,cap,old,'B',path);sx=tensor(oldtask['support_x'],device);sy=torch.tensor(oldtask['support_y'].tolist(),device=device,dtype=torch.long)
            target_seed=seed_for(seed,p.get('source_id',profile),job['capsule'],k,'B')
            B=module.fit_B(rt,bundle,sx,sy,target_seed,logger=emit)
            before_B={name:v.detach().clone() for name,v in B['model'].state_dict().items()}
            save(old,'B',profile,oldtask,B,[c['id'] for c in chains],nature=p['B_nature'])
            bmeta=dict(classes=B['classes'],profile=profile,input_path=path,k=k,
                source_checkpoint=bundle.get('source_checkpoint',bundle.get('source_path')),resources=plain(B.get('resources',{})),
                C_initialization='deepcopy_actual_B_for_each_independent_prefix',query_fit=False)
            write(root/f'B-K{k}.json',bmeta)
            if job['capsule']==rt['capsules'][0] and seed==rt['model_seeds'][0]:
                torch.save(dict(bmeta,model=B['model'].state_dict()),root/f'B-K{k}.pt')
            del oldtask
            for n in rt['new_counts']:
                if n==0:continue
                if not p['native_C'] and not chains:continue
                s=lookup[k,n];task=path_task(rt,cap,s,'C',path)
                full_x=tensor(task['support_x'],device);full_y=torch.tensor(task['support_y'].tolist(),device=device,dtype=torch.long)
                for chain in chains:
                    cseed=seed_for(seed,p.get('source_id',profile),job['capsule'],k,n,
                                   'native_C' if chain['kind'] in ('native','calibration_append') else chain['kind'])
                    if chain['kind'] in ('native','calibration_append'):
                        keep=full_y>=6
                        C=module.native_C(rt,bundle,copy.deepcopy(B),full_x[keep],full_y[keep],list(range(6,6+n)),cseed,logger=emit)
                    else:
                        C=build_control(chain['kind'],B,full_x,full_y,tuple(range(6+n)),cseed,logger=emit)
                    save(s,'C',chain['id'],task,C,[chain['id']],nature=chain['nature']);del C
                if p.get('auxiliary_native_C'):
                    keep=full_y>=6
                    C=module.native_C(rt,bundle,copy.deepcopy(B),full_x[keep],full_y[keep],list(range(6,6+n)),seed_for(seed,p.get('source_id',profile),job['capsule'],k,n,'native_C'),logger=emit)
                    save(s,'C',profile,task,C,nature=p['native_C_nature']);del C
                del task,full_x,full_y
            require(all(torch.equal(before_B[name],v) for name,v in B['model'].state_dict().items()),'C mutated common B snapshot')
            del B,sx,sy
        require(all(torch.equal(source_before[name],v) for name,v in bundle['model'].state_dict().items()),'Target fitting mutated source anchor')
        expected=2*len(rt['shots'])+len(rt['shots'])*(len(rt['new_counts'])-1)*(len(chains)+int(p.get('auxiliary_native_C',False)))
        require(len(rows)==expected,'Incomplete native/extension prediction matrix')
        write(root/'predictions.json',dict(capsule_id=m['capsule_id'],capsule=job['capsule'],profile=profile,model_seed=seed,rows=rows))
        write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=m['capsule_id'],profile=profile,model_seed=seed,input_path=path,
              rows=len(rows),truth_read=False,query_fit=False,source_weights_unchanged=True,seconds=time.monotonic()-started))
    finally:f.close()


def score_pair(rt,job):
    """Join already fixed P0/P1 outputs, then enter the independent truth side."""
    from .episodic_score import run
    parent=Path(job['prediction']);payloads=[];capsules=[]
    for path in ('P0','P1'):
        folder=parent/path;marker=read(folder/'complete.json');require(marker['status']=='PREDICTIONS_COMPLETE' and marker['truth_read'] is False,'Both paths must close before truth')
        item=read(folder/'predictions.json');capsules.append(item['capsule_id'])
        for row in item['rows']:payloads.append(dict(row,prediction=path+'/'+row['prediction']))
    require(len(set(capsules))==1,'Paired path capsule mismatch')
    sample=read(parent/'P0/predictions.json')
    write(parent/'predictions.json',dict(capsule_id=capsules[0],capsule=sample['capsule'],profile=sample['profile'],model_seed=sample['model_seed'],rows=payloads))
    write(parent/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=capsules[0],rows=len(payloads),model_seed=sample['model_seed'],truth_read=False))
    run(rt,dict(job,prediction=str(parent)))
