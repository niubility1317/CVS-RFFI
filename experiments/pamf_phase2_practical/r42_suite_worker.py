"""Missing-row execution for frozen R42 and the original LocalRidge control."""
import os
from pathlib import Path
import time
import numpy as np
from .io import read, write, require
from .milkr_screen import initialize, subset, extract as extract_baseline
from .features import tensor
from .episodic_io import load_task
from .episodic_experiment import split_rows, decide
from .teacher_features import AdaptiveResidualBridge, extract_orbit_ensemble
from .milkr.adaptive_residual import fit as fit_r42
from .milkr.prior_fusion import fuse, head_columns
from .milkr.d92_branch_local_ridge import fit_branch_local_ridge
from .group1_resources import Measure


def desired_rows(points):
    points=sorted(set(tuple(map(int,p)) for p in points))
    require(points and all(k in (1,5,10,20) and n in (2,5,10,20) for k,n in points),'Invalid suite work points')
    rows=[]
    def add(stage,k,n):
        for arm,paths in (('R42',range(3)),('LocalRidge',range(2))):
            rows.extend(dict(stage=stage,arm=arm,k=k,new_count=n,input_path=f'S{s}Q{q}') for s in (0,1) for q in paths)
    for k in sorted({p[0] for p in points}):
        add('B',k,0)
        rows.extend(dict(stage='A0',arm='LocalRidge',k=k,new_count=0,input_path=f'Q{q}') for q in (0,1))
    for k,n in points:add('C',k,n)
    return rows


def row_key(row):return row['arm'],row['stage'],int(row['k']),int(row['new_count']),row['input_path']


def _source_path(value):return value.replace('P0','Q0').replace('P1','Q1')


def _reuse(entries,wanted,specs,manifest,job,root):
    rows={};headers={}
    for entry in entries:
        key=tuple(entry['key'])
        require(key in wanted and key not in rows,'Unexpected or duplicate reuse row')
        desired=wanted[key];directory=Path(entry['prediction_dir']);given=entry['row']
        if str(directory) not in headers:
            index=read(directory/'predictions.json');done=read(directory/'complete.json')
            require(index['capsule']==job['capsule'] and index['capsule_id']==manifest['capsule_id'] and index['model_seed']==job['seed'],'Reuse capsule/seed mismatch')
            require(done['status']=='PREDICTIONS_COMPLETE' and done.get('truth_read') is False and done.get('query_fit') is False,'Reuse predictions not frozen')
            require(done['capsule_id']==manifest['capsule_id'],'Reuse completion capsule mismatch')
            headers[str(directory)]=index
        index=headers[str(directory)]
        matches=[r for r in index['rows'] if r['prediction']==given['prediction']]
        require(len(matches)==1 and matches[0]==given,'Reuse row differs from source prediction index')
        source=matches[0];spec=specs[desired['k'],desired['new_count']]
        for field in ('split_id','registered_classes','k','new_count'):
            require(source[field]==spec[field],'Reuse split/class/K/N mismatch: '+field)
        stage='A0' if source['stage'] in ('A','A0') else source['stage']
        source_path=source['input_path'] if stage=='A0' else _source_path(source['input_path'])
        require(stage==desired['stage'] and source_path==desired['input_path'],'Reuse stage/path mismatch')
        allowed=('R42',) if desired['arm']=='R42' else ('LocalRidge','local_ridge','default_head','pamf_tri_sig_time_ssl_milkr')
        require(source['arm'] in allowed and (source['arm']!='default_head' or stage=='A0'),'Reuse method mismatch')
        name=_name(desired);require((directory/source['prediction']).is_file(),'Reuse prediction missing')
        os.link(directory/source['prediction'],root/name)
        rows[key]=dict(source,**desired,prediction=name,original_arm=source['arm'],
                       reused_from=dict(prediction_dir=str(directory),prediction=source['prediction'],metrics_file=entry.get('metrics_file')))
    return rows


def _name(row):return f"{row['arm']}-{row['stage']}-K{row['k']}-N{row['new_count']}-{row['input_path']}.npz"


def array_bytes(value,seen=None):
    """Deployment arrays only; selection/audit records are not model state."""
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value))
    if isinstance(value,np.ndarray):return value.nbytes
    if isinstance(value,dict):return sum(array_bytes(v,seen) for k,v in value.items() if k not in ('selection','audit','training_summary','correlation_summary'))
    if isinstance(value,(tuple,list)):return sum(array_bytes(v,seen) for v in value)
    if hasattr(value,'__dict__'):return array_bytes(vars(value),seen)
    return 0


def _timed(function,device):
    import torch
    cuda=device.startswith('cuda') and torch.cuda.is_available()
    if cuda:torch.cuda.synchronize()
    begin=time.perf_counter();result=function()
    if cuda:torch.cuda.synchronize()
    return result,time.perf_counter()-begin


def _restore(engine,raw,device,batch,fast_transfer=False):
    import torch
    if fast_transfer:
        from .fast_transfer import to_tensor,to_numpy
        input_tensor=lambda value:to_tensor(value,device,np.float32)
        output_array=lambda value:to_numpy(value,np.float32)
    else:
        input_tensor=lambda value:tensor(value,device)
        output_array=lambda value:np.asarray(value.cpu().tolist(),np.float32)
    parts=[]
    with torch.inference_mode():
        for start in range(0,len(raw),batch):
            value=engine(input_tensor(np.asarray(raw[start:start+batch])))[0]
            parts.append(output_array(value))
    return np.concatenate(parts)


def _decisions(scores,classes):
    scores=np.asarray(scores);require(scores.ndim==2 and scores.shape[1]==len(classes) and np.isfinite(scores).all(),'Invalid suite all-class scores')
    order=np.asarray(sorted(range(len(classes)),key=lambda i:classes[i]));old=order[order<6]
    return order[scores[:,order].argmax(1)],old[scores[:,old].argmax(1)]


def local_ridge_scorer(state, device, enabled, pair_budget=2048, fast_transfer=False):
    if not enabled:
        return state
    from .milkr.cuda_local_ridge import CudaLocalRidge
    return CudaLocalRidge(state, device, pair_budget=pair_budget, fast_transfer=fast_transfer)


def orbit_features(reader, iq, device, batch, fast_transfer):
    if not fast_transfer:
        return extract_orbit_ensemble(reader,iq,device,None,batch=batch)
    from .phase_orbit import extract,ensemble_moments
    features,logits=extract(reader,iq,device,None,batch,aggregate=ensemble_moments,
                            return_phase_logits=True,phase_eighths=True,fast_transfer=True)
    return dict(features,teacher_phase_logits=logits),logits[:,0,:]


def baseline_features(reader, iq, device, batch, fast_transfer):
    if fast_transfer:
        return extract_baseline(reader,iq,device,None,batch=batch,fast_transfer=True)
    return extract_baseline(reader,iq,device,None,batch=batch)


def run(rt,job):
    import torch
    started=time.perf_counter();require(rt['model_seeds']==[job['seed']],'Suite F seed mismatch')
    cap=Path(rt['data_root'])/job['capsule'];manifest=read(cap/'manifest.json');rx=manifest['receiver']
    root=Path(rt['run_root'])/'predictions'/rx;root.mkdir(parents=True,exist_ok=False)
    desired=desired_rows(job['points']);wanted={row_key(r):r for r in desired}
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    require(all((r['k'],r['new_count']) in specs for r in desired),'Missing suite split')
    entries=read(job['reuse_file']) if job.get('reuse_file') else []
    rows=_reuse(entries,wanted,specs,manifest,job,root);reused=len(rows);states=[];query_times=[]
    missing=set(wanted)-set(rows);device=rt.get('device','cuda:0');batch=rt.get('batch_size',128);decision_batch=rt.get('decision_batch',512)
    fast_transfer=rt.get('fast_dlpack_transfer',False);pair_budget=rt.get('local_ridge_pair_budget',2048)
    reader=engine=before=g_before=bridge=None
    if missing:
        reader,engine=initialize(rt)
        before={k:v.detach().clone() for k,v in reader.state_dict().items()}
        g_before={k:v.detach().clone() for k,v in engine.pipeline.model.state_dict().items()}
        old_names=specs[min(r['k'] for r in desired),0]['registered_classes']
        columns=head_columns(read(rt['source_contract'])['classes'],old_names)
        require(all(specs[r['k'],r['new_count']]['registered_classes'][:6]==old_names for r in desired),'Suite old class order differs')
        if any(key[0]=='R42' for key in missing):
            bridge=AdaptiveResidualBridge.load(Path(rt['run_root'])/'source_orbit_cca.npz')
            rule=read(Path(rt['run_root'])/'source_selection.json')['K1_rule']
        # All current-SUP states freeze before the query feature bank is opened.
        state_keys=sorted({(r['arm'],r['stage'],r['k'],r['new_count'],int(r['input_path'][1])) for key,r in wanted.items() if key in missing and r['stage']!='A0'})
        for arm,stage,k,n,sp in state_keys:
            spec=specs[k,n];task=load_task(cap,spec['split_id'],stage=stage)
            context=dict(stage=stage,arm=arm,k=k,new_count=n,support_path=sp,physical_labels=k*(6+n),
                         native_optimizer_steps=0,F_updates=0,G_updates=0,query_fit=False,
                         cost_scope='cold_current_SUP_extraction_transform_and_fit; no SUP feature cache',support_paths=[sp])
            tag=f'{arm}-{stage}-K{k}-N{n}-S{sp}'
            with Measure(root/(tag+'-resources.json'),context=context,model=reader) as measure:
                iq=task['support_x'];g_seconds=0.
                if sp:iq,g_seconds=_timed(lambda:_restore(engine,iq,device,k,fast_transfer=True) if fast_transfer else _restore(engine,iq,device,k),device)
                if arm=='R42':
                    (features,_),f_seconds=_timed(lambda:orbit_features(reader,iq,device,k,fast_transfer),device)
                    mapped,nn_seconds=_timed(lambda:bridge.transform(features,restored=bool(sp),old_columns=columns),device)
                    state,fit_seconds=_timed(lambda:fit_r42(mapped,task['support_y'],task['support_ids'],fixed=rule),device)
                    audit=state.selection
                else:
                    (features,_),f_seconds=_timed(lambda:baseline_features(reader,iq,device,k,fast_transfer),device);nn_seconds=0.
                    state,fit_seconds=_timed(lambda:fit_branch_local_ridge(**features,support_labels=np.asarray(task['support_y'],np.int64),
                        support_ids=list(map(str,task['support_ids'])),classes=spec['registered_classes'],old_classes=old_names,arm='local_ridge'),device)
                    require(list(state.classes)==spec['registered_classes'],'LocalRidge class order differs');audit=state.audit_dict()
                state_bytes=array_bytes(state)
                measure.result.update(G_seconds=g_seconds,F_feature_seconds=f_seconds,source_encoder_seconds=nn_seconds,head_fit_seconds=fit_seconds,head_state_array_bytes=state_bytes)
            write(root/(tag+'-fit.json'),dict(selection=audit,config=getattr(state,'config',None),ridge=getattr(state,'ridge',1.),tau=getattr(state,'tau',getattr(state,'bandwidth_tau',None)),
                  physical_labels=k*(6+n),head_state_array_bytes=state_bytes,F_updates=0,G_updates=0,optimizer_updates=0,query_fit=False,C_inherits_B_kernel=False,support_paths=[sp]))
            states.append((arm,stage,k,n,sp,state));del task,features
            if arm=='R42':del mapped
        bank=np.load(cap/'received.npy',mmap_mode='r');allrows=np.load(cap/'stage_rows/N20.npy')
        positions=np.full(len(bank),-1,np.int64);positions[allrows]=np.arange(len(allrows));raw_cache={};local_heads={}
        def save(row,query,predicted,restricted=None):
            key=row_key(row)
            if key in rows:return
            name=_name(row);require(not (root/name).exists(),'Suite prediction overwrite')
            payload=dict(query_indices=query,predicted_indices=np.asarray(predicted,np.int16))
            if restricted is not None:payload['old_restricted_indices']=np.asarray(restricted,np.int16)
            np.savez_compressed(root/name,**payload);spec=specs[row['k'],row['new_count']]
            rows[key]=dict(row,split_id=spec['split_id'],registered_classes=spec['registered_classes'],query_count=len(query),prediction=name)
        def needs(arm,stage,k,n,sp,q):
            return (arm,stage,k,n,f'S{sp}Q{q}') in missing or (arm=='R42' and (arm,stage,k,n,f'S{sp}Q2') in missing)
        for qp in (0,1):
            active=[s for s in states if needs(*s[:5],qp)]
            a0=[r for key,r in wanted.items() if key in missing and r['stage']=='A0' and r['input_path']==f'Q{qp}']
            if not active and not a0:continue
            use_r42=any(s[0]=='R42' for s in active);use_native=bool(a0) or any(s[0]=='LocalRidge' for s in active)
            context=dict(stage='QUERY',query_path=qp,native_optimizer_steps=0,F_updates=0,G_updates=0,query_fit=False,
                         shared_scope='same canonical processed IQ shared across active methods; G cost recorded once',query_bank_rows=len(allrows))
            with Measure(root/f'query-Q{qp}-resources.json',context=context,model=reader) as measure:
                iq=np.asarray(bank[allrows]);g_seconds=0.
                if qp:iq,g_seconds=_timed(lambda:_restore(engine,iq,device,batch,fast_transfer=True) if fast_transfer else _restore(engine,iq,device,batch),device)
                orbit=base_features=logits=None;f_orbit=f_native=nn_seconds=head_seconds=0.
                if use_r42:
                    (orbit,_),f_orbit=_timed(lambda:orbit_features(reader,iq,device,batch,fast_transfer),device)
                    orbit,nn_seconds=_timed(lambda:bridge.transform(orbit,restored=bool(qp),old_columns=columns),device)
                if use_native:(base_features,logits),f_native=_timed(lambda:baseline_features(reader,iq,device,batch,fast_transfer),device)
                for row in a0:
                    spec=specs[row['k'],0];query=split_rows(cap,spec);pos=positions[query];require((pos>=0).all(),'A0 query bank mapping differs')
                    (pred,restricted),elapsed=_timed(lambda:_decisions(logits[pos][:,columns],spec['registered_classes']),device)
                    head_seconds+=elapsed;save(row,query,pred,restricted)
                for arm,stage,k,n,sp,state in active:
                    spec=specs[k,n];query=split_rows(cap,spec);pos=positions[query];require((pos>=0).all(),'Suite query bank mapping differs')
                    row=dict(arm=arm,stage=stage,k=k,new_count=n,input_path=f'S{sp}Q{qp}')
                    if arm=='LocalRidge':
                        def evaluate_local():
                            key=(stage,k,n,sp)
                            if key not in local_heads:
                                local_heads[key]=local_ridge_scorer(state,device,rt.get('local_ridge_cuda',False),pair_budget,fast_transfer)
                            return decide(local_heads[key],subset(base_features,pos),spec['registered_classes'],decision_batch)
                        (pred,restricted),elapsed=_timed(evaluate_local,device)
                        head_seconds+=elapsed;save(row,query,pred,restricted)
                    else:
                        def evaluate():
                            pieces=[state.evaluate(subset(orbit,pos[start:start+decision_batch])) for start in range(0,len(pos),decision_batch)]
                            return np.concatenate([p[0] for p in pieces]),np.concatenate([p[1] for p in pieces])
                        (scores,distance),elapsed=_timed(evaluate,device);head_seconds+=elapsed
                        pred,restricted=_decisions(scores,spec['registered_classes']);save(row,query,pred,restricted)
                        state_key=(stage,k,n,sp)
                        if ('R42',stage,k,n,f'S{sp}Q2') in missing:
                            if qp==0:raw_cache[state_key]=(query,scores,distance)
                            else:
                                q0,score0,d0=raw_cache.pop(state_key);require(np.array_equal(q0,query),'Fusion physical query alignment differs')
                                (fused,weight),elapsed=_timed(lambda:fuse((score0,d0),(scores,distance)),device);head_seconds+=elapsed
                                pred,restricted=_decisions(fused,spec['registered_classes']);save(dict(row,input_path=f'S{sp}Q2'),query,pred,restricted)
                                write(root/f'R42-{stage}-K{k}-N{n}-S{sp}-fusion.json',dict(query_count=len(query),rule=rule['fusion_policy'],
                                      nonzero_G_weight_count=int((weight>0).sum()),mean_G_weight=float(weight.mean()),query_fit=False,query_truth_read=False))
                values=dict(G_seconds=g_seconds,F_orbit_feature_seconds=f_orbit,F_native_feature_seconds=f_native,source_encoder_seconds=nn_seconds,head_seconds=head_seconds,
                            F_orbit_scope='8 frozen RF forwards and FFT, original 4-phase head payload',F_native_scope='one original F/default-head forward per record and FFT',
                            source_encoder_scope='frozen CCA/NN plus same-path BASE projections',
                            G_cost_scope='once for the shared processed query bank' if qp else 'not requested for raw queries')
                measure.result.update(values);query_times.append(dict(query_path=qp,**values))
            del iq,orbit,base_features,logits
        require(not raw_cache,'Suite fusion cache not consumed')
        require(all(torch.equal(before[k],v) for k,v in reader.state_dict().items()),'Frozen suite F changed')
        require(all(torch.equal(g_before[k],v) for k,v in engine.pipeline.model.state_dict().items()),'Frozen suite G changed')
    require(set(rows)==set(wanted),'Incomplete desired suite rows')
    ordered=[rows[row_key(r)] for r in desired]
    write(root/'predictions.json',dict(capsule=job['capsule'],capsule_id=manifest['capsule_id'],model_seed=job['seed'],rows=ordered))
    write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=manifest['capsule_id'],model_seed=job['seed'],rows=len(ordered),
          truth_read=False,query_fit=False,F_updates=0,G_updates=0,raw_score_cache_final_rows=0,support_paths_strict=True,
          new_prediction_rows=len(rows)-reused,reused_prediction_rows=reused,seconds=time.perf_counter()-started,query_stage_times=query_times))
