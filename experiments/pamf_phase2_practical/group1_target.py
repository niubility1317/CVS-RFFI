"""Inductive AB / direct AC comparison and PAMF support-query DURA factorial."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from .io import read,write,require,activate
from .group1 import row,source_scope,workpoints,registration_anchor
from .group1_source import load_source,embedding
from .group1_resources import Measure,tensor_bytes
from .episodic_io import load_task
from .episodic_experiment import split_rows
from .features import branches,fft96,tensor,load_G
from .external_execution import infer,plain,seed_for
from . import external_native as native,external_additional as extra


def decision_bytes(state):
    value=state.get('native_state')
    if value is not None and not isinstance(value,dict):value={k:v for k,v in vars(value).items() if k!='model'}
    scores=state.get('score_fn')
    readout=vars(scores) if hasattr(scores,'__dict__') else {}
    return tensor_bytes(dict(native=value,readout=readout))


def fit_B(rt,bundle,x,y,seed,logger):
    profile=bundle['profile']
    require(row(profile)['category'] in ('DA','both'),'CIL cannot execute B')
    if profile in ('poster_ft','feature_separation_hu2024'):
        return extra.fit_B(rt,bundle,x,y,seed,logger)
    cfg=dict(bundle,profile='twostage_sda' if profile=='twostage' else profile)
    return native.fit_B(rt,cfg,x,y,seed,logger)


def fit_C(rt,bundle,x,y,new_ids,seed,logger):
    """Every call starts at A, never a previous target-adapted checkpoint."""
    profile=bundle['profile'];method=bundle['method']
    require(row(profile)['category']=='CIL','DA has no C in Group1')
    if method=='orthogonal':
        from .vendor_native.comparison_suite import registration as cil
        anchor=registration_anchor(bundle)
        cfg=dict(seed=seed,old_adaptation_epochs=0,orthogonal_epochs=50,orthogonal_lr=.08)
        state=cil.register_new_classes(anchor,x,y,new_ids,config=cfg,logger=logger)
        return dict(model=state.model,score_fn=lambda model,iq:state.score(iq),classes=state.classes,
                    resources=state.resource,native_state=state,profile=profile)
    if method=='lora_triplet_knn':
        model=copy.deepcopy(bundle['model']).eval().requires_grad_(False)
        old=bundle['native_state']
        with torch.no_grad():z=embedding(model,method,x).detach().cpu()
        gallery=torch.cat((old['source_enrollment_features'],z))
        labels=torch.cat((old['source_enrollment_y'],y.detach().cpu()))
        state=extra._metric_state(dict(bundle,model=model),gallery,labels)
        state['resources'].update(actual_target_old_labels=0,actual_target_new_labels=len(y),
            source_gallery_count=len(old['source_enrollment_y']),optimizer_steps=0,
            C_initialization='A_source_gallery_no_B')
        return state
    if method=='issl_author_behavior':
        from .external_acceleration import issl_ssl,issl_transfer
        extra._seed(seed);model=copy.deepcopy(bundle['model']).requires_grad_(True)
        count=6+len(new_ids);extra._issl_expand(model,count)
        replay=bundle['native_state'];rx,ry=replay['source_replay_x'],replay['source_replay_y']
        transfer_x=torch.cat((rx,x.detach().cpu()));transfer_y=torch.cat((ry,y.detach().cpu()))
        spec=extra.source_training_spec(method)
        _,ssl_seconds=_timed(lambda:issl_ssl(model,torch.cat((rx,transfer_x)),6,spec,seed,logger))
        _,transfer_seconds=_timed(lambda:issl_transfer(model,transfer_x,transfer_y,300,128,.03,seed,logger,crop=218))
        return dict(model=model.eval(),score_fn=extra.native_logits,classes=tuple(range(count)),profile=profile,
            resources=dict(C_initialization='A_no_B',C_SSL_epochs=2000,C_transfer_epochs=300,
                C_SSL_seconds=ssl_seconds,C_transfer_seconds=transfer_seconds,
                source_replay_count=len(rx),actual_target_old_labels=0,actual_target_new_labels=len(y)))
    raise NotImplementedError('Unbound native direct C: '+method)


def _features(reader,raw,device,batch=128):
    arrays={k:[] for k in ('z_id','t_emb','f_emb','pa_local','fft')};defaults=[]
    with torch.inference_mode():
        for i in range(0,len(raw),batch):
            iq=np.asarray(raw[i:i+batch],np.float32)
            values,logits=branches(reader,tensor(iq,device))
            for k,v in values.items():arrays[k].append(np.asarray(v.cpu().tolist(),np.float32))
            arrays['fft'].append(fft96(iq));defaults.extend(logits.argmax(1).cpu().tolist())
    return {k:np.concatenate(v) for k,v in arrays.items()},np.asarray(defaults,np.int16)


def _restore(engine,raw,device,batch=128):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(raw),batch):
            restored=engine(tensor(raw[i:i+batch],device))[0]
            out.append(np.asarray(restored.cpu().tolist(),np.float32))
    return np.concatenate(out)


def _timed(fn):
    if torch.cuda.is_available():torch.cuda.synchronize()
    begin=time.perf_counter();result=fn()
    if torch.cuda.is_available():torch.cuda.synchronize()
    return result,time.perf_counter()-begin


def predict(rt,job):
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    profile,seed=job['profile'],job['seed'];r=row(profile);device='cuda:0'
    bundle=load_source(source_scope(rt),profile,seed,device)
    model=bundle['model'];model.eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=False)
    cap=Path(rt['data_root'])/job['capsule'];manifest=read(cap/'manifest.json')
    require(manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE','Wrong data handles')
    specs=[read(p) for p in (cap/'splits').glob('*.json')];lookup={(s['k'],s['new_count']):s for s in specs}
    Ks,Cs=workpoints(manifest['episode_id']);rows=[];B_support_features={}
    if job.get('scope')=='pilot_main':Ks,Cs=[5],[(5,20)]
    elif job.get('scope')=='curve_extra':Ks,Cs=[k for k in Ks if k!=5],[(k,n) for k,n in Cs if (k,n)!=(5,20)]
    engine=load_G(rt,device) if r['method']=='pamf' else None
    log=(out/'native_fit.jsonl').open('x',encoding='utf-8')
    def emit(record):
        log.write(json.dumps(dict(observed_at=time.time(),**plain(record)),allow_nan=False)+'\n');log.flush()
    def save(s,stage,path,pred,restricted=None,resources=None,**metadata):
        name=f'{stage}-K{s["k"]}-N{s["new_count"]}-{path}.npz'
        payload=dict(query_indices=split_rows(cap,s),predicted_indices=pred)
        if restricted is not None:payload['old_restricted_indices']=restricted
        np.savez_compressed(out/name,**payload)
        rows.append(dict(split_id=s['split_id'],stage=stage,arm=profile,input_path=path,
            prediction=name,k=s['k'],new_count=s['new_count'],registered_classes=s['registered_classes'],
            resources=plain(resources or {}),**metadata))
        print('GROUP1_PREDICTED',profile,stage,path,s['k'],s['new_count'],len(pred),flush=True)
    def pamf_stage(s,stage):
        from .milkr.d92_branch_local_ridge import fit_branch_local_ridge
        from .milkr.cuda_local_ridge import CudaLocalRidge
        task=load_task(cap,s['split_id'],stage=stage);states={};scorers={};costs={}
        for support_path in (0,1):
            path=out/f'{stage}-K{s["k"]}-N{s["new_count"]}-S{support_path}-fit.json'
            with Measure(path,dict(profile=profile,seed=seed,stage=stage,support_restored=support_path,
                         K=s['k'],N=s['new_count'],RX=manifest['receiver']),model) as probe:
                raw=task['support_x'];gsec=0.
                # Class-aligned K batches preserve identical old features in B/C.
                if support_path:raw,gsec=_timed(lambda:_restore(engine,raw,device,batch=s['k']))
                (features,_),fsec=_timed(lambda:_features(bundle['reader'],raw,device,batch=s['k']))
                ids=task['support_ids'].tolist();labels=task['support_y']
                state,ksec=_timed(lambda:fit_branch_local_ridge(**features,support_labels=labels,
                    support_ids=ids,classes=s['registered_classes'],old_classes=rt['old_classes'],arm='local_ridge'))
                states[support_path]=state
                scorers[support_path]=CudaLocalRidge(state,device)
                if stage=='B':B_support_features[s['k'],support_path]=(features,task['support_ids'].copy())
                probe.result.update(G_support_seconds=gsec,F_support_seconds=fsec,kernel_fit_seconds=ksec,
                    persistent_head_bytes=state.audit['persistent_state_bytes'],actual_target_old_labels=6*s['k'],
                    F_parameters=232993,G_parameters_if_deployed=266951 if support_path else 0,
                    measurement_memory_scope='paired_factorial_process_F_G_two_heads_residency_not_single_cell_deployment_peak',
                    actual_target_new_labels=s['new_count']*s['k'],native_optimizer_steps=0,
                    C_initialization='A_frozen_F_full_refit' if stage=='C' else 'A_frozen_F',
                    B_kernel_inherited=False,AC_equals_ABC_terminal=stage=='C',
                    old_support_feature_reuse_for_BC='potential_deployment_reuse_reported_separately_from_cold_measurement')
            costs[support_path]=probe.result
            np.savez_compressed(out/f'{stage}-K{s["k"]}-N{s["new_count"]}-S{support_path}-state.npz',
                background=state.support_background,auxiliary=state.support_auxiliary,alpha=state.alpha,
                reference_kernel=state.reference_kernel,center_mean=state.center_mean)
            write(out/f'{stage}-K{s["k"]}-N{s["new_count"]}-S{support_path}-state.json',
                dict(reference_self=state.reference_self,center_grand=state.center_grand,
                    bandwidth_tau=state.bandwidth_tau,trace_scale=state.trace_scale,classes=list(state.classes)))
            if (stage=='C' and s['k']==5 and s['new_count']==20 and seed==rt['model_seeds'][0]
                    and job['capsule']==rt['capsules'][0]):
                old_features,old_ids=B_support_features[s['k'],support_path]
                oldmask=task['support_y']<6;require(np.array_equal(old_ids,task['support_ids'][oldmask]),'B/C old support differs')
                for repeat in range(3):
                    file=out/f'BC-C-increment-S{support_path}-repeat{repeat}.json'
                    with Measure(file,dict(profile=profile,stage='C_AFTER_B_INCREMENT_RESOURCE_ONLY',
                                 cached_old_support=True,support_restored=support_path,repeat=repeat,
                                 concurrency='shared_run_not_isolated'),model) as incremental:
                        new=task['support_x'][~oldmask];gtime=0.
                        if support_path:new,gtime=_timed(lambda:_restore(engine,new,device,batch=s['k']))
                        (nf,_),ftime=_timed(lambda:_features(bundle['reader'],new,device,batch=s['k']))
                        merged={name:np.concatenate((old_features[name],nf[name])) for name in features}
                        require(all(np.array_equal(merged[name],features[name]) for name in features),'AC/ABC canonical feature identity differs')
                        check,ktime=_timed(lambda:fit_branch_local_ridge(**merged,support_labels=labels,
                            support_ids=ids,classes=s['registered_classes'],old_classes=rt['old_classes'],arm='local_ridge'))
                        require(np.array_equal(check.alpha,state.alpha),'AC/ABC terminal coefficient identity differs')
                        incremental.result.update(G_new_support_seconds=gtime,F_new_support_seconds=ftime,
                            full_kernel_refit_seconds=ktime,native_optimizer_steps=0,target_old_labels_newly_requested=0,
                            target_old_labels_previously_used=6*s['k'],target_new_labels=len(new),
                            terminal_state_identical=True,used_for_query=False)
        for query_path in (0,1):
            pred={p:[] for p in (0,1)};restricted={p:[] for p in (0,1)};default=[]
            tie_start={p:scorers[p].cpu_tie_rows for p in scorers}
            file=out/f'{stage}-K{s["k"]}-N{s["new_count"]}-Q{query_path}-inference.json'
            with Measure(file,dict(stage='QUERY',profile=profile,query_restored=query_path,
                         query_count=len(task['query_x']),support_states_shared=2),model) as probe:
                gtime=ftime=dtime=0.
                for start in range(0,len(task['query_x']),rt.get('batch_size',128)):
                    raw=task['query_x'][start:start+rt.get('batch_size',128)]
                    if query_path:raw,elapsed=_timed(lambda:_restore(engine,raw,device));gtime+=elapsed
                    (features,sourcepred),elapsed=_timed(lambda:_features(bundle['reader'],raw,device));ftime+=elapsed
                    default.extend(sourcepred.tolist())
                    for p,state in states.items():
                        scores,elapsed=_timed(lambda:scorers[p].score(**features));dtime+=elapsed
                        pred[p].extend(scores.argmax(1).tolist());restricted[p].extend(scores[:,:6].argmax(1).tolist())
                probe.result.update(G_query_seconds=gtime,F_query_seconds=ftime,decision_two_states_seconds=dtime,
                    shared_features_between_support_states=True,query_updates=0,
                    decision_backend='CUDA_FP64_rank2',
                    numerical_tie_cpu_rows={p:scorers[p].cpu_tie_rows-tie_start[p] for p in scorers})
            for p in (0,1):
                save(s,stage,f'S{p}Q{query_path}',np.asarray(pred[p],np.int16),
                    np.asarray(restricted[p],np.int16) if s['new_count'] else None,
                    dict(fit=costs[p],shared_query=probe.result),AC_ABC_terminal_shared=stage=='C')
            if stage=='B':save(s,'A0',f'Q{query_path}',np.asarray(default,np.int16),resources=probe.result)
    try:
        if r['method']=='pamf':
            for k in Ks:pamf_stage(lookup[k,0],'B')
            for k,n in Cs:pamf_stage(lookup[k,n],'C')
        else:
            for k in Ks:
                s=lookup[k,0];task=load_task(cap,s['split_id'],stage='A')
                with Measure(out/f'A0-K{k}.json',dict(stage='A0_QUERY',profile=profile),model):
                    pred,_=infer(bundle,task['query_x'],6,rt.get('batch_size',128))
                save(s,'A0','P0',pred)
                if r['category']=='DA':
                    task=load_task(cap,s['split_id'],stage='B');sx=tensor(task['support_x'],device)
                    sy=torch.tensor(task['support_y'].tolist(),device=device,dtype=torch.long)
                    with Measure(out/f'B-K{k}-fit.json',dict(stage='B',profile=profile,K=k,
                                 actual_target_old_labels=len(sy),actual_target_new_labels=0),optimizers=True) as probe:
                        state=fit_B(rt,bundle,sx,sy,seed_for(seed,job['capsule'],k,'B'),emit)
                        probe.model=state['model'];probe.result.update(native_resources=plain(state.get('resources',{})),
                            persistent_decision_tensor_bytes=decision_bytes(state),
                            source_history_tensor_bytes=tensor_bytes(bundle.get('native_state',{})))
                    with Measure(out/f'B-K{k}-inference.json',dict(stage='QUERY',query_count=len(task['query_x'])),state['model']):
                        pred,_=infer(state,task['query_x'],6,rt.get('batch_size',128))
                    save(s,'B','P0',pred,resources=probe.result);del state,sx,sy
            if r['category']=='CIL':
                for k,n in Cs:
                    s=lookup[k,n];task=load_task(cap,s['split_id'],stage='C');keep=task['support_y']>=6
                    sx=tensor(task['support_x'][keep],device)
                    sy=torch.tensor(task['support_y'][keep].tolist(),device=device,dtype=torch.long)
                    with Measure(out/f'C-K{k}-N{n}-fit.json',dict(stage='C',profile=profile,K=k,N=n,
                                 actual_target_old_labels=0,actual_target_new_labels=len(sy),initialization='A_no_B'),optimizers=True) as probe:
                        state=fit_C(rt,bundle,sx,sy,tuple(range(6,6+n)),seed_for(seed,job['capsule'],k,n,'C'),emit)
                        probe.model=state['model'];probe.result.update(native_resources=plain(state.get('resources',{})),
                            persistent_decision_tensor_bytes=decision_bytes(state),
                            source_history_tensor_bytes=tensor_bytes(bundle.get('native_state',{})))
                    with Measure(out/f'C-K{k}-N{n}-inference.json',dict(stage='QUERY',query_count=len(task['query_x'])),state['model']):
                        pred,restricted=infer(state,task['query_x'],6+n,rt.get('batch_size',128))
                    save(s,'C','P0',pred,restricted,resources=probe.result);del state,sx,sy
        require(all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'Target job changed A anchor')
        write(out/'predictions.json',dict(capsule=job['capsule'],capsule_id=manifest['capsule_id'],
            profile=profile,model_seed=seed,rows=rows))
        write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=manifest['capsule_id'],
            rows=len(rows),truth_read=False,query_fit=False,profile=profile,model_seed=seed))
    finally:log.close()
