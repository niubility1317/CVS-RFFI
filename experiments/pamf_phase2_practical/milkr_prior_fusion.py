"""Three query paths with strict single-path support and immutable states."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize,subset
from .episodic_io import load_task
from .episodic_experiment import split_rows
from .teacher_features import extract,TeacherBridge,bridge_type
from .milkr.prior_fusion import fit,fuse,head_columns


def worker(rt,capname):
    import torch
    extract_features=extract
    fit_model=fit
    if rt.get('kernel_method')=='time_orbit':
        from .teacher_features import extract_time as extract_features
    elif rt.get('kernel_method')=='phase8_prior':
        from .teacher_features import extract_phase8 as extract_features
    elif rt.get('kernel_method') in ('orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble'):
        from .teacher_features import extract_orbit_ensemble as extract_features
        from .milkr.orbit_ensemble import fit as fit_model
    reader,engine=initialize(rt);before={key:value.detach().clone() for key,value in reader.state_dict().items()}
    g_before={key:value.detach().clone() for key,value in engine.pipeline.model.state_dict().items()}
    cap=Path(rt['data_root'])/capname;m=read(cap/'manifest.json');root=Path(rt['run_root'])/'predictions'/m['receiver'];root.mkdir(parents=True,exist_ok=False)
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))};old_names=specs[rt['K'][0],0]['registered_classes']
    columns=head_columns(read(rt['source_contract'])['classes'],old_names)
    require(all(spec['registered_classes'][:6]==old_names for spec in specs.values()),'Old class order changes across rows')
    bridge=(bridge_type(rt['kernel_method']) if rt.get('kernel_method') in ('native_skip','residual_skip','episodic_metric','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble') else TeacherBridge).load(Path(rt['run_root'])/'source_orbit_cca.npz');rule=read(Path(rt['run_root'])/'source_selection.json')['K1_rule']
    states=[];rows=[];started=time.perf_counter();arm=rt['arm']
    for k in rt['K']:
        for n in [0]+rt['N']:
            stage='C' if n else 'B';spec=specs[k,n];task=load_task(cap,spec['split_id'],stage=stage)
            for sp,g in enumerate((None,engine)):
                features,_=extract_features(reader,task['support_x'],'cuda:0',g,batch=k)
                features=bridge.transform(features,restored=bool(sp),old_columns=columns);begin=time.perf_counter();state=fit_model(features,task['support_y'],task['support_ids'],fixed=rule)
                write(root/f'{stage}-K{k}-N{n}-S{sp}-{arm}-fit.json',dict(selection=state.selection,config=state.config,ridge=state.ridge,tau=state.tau,
                    seconds=time.perf_counter()-begin,F_updates=0,G_updates=0,query_fit=False,B_function_inherited=False,
                    coefficient_bytes=state.alpha.nbytes,physical_labels=len(task['support_y']),arm=arm,head_columns=columns,support_paths=[sp]))
                states.append((spec,stage,sp,state))
        print('FIT_K_COMPLETE',m['receiver'],k,flush=True)
    bank=np.load(cap/'received.npy',mmap_mode='r');allrows=np.load(cap/'stage_rows/N20.npy');pos=np.full(len(bank),-1,np.int64);pos[allrows]=np.arange(len(allrows));raw_cache={};cache_bytes=0
    def save(spec,stage,sp,qp,query,scores):
        classes=spec['registered_classes'];order=np.asarray(sorted(range(len(classes)),key=lambda i:classes[i]));old=order[order<6]
        prediction=order[scores[:,order].argmax(1)];restricted=old[scores[:,old].argmax(1)]
        name=f'{stage}-K{spec["k"]}-N{spec["new_count"]}-S{sp}Q{qp}-{arm}.npz'
        np.savez_compressed(root/name,query_indices=query,predicted_indices=prediction.astype(np.int16),old_restricted_indices=restricted.astype(np.int16))
        rows.append(dict(split_id=spec['split_id'],stage=stage,arm=arm,input_path=f'S{sp}Q{qp}',prediction=name,k=spec['k'],new_count=spec['new_count'],registered_classes=classes,
            query_semantics={0:'raw',1:'DURA_only',2:'same_record_raw_DURA_gated_half_fusion'}[qp]))
    for qp,g in enumerate((None,engine)):
        features,_=extract_features(reader,np.asarray(bank[allrows]),'cuda:0',g);features=bridge.transform(features,restored=bool(qp),old_columns=columns)
        for spec,stage,sp,state in states:
            query=split_rows(cap,spec);positions=pos[query];require((positions>=0).all(),'Invalid query bank mapping');score_parts=[];distance_parts=[]
            for begin in range(0,len(positions),rt['decision_batch']):
                score,distance=state.evaluate(subset(features,positions[begin:begin+rt['decision_batch']]))
                score_parts.append(score);distance_parts.append(distance)
            scores=np.concatenate(score_parts);distance=np.concatenate(distance_parts);key=(stage,spec['k'],spec['new_count'],sp)
            save(spec,stage,sp,qp,query,scores)
            if qp==0:
                raw_cache[key]=(query,scores,distance);cache_bytes+=scores.nbytes+distance.nbytes
            else:
                q0,score0,d0=raw_cache.pop(key);require(np.array_equal(q0,query),'Fusion physical query alignment differs')
                fused,weight=fuse((score0,d0),(scores,distance));save(spec,stage,sp,2,query,fused)
                order=np.asarray(sorted(range(len(spec['registered_classes'])),key=lambda i:spec['registered_classes'][i]))
                decision0=order[score0[:,order].argmax(1)];decisionG=order[scores[:,order].argmax(1)];decisionF=order[fused[:,order].argmax(1)]
                write(root/f'{stage}-K{spec["k"]}-N{spec["new_count"]}-S{sp}-fusion.json',dict(query_count=len(query),rule=rule['fusion_policy'],
                    nonzero_G_weight_count=int((weight>0).sum()),mean_G_weight=float(weight.mean()),
                    raw_G_disagreement_count=int((decision0!=decisionG).sum()),changed_from_raw_count=int((decisionF!=decision0).sum()),
                    query_fit=False,query_truth_read=False,weight_values=[0.,.5]))
        print('QUERY_PATH_COMPLETE',m['receiver'],qp,'rows',len(rows),flush=True)
    require(not raw_cache,'Incomplete same-record fusion cache consumption')
    require(all(torch.equal(before[key],value) for key,value in reader.state_dict().items()),'Frozen F changed')
    require(all(torch.equal(g_before[key],value) for key,value in engine.pipeline.model.state_dict().items()),'Frozen G changed')
    expected=len(rt['K'])*(1+len(rt['N']))*6;require(len(rows)==expected,'Incomplete three-path matrix')
    write(root/'predictions.json',dict(capsule=capname,capsule_id=m['capsule_id'],profile='MILKR-function-prior-fusion',model_seed=rt['model_seeds'][0],rows=rows))
    write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=m['capsule_id'],truth_read=False,query_fit=False,rows=len(rows),seconds=time.perf_counter()-started,
        raw_score_cache_bytes=cache_bytes,raw_score_cache_final_rows=0,support_paths_strict=True,query_paths=['Q0_raw','Q1_DURA_only','Q2_fusion']))
