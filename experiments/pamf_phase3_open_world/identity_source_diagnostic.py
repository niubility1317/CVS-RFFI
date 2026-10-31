"""Frozen source-L/OE identity linkage diagnostics, without fitting or queries."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from . import joint_identity_train as training
from . import joint_identity_model as joint
from . import star_dura_best_run as best
from . import star_representation_cache as IQ

RUN_ID='20261031-phase3-identity-source-diagnostic-s2026092701-r01'
TRAIN_RUN_ID='20261031-phase3-joint-identity-s2026092701-r01'
TRAIN_CODE='f540589c0478bed77a9fb4e664481c80ffe7eecc'
SCHEMA='pamf_phase3_identity_source_diagnostic_v1'
STATUS='IDENTITY_SOURCE_DIAGNOSTIC_COMPLETE'
FIELDS={'run_id','code_commit','train_config','output_root','batch_size'}
MODEL_NAMES=('F0','INIT','SOURCE')
COUNTS=dict(L=6300,OE=6134)
RX=(1,3,4,6,8)
CLASSES=('14-10','14-7','20-15','20-19','6-15','8-20')
PAIRS=((0,1),(0,2),(1,2))
_read,_write=best._read,best._write


def _source_train(c):return c['train_config']['reference_config']['reference_config']['reference_config']['train_config']


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact source-only diagnostic configuration required; no V/novel/U/Q/truth inputs')
    t=training.validate_config(c['train_config'])
    if t.get('run_id')!=TRAIN_RUN_ID or t.get('code_commit')!=TRAIN_CODE or t.get('steps')!=444:raise ValueError('Only the fixed completed f540 joint SOURCE444 parent is allowed')
    p=_source_train(c);out=Path(c['output_root']).resolve()
    roots=[t['output_root'],p['output_root'],p['IQ_root'],p['native_config']['source_output']]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New immutable diagnostic output cannot overlap source/model inputs')
    return c


def _backend(c,device):return training._backend(c['train_config'],device)


def _source_identity(i,c,b):
    expected=dict(run_id=TRAIN_RUN_ID,code_commit=TRAIN_CODE,arm='SOURCE_JOINT',updates=444,identity_head_updates=444,signal_parameters=335681,
        source_only=True,target_U_used=False,target_U_reads=0,identity_head_jointly_trained=True,identity_head_replaced=False,Sinc_frozen=True)
    if (any(i.get(k)!=v for k,v in expected.items()) or i.get('parent_signal_identity')!=b['model_identity']
            or i.get('parent_F0_identity')!=b['reader_identity'] or i.get('dura_identity')!=b['dura_identity']):
        raise ValueError('Exact source-only SOURCE_JOINT444 checkpoint and original F0/r03/G ancestors required')
    return i


def _snapshot(model):
    state=best._snapshot_model(model)
    return state,{k:v.detach().cpu().clone() for k,v in model.named_buffers()}


def _check(model,before):
    best._check_model(model,before[0]);buffers=dict(model.named_buffers())
    if set(buffers)!=set(before[1]) or any(not torch.equal(v,buffers[k].detach().cpu()) for k,v in before[1].items()):raise ValueError('Frozen runtime/nonpersistent model buffers changed')


def _smoke(b,device):return best._smoke(b,device)


def _load_role(root,role,b):
    if role not in COUNTS:raise ValueError('Only existing source L/OE P1 roles may be opened')
    folder=Path(root)/role;m=_read(folder/'manifest.json');n=COUNTS[role]
    expected=dict(schema=IQ.ROLE_SCHEMA,status='STAR_REPRESENTATION_ROLE_COMPLETE',role=role,count=n,iq_shape=[n,2,256],source_only=True,restored_input_path='P1',
        parent_F0_identity=b['reader_identity'],dura_identity=b['dura_identity'])
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Matching original source-only P1 role handle and F0/G identity required')
    ids=np.load(folder/'ids.npy',allow_pickle=False)
    with np.load(folder/'metadata.npz',allow_pickle=False) as z:meta={k:z[k].copy() for k in z.files}
    nkeys={'rx_i','day_i','domain'}|({'y'} if role=='L' else set())
    if (ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='') or set(meta)!=nkeys
            or any(v.shape!=(n,) or v.dtype!=np.int64 for v in meta.values()) or not np.isin(meta['rx_i'],RX).all() or not np.isin(meta['day_i'],(1,2,3)).all()
            or not np.array_equal(meta['domain'],np.searchsorted(RX,meta['rx_i'])*3+meta['day_i']-1) or set(meta['domain'].tolist())!=set(range(15))
            or (role=='L' and set(meta['y'].tolist())!=set(range(6)))):raise ValueError('Exact legal source labels/domains only; OE TX labels are forbidden')
    restored=np.load(folder/'restored.npy',allow_pickle=False,mmap_mode='r')
    if restored.dtype!=np.float32 or restored.shape!=(n,2,256):
        if hasattr(restored,'_mmap'):restored._mmap.close()
        raise ValueError('Readonly cached P1 FP32 full256 source IQ required')
    ids.setflags(write=False)
    for v in meta.values():v.setflags(write=False)
    return dict(manifest=m,ids=ids,metadata=meta,restored=restored)


def _encode(reader,init,source,values,device):
    a=np.asarray(values)
    if a.dtype!=np.float32 or a.shape!=(len(a),2,256) or not len(a) or not np.isfinite(a).all():raise ValueError('Finite cached source P1 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        x=torch.tensor(a.tolist(),device=device,dtype=torch.float32);f=reader.features(x);logits=[reader.classify_features(f)];raw=[f];z=[]
        for model in (init,source):
            v=model.features(x);head=model.signal.encoder.core.id_backbone.cls_head
            logits.append(type(head).classify(head,v['raw_fused160']));raw.append(v['raw_fused160']);z.append(v['z'])
    array=lambda values:np.stack([np.asarray(v.detach().cpu().tolist(),np.float32) for v in values],axis=1)
    raw,logits,z=array(raw),array(logits),array(z)
    if (raw.shape!=(len(a),3,160) or logits.shape!=(len(a),3,6) or z.shape!=(len(a),2,160)
            or not all(np.isfinite(v).all() for v in (raw,logits,z)) or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1.,rtol=0,atol=2e-6)):
        raise ValueError('Same-pass native raw160/logits6 and two open-unit160 outputs required')
    return dict(raw=raw,logits=logits,z=z)


def _unit(x):
    x=np.asarray(x,np.float64);n=np.linalg.norm(x,axis=-1,keepdims=True)
    return x/np.maximum(n,1e-12)


def _stats(a):
    a=np.asarray(a,np.float64)
    if not a.size:return None
    return dict(count=int(a.size),mean=float(a.mean()),std=float(a.std()),minimum=float(a.min()),maximum=float(a.max()),q05=float(np.quantile(a,.05)),median=float(np.median(a)),q95=float(np.quantile(a,.95)))


def _classification(logits,y,roster=tuple(range(6))):
    n=len(y)
    if not n:return dict(count=0,sample_accuracy=None,macro_accuracy=None,CE=None,macro_CE=None,coverage_complete=False,per_TX={})
    v=logits.astype(np.float64);maximum=v.max(1);ce=maximum+np.log(np.exp(v-maximum[:,None]).sum(1))-v[np.arange(n),y];correct=v.argmax(1)==y
    per={CLASSES[c]:dict(count=int((y==c).sum()),accuracy=float(correct[y==c].mean()) if np.any(y==c) else None,CE=float(ce[y==c].mean()) if np.any(y==c) else None) for c in roster}
    complete=all(r['count'] for r in per.values())
    return dict(count=n,sample_accuracy=float(correct.mean()),macro_accuracy=float(np.mean([r['accuracy'] for r in per.values()])) if complete else None,
        CE=float(ce.mean()),macro_CE=float(np.mean([r['CE'] for r in per.values()])) if complete else None,coverage_complete=bool(complete),per_TX=per)


def _pair_correct(closed,y,mask):
    result={}
    for old,new in PAIRS:
        a=closed[:,old]==y;b=closed[:,new]==y
        gain=int((mask&~a&b).sum());loss=int((mask&a&~b).sum())
        result[MODEL_NAMES[new]+'_vs_'+MODEL_NAMES[old]]=dict(count=int(mask.sum()),correct_gained=gain,correct_lost=loss,correct_net=gain-loss,closed_changed=int((mask&(closed[:,old]!=closed[:,new])).sum()))
    return result


def _confidence(logits):
    x=logits.astype(np.float64);m=x.max(1);logsum=m+np.log(np.exp(x-m[:,None]).sum(1));lp=x-logsum[:,None];prob=np.exp(lp);ordered=np.sort(x,axis=1)
    return dict(count=len(x),max_softmax=_stats(prob.max(1)),entropy=_stats(-(prob*lp).sum(1)),max_logit=_stats(m),logsumexp=_stats(logsum),top2_logit_margin=_stats(ordered[:,-1]-ordered[:,-2]),TX_labels_used=False)


def _geometry(values,y,rx,day):
    """Centroid geometry is descriptive arithmetic, not a fitted predictor."""
    x=_unit(values);cells=[];labels=[];receivers=[];days=[];class_means=[]
    for c in range(6):
        class_means.append(_unit(x[y==c].mean(0)))
        for r in RX:
            for d in (1,2,3):
                mask=(y==c)&(rx==r)&(day==d)
                if np.any(mask):cells.append(_unit(x[mask].mean(0)));labels.append(c);receivers.append(r);days.append(d)
    centers=np.asarray(cells);labels=np.asarray(labels);receivers=np.asarray(receivers);days=np.asarray(days)
    distance=np.clip(1-centers@centers.T,0,2);upper=np.triu(np.ones(distance.shape,bool),1);same=labels[:,None]==labels[None,:]
    both=(receivers[:,None]!=receivers[None,:])&(days[:,None]!=days[None,:])
    global_centers=np.asarray(class_means);between=np.clip(1-global_centers@global_centers.T,0,2)
    own=global_centers[y];record_d=np.clip(1-(x*own).sum(1),0,2)
    return dict(class_count=6,domain_centroid_count=len(cells),expected_domain_centroid_count=90,complete_90_groups=len(cells)==90,
        record_to_class_centroid_cosine_distance=_stats(record_d),same_TX_domain_centroid_distance=_stats(distance[upper&same]),
        same_TX_cross_RX_and_day_centroid_distance=_stats(distance[upper&same&both]),between_TX_centroid_cosine_distance=_stats(between[np.triu_indices(6,1)]),
        per_TX={CLASSES[c]:dict(record_count=int((y==c).sum()),domain_centroid_count=int((labels==c).sum()),record_to_class_centroid_cosine_distance=_stats(record_d[y==c]),
            same_TX_cross_RX_and_day_centroid_distance=_stats(distance[upper&same&both&(labels[:,None]==c)])) for c in range(6)},
        fitted_model=False,centroid_classifier_used=False,deployed=False)


def _role_summary(role,data,meta):
    logits,raw,z=data['logits'],data['raw'],data['z'];norm=np.linalg.norm(raw.astype(np.float64),axis=2);u=_unit(raw)
    cos=np.stack([np.einsum('nd,nd->n',u[:,a],u[:,b]) for a,b in PAIRS],axis=1).clip(-1,1)
    zcos=np.einsum('nd,nd->n',z[:,0].astype(np.float64),z[:,1].astype(np.float64)).clip(-1,1)
    drift={MODEL_NAMES[b]+'_vs_'+MODEL_NAMES[a]:dict(cosine=_stats(cos[:,j]),norm_difference=_stats(norm[:,b]-norm[:,a]),norm_ratio=_stats(norm[:,b]/np.maximum(norm[:,a],1e-12))) for j,(a,b) in enumerate(PAIRS)}
    result=dict(count=len(logits),raw_norm={name:_stats(norm[:,i]) for i,name in enumerate(MODEL_NAMES)},raw_drift=drift,z_SOURCE_vs_INIT_cosine=_stats(zcos))
    closed=logits.argmax(2).astype(np.int64)
    compact=dict(logits=logits,closed=closed,raw_cosines=cos,raw_norms=norm,z_SOURCE_vs_INIT_cosine=zcos,**meta)
    if role=='L':
        y=meta['y'];mask=np.ones(len(y),bool)
        group=lambda selected:dict(models={name:_classification(logits[selected,i],y[selected]) for i,name in enumerate(MODEL_NAMES)},paired_correct=_pair_correct(closed,y,selected))
        result.update(models={name:_classification(logits[:,i],y) for i,name in enumerate(MODEL_NAMES)},paired_correct=_pair_correct(closed,y,mask),
            by_TX={CLASSES[c]:dict(models={name:_classification(logits[y==c,i],y[y==c],(c,)) for i,name in enumerate(MODEL_NAMES)},paired_correct=_pair_correct(closed,y,y==c)) for c in range(6)},
            by_RX={str(r):group(meta['rx_i']==r) for r in RX},by_day={str(d):group(meta['day_i']==d) for d in (1,2,3)},by_scene=None,scene_status='metadata_missing',
            scene_reason='Original cached source-L metadata contains no scenario; no reconstruction or extra-role access',
            geometry={name+'.raw_fused160':_geometry(raw[:,i],y,meta['rx_i'],meta['day_i']) for i,name in enumerate(MODEL_NAMES)}|
                {name+'.z160':_geometry(z[:,i],y,meta['rx_i'],meta['day_i']) for i,name in enumerate(MODEL_NAMES[1:])})
    else:
        result.update(models={name:_confidence(logits[:,i]) for i,name in enumerate(MODEL_NAMES)},
            by_RX={str(r):{name:_confidence(logits[meta['rx_i']==r,i]) for i,name in enumerate(MODEL_NAMES)} for r in RX},
            by_day={str(d):{name:_confidence(logits[meta['day_i']==d,i]) for i,name in enumerate(MODEL_NAMES)} for d in (1,2,3)},TX_labels_used=False,TX_accuracy_computed=False)
    return result,compact


def _W_report(a,b):
    a,b=a.detach().double().cpu(),b.detach().double().cpu();cos=torch.nn.functional.cosine_similarity(a,b,dim=1)
    return dict(exact=torch.equal(a,b),difference_frobenius=float((a-b).norm()),relative_difference=float((a-b).norm()/a.norm().clamp_min(1e-12)),row_cosines=cos.tolist())


def diagnose(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source-diagnostic root required')
    b=_backend(c,device);source,identity=training.load_trained_model(c['train_config'],'SOURCE_JOINT',device);_source_identity(identity,c,b)
    reader=b['reader'];init=joint.JointIdentity(b['model'],reader).eval().requires_grad_(False)
    if (sum(p.numel() for p in reader.parameters())!=232993 or sum(p.numel() for p in b['model'].parameters())!=334721
            or any(sum(p.numel() for p in m.parameters())!=335681 for m in (init,source))):raise ValueError('Actual F0/r03/complete native-head student dimensions required')
    initial=init.state_dict();parent=b['model'].state_dict();key=joint.HEAD_WEIGHT
    if (set(initial)!=set(parent)|{key} or any(not torch.equal(initial[k],v) for k,v in parent.items())
            or not torch.equal(initial[key],reader.core.id_backbone.cls_head.weight)):raise ValueError('INIT must exactly retain r03 except for the same-source original F0 W')
    modules=[reader,b['model'],b['dura'][0],init,source];snapshots=[(m,_snapshot(m)) for m in modules]
    def check():
        for model,before in snapshots:_check(model,before)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    smoke=_smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual F0/r03/G synthetic checkpoint smoke must precede L/OE access')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    weights=[reader.core.id_backbone.cls_head.weight,init.signal.encoder.core.id_backbone.cls_head.weight,source.signal.encoder.core.id_backbone.cls_head.weight]
    summary=dict(schema=SCHEMA,run_id=RUN_ID,code_commit=c['code_commit'],model_names=list(MODEL_NAMES),W={MODEL_NAMES[b]+'_vs_'+MODEL_NAMES[a]:_W_report(weights[a],weights[b]) for a,b in PAIRS},
        original_F0_identity=b['reader_identity'],parent_r03_identity=b['model_identity'],source_model_identity=identity,dura_identity=b['dura_identity'],source_only=True,
        scientific_scope='Descriptive statistics on trained source L/OE roles; no new independent test, target performance, fitting, threshold or method selection')
    seen=set();handles={}
    for role in ('L','OE'):
        a=_load_role(_source_train(c)['IQ_root'],role,b)
        try:
            ids=set(a['ids'].tolist())
            if seen&ids:raise ValueError('Source L/OE physical roles overlap')
            seen.update(ids);handles[role]=a['manifest'];blocks=[]
            for start in range(0,len(a['ids']),128):
                blocks.append(_encode(reader,init,source,a['restored'][start:start+128],device));check()
                print('IDENTITY_SOURCE_DIAGNOSTIC_PROGRESS',role,min(start+128,len(a['ids'])),len(a['ids']),flush=True)
            data={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]};summary[role],rows=_role_summary(role,data,a['metadata'])
            best._save(out/(role+'_rows.npz'),dict(ids=a['ids'].copy(),**rows));check()
        finally:IQ.close_role(a)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,model_names=list(MODEL_NAMES),source_only=True,source_model_identity=identity,
        parent_F0_identity=b['reader_identity'],parent_r03_identity=b['model_identity'],dura_identity=b['dura_identity'],source_handles=handles,
        INIT_W_exact_F0=True,INIT_matches_r03_except_native_W=True,original_F0_unchanged=True,model_parameters_buffers_unchanged=True,
        source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],role_reads=['L','OE'],input_path='cached_P1_only',source_training_diagnostic=True,new_independent_records=0,
        real_F_forward_rows=sum(COUNTS.values()),real_signal_forward_rows=2*sum(COUNTS.values()),real_signal_forward_rows_by_model={name:sum(COUNTS.values()) for name in MODEL_NAMES[1:]},G_forward_rows=0,
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],
        inherited_SOURCE_joint_updates=444,inherited_original_F0_updates=11100,inherited_r03_updates=2220,optimizer_updates=0,fit_updates=0,threshold_updates=0,new_neural_updates=0,
        validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,target_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,OE_TX_labels_read=0,TARGET_JOINT_checkpoints_loaded=0,
        original_F0_parameters=232993,parent_r03_parameters=334721,INIT_parameters=335681,SOURCE_parameters=335681,
        L_scene_status='metadata_missing',summary_ref='summary.json',L_rows_ref='L_rows.npz',OE_rows_ref='OE_rows.npz',full_goal_status='ACTIVE',scientific_performance_conclusion=None,final_phase3_complete=False)
    check();_write(out/'summary.json',summary);_write(out/'complete.json',done);print(STATUS,flush=True);return done


def main():
    print(json.dumps(dict(event='IDENTITY_SOURCE_DIAGNOSTIC_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();diagnose(_read(a.config),a.device)


if __name__=='__main__':main()
