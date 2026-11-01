"""Frozen source-V mechanism comparison; independent CPU truth-last scoring."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import covariance_trial as physical
from . import covariance_feature_trial as guided
from .covariance_model import scratch_covariance
from .covariance_inverse import dense_action
from .source_inverse import R07,SCENES
from .source_pair import read,validate_source
from .source_validation import validation_roles
from .cache import tensor,load_F0,check_F0_state
from .variance_trial import load_scale,recovery_gpu_choice
from .main_split_eval import valid_identity
from .cost_probe import write

RUN='20261102-diagnostic-Restore-covariance-guidance-sourceV-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_guidance_source'
SPECS={'physical':(physical,'bf29554a03e28a012eaaa134f5c7b3f7856f2268'),
       'guided':(guided,'5359a1b82b1fa5d640a394c3e3a77ce3b945fe1e')}
METHODS=('received','physical','guided')
METRICS=('iq_nmse','feature_cosine_distance','feature_standardized_error','optimized_feature_objective',
         'Fclean_reference_margin','Fclean_margin_change','clean_decision_agreement','prediction')


def validate_parent_metadata(done,method):
    module,commit=SPECS[method]
    expected=dict(status='SOURCE_TRAINED',code_commit=commit,arm=module.ARM,epoch=30,updates=7410,
                  G_parameters=1230341,training_stages=1,checkpoint_selection='final30_only')
    if any(done.get(k)!=v for k,v in expected.items()) or any(done.get(k)!=0 for k in [
            'G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):
        raise ValueError('Exact original source-only final30 parent required')
    if done.get('F0_identity',{}).get('source_code_commit')!='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d':
        raise ValueError('Original ground Tri-SIG-TIME SSL required')


def load_parent(root,method,device):
    module,commit=SPECS[method];parent=root/'runs'/module.RUN
    if read(parent/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Complete parent Main required')
    folder=parent/'rows'/module.ARM/'training';done=read(folder/'complete.json');validate_parent_metadata(done,method)
    p=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    if (set(p)!={'config','code_commit','epoch','updates','state'} or p['config']!=module.configuration(module.ARM)
            or (p['code_commit'],p['epoch'],p['updates'])!=(commit,30,7410)):
        raise ValueError('Exact parent configuration and source final checkpoint required')
    G=scratch_covariance(2026092701);expected=G.state_dict()
    if set(p['state'])!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=torch.float32 or not torch.isfinite(v).all() for k,v in p['state'].items()):
        raise ValueError('Exact finite FP32 parent state required')
    G.load_state_dict(p['state'],strict=True)
    return G.to(device).eval().requires_grad_(False),done['F0_identity']


def source_metrics(restored,clean,features,logits,target,clean_logits,scale):
    reference=clean_logits.argmax(1);prediction=logits.argmax(1)
    def margin(value):
        chosen=value.gather(1,reference[:,None]).squeeze(1)
        other=value.clone();other.scatter_(1,reference[:,None],-torch.inf)
        return chosen-other.max(1).values
    cosine=1-torch.nn.functional.cosine_similarity(features,target,dim=1,eps=1e-8)
    standardized=((features-target).square()/scale).mean(1)
    return dict(iq_nmse=(restored-clean).square().mean((1,2))/clean.square().mean((1,2)).clamp_min(1e-4),
        feature_cosine_distance=cosine,feature_standardized_error=standardized,
        optimized_feature_objective=.01*(cosine+standardized),Fclean_reference_margin=margin(logits),
        Fclean_margin_change=margin(logits)-margin(clean_logits),clean_decision_agreement=(prediction==reference).float(),prediction=prediction)


def numpy_metrics(restored,clean,features,logits,target,clean_logits,scale):
    restored,clean,features,logits,target,clean_logits,scale=[np.asarray(x,np.float64) for x in [restored,clean,features,logits,target,clean_logits,scale]]
    reference=clean_logits.argmax(1);prediction=logits.argmax(1);ix=np.arange(len(reference))
    def margin(value):
        other=value.copy();other[ix,reference]=-np.inf
        return value[ix,reference]-other.max(1)
    cosine=1-(features*target).sum(1)/(np.maximum(np.linalg.norm(features,axis=1),1e-8)*np.maximum(np.linalg.norm(target,axis=1),1e-8))
    standardized=((features-target)**2/scale).mean(1)
    return dict(iq_nmse=((restored-clean)**2).mean((1,2))/np.maximum((clean**2).mean((1,2)),1e-4),
        feature_cosine_distance=cosine,feature_standardized_error=standardized,
        optimized_feature_objective=.01*(cosine+standardized),Fclean_reference_margin=margin(logits),
        Fclean_margin_change=margin(logits)-margin(clean_logits),clean_decision_agreement=(prediction==reference).astype(float),prediction=prediction)


def worker(args):
    start=time.perf_counter();torch.set_num_threads(2);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/'runs'/RUN;old=args.root/'runs'/R07/'cache';folder=old/'validation'
    meta=read(old/'complete.json');manifest=read(folder/'manifest.json')
    if (meta['status'],meta['train_count'],meta['source_V_count'],meta['epochs'],meta['TX_labels_read'],meta['target_reads'])!=('ACTION_SOURCE_CACHE_COMPLETE',63000,1800,20,0,0):raise ValueError('Original validated R07 cache required')
    if manifest['count']!=1800 or manifest['roles']!={'V':1800} or manifest['label_reads']!=0:raise ValueError('Fixed original source V required')
    ids,roles,rx,day=[np.load(folder/(k+'.npy'),allow_pickle=False) for k in ['physical_ids','roles','rx','day']]
    validation_roles(ids,roles,rx,day,np.load(old/'train/physical_ids.npy',allow_pickle=False))
    # Construct both original G models before the historical F0 package bootstrap.
    parents={m:load_parent(args.root,m,device) for m in SPECS};models={m:v[0] for m,v in parents.items()}
    if any(not torch.equal(models['physical'].state_dict()[k],models['guided'].state_dict()[k]) for k in ['prior_mean','prior_covariance']):raise ValueError('Same fixed source prior required')
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);valid_identity(identity)
    if any(v[1]!=identity for v in parents.values()):raise ValueError('Exact same frozen F0 identity required')
    snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in [F,*models.values()]]
    train_roles=np.load(old/'train/roles.npy',allow_pickle=False);take=np.flatnonzero(train_roles=='L_s')[:2]
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:source2=tensor(a[take],device)
    finally:a._mmap.close()
    with torch.inference_mode():
        for G in models.values():
            value,aux=G(source2,return_aux=True);logits=F.classify_features(F.features(value))
            if logits.shape!=(2,6) or not torch.isfinite(logits).all() or not torch.isfinite(value).all() or aux['relative_solve_residual'].max()>1e-8:raise ValueError('Actual frozen G/F0 source smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_Ls=2,both_final30=True,actual_original_F0=True,optimizer_updates=0,TX_labels_read=0,target_inputs=0))
    print('COVARIANCE_GUIDANCE_REAL_SOURCE_SMOKE_PASS',flush=True)
    a=np.load(folder/'clean.npy',mmap_mode='r',allow_pickle=False)
    try:clean=tensor(a,device)
    finally:a._mmap.close()
    scale=load_scale(args.root).to(device)
    with torch.inference_mode():
        target=torch.cat([F.features(clean[i:i+32]) for i in range(0,1800,32)])
        clean_logits=F.classify_features(target)
    def array(x):return np.asarray(x.cpu().tolist(),dtype=np.float32)
    with (out/'references.npz').open('xb') as f:np.savez(f,ids=ids,rx=rx,day=day,clean=array(clean),features=array(target),logits=array(clean_logits),scale=array(scale))
    for scene in SCENES:
        path=old/'validation_actions'/scene;done=read(path/'complete.json')
        if (done['count'],done['TX_labels_read'],done['target_reads'])!=(1800,0,0):raise ValueError('Fixed existing source V action trace required')
        arrays={k:np.load(path/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['received','kernel','coefficients','variance']}
        records={'ids':ids};lists={m+'__'+k:[] for m in METHODS for k in [*METRICS,'iq','features','logits']}
        lists.update({m+'__'+k:[] for m in SPECS for k in ['action_matrix_relative_mse','noise_log_mse','predicted_noise','true_noise','relative_solve_residual']})
        try:
            with torch.inference_mode():
                for offset in range(0,1800,32):
                    take=slice(offset,min(offset+32,1800));v={k:tensor(a[take],device) for k,a in arrays.items()}
                    action=dense_action(torch.complex(v['kernel'][:,0],v['kernel'][:,1]).to(torch.complex128),
                        torch.complex(v['coefficients'][:,0],v['coefficients'][:,1]).to(torch.complex128),torch.complex(v['coefficients'][:,2],v['coefficients'][:,3]).to(torch.complex128))
                    for method in METHODS:
                        if method=='received':value=v['received']
                        else:
                            value,aux=models[method](v['received'],return_aux=True)
                            errors=dict(action_matrix_relative_mse=(aux['action_matrix']-action).square().sum((1,2))/action.square().sum((1,2)).clamp_min(1e-8),
                                noise_log_mse=(aux['variance'].log()-v['variance'].clamp_min(1e-7).log()).square(),predicted_noise=aux['variance'],true_noise=v['variance'],relative_solve_residual=aux['relative_solve_residual'])
                            if errors['relative_solve_residual'].max()>1e-8:raise ValueError('Posterior residual too large')
                            for k,q in errors.items():lists[method+'__'+k].extend(q.cpu().tolist())
                        z=F.features(value);logits=F.classify_features(z)
                        metrics=source_metrics(value,clean[take],z,logits,target[take],clean_logits[take],scale)
                        for k,q in dict(metrics,iq=value,features=z,logits=logits).items():
                            if not torch.isfinite(q).all():raise ValueError('Nonfinite frozen source V result; preserve/no case omission')
                            lists[method+'__'+k].extend(q.cpu().tolist())
        finally:
            for a in arrays.values():a._mmap.close()
        records.update({k:np.asarray(v) for k,v in lists.items()})
        with (out/(scene+'.npz')).open('xb') as f:np.savez(f,**records)
        print('COVARIANCE_GUIDANCE_SOURCE_V_VIEW_COMPLETE',scene,1800,flush=True)
    for m,before in zip([F,*models.values()],snapshots):check_F0_state(m,before)
    write(out/'predictions_complete.json',dict(status='SOURCE_V_PREDICTIONS_COMPLETE',code_commit=args.commit,F0_identity=identity,
        methods=list(METHODS),views=list(SCENES),source_V_packets=1800,count=12600,TX_labels_read=0,target_inputs=0,optimizer_updates=0,model_state_updates=0,
        G_rows=25200,F0_output_rows=37800,F0_clean_rows=1800,extra_smoke_G_rows=4,extra_smoke_F0_rows=4,seconds=time.perf_counter()-start))


def fixed_predictions(out,commit):
    done=read(out/'predictions_complete.json')
    expected=dict(status='SOURCE_V_PREDICTIONS_COMPLETE',code_commit=commit,methods=list(METHODS),views=list(SCENES),source_V_packets=1800,count=12600,TX_labels_read=0,target_inputs=0,optimizer_updates=0,model_state_updates=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('All source V predictions required before labels')
    valid_identity(done['F0_identity'])
    with np.load(out/'references.npz',allow_pickle=False) as f:ref={k:f[k].copy() for k in f.files}
    if ref['ids'].shape!=(1800,) or len(set(ref['ids'].tolist()))!=1800:raise ValueError('Complete fixed V physical IDs required')
    data={};max_error=0.
    for scene in SCENES:
        with np.load(out/(scene+'.npz'),allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        if not np.array_equal(a['ids'],ref['ids']):raise ValueError('Source V prediction order differs')
        for method in METHODS:
            for key,shape in [('iq',(1800,2,256)),('features',(1800,160)),('logits',(1800,6))]:
                if a[method+'__'+key].shape!=shape or not np.isfinite(a[method+'__'+key]).all():raise ValueError('Incomplete or nonfinite frozen V outputs')
            m=numpy_metrics(a[method+'__iq'],ref['clean'],a[method+'__features'],a[method+'__logits'],ref['features'],ref['logits'],ref['scale'])
            for key,value in m.items():
                actual=a[method+'__'+key]
                if actual.shape!=(1800,) or not np.allclose(actual,value,rtol=2e-5,atol=2e-5):raise ValueError('Saved source metric or prediction differs from direct CPU calculation: '+key)
                max_error=max(max_error,float(np.max(np.abs(actual-value))))
            for key in METRICS:a[method+'__'+key]=m[key]
        data[scene]=a
    return done,ref,data,max_error


def paired_cells(physical_values,guided_values,labels,mask):
    delta=guided_values['optimized_feature_objective']-physical_values['optimized_feature_objective']
    old=physical_values['prediction']==labels;new=guided_values['prediction']==labels
    feature={'feature_better':delta < -1e-7,'feature_tied':abs(delta)<=1e-7,'feature_worse':delta > 1e-7}
    events={'TX_repair':~old&new,'TX_damage':old&~new,'TX_unchanged':old==new}
    return {f:{e:int((mask&fm&em).sum()) for e,em in events.items()} for f,fm in feature.items()}


def score(args,*,comparison='guided'):
    if os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('Independent CPU truth scorer required')
    if comparison not in METHODS or comparison in ('received','physical'):raise ValueError('Registered feature comparison method required')
    out=args.root/'runs'/RUN;done,ref,data,error=fixed_predictions(out,args.commit)
    # First truth read follows complete seven-view validation. Never load U_s labels or IQ here.
    c=validate_source(read(args.inputs));contract=read(Path(c['source_contract']))
    with np.load(Path(c['source_cache'])/'V_metadata.npz',allow_pickle=False) as f:
        ids=f['ids'].copy()
        if ids.tolist()!=contract['role_ids']['V'] or len(ids)!=27000:raise ValueError('Original V contract order required')
        lookup={str(v):i for i,v in enumerate(ids)}
        if any(str(v) not in lookup for v in ref['ids']):raise ValueError('Only original source V truth allowed')
        index=np.asarray([lookup[str(v)] for v in ref['ids']])
        if not np.array_equal(f['rx_i'][index],ref['rx']) or not np.array_equal(f['day_i'][index],ref['day']):raise ValueError('Original source V RX/day differs')
        labels=f['y'][index].copy()
    if labels.shape!=(1800,) or labels.dtype.kind not in 'iu' or (labels<0).any() or (labels>=6).any():raise ValueError('Six-class source V labels required')
    records=[];conditional=[];clean_correct=ref['logits'].argmax(1)==labels
    for scene,a in data.items():
        masks=[('overall','ALL',np.ones(1800,bool))]
        for key,v in [('RX',ref['rx']),('day',ref['day']),('TX',labels)]:masks.extend((key,str(x),v==x) for x in np.unique(v))
        for dim,value,mask in masks:
            n=int(mask.sum());pv={m:{k:a[m+'__'+k] for k in METRICS} for m in METHODS}
            conditional.append(dict(scene=scene,dimension=dim,stratum=value,count=n,cells=paired_cells(pv['physical'],pv[comparison],labels,mask)))
            for m in METHODS:
                correct=pv[m]['prediction']==labels;base=pv['physical']['prediction']==labels
                records.append(dict(scene=scene,dimension=dim,stratum=value,method=m,count=n,correct=int((correct&mask).sum()),accuracy_percent=100*float(correct[mask].mean()),
                    raw_clean_correct=int((clean_correct&mask).sum()),repair_vs_physical=int((~base&correct&mask).sum()),damage_vs_physical=int((base&~correct&mask).sum()),
                    metrics={k:float(pv[m][k][mask].mean()) for k in METRICS if k!='prediction'},
                    physical_errors={} if m=='received' else {k:float(a[m+'__'+k][mask].mean()) for k in ['action_matrix_relative_mse','noise_log_mse','predicted_noise','true_noise','relative_solve_residual']}))
    write(out/'results.json',dict(status='SOURCE_V_GUIDANCE_ANALYZED',run_id=RUN,code_commit=args.commit,F0_identity=done['F0_identity'],
        count=12600,source_V_packets=1800,source_V_views=7,records=records,conditional=conditional,conditional_comparison_vs_physical=comparison,max_metric_recomputation_error=error,
        source_TX_labels_scored=1800,truth_last=True,prediction_TX_labels_read=0,training_TX_labels_read=0,target_inputs=0,OE_reads=0,novel_reads=0,
        optimizer_updates=0,model_state_updates=0,checkpoint_selection=False,V_training_rows=0,V_historically_exposed=True,
        source_receiver_seed=2026101902,training_receiver_seed=2026101911,agreement_is_TX_accuracy=False,theoretical_ceiling=False,default_DURA_changed=False,goal_complete=False))
    print('COVARIANCE_GUIDANCE_SOURCE_V_CPU_SCORE_COMPLETE',flush=True)


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty launch owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    choices=recovery_gpu_choice()
    while not choices:time.sleep(5);choices=recovery_gpu_choice()
    admission=choices[0];common=['--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs)]
    for mode,env in [('worker',dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID'])),('score',dict(os.environ,CUDA_VISIBLE_DEVICES=''))]:
        command=[sys.executable,'-u','-m',MODULE,mode,*common]
        with (out/'logs'/(mode+'.log')).open('xb') as log:p=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/('submitted-'+mode+'.json'),dict(pid=p.pid,command=command,CWD=os.getcwd(),admission=admission if mode=='worker' else None))
        code=p.wait();write(out/('terminal-'+mode+'.json'),dict(pid=p.pid,exit_code=code))
        if code:raise RuntimeError('Frozen source V diagnostic failed; preserve/no automatic retry')
    if read(out/'results.json')['status']!='SOURCE_V_GUIDANCE_ANALYZED':raise ValueError('Complete independent source score required')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker','score']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual Git execution identity required')
    if args.mode=='run':controller(args)
    elif args.mode=='worker':worker(args)
    else:score(args)
