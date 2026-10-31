"""Second fixed screen: source K1 rule and support-only K>1 kernel calibration."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .io import read,write,require
from .features import source_smoke
from .milkr_screen import extract,subset,initialize
from .episodic_io import load_task
from .episodic_experiment import split_rows
from .milkr.support_calibrated import fit,CONFIGS,LAMBDAS,target_and_score


def fit_function(rt):
    if rt.get('kernel_method')=='kernel_margin':
        from .milkr.kernel_margin import fit as margin_fit
        return margin_fit
    if rt.get('kernel_method')=='balanced_h':
        from .milkr.balanced_h import fit as balanced_fit
        return balanced_fit
    if rt.get('kernel_method') in ('nested_fisher','source_bridge','paired_cca','class_whiten'):
        from .milkr.nested_fisher import fit as nested_fit
        return nested_fit
    if rt.get('kernel_method')=='fisher_signature':
        from .milkr.fisher_signature import fit as fisher_fit
        return fisher_fit
    if rt.get('kernel_method')=='task_signature':
        from .milkr.task_signature import fit as task_fit
        return task_fit
    if rt.get('kernel_method')=='signature_kernel':
        from .milkr.signature_kernel import fit as signature_fit
        return signature_fit
    if rt.get('kernel_method')=='group_calibrated':
        from .milkr.group_calibrated import fit as group_fit
        return group_fit
    if rt.get('kernel_method','support_calibrated')=='pooled_metric':
        from .milkr.pooled_metric import fit as metric_fit
        return metric_fit
    require(rt.get('kernel_method','support_calibrated')=='support_calibrated','Unknown kernel method')
    return fit


def kernel_configs(rt):
    if rt.get('kernel_method') in ('signature_kernel','task_signature','fisher_signature','nested_fisher','source_bridge','paired_cca','balanced_h','kernel_margin','class_whiten'):
        from .milkr.signature_kernel import CONFIGS as signature_configs
        return signature_configs
    return CONFIGS


def extract_for_method(rt,reader,raw,device,engine=None,batch=128):
    return extract(reader,raw,device,engine,batch,descriptor=rt.get('kernel_method') in ('signature_kernel','task_signature','fisher_signature','nested_fisher','source_bridge','paired_cca','balanced_h','kernel_margin','class_whiten'))


def source_episodes(rt):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    reader,engine=initialize(rt);ds=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=ds.metadata
    source=[]
    for rx in sorted(np.unique(m['rx_i']).tolist()):
        for episode,epoch in enumerate(rt['source_cache_epochs']):
            folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
            require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source cache')
            require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source cache physical alignment differs')
            bank=np.load(folder/'x.npy',mmap_mode='r');rng=np.random.default_rng(rt['source_proxy_seed']+rx*100+episode)
            chosen=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==c)))[:52] for c in range(6)]
            require(all(len(x)==52 for x in chosen),'Insufficient source physical count')
            indices=np.concatenate(chosen);features=[]
            for g in (None,engine):features.append(extract_for_method(rt,reader,np.asarray(bank[indices]),'cuda:0',g)[0])
            source.append((rx,episode,features))
    return source


def calibration(rt):
    if rt.get('kernel_method')=='class_whiten':
        from .milkr_class_source import calibrate
        return calibrate(rt)
    if rt.get('kernel_method')=='kernel_margin':
        from .milkr_margin_source import calibrate
        return calibrate(rt)
    if rt.get('kernel_method')=='balanced_h':
        from .milkr_h_source import calibrate
        return calibrate(rt)
    if rt.get('kernel_method')=='paired_cca':
        from .milkr_cca_source import calibrate
        return calibrate(rt)
    if rt.get('kernel_method')=='source_bridge':
        from .milkr_bridge_source import calibrate
        return calibrate(rt)
    if rt.get('kernel_method')=='nested_fisher':
        from .milkr_nested_source import calibrate
        return calibrate(rt)
    if rt.get('kernel_method')=='fisher_signature':
        from .milkr_fisher_source import calibrate
        return calibrate(rt)
    if rt.get('kernel_method')=='task_signature':
        from .milkr_task_source import calibrate
        return calibrate(rt)
    if rt.get('kernel_method')=='group_calibrated':
        from .milkr_group_source import calibrate
        return calibrate(rt)
    fit=fit_function(rt);started=time.perf_counter();source=source_episodes(rt)
    quality={};records=[]
    for config in kernel_configs(rt):
        for ridge in LAMBDAS:
            rule=dict(config=list(config),ridge=ridge);values=[]
            for rx,ep,features in source:
                sup=np.asarray([c*52 for c in range(6)]);held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)])
                y=np.arange(6);qy=np.repeat(np.arange(6),32)
                for sp in (0,1):
                    s=fit(subset(features[sp],sup),y,list(map(str,sup)),old_count=3,fixed=rule)
                    for qp in (0,1):
                        pred=s.score(subset(features[qp],held)).argmax(1)
                        values.append(target_and_score(qy,pred,3))
            objective=float(np.mean(values));quality[(config,ridge)]=objective
            records.append(dict(config=list(config),ridge=ridge,source_C_H=objective))
    # Fixed deterministic tie: lower source complexity then greater ridge.
    index=max(range(len(records)),key=lambda i:(records[i]['source_C_H'],records[i]['ridge'],-i))
    rule={k:records[index][k] for k in ('config','ridge')};validation=[]
    for rx,ep,features in source:
        held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)]);qy=np.repeat(np.arange(6),32)
        for k in (5,10,20):
            sup=np.concatenate([np.arange(c*52,c*52+k) for c in range(6)]);ys=np.repeat(np.arange(6),k)
            for sp in (0,1):
                s=fit(subset(features[sp],sup),ys,list(map(str,sup)),old_count=3,fixed=rule)
                for qp in (0,1):
                    pred=s.score(subset(features[qp],held)).argmax(1)
                    validation.append(dict(rx=rx,episode=ep,k=k,S=sp,Q=qp,H=target_and_score(qy,pred,3),config=s.config,ridge=s.ridge))
        print('SOURCE_POLICY',rx,ep,flush=True)
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=rule,K1_grid=records,
        support_policy_validation=validation,seconds=time.perf_counter()-started,target_reads=0,query_reads=0,U_truth_reads=0,
        scope='known-source proxy, full support-conditioned kernels; not unseen novel identity confirmation'))


def worker(rt,capname):
    fit=fit_function(rt);arm=rt.get('arm','R6')
    import torch
    reader,engine=initialize(rt);before={k:v.detach().clone() for k,v in reader.state_dict().items()}
    cap=Path(rt['data_root'])/capname;m=read(cap/'manifest.json');root=Path(rt['run_root'])/'predictions'/m['receiver'];root.mkdir(parents=True,exist_ok=False)
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    selected=read(Path(rt['run_root'])/'source_selection.json')
    rules=[(arm,selected['K1_rule'])]
    if rt.get('include_R9_control'):rules.append(('R9C',selected['control_K1_rule']))
    bridge=None
    if rt.get('kernel_method')=='source_bridge':
        from .milkr.source_bridge import FeatureBridge
        bridge=FeatureBridge.load(Path(rt['run_root'])/'source_bridge.npz')
        rules.append(('R12C',selected['control_K1_rule']))
    elif rt.get('kernel_method') in ('paired_cca','balanced_h','kernel_margin','class_whiten'):
        from .milkr.paired_cca import PairedCCA
        checkpoint='source_class_cca.npz' if rt.get('kernel_method')=='class_whiten' else 'source_cca.npz'
        bridge=PairedCCA.load(Path(rt['run_root'])/checkpoint)
    states=[];rows=[];started=time.perf_counter()
    for k in rt['K']:
        for n in [0]+rt['N']:
            stage='C' if n else 'B';s=specs[k,n];task=load_task(cap,s['split_id'],stage=stage)
            for sp,g in enumerate((None,engine)):
                f,_=extract_for_method(rt,reader,task['support_x'],'cuda:0',g,batch=k);begin=time.perf_counter()
                for state_arm,rule in rules:
                    if bridge is not None and state_arm==arm and rt.get('kernel_method') in ('paired_cca','balanced_h','kernel_margin','class_whiten'):features=bridge.transform(f,restored=bool(sp))
                    else:features=bridge.transform(f) if bridge is not None and sp==0 and state_arm==arm else f
                    begin=time.perf_counter();state=fit(features,task['support_y'],task['support_ids'],fixed=rule)
                    write(root/f'{stage}-K{k}-N{n}-S{sp}-{state_arm}-fit.json',dict(selection=state.selection,config=state.config,ridge=state.ridge,tau=state.tau,
                        seconds=time.perf_counter()-begin,F_updates=0,G_updates=0,query_fit=False,B_function_inherited=False,
                        coefficient_bytes=state.alpha.nbytes,physical_labels=len(task['support_y']),arm=state_arm,
                        source_bridge=bridge is not None and state_arm==arm and (sp==0 or rt.get('kernel_method') in ('paired_cca','balanced_h','kernel_margin','class_whiten'))))
                    states.append((s,stage,sp,state,state_arm))
        print('FIT_K_COMPLETE',m['receiver'],k,flush=True)
    bank=np.load(cap/'received.npy',mmap_mode='r');allrows=np.load(cap/'stage_rows/N20.npy');pos=np.full(len(bank),-1,np.int64);pos[allrows]=np.arange(len(allrows))
    for qp,g in enumerate((None,engine)):
        f,_=extract_for_method(rt,reader,np.asarray(bank[allrows]),'cuda:0',g)
        if rt.get('kernel_method') in ('paired_cca','balanced_h','kernel_margin','class_whiten'):bridged=bridge.transform(f,restored=bool(qp))
        else:bridged=bridge.transform(f) if bridge is not None and qp==0 else f
        for s,stage,sp,state,state_arm in states:
            query=split_rows(cap,s);positions=pos[query];require((positions>=0).all(),'Invalid query bank mapping')
            prediction=[];restricted=[];classes=s['registered_classes']
            order=np.asarray(sorted(range(len(classes)),key=lambda i:classes[i]));old=order[order<6]
            for start in range(0,len(positions),rt['decision_batch']):
                features=bridged if state_arm==arm else f
                scores=state.score(subset(features,positions[start:start+rt['decision_batch']]))
                prediction.extend(order[scores[:,order].argmax(1)].tolist());restricted.extend(old[scores[:,old].argmax(1)].tolist())
            name=f'{stage}-K{s["k"]}-N{s["new_count"]}-S{sp}Q{qp}-{state_arm}.npz'
            np.savez_compressed(root/name,query_indices=query,predicted_indices=np.asarray(prediction,np.int16),old_restricted_indices=np.asarray(restricted,np.int16))
            rows.append(dict(split_id=s['split_id'],stage=stage,arm=state_arm,input_path=f'S{sp}Q{qp}',prediction=name,
                k=s['k'],new_count=s['new_count'],registered_classes=s['registered_classes']))
        print('QUERY_PATH_COMPLETE',m['receiver'],qp,flush=True)
    require(all(torch.equal(before[k],v) for k,v in reader.state_dict().items()),'Frozen F changed')
    write(root/'predictions.json',dict(capsule=capname,capsule_id=m['capsule_id'],profile='MILKR-support-calibrated',model_seed=rt['model_seeds'][0],rows=rows))
    write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=m['capsule_id'],truth_read=False,query_fit=False,rows=len(rows),seconds=time.perf_counter()-started))


def dispatch(rt):
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    source_smoke(rt,root/'source_smoke.json');calibration(rt);active=[]
    selection=read(root/'source_selection.json')
    if selection.get('target_eligible') is False:
        write(root/'complete.json',dict(status='SOURCE_ONLY_SCIENTIFIC_STOP',target_rows=0,reason=selection['scientific_verdict']))
        return
    for i,cap in enumerate(rt['capsules']):
        gpu=rt['allowed_gpus'][i%len(rt['allowed_gpus'])];rx=cap.split('/')[-1]
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());require(free>=4096,'Insufficient GPU memory')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cmd=[sys.executable,'-u','-m',__package__+'.milkr_cv_screen','worker','--runtime',rt['runtime_file'],'--capsule',cap]
        log=(root/'logs'/(rx+'.log')).open('x');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);active.append((rx,p,log))
        write(root/(rx+'-launch.json'),dict(pid=p.pid,gpu=gpu,cwd=str(Path.cwd()),argv=cmd))
    failures=[]
    for rx,p,log in active:
        code=p.wait();log.close()
        if code:failures.append(dict(receiver=rx,exit_code=code))
    if failures:write(root/'failed.json',dict(status='TECHNICAL_FAILURE',failures=failures));raise RuntimeError('No blind retry')
    for rx,_,_ in active:
        subprocess.run([sys.executable,'-m',__package__+'.milkr_cv_screen','score','--runtime',rt['runtime_file'],'--capsule',rx],check=True)
    write(root/'complete.json',dict(status='SCORED_COMPLETE',truth_last=True,target_feedback=False,receivers=[rx for rx,_,_ in active]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker','score']);p.add_argument('--runtime',required=True);p.add_argument('--capsule');a=p.parse_args();rt=read(a.runtime)
    if a.action=='dispatch':dispatch(rt)
    elif a.action=='worker':worker(rt,a.capsule)
    else:
        from .episodic_score import run
        root=Path(rt['run_root']);run(rt,dict(prediction=str(root/'predictions'/a.capsule),output=str(root/'scores'/a.capsule)))
