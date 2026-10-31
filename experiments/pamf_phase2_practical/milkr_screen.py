"""Small preregistered K/N screen; frozen Tri-SIG-TIME SSL, source-only tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .io import read, write, require, activate
from .features import branches, fft96, tensor, load_G, source_smoke
from .trisig_source import load as load_source
from .episodic_io import load_task
from .episodic_experiment import split_rows
from .milkr.semantic_residual import fit, NAMES


def extract(reader, raw, device, engine=None, batch=128, descriptor=False):
    import torch
    names=NAMES+('sig_views',) if descriptor else NAMES
    features={k:[] for k in names};logits=[]
    with torch.inference_mode():
        for start in range(0,len(raw),batch):
            x=np.asarray(raw[start:start+batch],np.float32)
            if engine is not None:
                x=np.asarray(engine(tensor(x,device))[0].cpu().tolist(),np.float32)
            if descriptor:
                from .signature_features import capture
                v,l=capture(reader,tensor(x,device))
            else:v,l=branches(reader,tensor(x,device))
            for k in names:
                features[k].append(fft96(x) if k=='fft' else np.asarray(v[k].cpu().tolist(),np.float32))
            logits.append(np.asarray(l.cpu().tolist(),np.float32))
    return {k:np.concatenate(a) for k,a in features.items()},np.concatenate(logits)


def subset(features, rows):
    return {k:v[rows] for k,v in features.items()}


def initialize(rt):
    import torch
    activate(rt);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bundle=load_source(rt,rt['model_seeds'][0],'cuda:0')
    return bundle['reader'],load_G(rt,'cuda:0')


def calibration(rt):
    """Only legal source L; holdouts never enter a fitted state. No target handle."""
    from experiments.cric_dura_comparison.f_data import SourceDataset
    reader,engine=initialize(rt)
    c=read(rt['source_contract']);ds=SourceDataset(rt['source_cache'],'L_s',c)
    m=ds.metadata;rows=[];started=time.perf_counter()
    candidates=rt['lambda_candidates'];results={str(v):[] for v in candidates}
    for rx in sorted(np.unique(m['rx_i']).tolist()):
        for episode in range(rt['source_episodes_per_rx']):
            # Existing source-only, physically keyed augmentation; no new IQ view.
            epoch=rt['source_cache_epochs'][episode]
            folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
            require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source cache')
            require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source augmentation row order differs')
            bank=np.load(folder/'x.npy',mmap_mode='r')
            rng=np.random.default_rng(rt['source_proxy_seed']+rx*100+episode)
            chosen=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==j)))[:37] for j in range(6)]
            require(all(len(x)==37 for x in chosen),'Insufficient source proxy count')
            support=np.concatenate([x[:5] for x in chosen]);held=np.concatenate([x[5:] for x in chosen])
            require(not np.isin(support,held).any(),'Source support/holdout overlap')
            sx=np.array(bank[support]);qx=np.array(bank[held]);ys=np.repeat(np.arange(6),5);yq=np.repeat(np.arange(6),32)
            sf=[];sl=[];qf=[];ql=[]
            for g in (None,engine):
                f,l=extract(reader,sx,'cuda:0',g,batch=5);sf.append(f);sl.append(l)
                f,l=extract(reader,qx,'cuda:0',g);qf.append(f);ql.append(l)
            for lam in candidates:
                for paths in ((0,),(1,),(0,1)):
                    sv=[sf[p] for p in paths];ls=[sl[p] for p in paths]
                    ids=m['ids'][support].tolist()
                    b6=fit(sv,ls,ys,ids,regularization=lam)
                    b3=fit([subset(f,slice(0,15)) for f in sv],[l[:15,:3] for l in ls],ys[:15],ids[:15],old_count=3,regularization=lam)
                    c6=fit(sv,[l[:,:3] for l in ls],ys,ids,old_count=3,regularization=lam,previous=b3)
                    for p in (0,1):
                        a=(b6.score(qf[p],ql[p]).argmax(1)==yq).mean()
                        b=(b3.score(subset(qf[p],slice(0,96)),ql[p][:96,:3]).argmax(1)==yq[:96]).mean()
                        pred=c6.score(qf[p],ql[p][:,:3]).argmax(1)
                        old=(pred[:96]==yq[:96]).mean();new=(pred[96:]==yq[96:]).mean()
                        h=2*old*new/(old+new) if old+new else 0.
                        results[str(lam)].append(dict(rx=rx,episode=episode,support=list(paths),query=p,B6=float(a),B3=float(b),C_H=float(h)))
            print('SOURCE_PROXY',rx,episode,flush=True)
    objective={key:float(np.mean([np.mean([r['B6'],r['B3'],r['C_H']]) for r in vals])) for key,vals in results.items()}
    # Numeric tie break selects larger ridge. One value shared by every target arm/K/N.
    selected=max(candidates,key=lambda v:(objective[str(v)],v))
    value=dict(status='SOURCE_FROZEN',regularization=selected,objective=objective,rows=results,
        target_reads=0,query_reads=0,U_truth_reads=0,source_proxy_scope='known six source classes, not genuine novel identity validation',
        seconds=time.perf_counter()-started)
    write(Path(rt['run_root'])/'source_selection.json',value)
    return selected


def worker(rt, capsule):
    import torch
    reader,engine=initialize(rt);before={k:v.detach().clone() for k,v in reader.state_dict().items()}
    cap=Path(rt['data_root'])/capsule;m=read(cap/'manifest.json')
    root=Path(rt['run_root'])/'predictions'/m['receiver'];root.mkdir(parents=True,exist_ok=False)
    lam=read(Path(rt['run_root'])/'source_selection.json')['regularization']
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    seed=rt['model_seeds'][0];rows=[];states=[];started=time.perf_counter()
    for k in rt['K']:
        s=specs[k,0];task=load_task(cap,s['split_id'],stage='B');old_ids=task['support_ids'].tolist()
        old_views=[];old_logits=[]
        for g in (None,engine):
            f,l=extract(reader,task['support_x'],'cuda:0',g,batch=k);old_views.append(f);old_logits.append(l)
        B={}
        for path,paths in (('S0',(0,)),('S1',(1,)),('SD',(0,1))):
            views=[old_views[p] for p in paths];logits=[old_logits[p] for p in paths]
            B[path]=fit(views,logits,task['support_y'],old_ids,regularization=lam)
            states.append((s,'B',path,'R3' if path!='SD' else 'R5',B[path]))
        for n in rt['N']:
            s=specs[k,n];task=load_task(cap,s['split_id'],stage='C')
            require(task['support_ids'][:6*k].tolist()==old_ids,'B/C support physical IDs differ')
            ids=task['support_ids'].tolist();y=task['support_y'];views=[];logits=[]
            for p,g in enumerate((None,engine)):
                # Reuse the exact old-support feature rows, with new labels only in C.
                f,l=extract(reader,task['support_x'][6*k:],'cuda:0',g,batch=k)
                views.append({name:np.concatenate((old_views[p][name],f[name])) for name in NAMES})
                logits.append(np.concatenate((old_logits[p],l)))
            for path,paths in (('S0',(0,)),('S1',(1,)),('SD',(0,1))):
                sv=[views[p] for p in paths];sl=[logits[p] for p in paths]
                for arm,prior,inherit in (('R1',False,False),('R2',True,False),('R3',True,True)):
                    if path=='SD' and arm=='R1':continue
                    label={'R2':'R4','R3':'R5'}.get(arm,arm) if path=='SD' else arm
                    begin=time.perf_counter()
                    state=fit(sv,sl,y,ids,regularization=lam,previous=B[path] if inherit else None,
                              prior=prior,new_prototype=prior,tau=B[path].tau)
                    states.append((s,'C',path,label,state))
                    write(root/f'K{k}-N{n}-{path}-{label}-fit.json',dict(seconds=time.perf_counter()-begin,
                        tau=state.tau,regularization=lam,normal_equation_residual=state.residual,
                        physical_support=len(y),views=len(sv),F_updates=0,G_updates=0,query_fit=False,
                        B_inherited=inherit,persistent_array_bytes=sum(v.nbytes for f in state.views for v in f.values())+state.alpha.nbytes))
        print('FIT_K_COMPLETE',m['receiver'],k,flush=True)
    # All registered target states are fixed before any query features are used.
    bank=np.load(cap/'received.npy',mmap_mode='r');all_rows=np.load(cap/'stage_rows'/'N20.npy')
    position=np.full(len(bank),-1,np.int64);position[all_rows]=np.arange(len(all_rows))
    for qp,g in enumerate((None,engine)):
        f,l=extract(reader,np.asarray(bank[all_rows]),'cuda:0',g)
        for s,stage,path,arm,state in states:
            query=split_rows(cap,s);ix=position[query];require((ix>=0).all(),'Query outside registered bank')
            predictions=[];restricted=[]
            for start in range(0,len(ix),rt['decision_batch']):
                take=ix[start:start+rt['decision_batch']]
                scores=state.score(subset(f,take),l[take]);classes=s['registered_classes']
                order=np.asarray(sorted(range(len(classes)),key=lambda i:classes[i]))
                old=np.asarray(sorted(range(6),key=lambda i:classes[i]))
                predictions.extend(order[scores[:,order].argmax(1)].tolist())
                restricted.extend(old[scores[:,old].argmax(1)].tolist())
            name=f'{stage}-K{s["k"]}-N{s["new_count"]}-{path}Q{qp}-{arm}.npz'
            np.savez_compressed(root/name,query_indices=query,predicted_indices=np.asarray(predictions,np.int16),
                                old_restricted_indices=np.asarray(restricted,np.int16))
            rows.append(dict(split_id=s['split_id'],stage=stage,arm=arm,input_path=f'{path}Q{qp}',prediction=name,
                k=s['k'],new_count=s['new_count'],registered_classes=s['registered_classes']))
        print('QUERY_PATH_COMPLETE',m['receiver'],qp,len(rows),flush=True)
    require(all(torch.equal(before[k],v) for k,v in reader.state_dict().items()),'Frozen F changed')
    write(root/'predictions.json',dict(capsule=capsule,capsule_id=m['capsule_id'],profile='MILKR-semantic-residual',model_seed=seed,rows=rows))
    write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=m['capsule_id'],rows=len(rows),
        truth_read=False,query_fit=False,seconds=time.perf_counter()-started,model_seed=seed))


def dispatch(rt):
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv,run_id=rt['run_id']))
    source_smoke(rt,root/'source_smoke.json');calibration(rt)
    active=[]
    for i,cap in enumerate(rt['capsules']):
        gpu=rt['allowed_gpus'][i%len(rt['allowed_gpus'])]
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        require(free>=4096,'Insufficient remaining GPU memory before launch')
        output=Path(rt['data_root'])/cap;rx=read(output/'manifest.json')['receiver']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        cmd=[sys.executable,'-u','-m',__package__+'.milkr_screen','worker','--runtime',rt['runtime_file'],'--capsule',cap]
        log=(root/'logs'/(rx+'.log')).open('x');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        active.append((rx,p,log));write(root/(rx+'-launch.json'),dict(pid=p.pid,gpu=gpu,argv=cmd,cwd=str(Path.cwd())))
    failures=[]
    for rx,p,log in active:
        code=p.wait();log.close()
        if code:failures.append(dict(receiver=rx,exit_code=code))
    if failures:
        write(root/'failed.json',dict(status='TECHNICAL_FAILURE',failures=failures))
        raise RuntimeError('Screen workers failed; preserve outputs, no retry')
    # Separate scorer processes open truth only after every worker is complete.
    for rx,_,_ in active:
        subprocess.run([sys.executable,'-u','-m',__package__+'.milkr_screen','score','--runtime',rt['runtime_file'],'--capsule',rx],check=True)
    write(root/'complete.json',dict(status='SCORED_COMPLETE',receivers=[rx for rx,_,_ in active],
        truth_last=True,target_feedback=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker','score'])
    p.add_argument('--runtime',required=True);p.add_argument('--capsule');a=p.parse_args();rt=read(a.runtime)
    if a.action=='dispatch':dispatch(rt)
    elif a.action=='worker':worker(rt,a.capsule)
    else:
        from .episodic_score import run
        root=Path(rt['run_root']);run(rt,dict(prediction=str(root/'predictions'/a.capsule),output=str(root/'scores'/a.capsule)))


if __name__=='__main__':main()
