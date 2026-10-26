"""Frozen NoSIG source feature extraction and two-graph label-quality falsification."""
import argparse
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,snapshot,unchanged,source_channel
from .nosig_rgpl_base import RUN as BASE_RUN,SEED,check_state,validate_config as validate_base
from .u_realization_cache import RUN as CACHE_RUN,KEYS

RUN='20261026-diagnostic-PAMF-NoSIG-RGPL-sourceGraph-m1-r01'
RELEASE='20261026-diagnostic-PAMF-NoSIG-RGPL-sourceGraph-release-r01'
OLD_CACHE='20261025-phase1-PAMF-RestoredPL-m9-r01'
NAMESPACES=('PAMF/restored_dose/U/fixed',)+tuple('PAMF/realization_diversity/U/fresh'+str(r) for r in (1,2,3))


def teacher_folder(project):return project+'/runs/'+BASE_RUN+'/F/F0-NoSIG-PL-s2026092701/source/segment30'


def validate_runtime(rt):
    if (set(rt)!={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
            or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN or rt['release_root']!=rt['project']+'/releases/'+RELEASE
            or rt['allowed_gpus']!=list(range(8))):raise ValueError('One fixed source-only RGPL diagnostic required')


def load_teacher(rt,device):
    import torch
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload,load_G
    from experiments.pamf_repl_completion.source import reader_structure
    folder=Path(teacher_folder(rt['project']));done=read(folder/'complete.json');c=done['config'];validate_base(c)
    contract=read(c['source_contract']);cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(cp)
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=10 or done['updates']!=2220 or done['initialization']!='own_previous_segment'
            or done['ancestors']!=[c['shared_NoSIG_E20']] or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
            or c['epoch_offset']!=20 or c['step_offset']!=4440 or c['paper_method']!='NoSIG-PL' or contract!=read(c['expected_source_contract'])
            or cp['run_id']!=BASE_RUN or cp['candidate_id']!=c['row_id'] or cp['epoch']!=10 or cp['a1_ema_successful_updates']!=2220
            or cp['args']['baseline_ckpt']!=c['shared_NoSIG_E20']+'/final_ssdg.pth' or cp['args']['teacher_ckpt'] or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher']):
        raise ValueError('Actual own source-only NoSIG E30 graph teacher required')
    from .nosig_rgpl_base import load_prefix
    original=dict(c,output_root=str(folder.parent));[original.pop(k,None) for k in ('segment_epochs','epoch_offset','step_offset','parent_source')]
    _,parent=load_prefix(original);del parent
    with reader_structure(c):model=reader_from_payload(c,cp,device)
    del cp
    return model,load_G(c['source_G'],contract,device),c,contract


def U_caches(rt,contract,G,metadata):
    import numpy as np
    project=Path(rt['project']);old=project/'runs'/OLD_CACHE/'source_U_cache';done=read(old/'complete.json')
    if done['status']!='SOURCE_U_CACHE_COMPLETE' or done['source_contract']!=contract or done['G']!=G:raise ValueError('Original G IQ cache contract differs')
    caches=[np.load(old/'restored_U.npy',mmap_mode='r',allow_pickle=False)]
    physical=np.stack([metadata[k] for k in KEYS],axis=1);root=project/'runs'/CACHE_RUN
    if read(root/'complete.json')['status']!='SOURCE_U_REALIZATION_CACHE_COMPLETE':raise ValueError('Three complete fresh G IQ caches required')
    for replica in (1,2,3):
        folder=root/('fresh'+str(replica));d=read(folder/'complete.json')
        if d['source_contract']!=contract or d['G']!=G or d['namespace']!=NAMESPACES[replica]:raise ValueError('Same G/physical source realizations required')
        with np.load(folder/'physical_index.npz',allow_pickle=False) as a:
            if not np.array_equal(a['physical'],physical) or list(a['ids'])!=contract['role_ids']['U_s']:raise ValueError('Fresh source U physical order differs')
        caches.append(np.load(folder/'restored_U.npy',mmap_mode='r',allow_pickle=False))
    if any(a.shape!=(3,56700,2,256) or a.dtype!=np.float32 for a in caches):raise ValueError('Four complete FP32 G caches required')
    return caches


def features(rt):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    validate_runtime(rt);ctx=read(rt['context']);validate_context(ctx);model,g,c,contract=load_teacher(rt,'cuda:0')
    engine=InferenceEngine(g,reuse=True,graph=True,native=False);root=Path(rt['run_root']);start=time.perf_counter();cost=[]
    labeled=SourceDataset(c['source_cache'],'L_s',contract)
    with torch.inference_mode():
        x=labeled.x[:2].to('cuda:0');z=model.features(x.clone());restored=engine(x.clone())[0].clone();logits=model(restored)
        if z.shape!=(2,160) or logits.shape!=(2,6) or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(logits).all()):raise ValueError('Actual frozen NoSIG E30/defaulthead/G L2 smoke failed')
    write(root/'smoke.json',dict(status='PASS',NoSIG_E30=True,F_logical_packets=4,G_logical_packets=2,optimizer_updates=0,U_truth_read=False,target_reads=0))
    before=snapshot(model);gbefore=snapshot(g)
    for role,n in [('L_s',6300),('U_s',56700),('V',27000)]:
        data=labeled if role=='L_s' else SourceDataset(c['source_cache'],role,contract)
        if len(data)!=n or (role=='U_s' and 'y' in data.metadata):raise ValueError('Original source roles with hidden U required')
        out=root/role;out.mkdir(exist_ok=False)
        raw=np.lib.format.open_memmap(out/'raw_features.npy',mode='w+',dtype=np.float32,shape=(n,160))
        recovered=np.lib.format.open_memmap(out/'restored_features.npy',mode='w+',dtype=np.float32,shape=(n,160));confidence=np.empty(n,np.float32);raw_label=np.empty(n,np.int64)
        caches=U_caches(rt,contract,c['source_G'],data.metadata) if role=='U_s' else None;offset=0
        with torch.inference_mode():
            for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
                if role=='U_s' and (not bool((batch[1]==-1).all()) or bool(torch.as_tensor(batch[3]['tx_label_visible']).any())):raise ValueError('U true labels exposed')
                x=batch[0].to('cuda:0');b=len(x);z=model.features(x);p=model.classify_features(z).softmax(1)
                total=torch.zeros_like(z)
                for r,namespace in enumerate(NAMESPACES):
                    for k,scene in enumerate(SCENES):
                        if caches is not None:
                            a=np.ascontiguousarray(caches[r][k,offset:offset+b]);restored=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to('cuda:0')
                        else:
                            h=source_channel(x,batch[3],scene,contract,namespace=namespace);restored=engine(h.clone())[0].clone()
                        zz=model.features(restored)
                        if not bool(torch.isfinite(zz).all()) or bool((zz.norm(dim=1)<=1e-8).any()):raise ValueError('Finite nonzero frozen restored feature required')
                        total+=F.normalize(zz,dim=1)
                if not bool(torch.isfinite(z).all()) or bool((z.norm(dim=1)<=1e-8).any()) or bool((total.norm(dim=1)<=1e-8).any()):raise ValueError('Nonzero raw and twelve-view average required')
                raw[offset:offset+b]=fp32_copy(F.normalize(z,dim=1));recovered[offset:offset+b]=fp32_copy(F.normalize(total/12,dim=1))
                confidence[offset:offset+b]=fp32_copy(p.max(1).values);raw_label[offset:offset+b]=np.asarray(p.argmax(1).cpu().tolist(),np.int64);offset+=b
                write(root/'progress.json',dict(status='SOURCE_FEATURE_EXTRACTION',role=role,seen=offset,total=n,F_logical_packets=13*offset,U_truth_read=False,target_reads=0),exclusive=False)
                if offset%1024==0 or offset==n:print('SOURCE_RGPL_FEATURES',role,offset,n,flush=True)
        raw.flush();recovered.flush();del raw,recovered
        if offset!=n or not unchanged(model,before) or not unchanged(g,gbefore):raise ValueError('Source feature coverage or frozen F/G state differs')
        np.savez(out/'metadata.npz',**data.metadata,raw_confidence=confidence,raw_label=raw_label)
        write(out/'complete.json',dict(status='SOURCE_RGPL_FEATURES_COMPLETE',role=role,records=n,F_logical_packets=13*n,G_new_logical_packets=0 if role=='U_s' else 12*n,
            G_IQ_cache_views_reused=12*n if role=='U_s' else 0,U_truth_read=False,target_reads=0,optimizer_updates=0,parameters_and_registered_buffers_unchanged=True))
        cost.append(read(out/'complete.json'))
    write(root/'features_complete.json',dict(status='SOURCE_RGPL_FEATURE_EXTRACTION_COMPLETE',teacher_epoch=30,model_seed=SEED,source_contract=contract,roles=cost,
        source_F_logical_packets=1170000,new_G_logical_packets=399600,smoke_separate=True,U_truth_read=False,target_reads=0,optimizer_updates=0,seconds=time.perf_counter()-start))


def graph_diagnostic(rt):
    import numpy as np
    import torch
    from .rgpl_graph import geometry,propagate,extend,certify
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    root=Path(rt['run_root']);complete=read(root/'features_complete.json');contract=complete['source_contract'];allowed=contract['source_rxs']
    def tensor(a,dtype=torch.float32):
        value=np.ascontiguousarray(a,np.float32 if dtype==torch.float32 else np.int64)
        return torch.frombuffer(bytearray(value.tobytes()),dtype=dtype).reshape(value.shape).to('cuda:0')
    meta={}
    for role in ('L_s','U_s','V'):
        with np.load(root/role/'metadata.npz',allow_pickle=False) as a:meta[role]={k:a[k].copy() for k in a.files}
        if list(meta[role]['ids'])!=contract['role_ids'][role] or ('y' in meta[role])!=(role!='U_s'):raise ValueError('No V/target graph nodes or U labels permitted')
    rx=tensor(np.concatenate([meta[r]['rx_i'] for r in ('L_s','U_s')]),torch.long);vrx=tensor(meta['V']['rx_i'],torch.long)
    labels=tensor(np.concatenate([meta['L_s']['y'],np.full(56700,-1)]),torch.long);roles=tensor(np.concatenate([np.zeros(6300),np.ones(56700)]),torch.long)
    Q=[];VQ=[];stats=[]
    for name in ('raw','restored'):
        z=tensor(np.concatenate([np.load(root/r/(name+'_features.npy'),mmap_mode='r') for r in ('L_s','U_s')]))
        v=tensor(np.load(root/'V'/(name+'_features.npy'),mmap_mode='r'))
        graph=geometry(z,rx,allowed,k=20,block=512);prediction=propagate(graph,labels,roles,6);query=extend(graph,prediction,v,vrx)
        if prediction['V_or_target_nodes']!=0 or prediction['source_L_seeds']!=6300 or prediction['hidden_U_nodes']!=56700:raise ValueError('Exact L/U graph roles required')
        np.save(root/(name+'_source_Q.npy'),fp32_copy(prediction['Q']));np.save(root/(name+'_V_Q.npy'),fp32_copy(query))
        Q.append(prediction['Q']);VQ.append(query);stats.append(dict(graph=name,nodes=len(z),mutual_edges=graph['mutual_edges'],iterations=prediction['iterations'],residual=prediction['residual'],V_nodes=0))
        del graph,z,v
        write(root/'progress.json',dict(status='SOURCE_GRAPH_COMPLETE',graph=name,nodes=63000,V_nodes=0,U_truth_read=False,target_reads=0),exclusive=False)
    uconf=tensor(meta['U_s']['raw_confidence']);vconf=tensor(meta['V']['raw_confidence'])
    u=certify(Q[0][6300:],Q[1][6300:],rx[6300:],allowed,uconf);v=certify(VQ[0],VQ[1],vrx,allowed,vconf)
    ua=np.asarray(u['accepted'].cpu().tolist(),bool);ul=np.asarray(u['label'].cpu().tolist(),np.int64)
    va=np.asarray(v['accepted'].cpu().tolist(),bool);vl=np.asarray(v['label'].cpu().tolist(),np.int64);y=meta['V']['y'];old=meta['V']['raw_label']
    # Freeze all certification decisions before the source V truth-side quality arithmetic.
    np.savez(root/'source_U_graph_targets.npz',accepted=ua,label=ul,ids=meta['U_s']['ids'],physical=np.stack([meta['U_s'][k] for k in KEYS],axis=1),raw_confidence=meta['U_s']['raw_confidence'])
    np.savez(root/'source_V_graph_predictions.npz',accepted=va,label=vl,ids=meta['V']['ids'],raw_confidence=meta['V']['raw_confidence'])
    strata=[('overall','ALL',np.ones(27000,bool))]
    for dim,value in [('RX',meta['V']['rx_i']),('day',meta['V']['day_i']),('TX',y)]:strata.extend((dim,str(i),value==i) for i in np.unique(value))
    rows=[]
    for dim,name,mask in strata:
        m=mask&va;n=int(m.sum());correct=int((m&(vl==y)).sum());fixed=int((m&(old!=y)&(vl==y)).sum());broken=int((m&(old==y)&(vl!=y)).sum())
        rows.append(dict(dimension=dim,stratum=name,hard_pool=int((mask&(meta['V']['raw_confidence']<.95)).sum()),certified=n,correct=correct,
            precision=100*correct/n if n else None,corrected=fixed,broken=broken,net_correct=fixed-broken))
    n=int(va.sum());correct=int((va&(vl==y)).sum());passed=n>=200 and 100*correct>=99.5*n
    write(root/'complete.json',dict(status='SOURCE_RGPL_GRAPH_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],model_seed=SEED,teacher_epoch=30,
        graph_quality_pass=passed,quality_rule=dict(minimum_new_V_certified=200,minimum_precision_percent=99.5),graph_stats=stats,
        U_records=56700,U_hard_pool=int((meta['U_s']['raw_confidence']<.95).sum()),U_certified=int(ua.sum()),V_records=27000,V_certified=n,V_correct=correct,
        V_precision=100*correct/n if n else None,quality_cells=rows,U_truth_read=False,V_graph_nodes=0,target_reads=0,optimizer_updates=0,
        default_head_fit_updates=0,new_method_training=False,source_only=True,scope='Single frozen NoSIG E30;U precision unknown;no target result or automatic candidate training'))
    print('SOURCE_RGPL_GRAPH_DIAGNOSTIC_COMPLETE',passed,n,correct,flush=True)


def worker(rt):
    validate_runtime(rt)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in list(map(str,range(8))):raise ValueError('One admitted physical GPU required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);features(rt);graph_diagnostic(rt)


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes
    import subprocess
    validate_runtime(rt)
    while True:
        pending=processes();available=[]
        for gpu in rt['allowed_gpus']:
            pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
            if load<2 and free>=10000:available.append((load,-free,gpu))
        if available:break
        time.sleep(12)
    gpu=min(available)[2];release=Path(rt['release_root']);command=[rt['python'],'-u','-m','experiments.pamf_channel_pl.rgpl_source_graph','worker','--config',rt['runtime_file']]
    env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (release/'worker.log').open('x') as log:
        child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(release/'worker_submit.json',dict(pid=child.pid,gpu=gpu,cwd=str(release),argv=command,log=str(release/'worker.log')))
    code=child.wait();done=Path(rt['run_root'])/'complete.json';good=code==0 and done.is_file() and read(done)['status']=='SOURCE_RGPL_GRAPH_DIAGNOSTIC_COMPLETE'
    write(release/'controller_complete.json',dict(status='COMPLETE' if good else 'TECHNICAL_FAILURE',worker_exit_code=code,run_id=RUN))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--config',required=True)
    a=p.parse_args();rt=read(a.config);dispatch(rt) if a.action=='dispatch' else worker(rt)
