"""Matched raw/restored L-only adaptation of NoSIG label experts; source diagnostic."""
import argparse
from collections import Counter
import gzip
import math
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,snapshot,unchanged,source_channel
from .rgpl_source_graph import teacher_folder,load_teacher,NAMESPACES,RUN as GRAPH_RUN
from .nosig_rgpl_base import SEED,check_state

RUN='20261026-phase1-PAMF-NoSIG-RestorationLabeler-source-m2-r01'
RELEASE='20261026-phase1-PAMF-NoSIG-RestorationLabeler-source-release-r01'
METHODS=('RawLabeler','GLabeler')
UPDATES=2220
PHYSICAL=('eq_i','rx_i','day_i','sig_i','base_index')


def validate_runtime(rt):
    if (set(rt)!={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
            or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN or rt['release_root']!=rt['project']+'/releases/'+RELEASE
            or rt['allowed_gpus']!=list(range(8))):raise ValueError('One fixed source labeler diagnostic required')


def tensor(a,device):
    import numpy as np
    import torch
    value=np.ascontiguousarray(a,np.float32)
    return torch.frombuffer(bytearray(value.tobytes()),dtype=torch.float32).reshape(value.shape).to(device)


def finish_probe(model,before):
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    if not unchanged(model,before):raise ValueError('Smoke changed checkpoint parameters or registered buffers')


def make_model(rt):
    import torch
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from .nosig_rgpl_base import installed
    # Reuse source/ancestry checks, not graph labels or the Reader's no-grad Sinc override.
    reader,g,c,contract=load_teacher(rt,'cuda:0');del reader
    cp=torch.load(Path(teacher_folder(rt['project']))/'final_ssdg.pth',map_location='cpu',weights_only=False)
    with installed(c),f_train.class_dimension(c),native_installed(f_train.original_config(SEED)) as native:
        a=f_train.arguments(c,'cuda:0');a.input_len=256;a.baseline_ckpt=a.teacher_ckpt='';a.from_scratch=a.a1_scratch_only=True
        model=t1_model.install(native.build_baseline_model(a,'cuda:0'),c).to('cuda:0')
    model.load_state_dict(cp['model'],strict=True);check_state(dict(model=model.state_dict()))
    sinc=model.id_backbone.encoder.core.id_backbone.sinc
    if (sinc._filters.__func__ is not type(sinc)._filters or not sinc.low_coordinate.requires_grad
            or getattr(model,'_cd_U_loader',None) is not None or model._cd_U_seen!=0):
        raise ValueError('Joint native Sinc/body/default-head construction with no U iterator required')
    model._cd_U_loader=model._cd_U_iter=None
    return model,g,c,contract


def smoke_cache(rt):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    model,g,c,contract=make_model(rt);data=SourceDataset(c['source_cache'],'L_s',contract)
    if len(data)!=6300:raise ValueError('Original visible L only required')
    engine=InferenceEngine(g,reuse=True,graph=True,native=False);encoder=model.id_backbone.encoder
    with torch.no_grad():encoder(data.x[:2].to('cuda:0'));engine(data.x[:2].to('cuda:0'))
    before=snapshot(model);gbefore=snapshot(g);batch=next(iter(torch.utils.data.DataLoader(data,batch_size=128,shuffle=False)))
    x=batch[0].to('cuda:0');y=batch[1].to('cuda:0');proof=[]
    with torch.no_grad():restored=engine(source_channel(x,batch[3],SCENES[0],contract,namespace=NAMESPACES[0]).clone())[0].clone()
    for name,value in [('RawLabeler',x),('GLabeler',restored)]:
        model.zero_grad(set_to_none=True);model.train();loss=F.cross_entropy(encoder(value),y,label_smoothing=.01);loss.backward()
        head=encoder.core.id_backbone.cls_head.weight;sinc=encoder.core.id_backbone.sinc
        if (not bool(torch.isfinite(loss)) or head.grad is None or float(head.grad.norm())<=0
                or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                or sinc.low_coordinate.grad is None or sinc.band_coordinate.grad is None
                or float(sinc.low_coordinate.grad.norm()+sinc.band_coordinate.grad.norm())<=0):
            raise ValueError('Actual native NoSIG Sinc/body/default-head joint backward failed')
        proof.append(dict(method=name,loss=float(loss),head_gradient_norm=float(head.grad.norm()),
            Sinc_gradient_norm=float(sinc.low_coordinate.grad.norm()+sinc.band_coordinate.grad.norm()),optimizer_updates=0))
    finish_probe(model,before)
    root=Path(rt['run_root']);write(root/'smoke.json',dict(status='PASS',results=proof,F_logical_packets=258,G_logical_packets=130,
        actual_NoSIG_E30_default_head_G=True,native_Sinc_gradient_enabled=True,optimizer_updates=0,U_training_reads=0,V_reads=0,target_reads=0))
    del model,encoder;folder=root/'source_L_cache';folder.mkdir(exist_ok=False)
    values=np.lib.format.open_memmap(folder/'restored_L.npy',mode='w+',dtype=np.float32,shape=(4,3,6300,2,256));seen=0
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0');n=len(x)
            for r,namespace in enumerate(NAMESPACES):
                for s,scene in enumerate(SCENES):
                    h=source_channel(x,batch[3],scene,contract,namespace=namespace);gx=engine(h.clone())[0].clone()
                    if not bool(torch.isfinite(gx).all()):raise ValueError('Finite restored source L IQ required')
                    values[r,s,seen:seen+n]=fp32_copy(gx)
            seen+=n;write(folder/'progress.json',dict(seen=seen,total=6300,G_cache_packets=12*seen),exclusive=False)
            print('SOURCE_LABELER_L_CACHE',seen,6300,flush=True)
    values.flush();del values
    if seen!=6300 or not unchanged(g,gbefore):raise ValueError('Complete L cache and frozen G required')
    np.savez(folder/'physical_index.npz',physical=np.stack([data.metadata[k] for k in PHYSICAL],axis=1),ids=data.metadata['ids'])
    write(folder/'complete.json',dict(status='LABELER_SOURCE_L_CACHE_COMPLETE',source_contract=contract,G=c['source_G'],shape=[4,3,6300,2,256],
        namespaces=list(NAMESPACES),scenes=SCENES,new_G_cache_packets=75600,IQ_float32_bytes=154828800,G_parameters_and_registered_buffers_unchanged=True,
        F_weight_updates=0,G_updates=0,U_reads=0,V_reads=0,target_reads=0))


def cache(rt,c,contract,metadata):
    import numpy as np
    folder=Path(rt['run_root'])/'source_L_cache';d=read(folder/'complete.json')
    if d['status']!='LABELER_SOURCE_L_CACHE_COMPLETE' or d['source_contract']!=contract or d['G']!=c['source_G']:raise ValueError('Matching complete source L G cache required')
    physical=np.stack([metadata[k] for k in PHYSICAL],axis=1)
    with np.load(folder/'physical_index.npz',allow_pickle=False) as a:
        if not np.array_equal(a['physical'],physical) or list(a['ids'])!=contract['role_ids']['L_s']:raise ValueError('L physical order differs')
    values=np.load(folder/'restored_L.npy',mmap_mode='r',allow_pickle=False)
    if values.shape!=(4,3,6300,2,256) or values.dtype!=np.float32:raise ValueError('Complete FP32 twelve-view L required')
    return values,{tuple(row):i for i,row in enumerate(physical.tolist())}


def train(rt,method):
    import json
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset,FixedEpochLoader
    from experiments.cric_dura_comparison.f_optimizer import build,clip_physical
    model,g,c,contract=make_model(rt);del g
    data=SourceDataset(c['source_cache'],'L_s',contract)
    training_indices=[i for i,r in enumerate(data.metadata['rx_i']) if int(r)!=1]
    if len(training_indices)!=5040:raise ValueError('Fixed additional-adaptation RX1 holdout required')
    loader=FixedEpochLoader(torch.utils.data.Subset(data,training_indices),128,SEED,steps=222,drop_last=True)
    values,lookup=cache(rt,c,contract,data.metadata);root=Path(rt['run_root'])/method;root.mkdir(exist_ok=False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=build(model,parameters,'CENTER_UNITS',lr=.0002,weight_decay=.0001,eps=1e-8,betas=(.9,.999))
    encoder=model.id_backbone.encoder;head=encoder.core.id_backbone.cls_head.weight;old_head=head.detach().clone()
    seen=set();exposures=Counter();step=0;start=time.perf_counter();model.train()
    with gzip.open(root/'step_metrics.jsonl.gz','xt',encoding='utf-8') as stream:
        for epoch in range(1,11):
            for batch in loader:
                step+=1;indices=[lookup[tuple(int(batch[3][k][i]) for k in PHYSICAL)] for i in range(len(batch[0]))];seen.update(indices)
                if len(indices)!=128 or bool((batch[1]<0).any()) or bool((batch[3]['rx_i']==1).any()):raise ValueError('Actual visible L128 excluding adaptation RX1 required')
                view=(step-1)%12;r,s=divmod(view,3)
                x=batch[0].to('cuda:0') if method=='RawLabeler' else tensor(values[r,s,indices],'cuda:0')
                y=batch[1].to('cuda:0');exposures['raw' if method=='RawLabeler' else str(view)]+=len(x)
                lr=.0002*(.01+.99*.5*(1+math.cos(math.pi*(step-1)/(UPDATES-1))));opt.param_groups[0]['lr']=lr
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(encoder(x),y,label_smoothing=.01)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite label expert CE')
                loss.backward()
                if head.grad is None or not bool(torch.isfinite(head.grad).all()) or float(head.grad.norm())<=0:raise ValueError('Default head joint gradient absent')
                sinc=encoder.core.id_backbone.sinc
                if sinc.low_coordinate.grad is None or sinc.band_coordinate.grad is None:raise ValueError('Native Sinc gradient disabled')
                norm=clip_physical(opt,parameters,5.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite label expert gradient')
                head_norm=float(head.grad.norm());opt.step()
                record=dict(step=step,epoch=epoch,loss=float(loss),LR=lr,head_gradient_norm=head_norm,physical_gradient_norm=float(norm),
                    L_packets=128,U_packets=0,view='raw' if method=='RawLabeler' else view,native_U=False,EMA=False,target_reads=0)
                stream.write(json.dumps(record,allow_nan=False)+'\n')
                if step%32==0:stream.flush();write(root/'progress.json',dict(record,seconds=time.perf_counter()-start),exclusive=False)
            print('SOURCE_LABELER_EPOCH',method,epoch,step,flush=True)
    if (step!=UPDATES or opt._sinc_unit_steps!=UPDATES or opt._sinc_physical_clip_calls!=UPDATES or loader.actual_examples!=284160
            or len(seen)!=5040 or model._cd_U_seen or model._cd_U_loader is not None or float((head-old_head).norm())<=0):
        raise ValueError('Complete joint labeler adaptation and zero U exposure required')
    if method=='GLabeler' and dict(exposures)!={str(v):23680 for v in range(12)}:raise ValueError('Equal twelve-view training exposure required')
    cp=dict(schema='pamf_restoration_labeler_v1',run_id=RUN,method=method,model_seed=SEED,model=model.state_dict(),parent=teacher_folder(rt['project']),
        parent_config=c,source_contract=contract,code_commit=rt['code_commit'],new_epochs=10,new_updates=UPDATES,cumulative_epochs=40,
        checkpoint_selection='final_only',U_training_reads=0,target_reads=0,head_fit_updates=0,default_head_joint=True)
    torch.save(cp,root/'final.pt');check_state(cp)
    write(root/'training_complete.json',{k:v for k,v in cp.items() if k not in ('model','parent_config')})
    write(root/'training_evidence.json',dict(status='SOURCE_LABELER_TRAINED',L_exposures=284160,unique_L=5040,additional_adaptation_RX1_packets=0,exposures=dict(exposures),
        optimizer_updates=UPDATES,physical_clip_calls=opt._sinc_physical_clip_calls,head_change_norm=float((head-old_head).norm()),
        native_Sinc_gradient_enabled=True,signature_parameters=0,U_training_reads=0,V_training_reads=0,target_reads=0,seconds=time.perf_counter()-start))


def evaluate(rt,method):
    import numpy as np
    import torch
    from types import MethodType
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.cric_dura_comparison.frozen_inference import frozen_filters
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    model,g,c,contract=make_model(rt);root=Path(rt['run_root'])/method;cp=torch.load(root/'final.pt',map_location='cpu',weights_only=False);check_state(cp)
    if (cp['schema']!='pamf_restoration_labeler_v1' or cp['method']!=method or cp['run_id']!=RUN or cp['parent']!=teacher_folder(rt['project'])
            or cp['source_contract']!=contract or cp['new_updates']!=UPDATES or cp['new_epochs']!=10 or cp['cumulative_epochs']!=40
            or cp['U_training_reads'] or cp['target_reads'] or cp['head_fit_updates']):raise ValueError('Own fixed final joint label expert required')
    model.load_state_dict(cp['model'],strict=True);del cp;model.eval().requires_grad_(False);encoder=model.id_backbone.encoder
    sinc=encoder.core.id_backbone.sinc;sinc._filters=MethodType(frozen_filters,sinc)
    data=SourceDataset(c['source_cache'],'V',contract);n=len(data)
    if n!=27000:raise ValueError('Original full source V required')
    engine=InferenceEngine(g,reuse=True,graph=True,native=False) if method=='GLabeler' else None
    with torch.inference_mode():
        warm=data.x[:2].to('cuda:0');encoder(engine(warm.clone())[0].clone() if engine else warm)
    before=snapshot(model);gbefore=snapshot(g);logits=np.empty((12 if engine else 1,n,6),np.float32);offset=0
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0');count=len(x)
            for view in range(12 if engine else 1):
                value=x
                if engine:
                    r,s=divmod(view,3);h=source_channel(x,batch[3],SCENES[s],contract,namespace=NAMESPACES[r]);value=engine(h.clone())[0].clone()
                out=encoder(value)
                if not bool(torch.isfinite(out).all()):raise FloatingPointError('Nonfinite frozen label expert prediction')
                logits[view,offset:offset+count]=fp32_copy(out)
            offset+=count;write(root/'evaluation_progress.json',dict(seen=offset,total=n,target_reads=0),exclusive=False)
            if offset%2048==0 or offset==n:print('SOURCE_LABELER_V',method,offset,n,flush=True)
    if offset!=n or not unchanged(model,before) or not unchanged(g,gbefore):raise ValueError('Complete frozen labeler/G evaluation required')
    np.savez(root/'source_V_predictions.npz',logits=logits,ids=data.metadata['ids'])
    write(root/'complete.json',dict(status='SOURCE_LABELER_FROZEN_EVALUATED',method=method,model_seed=SEED,source_V=n,new_updates=UPDATES,
        V_label_quality_not_yet_scored=True,F_evaluation_packets=int(logits.shape[0]*n)+2,new_G_evaluation_packets=324002 if engine else 0,
        F_G_parameters_and_registered_buffers_unchanged=True,U_reads=0,target_reads=0))


def quality(rt):
    import numpy as np
    from .restoration_labeler_policy import certify,probability,quality_pass
    root=Path(rt['run_root']);graph=Path(rt['project'])/'runs'/GRAPH_RUN
    contract=read(root/'source_L_cache/complete.json')['source_contract']
    if read(graph/'features_complete.json')['source_contract']!=contract:raise ValueError('Same NoSIG E30/source V roles required')
    logits={}
    for method in METHODS:
        if read(root/method/'complete.json')['status']!='SOURCE_LABELER_FROZEN_EVALUATED':raise ValueError('Both labeler predictions must be fixed')
        with np.load(root/method/'source_V_predictions.npz',allow_pickle=False) as a:
            if list(a['ids'])!=contract['role_ids']['V']:raise ValueError('Labeler V physical order differs')
            logits[method]=a['logits'].copy()
    with np.load(graph/'V/metadata.npz',allow_pickle=False) as a:
        ids=a['ids'];old_conf=a['raw_confidence'];old=a['raw_label'];rx=a['rx_i'];day=a['day_i']
    result=certify(logits['RawLabeler'][0],logits['GLabeler'],old_conf);accepted=result['accepted'];label=result['label']
    np.savez(root/'source_V_labeler_predictions.npz',accepted=accepted,label=label,ids=ids,
        raw_probability=result['raw_probability'],G_mean_probability=result['G_mean_probability'],original_raw_confidence=old_conf)
    # Visible V truth is consumed only after both specialists and all certification predictions are fixed.
    with np.load(graph/'V/metadata.npz',allow_pickle=False) as a:y=a['y']
    raw=result['raw_probability'];gp=result['G_mean_probability'];hard=result['hard_pool']
    stages=[]
    for name,mask,pred in [('RawLabeler',hard&(raw.max(1)>=.95),raw.argmax(1)),('GLabeler',hard&(gp.max(1)>=.95),gp.argmax(1)),('CrossLabeler',accepted,label)]:
        n=int(mask.sum());correct=int((mask&(pred==y)).sum());fixed=int((mask&(old!=y)&(pred==y)).sum());broken=int((mask&(old==y)&(pred!=y)).sum())
        stages.append(dict(method=name,certified=n,correct=correct,precision=100*correct/n if n else None,corrected=fixed,broken=broken,net_correct=fixed-broken))
    cells=[]
    for dim,values in [('overall',np.zeros(len(y),int)),('RX',rx),('day',day),('TX',y)]:
        for value in np.unique(values):
            mask=(values==value)&accepted;n=int(mask.sum());correct=int((mask&(label==y)).sum());fixed=int((mask&(old!=y)&(label==y)).sum());broken=int((mask&(old==y)&(label!=y)).sum())
            cells.append(dict(dimension=dim,stratum='ALL' if dim=='overall' else str(value),certified=n,correct=correct,precision=100*correct/n if n else None,corrected=fixed,broken=broken,net_correct=fixed-broken))
    final=stages[-1];passed=quality_pass(final['certified'],final['correct'],final['net_correct'])
    write(root/'complete.json',dict(status='LABELER_SOURCE_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],model_seed=SEED,
        parent_E30=teacher_folder(rt['project']),new_epochs_per_teacher=10,new_updates_per_teacher=UPDATES,teacher_cumulative_epochs=40,
        source_V=27000,hard_pool=int(hard.sum()),label_quality_pass=passed,
        quality_rule=dict(minimum_certified=200,minimum_precision_percent=99.5,minimum_net_correct=100),stages=stages,quality_cells=cells,
        new_U_training_reads=0,U_truth_read=False,target_reads=0,new_G_training=0,new_student_training=False,U_label_generation=False,
        scope='Single source labeler pilot;ancestors used hiddenU;not target performance or a successful SSL method'))
    print('SOURCE_LABELER_QUALITY',passed,final,flush=True)


def prepare(rt):
    root=Path(rt['run_root']);common=dict(config=rt['runtime_file'],gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_channel_pl.restoration_labeler')
    jobs=[dict(common,id='smoke_cache',kind='smoke_cache',completion=str(root/'source_L_cache/complete.json'),successful_statuses=['LABELER_SOURCE_L_CACHE_COMPLETE'],dependencies=[])]
    jobs += [dict(common,id=m,kind=m,completion=str(root/m/'complete.json'),successful_statuses=['SOURCE_LABELER_FROZEN_EVALUATED'],dependencies=[str(root/'source_L_cache/complete.json')]) for m in METHODS]
    write(root/'jobs.json',jobs);return jobs


def dispatch(rt):
    import subprocess
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    validate_runtime(rt);ctx=read(rt['context']);validate_context(ctx);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    jobs=prepare(rt);by={j['id']:j for j in jobs};state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop=stop or not good;del active[key]
        pending=processes()
        for job in jobs:
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            available=[]
            for gpu in job['gpu_pool']:
                pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
                if load<2 and free>=10000:available.append((load,-free,gpu))
            if not available:continue
            gpu=min(available)[2];child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item;pending[child.pid]=gpu
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:quality(rt)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'LABELER_SOURCE_DIAGNOSTIC_COMPLETE';atomic(root/'pipeline_state.json',state)


def worker(rt,kind):
    validate_runtime(rt)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in list(map(str,range(8))):raise ValueError('One admitted physical GPU required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    if kind=='smoke_cache':return smoke_cache(rt)
    if kind in METHODS:train(rt,kind);return evaluate(rt,kind)
    raise ValueError('Fixed source expert row only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();rt=read(a.config);dispatch(rt) if a.action=='dispatch' else worker(rt,a.kind)
