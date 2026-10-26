"""Real SIG1 E20 source-only gradient/route diagnostic; no optimizer or target."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import time
from experiments.cric_dura_comparison.io import read,write

RUN='20261026-diagnostic-PAMF-SIG1-RXTrust-source-s2026092701-r01'
RELEASE='20261026-diagnostic-PAMF-SIG1-RXTrust-source-release-r01'
CACHE_RUN='20261026-diagnostic-PAMF-URealizations-cache-m3-r01'
RECIPE=dict(batches=9,L_batch=128,U_batch=256,weak_SNR_dB=35.,recovery_noise_SNR_dB=35.,
    hard_threshold=.95,hard_cap=.25,temperature=.1,scope=['U','LU'],optimizer_updates=0,
    V_records=27000,source_risk_precision=.995,minimum_trusted_V=200,
    gradient_coefficient='min(.25,.1*median(base_norm)/max(median(U_norm),median(LU_norm))) per component;not a performance guarantee')


def validate(c):
    if (c.get('run_id')!=RUN or c.get('recipe')!=RECIPE or c.get('run_root')!=c['project']+'/runs/'+RUN
        or c.get('release_root')!=c['project']+'/releases/'+RELEASE or c.get('allowed_gpus')!=list(range(8))
        or any(k in c for k in ('target_inputs','truth','target_truth','tests','dataset_path','checkpoint','teacher_checkpoint'))):
        raise ValueError('One immutable source-only SIG1 RX-trust probe required')
    return c


class Cache:
    def __init__(self,c,contract,metadata):
        import numpy as np
        from .history import KEYS
        from experiments.pamf_channel_pl.u_realization_cache import namespace
        from experiments.pamf_channel_pl.restored_source import SCENES
        root=Path(c['project'])/'runs'/CACHE_RUN;done=read(root/'complete.json')
        if (done['status']!='SOURCE_U_REALIZATION_CACHE_COMPLETE' or done['replicas']!=[1,2,3]
            or done['U_physical_records']!=56700 or not done['all_G_parameters_and_registered_buffers_unchanged']
            or any(done[k] for k in ('F_weight_reads','F_output_reads','F_forward_packets','U_truth_read','V_reads','target_reads','optimizer_updates','G_updates'))):
            raise ValueError('Completed G-only hidden-U cache required')
        if 'y' in metadata:raise ValueError('No U TX labels')
        physical=np.stack([metadata[k] for k in KEYS],axis=1)
        self.index={tuple(int(v) for v in row):i for i,row in enumerate(physical)};self.arrays={}
        if len(self.index)!=56700:raise ValueError('Complete unique U physical records required')
        for replica in (1,2,3):
            p=root/('fresh'+str(replica));header=read(p/'complete.json')
            if (header['status']!='SOURCE_U_HG_REALIZATION_COMPLETE' or header['source_contract']!=contract
                or header['G']!=c['G'] or header['U_seen']!=56700 or header['namespace']!=namespace(replica)
                or header['scenes']!=list(SCENES) or not header['G_parameters_and_registered_buffers_unchanged']):
                raise ValueError('Matched immutable U/G cache provenance required')
            with np.load(p/'physical_index.npz',allow_pickle=False) as a:
                if not np.array_equal(a['physical'],physical) or list(a['ids'])!=contract['role_ids']['U_s']:
                    raise ValueError('U cache physical order differs')
            a=np.load(p/'restored_U.npy',mmap_mode='r',allow_pickle=False)
            if a.shape!=(3,56700,2,256) or a.dtype!=np.float32:raise ValueError('Exact FP32 U view shape required')
            self.arrays[replica]=a

    def take(self,batch,replica,scene,device):
        import numpy as np
        from experiments.cric_dura_comparison.io import tensor
        from .history import KEYS
        if not (batch[1]==-1).all() or batch[3]['tx_label_visible'].any():raise ValueError('U truth exposed')
        rows=zip(*(batch[3][k].tolist() for k in KEYS));indices=[self.index[tuple(v)] for v in rows]
        a=np.array(self.arrays[replica][scene,indices],copy=True)
        if not np.isfinite(a).all():raise ValueError('Nonfinite cached IQ')
        return tensor(a,device)


def gradient_summary(loss,base,named):
    import torch
    values=torch.autograd.grad(loss,[p for _,p in named],retain_graph=True,allow_unused=True)
    norm=dot=base_norm=0.
    for g,b in zip(values,base):
        if b is not None:base_norm+=float(b.detach().double().square().sum())
        if g is not None:
            if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite actual auxiliary gradient')
            norm+=float(g.detach().double().square().sum())
            if b is not None:dot+=float((g.detach().double()*b.detach().double()).sum())
    return dict(loss=float(loss.detach()),norm=norm**.5,base_norm=base_norm**.5,
        cosine=dot/(norm*base_norm)**.5 if norm*base_norm>0 else None,dot=dot)


def probe(c):
    validate(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    from torch.nn import functional as F
    from .source import config,load_prefix,SEED
    from .model import install_single_view,validate_single_view
    from .receiver_trust import gaussian_view,instance_alignment,cross_receiver_relation,hard_view_trust
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cric_dura_comparison.f_data import SourceDataset,FixedEpochLoader
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison.frozen_inference import load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from experiments.pamf_channel_pl.restored_source import snapshot,source_channel,SCENES
    torch.set_num_threads(2);torch.manual_seed(2026102609);torch.cuda.manual_seed_all(2026102609)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ctx=read(c['context']);cfg=config(ctx,c['code_commit'],'PL-History');parent,payload=load_prefix(cfg)
    contract=read(cfg['source_contract'])
    if c['G']!=ctx['G']:raise ValueError('Frozen matched G required')
    dataL=SourceDataset(cfg['source_cache'],'L_s',contract);dataU=SourceDataset(cfg['source_cache'],'U_s',contract)
    cache=Cache(c,contract,dataU.metadata)
    # The History configuration supplies the already checked source skeleton only;
    # its optional objective installer is not called. t1_model.plain_pl stays native.
    with f_train.class_dimension(cfg),native_installed(f_train.original_config(SEED)) as native:
        args=f_train.arguments(cfg,'cuda:0');args.input_len=256;args.baseline_ckpt=args.teacher_ckpt=''
        args.from_scratch=args.a1_scratch_only=True
        student=install_single_view(t1_model.install(native.build_baseline_model(args,'cuda:0'),cfg),SEED).to('cuda:0')
    student.load_state_dict(payload['model'],strict=True);del payload;validate_single_view(student)
    g=load_G(c['G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    with torch.inference_mode():engine(dataL.x[:2].to('cuda:0'))
    before=snapshot(student);gbefore=snapshot(g);named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
    L=iter(FixedEpochLoader(dataL,128,2026102609,steps=9,drop_last=True));U=iter(FixedEpochLoader(dataU,256,2026102609,steps=9))
    noise=torch.Generator(device='cuda').manual_seed(2026102611);rows=[];out=Path(c['run_root']);start=time.monotonic()
    for step in range(9):
        lb=next(L);ub=next(U)
        x,y,d=[v.to('cuda:0') for v in lb[:3]];u=ub[0].to('cuda:0');ud=ub[2].to('cuda:0')
        student.train();student.zero_grad(set_to_none=True);student._feature_l_steps=4440+step
        student._cd_U_iter=iter([ub]);student._cd_U_loader=[ub]
        with student._phase1_labeled_feature_context():first=student(x,return_aux=True,domain_labels=d)
        ce=F.cross_entropy(first['tx_logits'],y,label_smoothing=.01)
        correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=first,labels=y,
            extra=(None,lb[3]),domains=d,optimizer_step=student._feature_l_steps,identity_loss=ce,labeled_count=128,labeled_inputs=x)
        base=ce+correction
        if metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or metrics['T1_PL_seen']!=(step+1)*256:
            raise ValueError('Unchanged full T1 L and ordinary PL must execute exactly once')
        base_grad=torch.autograd.grad(base,[p for _,p in named],retain_graph=True,allow_unused=True)
        student.eval()
        with torch.no_grad():
            lref=student.forward_identity_only(x,domain_labels=d)['z_id'].detach()
            uref=student.forward_identity_only(u,domain_labels=ud)['z_id'].detach()
            probabilities=[student(u).softmax(1)]
            for _ in range(2):probabilities.append(student(gaussian_view(u,snr_db=35.,generator=noise)).softmax(1))
            pseudo,trust,route=hard_view_trust(torch.stack(probabilities),ub[3]['rx_i'].to('cuda:0'))
            if not torch.isfinite(trust).all():raise ValueError('Nonfinite trust')
        a=step%3+1;b=a%3+1;k=step//3;other=(k+1)%3
        ug=[cache.take(ub,a,k,'cuda:0'),cache.take(ub,b,other,'cuda:0')];lg=[]
        with torch.inference_mode():
            for j,scene in enumerate((k,other)):
                h=source_channel(x,lb[3],SCENES[scene],contract,namespace='PAMF/RXTrust/L/probe/'+str(step)+'/'+str(j))
                lg.append(engine(h.clone())[0].clone())
        student.train();zs=[]
        for left,right in zip(lg,ug):
            view=gaussian_view(torch.cat((left,right)),snr_db=35.,generator=noise)
            zs.append(student.forward_identity_only(view,domain_labels=torch.cat((d,ud)))['z_id'])
        zs=torch.stack(zs);rx=torch.cat((lb[3]['rx_i'],ub[3]['rx_i'])).to('cuda:0')
        labels=torch.cat((y,torch.full_like(pseudo,-1)));isL=torch.arange(384,device='cuda:0')<128
        targets=torch.cat((torch.zeros_like(y),pseudo));weights=torch.cat((torch.ones_like(y,dtype=trust.dtype),trust))
        details={}
        for scope in ('U','LU'):
            selected=torch.cat((zs[:,:128].detach(),zs[:,128:]),dim=1) if scope=='U' else zs
            inst=instance_alignment(selected,torch.cat((lref,uref)))
            relation,counts=cross_receiver_relation(zs.mean(0),rx=rx,source_rxs=cfg['source_rxs'],labeled=isL,
                visible_tx=labels,pseudo=targets,trust=weights,scope=scope,temperature=.1)
            details[scope]=dict(instance=gradient_summary(inst,base_grad,named),relation=gradient_summary(relation,base_grad,named),counts=counts)
            if counts['anchors']==0 or details[scope]['instance']['norm']<=0 or details[scope]['relation']['norm']<=0:
                raise ValueError('Actual same-packet and cross-RX source gradients required')
        row=dict(step=step+1,L=128,U=256,replicas=[a,b],scenes=[SCENES[k],SCENES[other]],
            ordinary_PL_accepted=metrics['T1_PL_accepted']-(rows[-1]['ordinary_PL_cumulative_accepted'] if rows else 0),
            ordinary_PL_cumulative_accepted=metrics['T1_PL_accepted'],route=route,gradients=details)
        rows.append(row)
        with (out/'gradient_steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if step==0:write(out/'smoke.json',dict(status='PASS',actual_SIG1_E20=True,head_shape=[6,160],L_packets=768,U=256,optimizer_updates=0,U_truth_read=False,target_reads=0))
        print('RX_TRUST_SOURCE_STEP',step+1,'trusted',route['selected'],flush=True)
        del base_grad,base,correction,first,ce,zs,details,inst,relation,selected
    student.zero_grad(set_to_none=True);student.eval().requires_grad_(False)
    def parameters_unchanged(model,original):
        now=snapshot(model)
        return set(now)==set(original) and all(torch.equal(now[k],v) for k,v in original.items())
    if not parameters_unchanged(student,before) or not parameters_unchanged(g,gbefore):
        raise ValueError('Source diagnostic changed parameter or registered buffer')
    # Full V only evaluates the fixed shared-head route. It never enters gradients.
    v=SourceDataset(cfg['source_cache'],'V',contract);quality={key:dict(n=0,correct=0) for key in ('ordinary_PL','extra_hard')};byrx={str(r):dict(n=0,correct=0) for r in cfg['source_rxs']}
    loader=torch.utils.data.DataLoader(v,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(2026102613),num_workers=0)
    with torch.no_grad():
        for batch in loader:
            x=batch[0].to('cuda:0');truth=batch[1].to('cuda:0');r=batch[3]['rx_i'].to('cuda:0')
            p=[student(x).softmax(1)]+[student(gaussian_view(x,snr_db=35.,generator=noise)).softmax(1) for _ in range(2)]
            predicted,weight,_=hard_view_trust(torch.stack(p),r);masks={'ordinary_PL':p[0].max(1).values>=.95,'extra_hard':weight>0}
            for key,mask in masks.items():quality[key]['n']+=int(mask.sum());quality[key]['correct']+=int((mask&(predicted==truth)).sum())
            for receiver in cfg['source_rxs']:
                mask=masks['extra_hard']&(r==receiver);byrx[str(receiver)]['n']+=int(mask.sum());byrx[str(receiver)]['correct']+=int((mask&(predicted==truth)).sum())
    for record in [*quality.values(),*byrx.values()]:record['precision_percent']=100*record['correct']/record['n'] if record['n'] else None
    if not parameters_unchanged(student,before) or not parameters_unchanged(g,gbefore):raise ValueError('V changed model state')
    coefficients={}
    for component in ('instance','relation'):
        base_norm=statistics.median(r['gradients']['U'][component]['base_norm'] for r in rows)
        extra=max(statistics.median(r['gradients'][s][component]['norm'] for r in rows) for s in ('U','LU'))
        coefficients[component]=min(.25,.1*base_norm/max(extra,1e-12))
    write(out/'result.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=RUN,code_commit=c['code_commit'],recipe=RECIPE,
        source_prefix=parent['folder'],actual_checkpoint_epoch=20,actual_checkpoint_updates=4440,signature_views=1,signature_parameters=18976,
        L_seen=1152,U_seen=2304,V_seen=len(v),gradient_rows=rows,quality=quality,extra_hard_by_RX=byrx,gradient_scale_suggestion=coefficients,
        route_source_necessary_pass=quality['extra_hard']['n']>=200 and all(r['n']>=20 and r['precision_percent']>=99.5 for r in byrx.values()),
        all_parameters_and_registered_buffers_unchanged=True,optimizer_updates=0,U_truth_read=False,target_reads=0,G_updates=0,
        source_only=True,performance='UNTESTED',seconds=time.monotonic()-start))
    print('RX_TRUST_SOURCE_DIAGNOSTIC_COMPLETE',time.monotonic()-start,flush=True)


def dispatch(rt):
    validate(rt)
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.practical_full_2x2.cache import atomic
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    path=root/'config.json';write(path,rt);child=None;state=dict(status='QUEUED',controller_pid=os.getpid(),code_commit=rt['code_commit'])
    while child is None:
        pending=processes();free=[]
        for gpu in rt['allowed_gpus']:
            pids,memory=capacity(gpu)
            if not (pids|{pid for pid,g in pending.items() if g==gpu}) and memory>=10000:free.append((-memory,gpu))
        if free:
            gpu=min(free)[1];job=dict(id='probe',kind='probe',config=str(path),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.receiver_probe')
            child,item=launch(rt,job,gpu);state.update(status='RUNNING',worker=item)
        atomic(root/'pipeline_state.json',state)
        if child is None:time.sleep(12)
    code=child.wait();good=code==0 and (root/'result.json').exists()
    state.update(status='SOURCE_DIAGNOSTIC_COMPLETE' if good else 'FAILED',exit_code=code,finished_at=time.time());atomic(root/'pipeline_state.json',state)
    if not good:raise RuntimeError('Source probe failed;preserve output and do not relaunch blindly')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else probe(c)
