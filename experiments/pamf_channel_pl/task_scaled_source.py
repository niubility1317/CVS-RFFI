"""No-optimizer whole-model dose for uniform and task-scaled identity references."""
import argparse
import math
import os
from pathlib import Path
import statistics
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,snapshot,unchanged,SEEDS,SCENES
from .restored_dose_source import load_sources,no_native_U
from .receiver_reference_source import student_from_parent
from .receiver_reference import relation_loss,metadata_long
from .task_scaled_reference import task_scaled_loss,MAX_SCALE
from .receiver_reference_training import CACHE_RUN,DOSE_RUN

RUN='20261026-diagnostic-PAMF-TaskScaledRef-dose-m1-r01'
VARIANTS=('GUniform','GTaskScaled','RawTaskScaled')
BATCHES=12


def extra(student,view,domain,reference,rx,rxs,target,accepted,variant,check_head=False):
    import torch
    if variant not in VARIANTS:raise ValueError('Fixed uniform/Gtask/rawtask source variants required')
    out=student.forward_identity_only(view,domain_labels=domain);z=out['z_id']
    if variant=='GUniform':
        loss=relation_loss(z,reference,rx,rxs,target,accepted)
        return loss,dict(task_scale_mean=1.,task_scale_max=1.,task_scale_sum=float(accepted.sum()),task_scale_cap_fraction=0.)
    # Use the existing default head on the exact returned feature tensor; the
    # native forward may expose an equal-valued feature through another alias.
    logits=student.id_backbone.encoder.classify_features(z)
    if check_head and not torch.allclose(logits,out['tx_logits'],atol=1e-5,rtol=1e-5):
        raise ValueError('Recomputed actual default head differs from student forward')
    return task_scaled_loss(z,logits,reference,rx,rxs,target,accepted)


def decide(rows):
    if len(rows)!=BATCHES:raise ValueError('Twelve complete matched source batches required')
    main=[r['primary_norm'] for r in rows];extras={v:[r[v+'_norm'] for r in rows] for v in VARIANTS}
    medians={v:statistics.median(values) for v,values in extras.items()}
    norms=main+sum(extras.values(),[]);numbers=norms+[r[v+'_cosine'] for r in rows for v in VARIANTS]
    if any(not math.isfinite(v) for v in numbers) or any(v<=0 for v in norms):
        raise ValueError('Finite nonzero source gradient norms required')
    weight=min(.25,.10*statistics.median(main)/max(medians.values()))
    return dict(selected_weight=weight,median_primary_norm=statistics.median(main),median_extra_norm=medians,
        median_cosine={v:statistics.median(r[v+'_cosine'] for r in rows) for v in VARIANTS},
        weighted_median_norm_ratio={v:weight*x/statistics.median(main) for v,x in medians.items()},
        rule='min(.25,.10*median(primary_norm)/max_variant median(extra_norm));one common dose;no target feedback',
        scope='Whole-model preclip CENTER_UNITS physical gradient proxy;not AdamW update guarantee or target benefit')


def run(rt):
    if rt['run_id']!=RUN or rt['allowed_gpus']!=[0] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Fixed task-scaled source-dose/GPU0 run required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_restored_pl.cache import SourceCache
    torch.set_num_threads(2);start=time.perf_counter();root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    c=read(rt['context']);validate_context(c)
    write(root/'pipeline_state.json',dict(status='SOURCE_LOADING',code_commit=rt['code_commit'],pid=os.getpid()))
    teachers,configs,payloads,contract,g=load_sources(c,'cuda:0')
    torch.manual_seed(SEEDS[0]);torch.cuda.manual_seed_all(SEEDS[0])
    student,cfg=student_from_parent(configs[0],payloads[0]);del payloads,g
    folder=Path(rt['project'])/'runs';ref_header=torch.load(folder/DOSE_RUN/'references.pt',map_location='cpu',weights_only=False)
    cache_root=folder/CACHE_RUN/'source_U_cache';cache_header=read(cache_root/'complete.json')
    if (ref_header['source_contract']!=contract or ref_header['reference_role']!='L_s' or ref_header['teacher_epoch']!=30
            or ref_header['model_seeds']!=list(SEEDS) or ref_header['source_cases']!=cache_header['source_cases']
            or any(ref_header[k] for k in ('U_reads','V_reads','target_reads')) or cache_header['G']!=c['G']):
        raise ValueError('Fixed visible-L references/cached raw targets/same G required')
    reference=ref_header['references'][0].to('cuda:0').detach();rxs=tuple(sorted(contract['source_rxs']));held=rxs[0]
    cache=SourceCache(dict(configs[0],source_cache_output=str(cache_root)))
    labeled=SourceDataset(cfg['source_cache'],'L_s',contract);unlabeled=SourceDataset(cfg['source_cache'],'U_s',contract)
    x=labeled.x[:2].to('cuda:0');rx=metadata_long(labeled.metadata['rx_i'][:2]).to('cuda:0')
    target=F.one_hot(metadata_long(labeled.metadata['y'][:2]).to('cuda:0'),6).float();mask=torch.ones(2,device='cuda:0',dtype=torch.bool)
    student.eval()
    for variant in VARIANTS:
        value,metrics=extra(student,x,torch.zeros(2,device='cuda:0',dtype=torch.long),reference,rx,rxs,target,mask,variant,True)
        if not bool(torch.isfinite(value)):raise ValueError('Actual E30 task-scaled source smoke not finite')
    with torch.no_grad():
        out=student.forward_identity_only(x,domain_labels=torch.zeros(2,device='cuda:0',dtype=torch.long))
        if not torch.allclose(out['z_id'],teachers[0].features(x),atol=1e-5,rtol=1e-5):raise ValueError('Actual own E30 features differ')
    write(root/'smoke.json',dict(status='PASS',source_role='L_s',physical_packets=2,actual_E30=True,default_head=True,
        recomputed_head_matches=True,task_scaled_feature_graph=True,optimizer_updates=0,target_reads=0))
    student.train();before=snapshot(student);frozen=[snapshot(m) for m in teachers]
    named=[(name,p) for name,p in student.named_parameters() if p.requires_grad];params=[p for _,p in named]
    sinc=student.get_submodule('id_backbone.encoder.core.id_backbone.sinc')
    if sinc.scale_hz!=25000. or sinc.mode!='CENTER_25K':raise ValueError('Original Sinc physical coordinates required')
    def generator():return torch.Generator().manual_seed(SEEDS[0])
    ll=iter(torch.utils.data.DataLoader(labeled,batch_size=128,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    ul=iter(torch.utils.data.DataLoader(unlabeled,batch_size=256,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    def grad(loss):
        values=torch.autograd.grad(loss,params,allow_unused=True);parts=[];head=0.
        for (name,p),v in zip(named,values):
            value=torch.zeros_like(p) if v is None else v
            if name.endswith('cls_head.weight'):head=float(value.detach().norm())
            if name in ('id_backbone.encoder.core.id_backbone.sinc.low_coordinate','id_backbone.encoder.core.id_backbone.sinc.band_coordinate'):
                value=value/25000.
            parts.append(value.reshape(-1))
        result=torch.cat(parts).detach()
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite full source gradient')
        return result,head
    def restore(state):
        with torch.no_grad():
            for name,value in student.named_buffers():value.copy_(state[name])
    rows=[]
    with no_native_U():
        for step in range(1,BATCHES+1):
            lb,ub=next(ll),next(ul);lx,ly,ld=[v.to('cuda:0') for v in lb[:3]]
            ux=ub[0].to('cuda:0');ud=ub[2].to('cuda:0');urx=ub[3]['rx_i'].to('cuda:0')
            if not bool((ub[1]==-1).all()) or bool(torch.as_tensor(ub[3]['tx_label_visible']).any()):raise ValueError('Hidden source U required')
            q,accepted,gx=cache.take(ub[3],(step-1)%3,'cuda:0',True);accepted=accepted&(urx!=held)
            student.train()
            with student._phase1_labeled_feature_context():output=student(lx,return_aux=True,domain_labels=ld)
            ce=F.cross_entropy(output['tx_logits'],ly,label_smoothing=.01)
            correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=output,labels=ly,
                extra=(None,lb[3]),domains=ld,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=lx)
            gl,hl=grad(ce+correction);student.eval()
            with torch.no_grad():own_conf,own_label=student(ux).softmax(1).max(1)
            student.train();logits=student.forward_identity_only(ux,domain_labels=ud)['tx_logits']
            gu,hu=grad((F.cross_entropy(logits,own_label,reduction='none')*(own_conf>=.95)).mean());primary=gl+gu
            row=dict(step=step,scene=SCENES[(step-1)%3],L_seen=128,U_seen=256,extra_accepted=int(accepted.sum()),held_extra_RX=held,
                primary_norm=float(primary.norm()),L_norm=float(gl.norm()),raw_U_norm=float(gu.norm()),
                primary_L_head_norm=hl,primary_U_head_norm=hu,L_forward_packets=int(metrics['T1_L_packets']),As_KL_weight=float(metrics['T1_As_KL_weight']))
            temporary={name:value.detach().clone() for name,value in student.named_buffers()}
            cpu_rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
            for variant in VARIANTS:
                restore(temporary);torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                view=ux if variant=='RawTaskScaled' else gx;losses=[];detail=[]
                for _ in range(2):
                    loss,d=extra(student,view,ud,reference,urx,rxs,q,accepted,variant,True);losses.append(loss);detail.append(d)
                loss=sum(losses)/2;ge,head=grad(loss)
                if head!=0.:raise ValueError('Task-scaled auxiliary must not update default-head parameters')
                row.update({variant+'_norm':float(ge.norm()),variant+'_cosine':float(F.cosine_similarity(primary[None],ge[None],dim=1)[0]),
                    variant+'_loss':float(loss.detach()),variant+'_head_gradient':head})
                for key in ('task_scale_mean','task_scale_max','task_scale_sum','task_scale_cap_fraction'):
                    values=[float(d[key]) for d in detail]
                    row[variant+'_'+key]=max(values) if key=='task_scale_max' else sum(values)/2
            if row['L_forward_packets']!=768 or row['As_KL_weight']!=0 or hl<=0:raise ValueError('Original L/default-head primary differs')
            rows.append(row);restore(before)
            write(root/'progress.json',dict(status='SOURCE_TASK_SCALED_GRADIENTS',batches=step,total=BATCHES,optimizer_updates=0,V_reads=0,target_reads=0),exclusive=False)
            print('SOURCE_TASK_SCALED',step,BATCHES,flush=True)
    current=dict(student.state_dict());current.update(dict(student.named_buffers()))
    if (set(current)!=set(before) or any(not torch.equal(value,before[name]) for name,value in current.items())
            or any(p.grad is not None for p in student.parameters()) or any(not unchanged(m,b) for m,b in zip(teachers,frozen))):
        raise ValueError('Source calibration retained parameter/buffer/gradient changes')
    result=dict(status='SOURCE_TASK_SCALED_DOSE_COMPLETE',code_commit=rt['code_commit'],rows=rows,decision=decide(rows),
        variants=list(VARIANTS),model_seed=SEEDS[0],L_exposures=1536,U_exposures=3072,full_parameter_gradient_calls=60,
        task_scale_max=MAX_SCALE,held_extra_RX=held,original_L_primary=True,ordinary_raw_PL_primary=True,
        student_and_teacher_parameters_and_registered_buffers_unchanged=True,temporary_buffers_restored=True,
        default_head_auxiliary_parameter_gradients=0,optimizer_updates=0,V_reads=0,target_reads=0,U_truth_read=False,
        reference_updates=0,G_packet_forwards=0,head_fit_updates=0,seconds=time.perf_counter()-start,
        scope='No-update whole-model source dose andscale diagnostics;not training,AdamW guarantee or target benefit')
    write(root/'gradient_complete.json',result);write(root/'complete.json',result)
    write(root/'pipeline_state.json',dict(status=result['status'],code_commit=rt['code_commit'],pid=os.getpid()),exclusive=False)
    print('SOURCE_TASK_SCALED_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    a=p.parse_args();run(read(a.config))
