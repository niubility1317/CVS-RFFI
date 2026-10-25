"""Frozen source L/U feature-gradient alignment; no training or V/target access."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import statistics
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,snapshot,unchanged,SEEDS,SCENES
from .restored_dose_source import load_sources,soft_loss
from .receiver_reference import relation_loss,metadata_long
from .receiver_reference_training import RUN as RX_RUN,CACHE_RUN,DOSE_RUN,parents,VARIANTS

RUN='20261026-diagnostic-PAMF-RXReference-alignment-m1-r01'
BATCHES=12


def feature_gradients(model,features,references,rx,rxs,target):
    import torch
    from torch.nn import functional as F
    if (any(p.requires_grad for p in model.parameters()) or model.training or target.requires_grad or references.requires_grad):
        raise ValueError('Frozen eval/default head and detached source targets/references required')
    z=features.detach().clone().requires_grad_(True);mask=torch.ones(len(z),dtype=torch.bool,device=z.device)
    logits=model.classify_features(z);hard=target.argmax(1)
    gs=torch.autograd.grad(soft_loss(logits,target,mask),z,retain_graph=True)[0]
    gh=torch.autograd.grad(F.cross_entropy(logits,hard),z)[0]
    gr=torch.autograd.grad(relation_loss(z,references,rx,rxs,target,mask),z)[0]
    if any(not bool(torch.isfinite(v).all()) for v in (logits,gs,gh,gr)):
        raise ValueError('Finite source feature gradients required')
    ns=gs.norm(dim=1);nh=gh.norm(dim=1);nr=gr.norm(dim=1)
    valid_s=(ns>1e-12)&(nr>1e-12);valid_h=(nh>1e-12)&(nr>1e-12)
    return dict(soft_cosine=F.cosine_similarity(gs,gr,dim=1,eps=1e-12).clamp(-1,1).detach(),
        hard_cosine=F.cosine_similarity(gh,gr,dim=1,eps=1e-12).clamp(-1,1).detach(),
        soft_gradient_norm=ns.detach(),hard_gradient_norm=nh.detach(),reference_gradient_norm=nr.detach(),
        valid_soft=valid_s,valid_hard=valid_h,head_agrees_with_target=logits.detach().argmax(1)==hard,
        head_target_probability=logits.detach().softmax(1).gather(1,hard[:,None])[:,0],
        reference_to_hard_norm_ratio=(nr/nh.clamp_min(1e-12)).detach())


def run(rt):
    if rt['run_id']!=RUN or rt['allowed_gpus']!=[0] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Fixed source alignment diagnostic/GPU0 required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_repl_completion.source import load_reader
    from experiments.pamf_restored_pl.cache import SourceCache
    torch.set_num_threads(2);start=time.perf_counter();root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    c=read(rt['context']);validate_context(c)
    write(root/'pipeline_state.json',dict(status='SOURCE_LOADING',code_commit=rt['code_commit'],pid=os.getpid()))
    old,configs,payloads,contract,g=load_sources(c,'cuda:0');models=[old[0]];names=['E30'];del old,payloads,g
    froot=Path(rt['project'])/'runs';folder=froot/CACHE_RUN/'F'/('F0-PLControl-s'+str(SEEDS[0]))/'source'
    control,cfg=load_reader(folder,contract,'cuda:0');models.append(control);names.append('PLControl')
    if cfg['model_seed']!=SEEDS[0] or cfg['paper_method']!='PLControl':raise ValueError('Own original PLControl required')
    for method in VARIANTS:
        folder=froot/RX_RUN/'F'/('F0-'+method+'-sourceHold-s'+str(SEEDS[0]))/'source'
        done=read(folder/'complete.json');cfg=done['config'];chain,payload=parents(cfg);del chain,payload
        if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read']
                or read(cfg['source_contract'])!=contract or cfg['paper_method']!=method):raise ValueError('Three fixed source-only own E30+20 students required')
        cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if (cp['run_id']!=RX_RUN or cp['candidate_id']!=cfg['row_id'] or cp['epoch']!=20 or cp['a1_ema_successful_updates']!=4440
                or cp['checkpoint_selection']!='final_only' or cp['args']['use_unlabeled'] or cp['args']['use_ema_teacher'] or cp['args']['teacher_ckpt']):
            raise ValueError('Actual final own default-head student differs')
        models.append(reader_from_payload(cfg,cp,'cuda:0'));names.append(method);del cp
    refs=torch.load(froot/DOSE_RUN/'references.pt',map_location='cpu',weights_only=False)
    cache_root=froot/CACHE_RUN/'source_U_cache';cache_header=read(cache_root/'complete.json')
    if (refs['source_contract']!=contract or refs['model_seeds']!=list(SEEDS) or refs['teacher_epoch']!=30
            or refs['reference_role']!='L_s' or refs['U_reads'] or refs['V_reads'] or refs['target_reads']
            or refs['source_cases']!=cache_header['source_cases'] or cache_header['G']!=c['G']):
        raise ValueError('Original visible-L references/raw targets/same G cache required')
    reference=refs['references'][0].to('cuda:0').detach();rxs=tuple(sorted(contract['source_rxs']))
    cache=SourceCache(dict(configs[0],source_cache_output=str(cache_root)))
    labeled=SourceDataset(configs[0]['source_cache'],'L_s',contract);unlabeled=SourceDataset(configs[0]['source_cache'],'U_s',contract)
    def generator():return torch.Generator().manual_seed(SEEDS[0])
    ll=iter(torch.utils.data.DataLoader(labeled,batch_size=128,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    ul=iter(torch.utils.data.DataLoader(unlabeled,batch_size=256,shuffle=True,generator=generator(),num_workers=0,drop_last=True))
    x=labeled.x[:2].to('cuda:0');rx=metadata_long(labeled.metadata['rx_i'][:2]).to('cuda:0')
    target=F.one_hot(metadata_long(labeled.metadata['y'][:2]).to('cuda:0'),6).float()
    for m in models:
        with torch.no_grad():feature=m.features(x);logits=m(x)
        if feature.shape!=(2,160) or logits.shape!=(2,6) or not torch.allclose(m.classify_features(feature),logits,atol=1e-5,rtol=1e-5):
            raise ValueError('Five actual default-head source L2 feature/readout smoke differs')
        feature_gradients(m,feature,reference,rx,rxs,target)
    write(root/'smoke.json',dict(status='PASS',source_role='L_s',physical_packets=2,frozen_F=5,
        actual_default_heads=True,feature_gradients_finite=True,model_parameter_updates=0,optimizer_updates=0,target_reads=0))
    before=[snapshot(m) for m in models];buckets=defaultdict(lambda:defaultdict(list));batch_rows=[]
    for step in range(1,BATCHES+1):
        lb,ub=next(ll),next(ul);lx=lb[0].to('cuda:0');ux=ub[0].to('cuda:0')
        if not bool((ub[1]==-1).all()) or bool(torch.as_tensor(ub[3]['tx_label_visible']).any()):raise ValueError('Source U truth must stay hidden')
        q,accepted,gx=cache.take(ub[3],(step-1)%3,'cuda:0',True)
        ly=F.one_hot(lb[1].to('cuda:0'),6).float();lr=lb[3]['rx_i'].to('cuda:0');ur=ub[3]['rx_i'].to('cuda:0')
        views=[('L_raw',lx,lr,ly,torch.ones(len(lx),device='cuda:0',dtype=torch.bool),'visible_L'),
            ('U_raw',ux,ur,q,accepted,'frozen_raw_pseudo'),('U_G_'+SCENES[(step-1)%3],gx,ur,q,accepted,'frozen_raw_pseudo')]
        for name,m in zip(names,models):
            for view,signal,rx,target,accept,truth_kind in views:
                with torch.no_grad():feature=m.features(signal)
                values=feature_gradients(m,feature,reference,rx,rxs,target)
                base=[('all_seen',torch.ones(len(signal),device='cuda:0',dtype=torch.bool)),('accepted',accept),
                    ('accepted_held_RX',accept&(rx==rxs[0])),('accepted_trained_RX',accept&(rx!=rxs[0]))]
                agree=values['head_agrees_with_target']
                bucket_views=[view,'U_G_ALL'] if view.startswith('U_G_') else [view]
                for bucket_view in bucket_views:
                    for group,mask in base:
                        for decision,subset in [('all',mask),('agrees',mask&agree),('disagrees',mask&~agree)]:
                            b=buckets[name,bucket_view,truth_kind,group,decision];b['count'].append(int(subset.sum()))
                            for key,value in values.items():
                                if key in ('valid_soft','valid_hard','head_agrees_with_target'):continue
                                valid=values['valid_soft'] if key=='soft_cosine' else values['valid_hard'] if key in ('hard_cosine','reference_to_hard_norm_ratio') else torch.ones_like(subset)
                                b[key].extend(value[subset&valid].detach().cpu().tolist())
                batch_rows.append(dict(step=step,model=name,view=view,seen=len(signal),accepted=int(accept.sum()),
                    target_kind=truth_kind,agreement_with_target_count=int(agree.sum()),model_updates=0))
        write(root/'progress.json',dict(status='SOURCE_FEATURE_ALIGNMENT',batches=step,total=BATCHES,optimizer_updates=0,V_reads=0,target_reads=0),exclusive=False)
        print('SOURCE_ALIGNMENT',step,BATCHES,flush=True)
    if any(not unchanged(m,b) for m,b in zip(models,before)) or any(p.grad is not None for m in models for p in m.parameters()):
        raise ValueError('Frozen model parameters/registered buffers mutated or got gradients')
    rows=[]
    for (model,view,kind,group,decision),b in buckets.items():
        row=dict(model=model,model_seed=SEEDS[0],view=view,target_kind=kind,group=group,decision=decision,n=sum(b['count']))
        for key,values in b.items():
            if key=='count':continue
            row[key+'_valid_n']=len(values);row[key+'_mean']=statistics.mean(values) if values else None
            row[key+'_median']=statistics.median(values) if values else None
            if key.endswith('cosine'):row[key+'_positive_fraction']=sum(v>0 for v in values)/len(values) if values else None
        rows.append(row)
    result=dict(status='SOURCE_FEATURE_ALIGNMENT_COMPLETE',code_commit=rt['code_commit'],models=names,model_seeds=[SEEDS[0]],
        rows=rows,batches=batch_rows,L_exposures=BATCHES*128,U_exposures=BATCHES*256,F_packet_forwards=5*BATCHES*(128+2*256),
        feature_gradient_calls=5*BATCHES*3*3,source_L_smoke_packets=2,reference_role='L_s',reference_updates=0,G_packet_forwards=0,
        optimizer_updates=0,model_parameter_updates=0,model_parameter_gradients=0,head_fit_updates=0,V_reads=0,target_reads=0,U_truth_read=False,
        all_model_parameters_and_registered_buffers_unchanged=True,seconds=time.perf_counter()-start,
        scope='Frozen source L/U feature-gradient associations;U agreement is not truth accuracy;not full parameter/AdamW updates or target benefit;no automatic method expansion')
    write(root/'complete.json',result)
    write(root/'pipeline_state.json',dict(status=result['status'],code_commit=rt['code_commit'],pid=os.getpid()),exclusive=False)
    print('SOURCE_ALIGNMENT_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    a=p.parse_args();run(read(a.config))
