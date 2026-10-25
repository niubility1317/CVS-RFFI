"""Frozen matched F50 source diagnosis after the complete negative target test."""
import argparse
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import SCENES,VIEWS,validate_context,source_channel,snapshot,unchanged,paired_counts
from .restored_dose_source import load_sources,BASELINE_RUN
from experiments.pamf_restored_pl.plan import RUN as F_RUN,rows

RUN='20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01'


def phase_aligned_nmse(raw,view):
    """One complex scalar removes global phase/gain; inference IQ is untouched."""
    import torch
    x=torch.complex(raw[:,0],raw[:,1]);z=torch.complex(view[:,0],view[:,1])
    a=(z.conj()*x).sum(1)/(z.abs().square().sum(1)+1e-12)
    return ((x-a[:,None]*z).abs().square().sum(1)/(x.abs().square().sum(1)+1e-12)).clamp(0.,1.)


def run(rt):
    if rt['run_id']!=RUN or rt['allowed_gpus']!=[0] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Fixed frozen source-shift diagnostic/GPU0 required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from experiments.pamf_repl_completion.source import load_reader
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    torch.set_num_threads(2);start=time.perf_counter();root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    c=read(rt['context']);validate_context(c);froot=Path(rt['project'])/'runs'/F_RUN
    write(root/'pipeline_state.json',dict(status='SOURCE_LOADING',pid=os.getpid(),code_commit=rt['code_commit']))
    parents,configs,payloads,contract,g=load_sources(c,'cpu')
    parent_weights={case['model_seed']:payload['model']['id_backbone.encoder.core.id_backbone.cls_head.weight'].clone()
        for case,payload in zip(sorted(c['cases'],key=lambda x:x['model_seed']),payloads)}
    del parents,payloads
    g=g.to('cuda:0').eval().requires_grad_(False);engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    models=[];specs=[];head_changes=[]
    for row in rows():
        folder=froot/'F'/row['row_id']/'source';model,cfg=load_reader(folder,contract,'cuda:0')
        if (cfg['model_seed']!=row['model_seed'] or cfg['row_id']!=row['row_id'] or cfg['run_id']!=F_RUN
                or cfg['source_cache']!=configs[0]['source_cache'] or cfg['source_rxs']!=contract['source_rxs']
                or cfg['source_days']!=contract['source_days']):raise ValueError('Only nine fixed own-contract final F50 readers allowed')
        w=model.core.id_backbone.cls_head.weight.detach().cpu();old=parent_weights[row['model_seed']]
        cos=F.cosine_similarity(w,old,dim=1)
        head_changes.append(dict(method=row['method'],model_seed=row['model_seed'],
            mean_head_cosine_distance_from_own_E30=float((1-cos).mean()),
            max_head_cosine_distance_from_own_E30=float((1-cos).max()),
            head_relative_L2_from_own_E30=float((w-old).norm()/old.norm()),new_head=False))
        models.append(model);specs.append(row)
    labeled=SourceDataset(configs[0]['source_cache'],'L_s',contract);x=labeled.x[:2].to('cuda:0')
    with torch.inference_mode():
        gx=engine(x)[0].clone()
        if not torch.allclose(g(x),gx,atol=1e-5,rtol=1e-5):raise ValueError('Actual deployed G source-L smoke differs')
        for m in models:
            q=m(gx)
            if q.shape!=(2,6) or not bool(torch.isfinite(q).all()):raise ValueError('Nine actual final default-head source smoke failed')
    del labeled,x,gx
    write(root/'smoke.json',dict(status='PASS',source_role='L_s',physical_packets=2,frozen_F=9,frozen_G=1,
        parent_head_reference_epoch=30,optimizer_updates=0,target_reads=0,default_head=True))
    before=[snapshot(m) for m in models];gbefore=snapshot(g)
    data=SourceDataset(configs[0]['source_cache'],'V',contract);n=len(data)
    if n!=27000:raise ValueError('Complete original source V required')
    baseline_root=Path(rt['project'])/'runs'/BASELINE_RUN
    with np.load(baseline_root/'source_V_predictions.npz',allow_pickle=False) as old:
        if not np.array_equal(old['ids'],data.metadata['ids']) or not np.array_equal(old['y'],data.metadata['y']):
            raise ValueError('Identical source V physical order/labels for fixed E30 raw consensus required')
        accepted=old['raw_consensus_accepted'].copy()
    loader=torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0)
    logits=np.empty((9,8,n,6),np.float32);distance=np.empty((9,8,n),np.float32)
    power=np.empty((8,n),np.float32);nmse=np.empty((8,n),np.float32)
    rxs=sorted(contract['source_rxs']);geometry={(j,v):dict(sums=np.zeros((6,len(rxs),160),np.float64),
        counts=np.zeros((6,len(rxs)),np.int64)) for j in range(9) for v in range(8)}
    offset=0
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            x=batch[0].to('cuda:0');count=len(x);labels=batch[1].tolist();receivers=batch[3]['rx_i'].tolist()
            views=[x,engine(x)[0].clone()]
            for scene in SCENES:
                h=source_channel(x,batch[3],scene,contract);views.extend((h,engine(h)[0].clone()))
            for v,view in enumerate(views):
                if not bool(torch.isfinite(view).all()):raise ValueError('Finite source IQ required')
                power[v,offset:offset+count]=fp32_copy(view.square().sum(1).mean(1))
                nmse[v,offset:offset+count]=fp32_copy(phase_aligned_nmse(x,view))
            for j,model in enumerate(models):
                reference=None
                for v,view in enumerate(views):
                    feature=model.features(view);q=model.classify_features(feature);z=F.normalize(feature,dim=1)
                    if q.shape!=(count,6) or not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(z).all()):
                        raise ValueError('Finite original head/features required')
                    if reference is None:reference=z
                    logits[j,v,offset:offset+count]=fp32_copy(q)
                    distance[j,v,offset:offset+count]=fp32_copy(1-(z*reference).sum(1))
                    values=fp32_copy(z);s=geometry[j,v]
                    for k,(label,rx) in enumerate(zip(labels,receivers)):
                        r=rxs.index(rx);s['sums'][label,r]+=values[k];s['counts'][label,r]+=1
            offset+=count
            if i%5==0 or offset==n:
                write(root/'progress.json',dict(status='FROZEN_SOURCE_SHIFT',source_V_seen=offset,total=n,
                    F_packet_forwards=9*8*offset,G_packet_forwards=4*offset,optimizer_updates=0,target_reads=0,
                    seconds=time.perf_counter()-start),exclusive=False)
                print('SOURCE_SHIFT',offset,n,flush=True)
    if offset!=n or not unchanged(g,gbefore) or any(not unchanged(m,b) for m,b in zip(models,before)):
        raise ValueError('Incomplete source role or frozen parameter/buffer mutation')
    y=data.metadata['y'];pred=logits.argmax(-1);result_rows=[];paired=[];fidelity_edges=(0.,.01,.05,.1,.2,.4,1.00001)
    for j,row in enumerate(specs):
        control=next(k for k,r in enumerate(specs) if r['method']=='PLControl' and r['model_seed']==row['model_seed'])
        for v,name in enumerate(VIEWS):
            q=logits[j,v];margin=q[np.arange(n),y]-np.max(np.where(np.eye(6,dtype=bool)[y],-np.inf,q),axis=1)
            dimensions=[('overall','ALL',np.ones(n,bool)),('raw_E30_consensus','accepted',accepted)]
            for dim,values in [('TX',y),('RX',data.metadata['rx_i']),('day',data.metadata['day_i'])]:
                dimensions.extend((dim,str(value),values==value) for value in np.unique(values))
            if name.startswith('G_'):
                dimensions.extend(('aligned_NMSE',str(lo)+'to'+str(hi),(nmse[v]>=lo)&(nmse[v]<hi)) for lo,hi in zip(fidelity_edges[:-1],fidelity_edges[1:]))
            for dim,stratum,mask in dimensions:
                count=int(mask.sum())
                result_rows.append(dict(method=row['method'],model_seed=row['model_seed'],view=name,dimension=dim,stratum=stratum,
                    n=count,correct=int(((pred[j,v]==y)&mask).sum()),accuracy_percent=100.*float((pred[j,v,mask]==y[mask]).mean()) if count else None,
                    mean_true_class_margin=float(margin[mask].mean()) if count else None,
                    mean_raw_feature_cosine_distance=float(distance[j,v,mask].mean()) if count else None,
                    mean_IQ_power=float(power[v,mask].mean()) if count else None,mean_aligned_NMSE=float(nmse[v,mask].mean()) if count else None))
                if row['method']!='PLControl':
                    counts=paired_counts(pred[control,v],pred[j,v],y,mask)
                    counts['control_accuracy']=counts.pop('raw_accuracy')
                    counts['candidate_accuracy']=counts.pop('restored_accuracy')
                    paired.append(dict(method=row['method'],control='PLControl',model_seed=row['model_seed'],view=name,dimension=dim,stratum=stratum,
                        **counts))
    geom=[]
    for (j,v),s in geometry.items():
        centres=s['sums']/np.maximum(s['counts'][...,None],1);norm=np.linalg.norm(centres,axis=-1);unit=centres/np.maximum(norm[...,None],1e-12)
        same=[];different=[]
        for tx in range(6):
            for rx in range(len(rxs)):
                for other in range(rx+1,len(rxs)):same.append(1-float(unit[tx,rx]@unit[tx,other]))
                for other_tx in range(tx+1,6):different.append(1-float(unit[tx,rx]@unit[other_tx,rx]))
        geom.append(dict(method=specs[j]['method'],model_seed=specs[j]['model_seed'],view=VIEWS[v],
            class_RX_resultant_length=float(norm.mean()),same_TX_cross_RX_centroid_distance=float(np.mean(same)),
            different_TX_same_RX_centroid_distance=float(np.mean(different)),fitted_probe_updates=0))
    np.savez(root/'source_V_predictions.npz',logits=logits,y=y,ids=data.metadata['ids'],rx=data.metadata['rx_i'],day=data.metadata['day_i'],
        methods=np.asarray([r['method'] for r in specs]),seeds=np.asarray([r['model_seed'] for r in specs]),views=np.asarray(VIEWS),
        aligned_NMSE=nmse,IQ_power=power,raw_E30_consensus=accepted)
    result=dict(status='SOURCE_SHIFT_DIAGNOSTIC_COMPLETE',code_commit=rt['code_commit'],source_V_seen=n,frozen_F=9,
        model_specs=specs,views=list(VIEWS),results=result_rows,paired_vs_PLControl=paired,geometry=geom,head_changes=head_changes,
        F_packet_forwards=9*8*n,G_packet_forwards=4*n,source_L_smoke_packets=2,parent_head_reference_epoch=30,
        optimizer_updates=0,F_G_updates=0,head_fit_updates=0,U_IQ_reads=0,U_truth_read=False,target_reads=0,
        source_V_state_updates=0,all_parameters_and_registered_buffers_unchanged=True,seconds=time.perf_counter()-start,
        scope='Post-target frozen source diagnosis;no new training/target access;geometry/NMSE associations do not prove causality')
    write(root/'complete.json',result)
    write(root/'pipeline_state.json',dict(status=result['status'],pid=os.getpid(),code_commit=rt['code_commit']),exclusive=False)
    print('SOURCE_SHIFT_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    a=p.parse_args();run(read(a.config))
