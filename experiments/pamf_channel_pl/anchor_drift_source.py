"""Frozen visible-L old/current-anchor comparison across legal E30/F50 readers."""
import argparse
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,SEEDS,VIEWS,source_channel,snapshot,unchanged
from .restored_dose_source import load_sources
from .receiver_reference import metadata_long,build_references
from .set_reference import mean_logits
from .set_geometry_source import summarize

RUN='20261026-diagnostic-PAMF-LAnchorDrift-m5-r01'
ASSET_RUN='20261025-diagnostic-PAMF-RXReference-dose-m1-r01'
NAMESPACE='PAMF/set_reference/L/fixed'
ANCHORS=('FixedE30','CurrentL')
F50_ROWS=(
 ('PLControlF50','20261025-phase1-PAMF-RestoredPL-m9-r01','F0-PLControl-s2026092701'),
 ('GTask1xF50','20261026-phase1-PAMF-TaskScaledRef-sourceHold-m3-r01','F0-GTaskScaled-sourceHold-s2026092701'),
 ('GTask2xF50','20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01','F0-GTask2x-sourceHold-s2026092701'),
 ('GTask3xF50','20261026-phase1-PAMF-TaskRefDose-sourceHold-m4-r01','F0-GTask3x-sourceHold-s2026092701'))


def validate_final(done,actual,expected,row,source_cache):
    c=done['config']
    if (done.get('status')!='SOURCE_FROZEN' or done.get('epoch')!=50 or done.get('updates')!=11100
            or done.get('target_access') is not False or done.get('U_truth_read') is not False or done.get('head_fit_updates')!=0
            or c.get('run_id')!=row[1] or c.get('row_id')!=row[2] or c.get('model_seed')!=SEEDS[0]
            or c.get('protocol_id')!='F0' or c.get('num_classes')!=6 or c.get('method_id')!='pamf_t1_e50'
            or not c.get('default_head_joint') or c.get('posthoc_head_fit') or c.get('source_cache')!=source_cache
            or actual!=expected or c.get('source_rxs')!=expected['source_rxs'] or c.get('source_days')!=expected['source_days']
            or any(k in c for k in ('tests','target_inputs','target_truth','truth','dataset_path','teacher_checkpoint','checkpoint'))):
        raise ValueError('Known same-contract source-only F50/default-head ancestry required')


def run(rt):
    if rt['run_id']!=RUN or rt['allowed_gpus']!=[0] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Fixed frozen source-L anchor diagnostic/GPU0 required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_repl_completion.source import load_reader
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    torch.set_num_threads(2);torch.manual_seed(SEEDS[0]);torch.cuda.manual_seed_all(SEEDS[0])
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    c=read(rt['context']);validate_context(c)
    write(root/'pipeline_state.json',dict(status='SOURCE_L_ANCHOR_LOADING',pid=os.getpid(),code_commit=rt['code_commit']))
    teachers,configs,payloads,contract,g=load_sources(c,'cuda:0');del payloads
    models=[teachers[0]];names=['E30'];origins=[dict(name='E30',epoch=30,model_seed=SEEDS[0])]
    for row in F50_ROWS:
        path=Path(rt['project'])/'runs'/row[1]/'F'/row[2]/'source';done=read(path/'complete.json')
        validate_final(done,read(done['config']['source_contract']),contract,row,configs[0]['source_cache'])
        model,_=load_reader(path,contract,'cuda:0');models.append(model);names.append(row[0])
        origins.append(dict(name=row[0],run_id=row[1],row_id=row[2],epoch=50,model_seed=SEEDS[0]))
    data=SourceDataset(configs[0]['source_cache'],'L_s',contract);n=len(data)
    if n!=6300:raise ValueError('Complete original L6300 required')
    engine=InferenceEngine(g,reuse=True,graph=True,native=False);x=data.x[:2].to('cuda:0')
    with torch.no_grad():
        gx=engine(x)[0].clone()
        if not torch.allclose(g(x),gx,atol=1e-5,rtol=1e-5):raise ValueError('Deployed same G differs')
        for m in models:
            z=m.features(x);q=m.classify_features(z)
            if z.shape!=(2,160) or q.shape!=(2,6) or not bool(torch.isfinite(q).all()) or not torch.allclose(q,m(x),atol=1e-5,rtol=1e-5):
                raise ValueError('Actual checkpoint default-head/feature source L2 smoke differs')
    write(root/'smoke.json',dict(status='PASS',source_role='L_s',physical_packets=2,query_models=names,
        default_head=True,full_F50_ancestry=True,optimizer_updates=0,U_reads=0,V_reads=0,target_reads=0))
    unique=teachers+models[1:];before=[snapshot(m) for m in unique];gbefore=snapshot(g)
    old=torch.load(Path(rt['project'])/'runs'/ASSET_RUN/'references.pt',map_location='cpu',weights_only=False)
    if (old['source_contract']!=contract or old['source_cases']!=c['cases'] or old['teacher_epoch']!=30
            or old['model_seeds']!=list(SEEDS) or old['reference_role']!='L_s' or any(old[k] for k in ('U_reads','V_reads','target_reads'))):
        raise ValueError('Original same-source E30 L references required')
    fixed=old['references'][0].to('cuda:0');rxs=tuple(contract['source_rxs'])
    labels=metadata_long(data.metadata['y']);rx=metadata_long(data.metadata['rx_i']);features=[[] for _ in models]
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            if not bool(torch.as_tensor(batch[3]['tx_label_visible']).all()) or bool((batch[1]<0).any()):raise ValueError('Visible source L only')
            for j,m in enumerate(models):features[j].append(m.features(batch[0].to('cuda:0')).detach().cpu())
    centers=[];counts=[];drift=[]
    for j,parts in enumerate(features):
        ref,count=build_references(torch.cat(parts),labels,rx,rxs,6,'L_s');centers.append(ref.to('cuda:0'));counts.append(count)
        if not torch.equal(count,old['counts'][0]):raise ValueError('Same source L class/RX counts required')
        for cl in range(6):
            for k,receiver in enumerate(rxs):
                drift.append(dict(model=names[j],TX_class_index=cl,RX=receiver,
                    center_cosine=float(F.cosine_similarity(fixed[cl,k][None],centers[j][cl,k][None],dim=1)[0]),
                    center_max_abs_difference=float((fixed[cl,k]-centers[j][cl,k]).abs().max())))
    control_error=float((fixed-centers[0]).abs().max())
    if control_error>1e-5:raise ValueError('E30 zero-drift control mismatch')
    del features
    torch.save(dict(schema='pamf_visible_L_anchor_drift_v1',models=names,origins=origins,fixed=fixed.cpu(),
        current=torch.stack([r.cpu() for r in centers]),counts=torch.stack(counts),source_contract=contract,
        reference_role='L_s',U_reads=0,V_reads=0,target_reads=0),root/'L_anchor_centers.pt')
    pred=np.empty((5,8,2,n),np.int64);head=np.empty((5,8,n),np.int64)
    cos=np.empty((5,8,2,n),np.float32);nr=np.empty_like(cos);nh=np.empty((5,8,n),np.float32);offset=0;batches=0
    for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
        x,y=[v.to('cuda:0') for v in batch[:2]];r=batch[3]['rx_i'].to('cuda:0');bs=len(x)
        with torch.no_grad():
            views=[x,engine(x)[0].clone()]
            for scene in SCENES:
                h=source_channel(x,batch[3],scene,contract,namespace=NAMESPACE);views.extend((h,engine(h)[0].clone()))
        for j,m in enumerate(models):
            for k,view in enumerate(views):
                with torch.no_grad():feature=m.features(view)
                z=feature.detach().clone().requires_grad_(True);logits=m.classify_features(z)
                if not bool(torch.isfinite(logits).all()):raise ValueError('Finite frozen actual default head required')
                gh=torch.autograd.grad(F.cross_entropy(logits,y,reduction='sum'),z)[0]
                head[j,k,offset:offset+bs]=np.asarray(logits.argmax(1).tolist(),np.int64);nh[j,k,offset:offset+bs]=fp32_copy(gh.norm(dim=1))
                for a,ref in enumerate((fixed,centers[j])):
                    score=mean_logits(z,ref,r,rxs)
                    if not bool(torch.isfinite(score).all()):raise ValueError('Finite old/current reference scores required')
                    gr=torch.autograd.grad(F.cross_entropy(score,y,reduction='sum'),z)[0]
                    if not bool(torch.isfinite(gr).all()):raise ValueError('Finite source feature gradients required')
                    pred[j,k,a,offset:offset+bs]=np.asarray(score.argmax(1).tolist(),np.int64)
                    cos[j,k,a,offset:offset+bs]=fp32_copy(F.cosine_similarity(gr,gh,dim=1,eps=1e-12))
                    nr[j,k,a,offset:offset+bs]=fp32_copy(gr.norm(dim=1))
        offset+=bs;batches+=1
        write(root/'progress.json',dict(status='SOURCE_L_ANCHOR_DRIFT',seen=offset,total=n,query_models=names,
            optimizer_updates=0,U_reads=0,V_reads=0,target_reads=0),exclusive=False)
        print('SOURCE_L_ANCHOR_DRIFT',offset,n,flush=True)
    if offset!=n or not unchanged(g,gbefore) or any(not unchanged(m,b) for m,b in zip(unique,before)):
        raise ValueError('Frozen source state changed or source diagnostic incomplete')
    if not np.array_equal(pred[0,:,0],pred[0,:,1]) or not np.allclose(cos[0,:,0],cos[0,:,1],atol=1e-5,rtol=1e-5):
        raise ValueError('E30 identical-anchor prediction/gradient control differs')
    meta=data.metadata;y=meta['y'];strata=[('overall','ALL',np.ones(n,bool))];records=[]
    for dimension,values in [('TX',y),('RX',meta['rx_i']),('day',meta['day_i'])]:
        strata.extend((dimension,str(value),values==value) for value in np.unique(values))
    groups=[(view,[k]) for k,view in enumerate(VIEWS)]+[('G_three_mean',[3,5,7]),('H_three_mean',[2,4,6])]
    for j,name in enumerate(names):
        for view,indices in groups:
            repeat=len(indices)
            for a,anchor in enumerate(ANCHORS):
                for dimension,stratum,mask in strata:
                    records.append(dict(model=name,view=view,anchor=anchor,dimension=dimension,stratum=stratum,
                        **summarize(pred[j,indices,a].reshape(-1),head[j,indices].reshape(-1),np.tile(y,repeat),
                            cos[j,indices,a].reshape(-1),nr[j,indices,a].reshape(-1),nh[j,indices].reshape(-1),np.tile(mask,repeat))))
    np.savez_compressed(root/'source_L_anchor_drift.npz',reference_prediction=pred,default_head_prediction=head,y=y,ids=meta['ids'],
        rx=meta['rx_i'],day=meta['day_i'],models=np.asarray(names),anchors=np.asarray(ANCHORS),views=np.asarray(VIEWS),
        cosine=cos,reference_feature_gradient_norm=nr,head_feature_gradient_norm=nh)
    result=dict(status='SOURCE_L_ANCHOR_DRIFT_COMPLETE',code_commit=rt['code_commit'],results=records,center_drift=drift,
        L_records=n,query_models=origins,query_model_seeds=[SEEDS[0]],anchors=list(ANCHORS),views=list(VIEWS),
        source_channel_namespace=NAMESPACE,zero_drift_control_max_abs_difference=control_error,zero_drift_predictions_and_gradients_match=True,
        all_model_parameters_and_registered_buffers_unchanged=True,all_model_parameter_gradients_none=True,
        reference_F_packet_forwards=5*n,diagnostic_F_packet_forwards=5*8*n,G_packet_forwards=4*n,feature_gradient_calls=batches*5*8*3,
        optimizer_updates=0,U_reads=0,U_truth_read=False,V_reads=0,target_reads=0,head_fit_updates=0,
        seconds=time.perf_counter()-start,
        scope='Frozen visible-L anchor geometry only;five checkpoints ofone modelseed,not five seeds;defaultheads unchanged;not new F training or target benefit')
    write(root/'complete.json',result);write(root/'pipeline_state.json',dict(status=result['status'],pid=os.getpid(),code_commit=rt['code_commit']),exclusive=False)
    print('SOURCE_L_ANCHOR_DRIFT_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    run(read(p.parse_args().config))
