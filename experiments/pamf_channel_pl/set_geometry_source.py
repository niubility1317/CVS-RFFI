"""Frozen visible-L comparison of mean and multi-sample cross-RX references."""
import argparse
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,SEEDS,source_channel,snapshot,unchanged
from .restored_dose_source import load_sources
from .receiver_reference_source import student_from_parent
from .receiver_reference import metadata_long,build_references
from .set_reference import build_bank,set_logits,mean_logits

RUN='20261026-diagnostic-PAMF-LSetGeometry-m1-r01'
ASSET_RUN='20261025-diagnostic-PAMF-RXReference-dose-m1-r01'
METHODS=('Mean','Set05','Set10')
VIEWS=('raw','G_raw')+tuple('G_'+s for s in SCENES)
NAMESPACE='PAMF/set_reference/L/fixed'


def summarize(pred,head,y,cos,nr,nh,mask):
    import numpy as np
    a=pred[mask]==y[mask];b=head[mask]==y[mask];defined=(nr[mask]>1e-12)&(nh[mask]>1e-12)
    valid_cos=cos[mask][defined];wrong=(~b)&defined;right=b&defined
    return dict(n=int(mask.sum()),reference_correct=int(a.sum()),default_head_correct=int(b.sum()),
        reference_correct_head_wrong=int((a&~b).sum()),reference_wrong_head_correct=int((~a&b).sum()),
        reference_accuracy=100*float(a.mean()),default_head_accuracy=100*float(b.mean()),
        gradient_defined=int(defined.sum()),gradient_positive=int((valid_cos>0).sum()),
        gradient_cosine_median=float(np.median(valid_cos)) if len(valid_cos) else None,
        gradient_positive_fraction=float((valid_cos>0).mean()) if len(valid_cos) else None,
        gradient_cosine_head_wrong_median=float(np.median(cos[mask][wrong])) if wrong.any() else None,
        gradient_cosine_head_correct_median=float(np.median(cos[mask][right])) if right.any() else None)


def run(rt):
    if rt['run_id']!=RUN or rt['allowed_gpus']!=[0] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Fixed visible-L geometry diagnostic/GPU0 required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    torch.set_num_threads(2);torch.manual_seed(SEEDS[0]);torch.cuda.manual_seed_all(SEEDS[0])
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    c=read(rt['context']);validate_context(c)
    write(root/'pipeline_state.json',dict(status='SOURCE_L_LOADING',code_commit=rt['code_commit'],pid=os.getpid()))
    teachers,configs,payloads,contract,g=load_sources(c,'cuda:0')
    student,_=student_from_parent(configs[0],payloads[0]);del payloads
    student.eval()
    for parameter in student.parameters():parameter.requires_grad_(False)
    engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    data=SourceDataset(configs[0]['source_cache'],'L_s',contract)
    if len(data)!=6300:raise ValueError('Complete original visible L6300 required')
    x=data.x[:2].to('cuda:0');domain=torch.zeros(2,device='cuda:0',dtype=torch.long)
    with torch.no_grad():
        out=student.forward_identity_only(x,domain_labels=domain);gx=engine(x)[0].clone()
        if (not torch.allclose(g(x),gx,atol=1e-5,rtol=1e-5) or out['z_id'].shape!=(2,160)
                or not torch.allclose(out['z_id'],teachers[0].features(x),atol=1e-5,rtol=1e-5)
                or not torch.allclose(student.id_backbone.encoder.classify_features(out['z_id']),out['tx_logits'],atol=1e-5,rtol=1e-5)):
            raise ValueError('Actual own E30/default head/G source L2 smoke differs')
    write(root/'smoke.json',dict(status='PASS',source_role='L_s',physical_packets=2,actual_E30=True,
        own_seed_features_match=True,default_head_recompute_matches=True,optimizer_updates=0,U_reads=0,V_reads=0,target_reads=0))
    before=[snapshot(m) for m in teachers];sbefore=snapshot(student);gbefore=snapshot(g)
    features=[[] for _ in teachers]
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            if not bool(torch.as_tensor(batch[3]['tx_label_visible']).all()) or bool((batch[1]<0).any()):
                raise ValueError('Only truly visible source L records may build references')
            for j,model in enumerate(teachers):features[j].append(model.features(batch[0].to('cuda:0')).detach().cpu())
    old=torch.load(Path(rt['project'])/'runs'/ASSET_RUN/'references.pt',map_location='cpu',weights_only=False)
    if (old['source_contract']!=contract or old['source_cases']!=c['cases'] or old['teacher_epoch']!=30
            or old['model_seeds']!=list(SEEDS) or old['reference_role']!='L_s' or any(old[k] for k in ('U_reads','V_reads','target_reads'))):
        raise ValueError('Same original visible-L own-E30 reference source required')
    labels=metadata_long(data.metadata['y']);rx=metadata_long(data.metadata['rx_i']);rxs=tuple(sorted(contract['source_rxs']))
    banks=[];valids=[];centers=[];resultants=[];max_difference=0.
    for j,parts in enumerate(features):
        z=torch.cat(parts);bank,valid=build_bank(z,labels,rx,rxs,6,'L_s');ref,count=build_references(z,labels,rx,rxs,6,'L_s')
        difference=float((ref-old['references'][j]).abs().max());max_difference=max(max_difference,difference)
        if difference>1e-5 or not torch.equal(count,old['counts'][j]) or not torch.equal(valid.sum(-1),count):
            raise ValueError('New sample bank no longer matches the original mean-reference coordinates/counts')
        mean=bank.double().sum(2)/valid.sum(2)[...,None]
        resultants.append(mean.norm(dim=-1).float().tolist());banks.append(bank);valids.append(valid);centers.append(ref)
    del features
    torch.save(dict(schema='pamf_visible_L_set_reference_v1',bank=torch.stack(banks),valid=torch.stack(valids),
        mean_references=torch.stack(centers),source_contract=contract,source_cases=c['cases'],source_rxs=list(rxs),
        model_seeds=list(SEEDS),teacher_epoch=30,reference_role='L_s',L_records=6300,U_reads=0,V_reads=0,target_reads=0,
        original_mean_reference_max_abs_difference=max_difference),root/'visible_L_bank.pt')
    bank=banks[0].to('cuda:0');valid=valids[0].to('cuda:0');center=centers[0].to('cuda:0')
    n=len(data);pred=np.empty((5,3,n),np.int64);head=np.empty((5,n),np.int64)
    cos=np.empty((5,3,n),np.float32);nr=np.empty_like(cos);nh=np.empty((5,n),np.float32);offset=0;batches=0
    for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
        x,y,d=[v.to('cuda:0') for v in batch[:3]];r=batch[3]['rx_i'].to('cuda:0');bs=len(x)
        with torch.no_grad():
            views=[x,engine(x)[0].clone()]
            for scene in SCENES:views.append(engine(source_channel(x,batch[3],scene,contract,namespace=NAMESPACE))[0].clone())
        for k,view in enumerate(views):
            with torch.no_grad():out=student.forward_identity_only(view,domain_labels=d)
            z=out['z_id'].detach().clone().requires_grad_(True)
            logits=student.id_backbone.encoder.classify_features(z)
            if (not torch.allclose(logits,out['tx_logits'],atol=1e-5,rtol=1e-5)
                    or not bool(torch.isfinite(logits).all())):raise ValueError('Exact frozen actual default head required')
            gh=torch.autograd.grad(F.cross_entropy(logits,y,reduction='sum'),z)[0]
            head[k,offset:offset+bs]=np.asarray(logits.argmax(1).tolist(),np.int64);nh[k,offset:offset+bs]=fp32_copy(gh.norm(dim=1))
            for j,method in enumerate(METHODS):
                score=mean_logits(z,center,r,rxs) if method=='Mean' else set_logits(z,bank,valid,r,rxs,.05 if method=='Set05' else .1)
                if not bool(torch.isfinite(score).all()):raise ValueError('Finite source reference scores required')
                gr=torch.autograd.grad(F.cross_entropy(score,y,reduction='sum'),z)[0]
                if not bool(torch.isfinite(gr).all()):raise ValueError('Finite source reference feature gradients required')
                pred[k,j,offset:offset+bs]=np.asarray(score.argmax(1).tolist(),np.int64)
                cos[k,j,offset:offset+bs]=fp32_copy(F.cosine_similarity(gr,gh,dim=1,eps=1e-12))
                nr[k,j,offset:offset+bs]=fp32_copy(gr.norm(dim=1))
        offset+=bs;batches+=1
        write(root/'progress.json',dict(status='SOURCE_L_GEOMETRY',seen=offset,total=n,reference_variants=list(METHODS),
            optimizer_updates=0,U_reads=0,V_reads=0,target_reads=0),exclusive=False)
        print('SOURCE_L_SET_GEOMETRY',offset,n,flush=True)
    if (offset!=n or not unchanged(student,sbefore) or not unchanged(g,gbefore)
            or any(not unchanged(m,b) for m,b in zip(teachers,before))):
        raise ValueError('Frozen L diagnostic changed model parameters/registered buffers or was incomplete')
    meta=data.metadata;y=meta['y'];records=[]
    strata=[('overall','ALL',np.ones(n,bool))]
    for dimension,values in [('TX',y),('RX',meta['rx_i']),('day',meta['day_i'])]:
        strata.extend((dimension,str(value),values==value) for value in np.unique(values))
    for k,view in enumerate(VIEWS):
        for j,method in enumerate(METHODS):
            for dimension,stratum,mask in strata:
                records.append(dict(view=view,reference=method,dimension=dimension,stratum=stratum,
                    **summarize(pred[k,j],head[k],y,cos[k,j],nr[k,j],nh[k],mask)))
    for j,method in enumerate(METHODS):
        for dimension,stratum,mask in strata:
            records.append(dict(view='G_three_mean',reference=method,dimension=dimension,stratum=stratum,
                **summarize(pred[2:,j].reshape(-1),head[2:].reshape(-1),np.tile(y,3),cos[2:,j].reshape(-1),
                    nr[2:,j].reshape(-1),nh[2:].reshape(-1),np.tile(mask,3))))
    np.savez_compressed(root/'source_L_geometry.npz',reference_prediction=pred,default_head_prediction=head,y=y,ids=meta['ids'],
        rx=meta['rx_i'],day=meta['day_i'],reference=np.asarray(METHODS),views=np.asarray(VIEWS),cosine=cos,
        reference_feature_gradient_norm=nr,head_feature_gradient_norm=nh)
    result=dict(status='SOURCE_L_SET_GEOMETRY_COMPLETE',code_commit=rt['code_commit'],results=records,
        L_records=6300,query_student_seed=SEEDS[0],query_epoch=30,reference_teacher_seeds=list(SEEDS),reference_role='L_s',
        bank_shape=list(torch.stack(banks).shape),bank_counts=[v.sum(-1).tolist() for v in valids],
        source_RXs=list(rxs),views=list(VIEWS),reference_methods=list(METHODS),outer_temperature=.1,
        set_inner_temperatures=[.05,.1],source_channel_namespace=NAMESPACE,
        original_mean_reference_max_abs_difference=max_difference,mean_resultant_lengths=resultants,
        reference_F_packet_forwards=3*n,diagnostic_F_packet_forwards=5*n,G_packet_forwards=4*n,
        feature_gradient_calls=batches*5*4,gradient_units='Derivative of summed per-packet CE at independent detached features',
        optimizer_updates=0,U_reads=0,U_truth_read=False,V_reads=0,target_reads=0,head_fit_updates=0,
        all_model_parameters_and_registered_buffers_unchanged=True,all_model_parameter_gradients_none=True,
        seconds=time.perf_counter()-start,
        scope='Frozen visible-L relation geometry only;nonparametric scores are diagnostics,not a replacement head or deployment result;no new F training or goal achievement')
    write(root/'complete.json',result)
    write(root/'pipeline_state.json',dict(status=result['status'],pid=os.getpid(),code_commit=rt['code_commit']),exclusive=False)
    print('SOURCE_L_SET_GEOMETRY_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    run(read(p.parse_args().config))
