"""Visible-source-L diagnostic: deterministic repeat versus fresh channel realizations."""
import argparse
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,SEEDS,source_channel,snapshot,unchanged,paired_counts

RUN='20261026-diagnostic-PAMF-DURARealizations-source-m1-r01'
RELEASE='20261026-diagnostic-PAMF-DURARealizations-source-release-r01'
FIXED_NAMESPACE='PAMF/restored_dose/U/fixed'
REPLICAS=('fixed','fresh1','fresh2','fresh3')
METRICS=('G_relative_L2_vs_fixed','G_real_cos_vs_fixed','G_RMS_vs_fixed','G_relative_L2_vs_raw','H_relative_L2_vs_fixed','G_max_abs_vs_fixed')


def namespace(replica):
    if replica not in REPLICAS:raise ValueError('Fixed registered realization set required')
    return FIXED_NAMESPACE if replica=='fixed' else 'PAMF/realization_diversity/L/'+replica


def views():
    return ['raw']+[scene+'::'+replica+'::'+stage for scene in SCENES for replica in REPLICAS for stage in ('H','G')]


def validate_runtime(rt):
    required={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner','entrypoint'}
    if (set(rt)!=required or rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8))
            or rt['release_root']!=rt['project']+'/releases/'+RELEASE or rt['run_root']!=rt['project']+'/runs/'+RUN
            or rt['context']!=rt['release_root']+'/context.json' or rt['runtime_file']!=rt['release_root']+'/runtime.json'
            or rt['entrypoint']!='experiments.pamf_channel_pl.realization_diversity_source'
            or os.environ.get('CUDA_VISIBLE_DEVICES') not in list(map(str,range(8)))):
        raise ValueError('Registered source-L-only realization diagnostic and one physical GPU required')


def geometry(raw,h0,h,g0,g):
    import torch
    if (raw.ndim!=3 or raw.shape[1:]!=(2,256) or any(x.shape!=raw.shape for x in (h0,h,g0,g))
            or any(not bool(torch.isfinite(x).all()) for x in (raw,h0,h,g0,g))):
        raise ValueError('Aligned finite source IQ pairs required')
    x=raw.flatten(1);a=g0.flatten(1);b=g.flatten(1)
    an=a.norm(dim=1).clamp_min(1e-12);bn=b.norm(dim=1).clamp_min(1e-12);xn=x.norm(dim=1).clamp_min(1e-12)
    hn=h0.flatten(1).norm(dim=1).clamp_min(1e-12)
    return torch.stack(((b-a).norm(dim=1)/an,(a*b).sum(1)/(an*bn),bn/an,(b-x).norm(dim=1)/xn,
        (h-h0).flatten(1).norm(dim=1)/hn,(b-a).abs().max(1).values),1)


def summarize(values):
    import numpy as np
    a=np.asarray(values,np.float64)
    if not a.size or not np.isfinite(a).all():raise ValueError('Complete finite source geometry required')
    return dict(mean=float(a.mean()),minimum=float(a.min()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()))


def run(rt):
    validate_runtime(rt)
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import numpy as np
    import torch
    from .restored_dose_source import load_sources
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    torch.set_num_threads(2);torch.manual_seed(SEEDS[0]);torch.cuda.manual_seed_all(SEEDS[0])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    write(root/'pipeline_state.json',dict(status='SOURCE_L_REALIZATION_LOADING',pid=os.getpid(),code_commit=rt['code_commit']))
    c=read(rt['context']);validate_context(c)
    models,configs,payloads,contract,g=load_sources(c,'cuda:0');del payloads
    model=models[0];data=SourceDataset(configs[0]['source_cache'],'L_s',contract);n=len(data)
    if n!=6300 or list(data.metadata['ids'])!=contract['role_ids']['L_s']:raise ValueError('Original complete visible L6300 required')
    engine=InferenceEngine(g,reuse=True,graph=True,native=False);x=data.x[:2].to('cuda:0')
    with torch.no_grad():
        gx=engine(x.clone())[0].clone()
        q0,qg=model(x),model(gx)
        if (not torch.allclose(g(x.clone()),gx,atol=1e-5,rtol=1e-5) or q0.shape!=(2,6) or qg.shape!=(2,6)
                or not bool(torch.isfinite(q0).all()) or not bool(torch.isfinite(qg).all())):
            raise ValueError('Actual original E30/default-head/G source L2 smoke failed')
    write(root/'real_checkpoint_smoke.json',dict(status='PASS',role='L_s',physical_packets=2,query_seed=SEEDS[0],
        actual_default_head=True,G_eager_graph_close=True,optimizer_updates=0,U_reads=0,V_reads=0,target_reads=0))
    before=[snapshot(m) for m in models];gbefore=snapshot(g)
    names=views();lookup={name:i for i,name in enumerate(names)};logits=np.empty((len(names),n,6),np.float32)
    geom=np.empty((3,3,n,len(METRICS)),np.float32);repeat=np.empty((3,n),np.float32);offset=0
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0');count=len(x)
            if not bool((batch[1]>=0).all()) or not bool(torch.as_tensor(batch[3]['tx_label_visible']).all()):
                raise ValueError('Visible source L only;U not permitted in this diagnostic')
            def predict(view,index):
                q=model(view)
                if q.shape!=(count,6) or not bool(torch.isfinite(q).all()):raise ValueError('Finite frozen six-class default head required')
                logits[index,offset:offset+count]=fp32_copy(q)
            predict(x,0)
            for s,scene in enumerate(SCENES):
                h0=source_channel(x,batch[3],scene,contract,namespace=namespace('fixed'));g0=engine(h0.clone())[0].clone()
                gr=engine(h0.clone())[0].clone();repeat[s,offset:offset+count]=fp32_copy((gr-g0).flatten(1).abs().max(1).values)
                predict(h0,lookup[scene+'::fixed::H']);predict(g0,lookup[scene+'::fixed::G'])
                for r,replica in enumerate(REPLICAS[1:]):
                    h=source_channel(x,batch[3],scene,contract,namespace=namespace(replica));restored=engine(h.clone())[0].clone()
                    geom[s,r,offset:offset+count]=fp32_copy(geometry(x,h0,h,g0,restored))
                    predict(h,lookup[scene+'::'+replica+'::H']);predict(restored,lookup[scene+'::'+replica+'::G'])
            offset+=count
            progress=dict(status='SOURCE_L_REALIZATIONS',L_seen=offset,total=n,pid=os.getpid(),new_F_logical_packet_calls=25*offset,
                new_G_logical_packet_calls=15*offset,optimizer_updates=0,U_reads=0,V_reads=0,target_reads=0)
            write(root/'progress.json',progress,exclusive=False);print('SOURCE_L_REALIZATIONS',offset,n,flush=True)
    if offset!=n or not np.isfinite(logits).all() or not np.isfinite(geom).all() or not np.isfinite(repeat).all():raise ValueError('Complete finite source matrix required')
    if any(not unchanged(m,b) for m,b in zip(models,before)) or not unchanged(g,gbefore):raise ValueError('Frozen source F/G parameters or registered buffers changed')
    y=data.metadata['y'];rx=data.metadata['rx_i'];day=data.metadata['day_i'];pred=logits.argmax(-1)
    strata=[('overall','ALL',np.ones(n,bool))]
    for dimension,values in [('TX',y),('RX',rx),('day',day)]:strata.extend((dimension,str(v),values==v) for v in np.unique(values))
    if len(strata)!=15:raise ValueError('Original sixTX/fiveRX/threeDay source strata required')
    cells=[];restoration=[];geometry_cells=[];repeat_cells=[]
    for j,name in enumerate(names):
        baseline=0 if name=='raw' else lookup[name.split('::')[0]+'::fixed::'+name.split('::')[2]]
        for dimension,stratum,mask in strata:
            counts=paired_counts(pred[baseline],pred[j],y,mask)
            counts['baseline_accuracy']=counts.pop('raw_accuracy');counts['candidate_accuracy']=counts.pop('restored_accuracy')
            cells.append(dict(view=name,dimension=dimension,stratum=stratum,baseline_view=names[baseline],
                **counts))
    for s,scene in enumerate(SCENES):
        for replica in REPLICAS:
            hi=lookup[scene+'::'+replica+'::H'];gi=lookup[scene+'::'+replica+'::G']
            for dimension,stratum,mask in strata:
                counts=paired_counts(pred[hi],pred[gi],y,mask)
                counts['H_accuracy']=counts.pop('raw_accuracy');counts['G_accuracy']=counts.pop('restored_accuracy')
                restoration.append(dict(scene=scene,replica=replica,dimension=dimension,stratum=stratum,**counts))
        for dimension,stratum,mask in strata:
            repeat_cells.append(dict(scene=scene,dimension=dimension,stratum=stratum,n=int(mask.sum()),
                exact_repeat_packets=int((repeat[s,mask]==0).sum()),repeat_max_absolute_error=float(repeat[s,mask].max())))
        for r,replica in enumerate(REPLICAS[1:]):
            for dimension,stratum,mask in strata:
                for k,metric in enumerate(METRICS):
                    geometry_cells.append(dict(scene=scene,replica=replica,dimension=dimension,stratum=stratum,metric=metric,n=int(mask.sum()),**summarize(geom[s,r,mask,k])))
    np.savez(root/'source_L_predictions.npz',logits=logits,y=y,ids=data.metadata['ids'],views=np.asarray(names))
    np.savez(root/'source_L_geometry.npz',geometry=geom,repeat_max_absolute_error=repeat,metrics=np.asarray(METRICS),
        scenes=np.asarray(SCENES),replicas=np.asarray(REPLICAS[1:]),ids=data.metadata['ids'])
    result=dict(status='SOURCE_L_REALIZATIONS_ANALYZED',code_commit=rt['code_commit'],source_role='L_s',L_records=n,query_model_seed=SEEDS[0],
        query_model_epoch=30,query_models=1,default_head=True,source_G=c['G'],source_contract=configs[0]['source_contract'],
        scenes=list(SCENES),replicas=list(REPLICAS),namespaces={r:namespace(r) for r in REPLICAS},views=names,
        processing_route='residual/post_sync/noEQ/reflect/SNR30',channel_seed=20261016421,hardware_seed=20261016423,
        fixed_receiver_day_hardware=True,source_IQ_secondary_normalization=False,results=cells,restoration=restoration,
        geometry=geometry_cells,repeats=repeat_cells,F_evaluation_logical_packet_calls=25*n,G_evaluation_logical_packet_calls=15*n,
        F_smoke_logical_packet_calls=4,G_smoke_logical_packet_calls=4,student_teacher_G_parameters_and_registered_buffers_unchanged=True,
        optimizer_updates=0,U_reads=0,V_reads=0,target_reads=0,U_truth_read=False,head_fit_updates=0,
        seconds=time.perf_counter()-start,scope='Frozen source-L view diversity/identity diagnostic,not U accuracy or target performance;no IQ cache promotion or seed selection')
    write(root/'complete.json',result);write(root/'pipeline_state.json',dict(status=result['status'],pid=os.getpid(),code_commit=rt['code_commit'],seconds=result['seconds']),exclusive=False)
    print('SOURCE_L_REALIZATIONS_ANALYZED',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True);a=p.parse_args()
    run(read(a.config))
