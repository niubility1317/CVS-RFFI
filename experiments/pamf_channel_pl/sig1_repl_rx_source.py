"""Frozen single-view PAMF source geometry; never trains a control or student."""
import argparse
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,snapshot,unchanged,source_channel

RUN='20261026-diagnostic-PAMF-SIG1-REPL-RXGeometry-m1-r01'
RELEASE='20261026-diagnostic-PAMF-SIG1-REPL-RXGeometry-release-r01'
SEED=2026092701
NAMESPACES=('PAMF/restored_dose/U/fixed',)+tuple('PAMF/realization_diversity/U/fresh'+str(r) for r in (1,2,3))


def validate_runtime(rt):
    if (set(rt)!={'run_id','project','python','release_root','run_root','context','runtime_file','code_commit','frozen_sources','allowed_gpus','launch_owner'}
            or rt['run_id']!=RUN or rt['run_root']!=rt['project']+'/runs/'+RUN or rt['release_root']!=rt['project']+'/releases/'+RELEASE
            or rt['context']!=rt['release_root']+'/context.json' or rt['runtime_file']!=rt['release_root']+'/runtime.json'
            or rt['allowed_gpus']!=list(range(8)) or rt['launch_owner']!='codex-pamf-sig1-repl-rx-geometry-20261026'):
        raise ValueError('One immutable source-only single-view RX geometry run required')


def load_reader(rt):
    import torch
    from .sig1_repl_base import config,load_prefix,check_state
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload,load_G
    from experiments.pamf_repl_completion.source import reader_structure
    context=read(rt['context']);validate_context(context)
    # Only the known completed m201 E20 is opened. The stopped control's outputs
    # and partial tensors are never inputs; its pure config builder validates ancestry.
    probe=config(context,rt['code_commit'],'SIG1-PL');parent,payload=load_prefix(probe);check_state(payload)
    c=parent['config'];contract=read(c['source_contract'])
    with reader_structure(c):model=reader_from_payload(c,payload,'cuda:0')
    del payload
    pool=model.sequence_pool
    if (len(pool.views)!=1 or tuple(pool.output_projection.weight.shape)!=(160,112)
            or sum(p.numel() for p in pool.parameters())!=18976 or tuple(model.core.id_backbone.cls_head.weight.shape)!=(6,160)):
        raise ValueError('Actual single SIG_TIME plus original default head required')
    return model,load_G(context['G'],contract,'cuda:0'),c,contract


def extract(rt):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    validate_runtime(rt);model,g,c,contract=load_reader(rt);root=Path(rt['run_root']);start=time.perf_counter()
    engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    labeled=SourceDataset(c['source_cache'],'L_s',contract)
    with torch.inference_mode():
        x=labeled.x[:2].to('cuda:0');z=model(x.clone());restored=engine(x.clone())[0].clone();zz=model(restored)
        if z.shape!=(2,6) or zz.shape!=(2,6) or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(zz).all()):
            raise ValueError('Actual frozen single-view E20/G/default-head source L2 smoke failed')
    write(root/'smoke.json',dict(status='PASS',teacher_epoch=20,signature_views=1,signature_parameters=18976,default_head_shape=[6,160],
        F_logical_packets=4,G_logical_packets=2,optimizer_updates=0,U_reads=0,U_truth_reads=0,target_reads=0))
    before=snapshot(model);gbefore=snapshot(g);cost=[]
    for role,n in [('L_s',6300),('V',27000)]:
        data=labeled if role=='L_s' else SourceDataset(c['source_cache'],role,contract)
        if len(data)!=n or list(data.metadata['ids'])!=contract['role_ids'][role] or 'y' not in data.metadata:
            raise ValueError('Exact original source role/order with visible L/V labels required')
        out=root/'features'/role;out.mkdir(parents=True,exist_ok=False)
        raw=np.lib.format.open_memmap(out/'raw_features.npy',mode='w+',dtype=np.float32,shape=(n,160))
        avg=np.lib.format.open_memmap(out/'restored_features.npy',mode='w+',dtype=np.float32,shape=(n,160));offset=0
        with torch.inference_mode():
            for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
                x=batch[0].to('cuda:0');b=len(x);z=model.features(x);total=torch.zeros_like(z)
                for namespace in NAMESPACES:
                    for scene in SCENES:
                        h=source_channel(x,batch[3],scene,contract,namespace=namespace);restored=engine(h.clone())[0].clone()
                        value=model.features(restored)
                        if not bool(torch.isfinite(value).all()) or bool((value.norm(dim=1)<=1e-8).any()):raise ValueError('Finite nonzero source G features required')
                        total+=F.normalize(value,dim=1)
                if not bool(torch.isfinite(z).all()) or bool((z.norm(dim=1)<=1e-8).any()) or bool((total.norm(dim=1)<=1e-8).any()):
                    raise ValueError('Finite nonzero frozen raw/Gmean12 feature required')
                raw[offset:offset+b]=fp32_copy(F.normalize(z,dim=1));avg[offset:offset+b]=fp32_copy(F.normalize(total/12,dim=1));offset+=b
                write(root/'progress.json',dict(status='SOURCE_SIG1_FEATURES',role=role,seen=offset,total=n,U_reads=0,target_reads=0),exclusive=False)
                if offset%1024==0 or offset==n:print('SOURCE_SIG1_FEATURES',role,offset,n,flush=True)
        raw.flush();avg.flush();del raw,avg
        if offset!=n or not unchanged(model,before) or not unchanged(g,gbefore):raise ValueError('Complete source coverage and unchanged F/G required')
        np.savez(out/'metadata.npz',**{key:data.metadata[key] for key in ('ids','y','rx_i','day_i')})
        item=dict(role=role,records=n,F_logical_packets=13*n,G_logical_packets=12*n,optimizer_updates=0,parameters_and_registered_buffers_unchanged=True)
        write(out/'complete.json',dict(status='SOURCE_SIG1_FEATURES_COMPLETE',**item));cost.append(item)
    write(root/'features/features_complete.json',dict(status='SOURCE_SIG1_FEATURE_EXTRACTION_COMPLETE',source_contract=contract,teacher_epoch=20,model_seed=SEED,
        roles=cost,source_F_logical_packets=432900,new_G_logical_packets=399600,smoke_separate=True,U_reads=0,U_truth_read=False,target_reads=0,
        optimizer_updates=0,seconds=time.perf_counter()-start,source='known completed m201 single-view ordinaryPL E20;no stopped-run partial'))


def worker(rt):
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    extract(rt)
    from .sig1_repl_rx_geometry import main
    main(Path(rt['run_root'])/'features',Path(rt['run_root'])/'results',rt['code_commit'])


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    validate_runtime(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    job=dict(id='RXGeometry',kind='geometry',config=rt['runtime_file'],entrypoint='experiments.pamf_channel_pl.sig1_repl_rx_source',
        execution_release=rt['release_root'],execution_python=rt['python'],row_id='SIG1-E20-RXGeometry-s2026092701')
    write(root/'jobs.json',[job]);state=dict(status='QUEUED',pid=os.getpid(),code_commit=rt['code_commit'],jobs={})
    while True:
        pending=processes();available=[]
        for gpu in range(8):
            pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
            if load<2 and free>=10000:available.append((load,-free,gpu))
        if available:break
        state.update(updated_at=time.time());atomic(root/'pipeline_state.json',state);time.sleep(12)
    child,item=launch(rt,job,min(available)[2]);state.update(status='RUNNING',jobs={'RXGeometry':item});atomic(root/'pipeline_state.json',state)
    code=child.wait();complete=root/'results/complete.json'
    good=code==0 and complete.is_file() and read(complete)['status']=='SOURCE_REPL_RX_GEOMETRY_COMPLETE'
    state.update(status='SOURCE_GEOMETRY_COMPLETE' if good else 'STOPPED_FOR_TECHNICAL_FAILURE',updated_at=time.time(),
        jobs={'RXGeometry':dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code)})
    atomic(root/'pipeline_state.json',state)
    if good:write(root/'controller_complete.json',dict(status='SOURCE_GEOMETRY_COMPLETE',geometry_pass=read(complete)['geometry_pass'],
        optimizer_updates=0,U_reads=0,target_reads=0,student_or_control_training=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    args=p.parse_args();rt=read(args.config)
    if args.action=='dispatch':dispatch(rt)
    else:
        if args.kind!='geometry':raise ValueError('Only frozen source geometry, no training stage')
        worker(rt)
