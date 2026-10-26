"""Three new hidden-source-U H/G realizations,with frozen G and no F access."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time
from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,SCENES,source_channel,snapshot,unchanged

RUN='20261026-diagnostic-PAMF-URealizations-cache-m3-r01'
RELEASE='20261026-diagnostic-PAMF-URealizations-cache-release-r01'
REPLICAS=(1,2,3)
KEYS=('eq_i','rx_i','day_i','sig_i','base_index')


def namespace(replica):
    if type(replica) is not int or replica not in REPLICAS:raise ValueError('Three fixed U realizations required')
    return 'PAMF/realization_diversity/U/fresh'+str(replica)


def validate_config(c):
    if set(c)!={'run_id','source_context','frozen_sources','mode','replica','output_root','code_commit'}:
        raise ValueError('G-only source cache rejects target/F/foreign handles')
    ctx=c['source_context'];validate_context(ctx);root=ctx['project']+'/runs/'+RUN
    if (c['run_id']!=RUN or c['frozen_sources']!=ctx['frozen_sources'] or c['mode'] not in ('smoke','cache')
            or type(c['replica']) is not int or (c['replica']!=0 if c['mode']=='smoke' else c['replica'] not in REPLICAS)
            or Path(c['output_root'])!=Path(root if c['mode']=='smoke' else root+'/fresh'+str(c['replica']))):
        raise ValueError('Registered hidden-U realization output/mode required')
    return c


def hidden(batch,contract):
    import torch
    if (not bool((batch[1]==-1).all()) or bool(torch.as_tensor(batch[3]['tx_label_visible']).any())
            or set(torch.as_tensor(batch[3]['rx_i']).tolist())-set(contract['source_rxs'])
            or set(torch.as_tensor(batch[3]['day_i']).tolist())-set(contract['source_days'])
            or set(torch.as_tensor(batch[3]['eq_i']).tolist())!={1}):
        raise ValueError('Hidden-label original source U only')


def source(c):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    ctx=c['source_context'];case=next(v for v in ctx['cases'] if v['model_seed']==2026092701)
    # Metadata only;never open final_ssdg.pth or a frozen F output.
    cfg=read(Path(case['parent_source']).parent/'segment30/complete.json')['config']
    contract=read(cfg['source_contract'])
    if contract!=read(case['expected_source_contract']) or contract['counts']!=dict(L_s=6300,U_s=56700,V=27000):
        raise ValueError('Original F0 source data handles required')
    data=SourceDataset(cfg['source_cache'],'U_s',contract)
    if len(data)!=56700 or 'y' in data.metadata or list(data.metadata['ids'])!=contract['role_ids']['U_s']:
        raise ValueError('Complete original U physical role with TX labels absent required')
    return data,contract,cfg['source_cache']


def setup(c):
    validate_config(c)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in list(map(str,range(8))):raise ValueError('One admitted physical GPU required')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    torch.set_num_threads(2);torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from experiments.cric_dura_comparison.frozen_inference import load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    data,contract,cache=source(c);g=load_G(c['source_context']['G'],contract,'cuda:0');g.eval()
    return data,contract,cache,g,InferenceEngine(g,reuse=True,graph=True,native=False)


def smoke(c):
    import torch
    data,contract,cache,g,engine=setup(c)
    batch=next(iter(torch.utils.data.DataLoader(data,batch_size=2,shuffle=False,num_workers=0)));hidden(batch,contract)
    x=batch[0].to('cuda:0');before={name:p.detach().clone() for name,p in g.named_parameters()}
    with torch.inference_mode():
        a=g(x.clone());b=engine(x.clone())[0].clone()
    if (a.shape!=(2,2,256) or b.dtype!=torch.float32 or not bool(torch.isfinite(b).all()) or not torch.allclose(a,b,atol=1e-5,rtol=1e-5)
            or any(not torch.equal(p,before[name]) for name,p in g.named_parameters())):
        raise ValueError('Real frozen same-contract G hidden-U2 graph/eager smoke failed')
    write(Path(c['output_root'])/'smoke.json',dict(status='PASS',source_role='U_s',physical_packets=2,G_logical_packet_calls=4,
        G_eager_graph_close=True,G_parameters_unchanged=True,F_weight_reads=0,F_output_reads=0,F_forward_packets=0,
        U_truth_read=False,V_reads=0,target_reads=0,optimizer_updates=0))


def cache(c):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    start=time.perf_counter();data,contract,source_cache,g,engine=setup(c)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);n=len(data);stream=namespace(c['replica'])
    with torch.inference_mode():warm=engine(data.x[:2].to('cuda:0').clone())[0].clone()
    if not bool(torch.isfinite(warm).all()):raise ValueError('Finite frozen G warmup required')
    before=snapshot(g)
    H=np.lib.format.open_memmap(root/'channel_U.npy',mode='w+',dtype=np.float32,shape=(3,n,2,256))
    G=np.lib.format.open_memmap(root/'restored_U.npy',mode='w+',dtype=np.float32,shape=(3,n,2,256));offset=0
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            hidden(batch,contract);x=batch[0].to('cuda:0');count=len(x)
            for k,scene in enumerate(SCENES):
                h=source_channel(x,batch[3],scene,contract,namespace=stream);restored=engine(h.clone())[0].clone()
                if (h.shape!=(count,2,256) or restored.shape!=h.shape or h.dtype!=torch.float32 or restored.dtype!=torch.float32
                        or not bool(torch.isfinite(h).all()) or not bool(torch.isfinite(restored).all())):
                    raise ValueError('Complete finite paired FP32 H/G source views required')
                H[k,offset:offset+count]=fp32_copy(h);G[k,offset:offset+count]=fp32_copy(restored)
            offset+=count
            write(root/'progress.json',dict(status='SOURCE_U_HG_REALIZATION',replica=c['replica'],U_seen=offset,total=n,
                G_cache_logical_packet_calls=3*offset,F_forward_packets=0,U_truth_read=False,V_reads=0,target_reads=0,
                seconds=time.perf_counter()-start),exclusive=False)
            if offset%2560==0 or offset==n:print('SOURCE_U_HG_REALIZATION',c['replica'],offset,n,flush=True)
    H.flush();G.flush();logical_bytes=int(H.nbytes+G.nbytes);del H,G
    physical=np.stack([data.metadata[key] for key in KEYS],axis=1)
    if offset!=56700 or len({tuple(v) for v in physical.tolist()})!=56700 or not unchanged(g,before):
        raise ValueError('U coverage/physical alignment/frozen G state differs')
    np.savez(root/'physical_index.npz',physical=physical,ids=data.metadata['ids'])
    for name in ['channel_U.npy','restored_U.npy']:
        value=np.load(root/name,mmap_mode='r',allow_pickle=False)
        if value.shape!=(3,56700,2,256) or value.dtype!=np.float32:raise ValueError('Written paired cache header differs')
    write(root/'complete.json',dict(status='SOURCE_U_HG_REALIZATION_COMPLETE',run_id=RUN,code_commit=c['code_commit'],replica=c['replica'],namespace=stream,
        source_role='U_s',U_seen=n,scenes=list(SCENES),source_cache=source_cache,source_contract=contract,G=c['source_context']['G'],
        shape=[3,n,2,256],dtype='float32',physical_keys=list(KEYS),IQ_float32_bytes=logical_bytes,
        G_cache_logical_packet_calls=3*n,G_warmup_logical_packet_calls=2,F_weight_reads=0,F_output_reads=0,F_forward_packets=0,
        U_truth_read=False,V_reads=0,target_reads=0,optimizer_updates=0,G_updates=0,
        G_parameters_and_registered_buffers_unchanged=True,seconds=time.perf_counter()-start))
    print('SOURCE_U_HG_REALIZATION_COMPLETE',c['replica'],time.perf_counter()-start,flush=True)


def prepare(rt):
    ctx=read(rt['context']);validate_context(ctx);root=Path(rt['run_root']);jobs=[]
    for mode,replica in [('smoke',0)]+[('cache',r) for r in REPLICAS]:
        name='smoke' if mode=='smoke' else 'fresh'+str(replica)
        c=dict(run_id=RUN,source_context=ctx,frozen_sources=rt['frozen_sources'],mode=mode,replica=replica,
            output_root=str(root if mode=='smoke' else root/name),code_commit=rt['code_commit']);validate_config(c)
        path=root/'configs'/(name+'.json');write(path,c)
        jobs.append(dict(id=name,kind=mode,config=str(path),completion=str(root/'smoke.json' if mode=='smoke' else root/name/'complete.json'),
            successful_statuses=['PASS' if mode=='smoke' else 'SOURCE_U_HG_REALIZATION_COMPLETE'],
            dependencies=[] if mode=='smoke' else [str(root/'smoke.json')],gpu_pool=list(range(8)),execution_release=rt['release_root'],execution_python=rt['python'],
            entrypoint='experiments.pamf_channel_pl.u_realization_cache',row_id=name))
    write(root/'jobs.json',jobs);return jobs


def summarize(rt,jobs):
    values=[read(j['completion']) for j in jobs if j['kind']=='cache']
    if len(values)!=3 or any(v['status']!='SOURCE_U_HG_REALIZATION_COMPLETE' for v in values):raise ValueError('All three caches required')
    first=values[0]
    if any(v['source_contract']!=first['source_contract'] or v['G']!=first['G'] or v['U_seen']!=56700 for v in values):
        raise ValueError('One original U/G contract required')
    write(Path(rt['run_root'])/'complete.json',dict(status='SOURCE_U_REALIZATION_CACHE_COMPLETE',run_id=RUN,code_commit=rt['code_commit'],
        realizations=values,U_physical_records=56700,replicas=list(REPLICAS),new_G_cache_logical_packet_calls=510300,
        G_worker_warmup_logical_packet_calls=6,G_shared_smoke_logical_packet_calls=4,new_IQ_float32_bytes=2090188800,
        F_weight_reads=0,F_output_reads=0,F_forward_packets=0,U_truth_read=False,V_reads=0,target_reads=0,optimizer_updates=0,G_updates=0,
        all_G_parameters_and_registered_buffers_unchanged=True,scope='Three source U augmentations,not three F seeds;no F training,promotions or target predictions'))


def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    if rt['run_id']!=RUN or rt['allowed_gpus']!=list(range(8)):raise ValueError('Fixed source U cache matrix required')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=prepare(rt);by={j['id']:j for j in jobs}
    state=dict(status='RUNNING',pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs},started_at=time.time())
    active={};stop=False;atomic(root/'pipeline_state.json',state)
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code)
            if not good:stop=True
            del active[key]
        pending=processes()
        for job in jobs:
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job) or len(active)>=3:continue
            available=[]
            for gpu in job['gpu_pool']:
                pids,free=capacity(gpu);load=len(pids|{pid for pid,g in pending.items() if g==gpu})
                if load<2 and free>=10000:available.append((load,-free,gpu))
            if not available:continue
            gpu=min(available)[2];child,item=launch(rt,job,gpu);active[key]=(child,item);state['jobs'][key]=item;pending[child.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),systemic_stop_new_launches=stop)
        atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    if not stop:summarize(rt,jobs)
    state['status']='STOPPED_FOR_TECHNICAL_FAILURE' if stop else 'SOURCE_U_REALIZATION_CACHE_COMPLETE';atomic(root/'pipeline_state.json',state)


def worker(c,kind):
    validate_config(c)
    if kind!=c['mode']:raise ValueError('Declared G-only source stage required')
    return smoke(c) if kind=='smoke' else cache(c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else worker(c,a.kind)
