"""Exclusive-device FP32 online G/F timing, including graph/copy setup costs."""
from pathlib import Path
import os
import time
import subprocess
import numpy as np
import torch
from .io import read,write,tensor

EXECUTIONS=(('T0_native',True,False,False),('T1_tensor',False,False,False),
            ('T2_reuse',False,True,False),('T3_graph',False,False,True),('T4_graph_reuse',False,True,True))


def exclusive():
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    pids=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    if set(pids)!={str(os.getpid())}:raise RuntimeError('TIMING_INVALID_SHARED_GPU: retain partial samples, no latency claim')


def run(c):
    from .frozen_inference import load_original_F,load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from experiments.pamf_r46_cuda_graph.graph import GraphCache
    exclusive()
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    contract=read(c['source_contract'])
    data=np.load(Path(c['source_cache'])/'L_s_x.npy',mmap_mode='r',allow_pickle=False)[:128]
    tic=time.perf_counter();reader=load_original_F(2026092701,contract,'cuda:0')
    g=load_G(c['G'],contract,'cuda:0');load_seconds=time.perf_counter()-tic
    cases={};engines=[]
    def f(x):
        features=reader.features(x);logits=reader.classify_features(features)
        return (logits,torch.isfinite(logits).all())
    cases['F_eager']=lambda x:f(x)[0]
    fgraph=GraphCache(f)
    def graph_f(x):
        logits,valid=fgraph(x)
        if not bool(valid):raise FloatingPointError('Nonfinite F timing')
        return logits
    cases['F_graph']=graph_f
    for name,native,reuse,graph in EXECUTIONS:
        for mode,chosen in (('G',None),('GF',reader)):
            engine=InferenceEngine(g,chosen,native=native,reuse=reuse,graph=graph)
            engines.append(engine)
            cases[name+'_'+mode]=(lambda x,engine=engine,mode=mode:engine(x)[3 if mode=='GF' else 0])
    # The external methods are the exact frozen full/own-CE models. No model
    # fitting, target scores or data-loading duration enters deployment timing.
    import experiments
    experiments.__path__.insert(0,c['external_release']+'/experiments')
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
    sys.path.insert(0,c['external_release'])
    from experiments.phase1_external_dg_aug.source import frozen_model,forward
    external=[]
    for item in c['external_models']:
        frozen=read(Path(item['source_output'])/'freeze.json')
        net,sc=frozen_model(item['source_output'],frozen['source_contract'],'cuda:0')
        net.requires_grad_(False);external.append(net)
        cases[item['name']]=(lambda x,net=net,sc=sc:forward(net,sc,x))
    write(root/'resolved.json',dict(config=c,device=torch.cuda.get_device_name(),python=__import__('sys').version,
        torch=torch.__version__,CPU_threads=torch.get_num_threads(),precision='FP32',TF32=False,
        F_parameters=sum(p.numel() for p in reader.parameters()),G_parameters=sum(p.numel() for p in g.parameters()),
        model_load_seconds=load_seconds,cases=list(cases),source_IQ_only=True,restored_IQ_cache=False))
    results=[]
    with (root/'samples.jsonl').open('x',encoding='utf-8') as stream:
        for batch in (1,8,32,64,128):
            cpu=tensor(np.ascontiguousarray(data[:batch]));x=cpu.to('cuda:0')
            warm,repeats=(100,1000) if batch==1 else (20,100)
            cold={}
            for round_index in range(5):
                names=list(cases);shift=round_index%len(names);names=names[shift:]+names[:shift]
                if round_index%2:names.reverse()
                for name in names:
                    exclusive();call=cases[name]
                    with torch.inference_mode():
                        torch.cuda.synchronize();start=time.perf_counter();value=call(x);torch.cuda.synchronize()
                        if round_index==0:cold[name]=time.perf_counter()-start
                        if not bool(torch.isfinite(value).all()):raise FloatingPointError('Invalid timing output')
                        for _ in range(warm):call(x)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        wall=[];device=[]
                        for _ in range(repeats):
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start=time.perf_counter();a.record();value=call(x);b.record();b.synchronize()
                            wall.append((time.perf_counter()-start)*1000);device.append(a.elapsed_time(b))
                        copies=[]
                        for _ in range(20):
                            torch.cuda.synchronize();start=time.perf_counter();v=call(cpu.to('cuda:0'));v.cpu();torch.cuda.synchronize()
                            copies.append((time.perf_counter()-start)*1000)
                    exclusive()
                    item=dict(case=name,batch=batch,round=round_index,warmup=warm,repeats=repeats,
                        wall_ms=wall,device_ms=device,H2D_D2H_ms=copies,cold_first_call_ms=cold[name]*1000,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),exclusive_GPU=True)
                    stream.write(__import__('json').dumps(item,allow_nan=False)+'\n');stream.flush()
                    results.append(dict(case=name,batch=batch,round=round_index,median_ms=float(np.median(wall)),
                        P95_ms=float(np.percentile(wall,95)),throughput_IQ_s=batch*1000/float(np.median(wall))))
                    print('TIMING_ROUND',name,batch,round_index,results[-1]['median_ms'],flush=True)
    write(root/'summary.json',dict(results=results,graph_setup=[e.setup() for e in engines],exclusive_GPU=True,
        training_only_mechanisms_incremental_deployment_cost=0,mechanism_note='EMA/RDrop/SupCon/U filtering are absent from frozen deployment; their training costs are separately logged'))
    write(root/'complete.json',dict(status='TIMING_COMPLETE',batches=[1,8,32,64,128],rounds=5,cases=list(cases),source_IQ_only=True,exclusive_GPU=True))
