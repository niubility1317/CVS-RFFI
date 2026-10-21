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


def profile_module(net,x,call):
    """Source-only eager operator counts; no latency inferred from these counts."""
    totals=dict(conv_linear_MACs=0,FFT_IFFT_calls=0);hooks=[];original={}
    def hook(module,inputs,output):
        if isinstance(module,torch.nn.Linear):totals['conv_linear_MACs']+=output.numel()*module.in_features
        else:totals['conv_linear_MACs']+=output.numel()*(module.in_channels//module.groups)*int(np.prod(module.kernel_size))
    for module in net.modules():
        if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d)):hooks.append(module.register_forward_hook(hook))
    for name in ('fft','ifft','rfft','irfft','fft2','ifft2'):
        original[name]=getattr(torch.fft,name)
        def counted(*args,_fn=original[name],**kwargs):
            totals['FFT_IFFT_calls']+=1;return _fn(*args,**kwargs)
        setattr(torch.fft,name,counted)
    try:
        with torch.inference_mode():call(x)
        torch.cuda.synchronize()
    finally:
        for h in hooks:h.remove()
        for name,fn in original.items():setattr(torch.fft,name,fn)
    totals.update(parameters=sum(p.numel() for p in net.parameters()),FP32_parameter_bytes=sum(p.numel()*p.element_size() for p in net.parameters()),
        note='Conv/Linear real MAC convention only; functional cosine head, FFT complex arithmetic, CG reductions, normalization and elementwise work excluded. FFT counts are call counts at B1, not FLOPs.')
    return totals


def exclusive():
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    pids=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
    if set(pids)!={str(os.getpid())}:raise RuntimeError('TIMING_INVALID_SHARED_GPU: retain partial samples, no latency claim')


def expose_external_baselines(release):
    """Native F imported its own baselines package; extend only this namespace."""
    import baselines
    folder=str(Path(release)/'baselines')
    if not Path(folder,'riei_fd','model.py').is_file():raise FileNotFoundError('Frozen comparator baselines missing')
    if folder not in baselines.__path__:baselines.__path__.insert(0,folder)


def run(c):
    from .frozen_inference import load_original_F,load_F,load_G
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
    expose_external_baselines(c['external_release'])
    from experiments.phase1_external_dg_aug.source import frozen_model,forward
    external=[];mechanisms=[];mechanism_parameters={}
    for item in c['external_models']:
        frozen=read(Path(item['source_output'])/'freeze.json')
        net,sc=frozen_model(item['source_output'],frozen['source_contract'],'cuda:0')
        net.requires_grad_(False);external.append(net)
        cases[item['name']]=(lambda x,net=net,sc=sc:forward(net,sc,x))
    for item in c.get('F_mechanisms',[]):
        net,_=load_F(item['source_output'],contract,'cuda:0');mechanisms.append(net)
        mechanism_parameters[item['name']]=sum(p.numel() for p in net.parameters())
        cases[item['name']+'_F']=(lambda x,net=net:net.classify_features(net.features(x)))
        engine=InferenceEngine(g,net,native=False,reuse=True,graph=True);engines.append(engine)
        cases[item['name']+'_GF']=(lambda x,engine=engine:engine(x)[3])
    for item in c.get('G_mechanisms',[]):
        net=load_G(item['spec'],contract,'cuda:0');mechanisms.append(net)
        mechanism_parameters[item['name']]=sum(p.numel() for p in net.parameters())
        cases[item['name']+'_G']=(lambda x,net=net:net(x))
        cases[item['name']+'_GF']=(lambda x,net=net:reader.classify_features(reader.features(net(x))))
    source_one=tensor(np.ascontiguousarray(data[:1]),'cuda:0')
    profiles={'T1_F':profile_module(reader,source_one,lambda x:reader.classify_features(reader.features(x))),
        'DURA_G':profile_module(g,source_one,g)}
    write(root/'operator_profiles.json',profiles)
    write(root/'resolved.json',dict(config=c,device=torch.cuda.get_device_name(),python=__import__('sys').version,
        torch=torch.__version__,CPU_threads=torch.get_num_threads(),precision='FP32',TF32=False,
        F_parameters=sum(p.numel() for p in reader.parameters()),G_parameters=sum(p.numel() for p in g.parameters()),
        mechanism_parameters=mechanism_parameters,model_load_seconds=load_seconds,cases=list(cases),source_IQ_only=True,restored_IQ_cache=False,
        scheduler_reservation_bytes=c.get('scheduler_reservation_bytes',0),
        memory_note='All listed models/graphs plus explicitly recorded scheduler VRAM reservation resident; incremental peak excludes baseline. Absolute peak is not isolated model memory.'))
    results=[]
    with (root/'samples.jsonl').open('x',encoding='utf-8') as stream:
        for batch in (1,8,32,64,128):
            cpu=tensor(np.ascontiguousarray(data[:batch])).pin_memory();x=cpu.to('cuda:0')
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
                        torch.cuda.synchronize();baseline_allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                        wall=[];device=[]
                        for _ in range(repeats):
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start=time.perf_counter();a.record();value=call(x);b.record();b.synchronize()
                            wall.append((time.perf_counter()-start)*1000);device.append(a.elapsed_time(b))
                        copies=[]
                        for _ in range(20):
                            torch.cuda.synchronize();start=time.perf_counter();v=call(cpu.to('cuda:0',non_blocking=True))
                            (v.argmax(1) if v.ndim==2 else v).cpu();torch.cuda.synchronize()
                            copies.append((time.perf_counter()-start)*1000)
                    exclusive()
                    item=dict(case=name,batch=batch,round=round_index,warmup=warm,repeats=repeats,
                        wall_ms=wall,device_ms=device,H2D_D2H_ms=copies,cold_first_call_ms=cold[name]*1000,
                        resident_baseline_bytes=baseline_allocated,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        incremental_peak_bytes=max(0,torch.cuda.max_memory_allocated()-baseline_allocated),exclusive_GPU=True)
                    stream.write(__import__('json').dumps(item,allow_nan=False)+'\n');stream.flush()
                    results.append(dict(case=name,batch=batch,round=round_index,median_ms=float(np.median(wall)),
                        P95_ms=float(np.percentile(wall,95)),throughput_IQ_s=batch*1000/float(np.median(wall))))
                    print('TIMING_ROUND',name,batch,round_index,results[-1]['median_ms'],flush=True)
    write(root/'summary.json',dict(results=results,graph_setup=[e.setup() for e in engines],exclusive_GPU=True,
        training_only_mechanisms_incremental_deployment_cost=0,mechanism_note='EMA/RDrop/SupCon/U filtering are absent from frozen deployment; their training costs are separately logged'))
    write(root/'complete.json',dict(status='TIMING_COMPLETE',batches=[1,8,32,64,128],rounds=5,cases=list(cases),source_IQ_only=True,exclusive_GPU=True))
