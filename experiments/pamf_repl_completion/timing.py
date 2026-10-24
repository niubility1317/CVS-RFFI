"""Exclusive source-IQ timing for updated F; existing pure-G results are retained."""
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from .source import load_reader

def run(c):
    import torch
    from experiments.cric_dura_comparison.timing import exclusive,profile_module
    from experiments.cric_dura_comparison.frozen_inference import load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from .plan import STRUCTURES
    # The scheduler has admitted an empty card. Register this process with CUDA
    # before asking the existing verifier to require exactly our compute PID.
    torch.cuda.init();reservation=torch.empty(1,device='cuda:0')
    exclusive();context=read(c['context']);sources=read(c['sources']);contract=read(c['source_contract'])
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    data=np.load(Path(c['source_cache'])/'L_s_x.npy',mmap_mode='r',allow_pickle=False)[:128]
    cases={};engines=[];models=[];parameters={}
    if c['panel']=='G_combinations':
        reader,_=load_reader(context['sources']['2026092701']['REPL'],contract,'cuda:0')
        models.append(reader)
        for item in context['G_components']:
            if item['seed']!=2026092701:continue
            g=load_G(item['spec'],contract,'cuda:0');models.append(g)
            if getattr(g,'_cd_eager_variant',False):
                cases[item['arm']+'_GF']=lambda x,g=g:reader.classify_features(reader.features(g(x)))
            else:
                engine=InferenceEngine(g,reader,native=False,reuse=True,graph=True);engines.append(engine)
                cases[item['arm']+'_GF']=lambda x,e=engine:e(x)[3]
            parameters[item['arm']]=sum(p.numel() for p in g.parameters())
    else:
        methods=[('REPL' if a=='full' else 'REPL_'+a) for a in STRUCTURES] if c['panel']=='REPL_structure' else ['CE_'+a for a in STRUCTURES] if c['panel']=='strict_structure' else ['supervised','ordinary_PL']
        restorers={}
        for method in methods:
            source=sources['F0-'+method+'-s2026092701']['source_output']
            reader,cfg=load_reader(source,contract,'cuda:0');models.append(reader)
            role='LU' if cfg.get('U_use') and cfg.get('ablation')!='F20' else 'L'
            if role not in restorers:restorers[role]=load_G(context['caches']['F0'][role]['G'],contract,'cuda:0')
            cases[method+'_F']=lambda x,f=reader:f.classify_features(f.features(x))
            engine=InferenceEngine(restorers[role],reader,native=False,reuse=True,graph=True);engines.append(engine)
            cases[method+'_GF']=lambda x,e=engine:e(x)[3]
            parameters[method]=sum(p.numel() for p in reader.parameters())
    if len(cases)*25!=c['case_rounds']:raise ValueError('Timing panel size differs')
    write(out/'resolved.json',dict(config=c,device=torch.cuda.get_device_name(),cases=list(cases),parameters=parameters,
        physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),FP32=True,TF32=False,
        source_IQ_only=True,exclusive_GPU=True,model_memory_note='Co-resident models/graphs; baseline and incremental allocation separate; no isolated-model VRAM claim',
        pure_G_retimed=False,training_only_REPL_blocks_have_no_added_inference_parameters=True))
    summary=[]
    with (out/'samples.jsonl').open('x',encoding='utf-8') as stream:
        import json
        for batch in (1,8,32,64,128):
            cpu=tensor(np.ascontiguousarray(data[:batch])).pin_memory();x=cpu.to('cuda:0')
            warm,repeats=(100,1000) if batch==1 else (20,100)
            for round_index in range(5):
                names=list(cases);shift=round_index%len(names);names=names[shift:]+names[:shift]
                if round_index%2:names.reverse()
                for name in names:
                    exclusive();call=cases[name]
                    with torch.inference_mode():
                        torch.cuda.synchronize();tic=time.perf_counter();value=call(x);torch.cuda.synchronize();cold=(time.perf_counter()-tic)*1000
                        if not bool(torch.isfinite(value).all()):raise FloatingPointError('Timing output invalid')
                        for _ in range(warm):call(x)
                        torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();wall=[];device=[];copies=[]
                        for _ in range(repeats):
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            tic=time.perf_counter();a.record();call(x);b.record();b.synchronize()
                            wall.append((time.perf_counter()-tic)*1000);device.append(a.elapsed_time(b))
                        for _ in range(20):
                            torch.cuda.synchronize();tic=time.perf_counter();v=call(cpu.to('cuda:0',non_blocking=True));v.argmax(1).cpu();torch.cuda.synchronize();copies.append((time.perf_counter()-tic)*1000)
                    exclusive();rec=dict(case=name,batch=batch,round=round_index,warmup=warm,repeats=repeats,wall_ms=wall,device_ms=device,H2D_D2H_ms=copies,
                        first_call_ms=cold,resident_baseline_bytes=baseline,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        incremental_peak_bytes=max(0,torch.cuda.max_memory_allocated()-baseline),exclusive_GPU=True)
                    stream.write(json.dumps(rec)+'\n');stream.flush()
                    summary.append(dict(case=name,batch=batch,round=round_index,median_ms=float(np.median(wall)),P95_ms=float(np.percentile(wall,95)),throughput_IQ_s=batch*1000/float(np.median(wall))))
                    print('TIMING_ROUND',name,batch,round_index,flush=True)
    write(out/'summary.json',dict(results=summary,graph_setup=[e.setup() for e in engines],pure_G_retimed=False))
    write(out/'complete.json',dict(status='TIMING_COMPLETE',case_rounds=len(summary),cases=list(cases),rounds=5,batches=[1,8,32,64,128],exclusive_GPU=True,source_IQ_only=True))
