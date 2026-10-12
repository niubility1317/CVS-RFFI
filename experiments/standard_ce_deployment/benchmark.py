"""Truth-free CPU-IQ -> CPU-label deployment measurement on frozen CE models."""
import argparse
import csv
import gc
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.standard_ce_baselines.common import PROJECT,RUN,EXTENSION_RUN,read,write

VARIANTS=('residual_fusion','resnet50','densenet121','cvresnet18_1d','resnet18','resnet34',
          'vgg16_bn','bilstm3','convnext_tiny','resnet50_2d','resnet18_2d','cvresnet18_2d','transformer6')
BENCH_RUN='20261012-phase1-standard-ce-deployment-m104-r02'

def source_folder(variant,seed):
    if variant not in VARIANTS or seed not in range(2026092701,2026092709):raise ValueError('Invalid fixed row')
    return Path(PROJECT)/'runs'/(RUN if seed<=2026092704 else EXTENSION_RUN)/f'{variant}-s{seed}'/'source'

def row_paths(variant,seed,out):
    source=source_folder(variant,seed)
    return source,Path(out)/f'{variant}-s{seed}'

def summary(values):
    import numpy as np
    return dict(mean_ms=statistics.mean(values),sd_ms=statistics.stdev(values),
        p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),
        p99_ms=float(np.percentile(values,99)),minimum_ms=min(values),maximum_ms=max(values),trials=len(values))

def worker(variant,seed,out,smoke=False):
    import numpy as np
    import torch
    from experiments.standard_ce_baselines.model import build
    from experiments.standard_ce_baselines.predict import check
    source,folder=row_paths(variant,seed,out)
    folder.mkdir(parents=True,exist_ok=False)
    meta={f:read(source/f'{f}.json') for f in ('completion','initialization','source_contract','resolved_config','freeze')}
    c=dict(variant=variant,model_seed=seed,source_output=str(source))
    expected=read(meta['resolved_config']['source_contract'])
    payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
    check(c,meta['completion'],meta['initialization'],meta['source_contract'],expected,meta['resolved_config'],payload,meta['freeze'])
    del meta,expected,c
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=build(variant);model.load_state_dict(payload['model'],strict=True);del payload
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in model.state_dict().values()):raise ValueError('Unexpected precision')
    with io.BytesIO() as buffer:
        torch.save(model.state_dict(),buffer)
        storage_bytes=buffer.tell()
    resident_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values())
    model.to('cuda:0').eval();gc.collect();torch.cuda.empty_cache()
    with torch.no_grad():
        y=model(torch.zeros(2,2,256,device='cuda:0'))
        if y.shape!=(2,6) or not torch.isfinite(y).all():raise ValueError('No-query checkpoint smoke failed')
        del y
    header=dict(variant=variant,model_seed=seed,checkpoint=str(source/'last.pt'),
        checkpoint_provenance='VERIFIED_SCRATCH_MATCHED_CONTRACT_FIXED_E200',
        commit=(ROOT/'release_commit.txt').read_text().strip(),hardware=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,precision='FP32',tf32=False,
        cpu_threads=2,model_storage_bytes=storage_bytes,resident_state_bytes=resident_bytes,
        total_parameters=sum(p.numel() for p in model.parameters()),truth_read=False,dataset_read=False,
        input_seed=20261012,input_kind='deterministic synthetic already-synchronized/RMS-normalized CPU IQ',
        timing_scope='CPU contiguous input/byte-buffer construction -> H2D -> complete model (including FFT/tokenization/reshape) -> finite check -> argmax -> CPU labels/list; no disk/network/acquisition/synchronization front-end/batch queue',
        storage_scope='Actual uncompressed torch.save(FP32 state_dict) serialization byte length including buffers/metadata; no optimizer, training history or model code')
    if smoke:
        write(folder/'smoke.json',dict(header,status='PASS'));print('NO_QUERY_SMOKE_PASS',flush=True);return
    results=[]
    for batch,trials in ((1,300),(128,100)):
        rng=np.random.default_rng(20261012)
        bank=rng.standard_normal((16,batch,2,256),dtype=np.float32)
        bank/=np.sqrt(np.mean(bank**2,axis=(-2,-1),keepdims=True))
        def inference(index):
            data=np.ascontiguousarray(bank[index%16],dtype=np.float32)
            x=torch.frombuffer(bytearray(data.tobytes()),dtype=torch.float32).reshape(data.shape).to('cuda:0')
            logits=model(x)
            if logits.shape!=(batch,6) or not torch.isfinite(logits).all():raise ValueError('Invalid finite prediction')
            return logits.argmax(1).cpu().tolist()
        with torch.no_grad():
            for index in range(30):inference(index)
            torch.cuda.synchronize()
            elapsed=[]
            for index in range(trials):
                start=time.perf_counter_ns();labels=inference(index);torch.cuda.synchronize()
                elapsed.append((time.perf_counter_ns()-start)/1e6)
                if len(labels)!=batch:raise ValueError('Incomplete labels')
            # Memory is measured separately so sampling does not perturb latency.
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            def process_rss():
                return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            baseline_rss=process_rss();rss=[baseline_rss];stop=threading.Event()
            def sample_rss():
                while not stop.is_set():rss.append(process_rss());stop.wait(.001)
            thread=threading.Thread(target=sample_rss,daemon=True);thread.start()
            try:
                for index in range(50):inference(index)
                torch.cuda.synchronize();rss.append(process_rss())
            finally:stop.set();thread.join()
        item=dict(header,batch_size=batch,warmups=30,**summary(elapsed),
            latency_trials_ms=elapsed,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cpu_baseline_rss_bytes=baseline_rss,cpu_inference_peak_rss_bytes=max(rss),cpu_rss_sample_interval_ms=1,
            memory_trials=50,mean_batch_throughput_samples_s=batch*1000/statistics.mean(elapsed))
        results.append(item)
        with (folder/f'latency_b{batch}.csv').open('x',encoding='utf-8',newline='') as file:
            writer=csv.writer(file);writer.writerow(['trial','end_to_end_ms']);writer.writerows(enumerate(elapsed,1))
        write(folder/f'b{batch}.json',item)
        print(json.dumps(dict(variant=variant,seed=seed,batch=batch,mean_ms=item['mean_ms'],p95_ms=item['p95_ms'])),flush=True)
    write(folder/'complete.json',dict(status='MEASURED',results=results,truth_read=False,dataset_read=False))

def gpu_processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    uuid=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    return [line for line in output.splitlines() if uuid in line]

def dispatch(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    state=dict(status='RUNNING',pid=os.getpid(),cwd=str(ROOT),completed=0,total=104,rows=[],gpu=0,
        commit=(ROOT/'release_commit.txt').read_text().strip(),truth_read=False,dataset_read=False)
    write(out/'pipeline_state.json',state)
    command=[sys.executable,'-m','experiments.standard_ce_deployment.benchmark']
    if gpu_processes():raise RuntimeError('GPU0 is occupied; no workloads are stopped')
    smoke=subprocess.run(command+['--worker','--variant','residual_fusion','--seed','2026092701','--output',str(out/'smoke'),'--smoke'],cwd=ROOT)
    if smoke.returncode:raise RuntimeError('No-query smoke failed')
    all_results=[]
    for variant in VARIANTS:
        for seed in range(2026092701,2026092709):
            if gpu_processes():
                state.update(status='PARTIAL_GPU_BUSY');write(out/'pipeline_state.json',state);return
            row=dict(variant=variant,model_seed=seed,status='RUNNING')
            with (out/f'{variant}-s{seed}.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command+['--worker','--variant',variant,'--seed',str(seed),'--output',str(out/'rows')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state.update(current_row=dict(row,pid=child.pid));write(out/'pipeline_state.json',state)
                contaminated=False
                while child.poll() is None:
                    if any(int(line.split(',')[0])!=child.pid for line in gpu_processes()):contaminated=True
                    time.sleep(1)
            if child.returncode or contaminated:
                row['status']='FAILED' if child.returncode else 'INVALID_GPU_INTERFERENCE'
                state['rows'].append(row);state.update(status='PARTIAL',current_row=None);write(out/'pipeline_state.json',state);return
            done=read(out/'rows'/f'{variant}-s{seed}'/'complete.json')
            if done['status']!='MEASURED' or len(done['results'])!=2:raise ValueError('Missing row artifacts')
            all_results.extend(done['results']);row['status']='MEASURED';state['rows'].append(row)
            state.update(completed=len(state['rows']),current_row=None);write(out/'pipeline_state.json',state)
            print(f"MEASURED {state['completed']}/104 {variant} {seed}",flush=True)
    write(out/'results.json',dict(status='VERIFIED',run_id=BENCH_RUN,results=all_results,models=104,networks=13,
        hardware='NVIDIA GeForce RTX 3090',gpu=0,exclusive_during_measurement=True,truth_read=False,dataset_read=False))
    state.update(status='ARTIFACTS_COMPLETE');write(out/'pipeline_state.json',state)
    print('DEPLOYMENT_MEASUREMENTS_COMPLETE rows=208',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--seed',type=int);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.worker:worker(args.variant,args.seed,args.output,args.smoke)
    else:dispatch(args.output)
