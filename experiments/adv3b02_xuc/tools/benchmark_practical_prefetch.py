"""Test bounded overlap with process workers, avoiding a Python-GIL producer."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from leo_practical import Config
from leo_practical.execution import FAST
from cvsrffi.practical_parallel import parallel_batch,shutdown
from cvsrffi.practical_adapter import _output_tensor
from cvsrffi.bounded_prefetch import prefetch_map
from scripts.train_rc4_practical import native,build_args
from contextlib import closing,nullcontext

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    device=torch.device('cuda:0')
    args=build_args(ROOT/'configs/rc4_practical_full_noeq_20260918.json','unused','unused','unused','unused','unused','synthetic')
    torch.manual_seed(23)
    ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},args,input_len=256,num_domains=15)
    model=native.build_baseline_model(native._apply_model_cli_args(ma,args),device).eval()
    x=np.random.default_rng(41).normal(size=(128,2,256))
    kw=dict(sample_ids=[f'synthetic{i}' for i in range(len(x))],session_ids=['rx']*len(x),realization_namespace='target_fixed_v3')
    rows=[]
    for route in ('full','residual'):
        cfg=Config(fs_hz=25e6,fc_hz=2.462e9,scenario='practical_mid',processing_route=route)
        def produce(index):return parallel_batch(x,cfg,workers=2,execution=FAST,seed=392005,**kw)[0]
        def pipeline(prefetch):
            sequence=prefetch_map(produce,range(8)) if prefetch else map(produce,range(8))
            with closing(sequence) if prefetch else nullcontext(sequence):
                return [model(_output_tensor(y,device,torch.float32,True),y_tx=None,return_aux=True)['tx_logits'].cpu() for y in sequence]
        def measure(enabled):
            pipeline(enabled);times=[]
            for _ in range(5):
                torch.cuda.synchronize();t=time.perf_counter();z=pipeline(enabled);torch.cuda.synchronize();times.append(time.perf_counter()-t)
            return dict(median_s=statistics.median(times),samples_s=times),z
        with torch.no_grad():
            before,old=measure(False);after,new=measure(True)
        assert all(torch.equal(u,v) for u,v in zip(old,new))
        rows.append(dict(route=route,baseline=before,optimized=after,speedup=before['median_s']/after['median_s'],exact=True))
    shutdown();out.write_text(json.dumps(dict(scope='fixed evaluation pipeline; two CPU workers, eight batches, five repeats',cases=rows),indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
