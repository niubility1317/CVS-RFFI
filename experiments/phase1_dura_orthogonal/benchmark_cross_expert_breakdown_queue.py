"""Wait for GPU capacity and run a fixed source-only training timing breakdown."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.phase1_dura_orthogonal.gpu_capacity_queue import capacity


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args()
    if args.out.exists() or not args.cache.is_file():raise ValueError('Unique output and fixed source cache required')
    print('CROSS_BREAKDOWN_GPU_WAITING',args.gpu,flush=True)
    while True:
        if args.out.exists():raise FileExistsError('Another owner created benchmark output')
        g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        ready,free,count=capacity(g,a,args.gpu,minimum_mib=8192)
        if ready:
            print('CROSS_BREAKDOWN_GPU_SLOT',args.gpu,'free_mib',free,'other_apps',count,flush=True)
            break
        time.sleep(20)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    cmd=[sys.executable,'-X','utf8','-u','-m','experiments.phase1_dura_orthogonal.benchmark_cross_expert_breakdown',
        '--root',str(args.root),'--cache',str(args.cache),'--out',str(args.out),'--commit',args.commit]
    child=subprocess.Popen(cmd,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL)
    print('CROSS_BREAKDOWN_BENCHMARK_STARTED',child.pid,flush=True)
    if child.wait():raise RuntimeError('Owned source benchmark failed')
    print('CROSS_BREAKDOWN_QUEUE_COMPLETE',flush=True)


if __name__=='__main__':main()
