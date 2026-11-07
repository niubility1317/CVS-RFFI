"""Wait for the registered GPU and launch one confidence-gate controller."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.phase1_dura_orthogonal.gpu_capacity_queue import capacity


def owned_gpu_conflict(gpu_csv, app_csv, gpu_index):
    rows=[tuple(x.strip() for x in row.split(',')) for row in gpu_csv.splitlines() if row.strip()]
    match=[row for row in rows if len(row)==3 and row[0]==str(gpu_index)]
    if len(match)!=1:
        raise ValueError('Selected physical GPU must be present')
    uuid=match[0][1]
    for row in app_csv.splitlines():
        parts=[x.strip() for x in row.split(',')]
        if len(parts)!=2 or parts[0]!=uuid:
            continue
        cmd=Path('/proc')/parts[1]/'cmdline'
        if cmd.exists() and b'20261108-phase1-dura-context-wide-residual-source-s2026092701-r01' in cmd.read_bytes():
            return True
    return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--ground',required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    runs=args.root/'runs'
    paths=[runs/args.source,runs/(args.source+'-controller'),runs/args.ground]
    if any(x.exists() for x in paths) or not args.cache.is_file():
        raise ValueError('Unique output roots and source cache required')
    print('SIGNAL_GATE_GPU_WAITING',args.gpu,flush=True)
    while True:
        if any(x.exists() for x in paths):
            raise FileExistsError('Another owner created an output while queued')
        g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        ready,free,count=capacity(g,a,args.gpu)
        if ready and not owned_gpu_conflict(g,a,args.gpu):
            print('SIGNAL_GATE_GPU_SLOT',args.gpu,'free_mib',free,'other_apps',count,flush=True)
            break
        time.sleep(20)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    cmd=[sys.executable,'-X','utf8','-u','-m','experiments.phase1_dura_orthogonal.signal_gate_controller',
         '--root',str(args.root),'--source',args.source,'--ground',args.ground,'--cache',str(args.cache),'--commit',args.commit]
    child=subprocess.Popen(cmd,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL)
    print('SIGNAL_GATE_CONTROLLER_STARTED',child.pid,flush=True)
    code=child.wait()
    if code:raise RuntimeError('Owned controller failed: '+str(code))
    print('SIGNAL_GATE_CONTROLLER_COMPLETE',flush=True)


if __name__=='__main__':main()
