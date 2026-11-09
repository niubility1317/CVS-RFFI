"""Wait for both fixed source checkpoints, then run source-only comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiments.phase1_cumulant_rank.source_pair_probe import BASE,RANK
from experiments.phase1_joint_particle.runner import gpu_available


def wait_for_project_slot(root,gpu):
    """Honor total GPU capacity and avoid another same-project GPU worker."""
    while True:
        ready,free,count=gpu_available(gpu)
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid',
            '--format=csv,noheader'],text=True).splitlines()
        matches=[row.split(',')[1].strip() for row in rows if row.split(',')[0].strip()==str(gpu)]
        if len(matches)!=1:raise ValueError('Exactly one GPU identity required')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader'],text=True).splitlines()
        project_busy=False
        for row in apps:
            fields=[x.strip() for x in row.split(',')]
            if len(fields)!=2 or fields[0]!=matches[0]:continue
            if not fields[1].isdigit():
                project_busy=True
                continue
            proc=Path('/proc')/fields[1]
            try:
                command=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
            except (OSError,PermissionError):
                if proc.exists():project_busy=True
                continue
            if str(root) in command:project_busy=True
        if ready and not project_busy:
            print('CUMULANT_PAIR_GPU_READY',gpu,free,count,flush=True)
            return
        print('CUMULANT_PAIR_GPU_WAITING',gpu,free,count,'project_busy',project_busy,flush=True)
        time.sleep(20)


def execute(root,cache,out,commit,gpu):
    if (Path.cwd()/'release_commit.txt').read_text(encoding='ascii').strip()!=commit:
        raise ValueError('Registered release commit required')
    controller=root/'runs'/(out.name+'-controller')
    if controller.exists() or out.exists():raise FileExistsError('New diagnostic output root required')
    controller.mkdir(parents=True)
    def status(state,**kwargs):
        tmp=controller/'status.tmp';tmp.write_text(json.dumps(dict(state=state,**kwargs)),encoding='utf8')
        os.replace(tmp,controller/'status.json')
    try:
        while True:
            missing=[]
            for source in (BASE,RANK):
                folder=root/'runs'/source
                controller_status=root/'runs'/(source+'-controller')/'status.json'
                if controller_status.exists():
                    parent=json.loads(controller_status.read_text(encoding='utf8'))
                    if parent.get('state') in ('FAILED','STOPPED'):
                        raise RuntimeError('Source training not completed: '+source)
                if not (folder/'final.pt').is_file() or not (folder/'source_summary.json').is_file():missing.append(source)
            if not missing:break
            status('WAITING_SOURCE_CHECKPOINTS',pid=os.getpid(),missing=missing)
            print('CUMULANT_PAIR_WAITING',','.join(missing),flush=True)
            time.sleep(30)
        status('WAITING_GPU',pid=os.getpid());wait_for_project_slot(root,gpu)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        command=[sys.executable,'-X','utf8','-u','-m','experiments.phase1_cumulant_rank.source_pair_probe',
                 '--root',str(root),'--cache',str(cache),'--out',str(out),'--commit',commit]
        with (controller/'source_probe.log').open('x',encoding='utf8') as log:
            process=subprocess.Popen(command,cwd=Path.cwd(),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status('RUNNING_SOURCE_DIAGNOSTIC',pid=process.pid)
            if process.wait()!=0:raise RuntimeError('Source diagnostic exited '+str(process.returncode))
        status('SOURCE_DIAGNOSTIC_COMPLETE')
    except BaseException as e:
        (controller/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc())),encoding='utf8')
        status('FAILED',error=repr(e));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args();execute(a.root,a.cache,a.out,a.commit,a.gpu)

if __name__=='__main__':main()
