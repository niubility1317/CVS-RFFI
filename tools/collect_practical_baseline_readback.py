"""Read state/process/log evidence through short SSH calls, without mutation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prefix = ['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    base = '/home/szu2070436088/2510044040/CV-SincNet'
    run = '20260927-phase1-baselines-practical-manysig-m5-r01'
    def read(cmd):
        result = subprocess.run([*prefix,cmd],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout
    state = json.loads(read(f'cat {base}/runs/{run}/state.json'))
    active = [r['pid'] for r in state.values() if r['status']=='RUNNING']
    evidence = dict(time=datetime.now(timezone.utc).isoformat(), state=state,
        counts=dict(Counter(r['status'] for r in state.values())),
        processes=read('ps -p '+','.join(str(i) for i in active)+' -o pid,ppid,etime,args') if active else '',
        cwd=read('readlink '+' '.join(f'/proc/{i}/cwd' for i in active)) if active else '',
        gpu=read('nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader'),
        log_tail=read(f'tail -n 3 {base}/logs/{run}/*.log'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:
        json.dump(evidence,f,indent=2)
    print(json.dumps(dict(output=str(a.output),counts=evidence['counts'],active_pids=active)))


if __name__ == '__main__':
    main()
