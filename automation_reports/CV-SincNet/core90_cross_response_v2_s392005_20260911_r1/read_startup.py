"""Read-only short SSH snapshots and independent owned-process verification."""
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
RELEASE = PROJECT + '/releases/' + (sys.argv[3] if len(sys.argv)>3 else 'core90_cross_response_v2_392005_e5bffaa3')
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
RUN = sys.argv[2] if len(sys.argv)>2 else HERE.name
SSH = ['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607']
command = f'{PYTHON} {RELEASE}/code/scripts/inspect_core90_cross_response_experiment.py --project-root {PROJECT} --run-id {RUN}'
raw = subprocess.check_output(SSH + [command], timeout=45)
data = json.loads(raw)
dest = HERE / ('inspection_' + sys.argv[1] + '.json')
dest.write_bytes(raw)
state = data.get('pipeline', {})
processes = {p['pid']:p for p in data['processes']}
gpu_map = {int(line.split(',')[0]):line.split(',')[1].strip() for line in data['compute_processes'] if line}
rows = []
for job in state.get('jobs', []):
    proc = processes.get(job.get('pid'))
    checks = dict(pid_present=bool(proc), argv=bool(proc and proc['argv']==job['argv']),
        cwd=bool(proc and proc['cwd']==RELEASE+'/code'),
        ppid=bool(proc and proc['ppid']==state['dispatcher_pid']),
        cuda=bool(proc and proc['cuda_visible_devices']==job.get('gpu')),
        registered=gpu_map.get(job.get('pid'))==job.get('gpu'))
    log = next((r for r in data['logs'] if r['path']==job['log']), {})
    metric = next((r for r in data['metrics'] if '/'+job['variant']+'/' in r['path']), {})
    rows.append(dict(variant=job['variant'],status=job['status'],pid=job.get('pid'),gpu=job.get('gpu'),
        checks=checks, log_bytes=log.get('bytes'), epochs=metric.get('epochs_logged'),
        init_scratch=any('init=scratch' in line for line in log.get('initialization_and_progress', []))))
summary = dict(status=state.get('status'), dispatcher_pid=state.get('dispatcher_pid'),
    dispatcher_error=state.get('dispatcher_error'), rows=rows,
    verified_running=bool(rows) and all(all(r['checks'].values()) and r['log_bytes'] for r in rows))
(HERE / ('startup_' + sys.argv[1] + '.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
