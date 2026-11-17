"""Stop only the verified priority dispatcher; never signal its workers."""
import json
import os
import signal
import time
from pathlib import Path


def main():
    project=Path('/home/szu2070436088/2510044040/CV-SincNet')
    name='20261117-phase1-pamf-repl-priority-clean-m20-r01'
    root=project/'runs'/name;release=project/'releases'/(name+'-release')
    state=json.loads((root/'pipeline_state.json').read_text());pid=state['pid'];proc=Path('/proc')/str(pid)
    argv=proc.joinpath('cmdline').read_bytes().decode().split('\0')
    if (state['status']!='RUNNING' or state.get('smoke',{}).get('status')!='PASS'
            or os.readlink(proc/'cwd')!=str(release)
            or 'experiments.pamf_repl_priority.runner' not in argv or 'dispatch' not in argv
            or str(release/'runtime.json') not in argv):
        raise ValueError('Exact healthy priority dispatcher required')
    os.kill(pid,signal.SIGSTOP);terminated=False
    try:
        for _ in range(30):
            if proc.joinpath('stat').read_text().split(') ',1)[1][0]=='T':break
            time.sleep(.1)
        else:raise RuntimeError('Dispatcher did not stop for safe snapshot')
        state=json.loads((root/'pipeline_state.json').read_text())
        if state['pid']!=pid or state['status']!='RUNNING':raise ValueError('Dispatcher state changed')
        children_text=proc.joinpath('task',str(pid),'children').read_text().strip()
        children={int(x) for x in children_text.split()} if children_text else set()
        recorded={j['pid'] for j in state['jobs'].values() if j['status']=='RUNNING'}
        if not children.issubset(recorded):raise ValueError('Unrecorded child; resume original owner')
        for key,item in state['jobs'].items():
            if item['status']=='QUEUED' and (root/'logs'/(key+'.log')).exists():
                raise ValueError('Queued job already launched; resume original owner')
        os.kill(pid,signal.SIGTERM);terminated=True
    finally:
        os.kill(pid,signal.SIGCONT)
    if terminated:
        for _ in range(100):
            if not proc.exists():break
            stat=proc.joinpath('stat').read_text().split(') ',1)[1][0]
            if stat=='Z':break
            time.sleep(.1)
        else:raise RuntimeError('Dispatcher did not terminate; do not start takeover')
    print(json.dumps(dict(status='RETIRED',pid=pid,recorded_workers=sorted(recorded),
                          signal_scope='dispatcher_only',original_run=name)))


if __name__=='__main__':main()
