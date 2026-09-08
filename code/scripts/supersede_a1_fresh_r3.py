"""Stop only the superseded fresh dependency chain after checking ownership."""
import json
import os
from pathlib import Path
import signal
import time

PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='a1_fast_matched_core90_s392005_20260908_r3'
RELEASE=PROJECT/'releases/a1_fast_matched_ab8a5778'

def main():
    path=PROJECT/'runs'/RUN/'pipeline_state.json'
    state=json.loads(path.read_text())
    if state.get('status')=='SUPERSEDED_BY_USER':
        print(json.dumps(state)); return
    if state['status']!='CORE90_RUNNING' or state['rows']:
        raise RuntimeError('Chain changed; reconcile before cancelling')
    pids=[state['pid'],state['core90_pid']]
    for i,pid in enumerate(pids):
        proc=Path('/proc')/str(pid)
        cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if RUN not in cmd or str(RELEASE) not in cmd: raise RuntimeError('PID ownership mismatch')
        if i==1:
            if (proc/'cwd').resolve()!=RELEASE: raise RuntimeError('CWD mismatch')
            status=(proc/'status').read_text()
            ppid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
            if ppid!=pids[0]: raise RuntimeError('Parent mismatch')
    # Prevent queued old A1 launches first, then terminate its current child.
    os.kill(pids[0],signal.SIGTERM)
    os.kill(pids[1],signal.SIGTERM)
    for _ in range(20):
        if all(not (Path('/proc')/str(pid)).exists() for pid in pids): break
        time.sleep(0.5)
    else: raise RuntimeError('Termination not confirmed; no automatic force kill')
    state.update(status='SUPERSEDED_BY_USER',reason='User selected existing ADV3B02 E200, seed392005',stopped_pids=pids)
    path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(state))

if __name__=='__main__': main()
