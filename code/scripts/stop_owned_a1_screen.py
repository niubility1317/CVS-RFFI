"""Replace only this task's original screen after the periodic-test request."""
import json
import os
from pathlib import Path
import signal
import time


def main():
    project=Path('/home/szu2070436088/2510044040/CV-SincNet')
    root=project/'runs/a1_mechanism_screen_s392005_20260910_r2'
    state=json.loads((root/'pipeline_state.json').read_text())
    pid=int(state['pid'])
    proc=Path('/proc')/str(pid)
    assert proc.exists() and os.getpgid(pid)==pid and os.getsid(pid)==pid
    release=str(project/'releases/a1_mechanism_screen_75c07e46_r2')
    assert os.readlink(proc/'cwd')==release
    assert root.name in (proc/'cmdline').read_bytes().decode()
    processes={}
    for item in Path('/proc').iterdir():
        if not item.name.isdigit(): continue
        try:
            number=int(item.name)
            if os.getpgid(number)!=pid: continue
            status=(item/'status').read_text()
            parent=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
            processes[number]={'parent':parent,'cwd':os.readlink(item/'cwd'),
                'argv':(item/'cmdline').read_bytes().decode().split('\0')[:-1]}
        except (FileNotFoundError,ProcessLookupError): continue
    assert pid in processes
    for number,item in processes.items():
        assert item['cwd']==release
        if number==pid: continue
        ancestor=item['parent'];seen=set()
        while ancestor!=pid:
            assert ancestor in processes and ancestor not in seen
            seen.add(ancestor);ancestor=processes[ancestor]['parent']
    for row in state['rows'].values():
        if row['status']=='RUNNING':
            assert row['pid'] in processes
            assert str(root) in ' '.join(processes[row['pid']]['argv'])
    evidence=root/'user_periodic_reconfiguration_stop.json'
    with evidence.open('x',encoding='utf-8') as output:
        json.dump({'reason':'USER_REQUEST_PERIODIC_TEST_RECONFIGURATION','processes':processes},output,indent=2)
    os.killpg(pid,signal.SIGTERM)
    deadline=time.monotonic()+15
    remaining=[]
    while time.monotonic()<deadline:
        remaining=[]
        for number in processes:
            try:
                status=(Path('/proc')/str(number)/'status').read_text()
                if '\nState:\tZ' not in status: remaining.append(number)
            except FileNotFoundError: pass
        if not remaining: break
        time.sleep(.5)
    result={'status':'VERIFIED' if not remaining else 'UNKNOWN','stopped_pids':list(processes),'remaining':remaining}
    (root/'user_periodic_reconfiguration_stop_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if remaining: raise RuntimeError('Owned process termination not yet verified')


if __name__=='__main__':main()
