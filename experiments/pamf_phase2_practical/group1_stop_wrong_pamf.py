"""Stop only the superseded Group1 PAMF recipe after a user correction."""
import argparse
import os
from pathlib import Path
import signal
import time
from .io import read,write,require
from .external_takeover import process


def stop(rt,reason):
    root=Path(rt['run_root']);state=read(root/'pipeline_state.json')
    parent=process(state['controller_pid'])
    require(parent and parent['uid']==os.getuid() and parent['cwd']==rt['release_root']
        and rt['runtime_file'] in parent['argv']
        and 'experiments.pamf_phase2_practical.group1_recovery' in parent['argv']
        and '--job' not in parent['argv'],'Current Group1 CPU controller mismatch')
    marker=root/'PAMF_METHOD_CORRECTION.json'
    require(not marker.exists(),'Reconcile existing stop instead of repeating')
    os.kill(parent['pid'],signal.SIGSTOP);time.sleep(.2)
    stopped=[];retained=[]
    try:
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            item=process(int(entry.name))
            if not item or item['uid']!=os.getuid():continue
            argv=item['argv']
            if rt['runtime_file'] not in argv or '--job' not in argv:continue
            require(item['cwd']==rt['release_root'] and
                'experiments.pamf_phase2_practical.group1_dispatch' in argv,'Unexpected correction worker')
            key=argv[argv.index('--job')+1]
            require('pamf_pl_milkr' in key,'Refuse to stop a different method')
            os.kill(item['pid'],signal.SIGTERM);stopped.append(dict(item,job=key))
        for item in state['active'].values():
            if item['profile']!='pamf_pl_milkr':
                observed=process(item['pid'])
                if observed:retained.append(observed)
        deadline=time.monotonic()+5
        while any(process(p['pid']) for p in stopped) and time.monotonic()<deadline:time.sleep(.1)
        require(not any(process(p['pid']) for p in stopped),'PAMF worker still exiting; preserve suspended controller')
        os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT)
        deadline=time.monotonic()+5
        while process(parent['pid']) and time.monotonic()<deadline:time.sleep(.1)
        require(process(parent['pid']) is None,'Old CPU controller did not exit')
        record=dict(status='VERIFIED',reason=reason,observed_at=time.time(),
            stopped_PAMF_workers=stopped,retained_external_workers=retained,
            stopped_controller=parent,controller_exited=True,outputs_preserved=True,
            incorrect_recipe='ordinary_midpoint_PL_is_not_Tri_SIG_TIME_SSL',
            next_controller_pending=True)
        write(marker,record);return record
    except BaseException:
        # Keep the verified wrong-recipe dispatcher suspended, preventing more work.
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--reason',required=True)
    a=p.parse_args();value=stop(read(a.runtime),a.reason)
    import json
    print(json.dumps(value))
