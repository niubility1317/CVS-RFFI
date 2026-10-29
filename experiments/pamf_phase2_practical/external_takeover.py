"""Move only this chat's core dispatcher under the shared Phase2 scheduler.

Running model workers keep their original code, command, working directory,
outputs and GPU. Only the exact configured old CPU dispatcher is retired.
"""
import os
from pathlib import Path
import signal
import time
from .io import read,write,require


def process(pid):
    p=Path('/proc')/str(pid)
    try:
        state=(p/'stat').read_text().split(') ',1)[1].split()[0]
        if state=='Z':return None
        return dict(pid=int(pid),uid=p.stat().st_uid,cwd=os.readlink(p/'cwd'),
                    argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'))
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def adopt(rt,output):
    old=read(rt['shared_core_runtime'])
    require(old['run_id']=='20261029-phase2-milkr-threepool-m4-r01','Unexpected core owner/run')
    root=Path(old['run_root']);state=read(root/'pipeline_state.json');pid=state['controller_pid']
    parent=process(pid)
    if parent:
        require(parent['uid']==os.getuid() and parent['cwd']==old['release_root'] and old['runtime_file'] in parent['argv']
                and 'experiments.pamf_phase2_practical.episodic_dispatch' in parent['argv'] and '--job' not in parent['argv'],
                'Old dispatcher identity does not match the authorized core run')
        os.kill(pid,signal.SIGSTOP)
        # Let an already-forked worker reach exec before enumerating commands.
        time.sleep(.2)
    workers={}
    try:
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            item=process(int(p.name))
            if not item or item['uid']!=os.getuid() or item['cwd']!=old['release_root']:continue
            argv=item['argv']
            if old['runtime_file'] not in argv or 'experiments.pamf_phase2_practical.episodic_dispatch' not in argv or '--job' not in argv:continue
            key=argv[argv.index('--job')+1]
            env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
            gpu=env.get('CUDA_VISIBLE_DEVICES','')
            workers[key]=dict(item,gpu=int(gpu) if gpu.isdigit() else None,kind='core',profile='PAMF_core',
                              original_id=key,log=str(root/'logs'/(key+'.log')),adopted=True)
        if parent:
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
            deadline=time.monotonic()+5
            while process(pid) and time.monotonic()<deadline:time.sleep(.1)
            require(process(pid) is None,'Old dispatcher did not exit; no second owner may start')
    finally:
        if parent and process(pid):os.kill(pid,signal.SIGCONT)
    write(output,dict(status='VERIFIED',old_run_id=old['run_id'],old_dispatcher_pid=pid,
        old_dispatcher_exited=process(pid) is None,model_workers_restarted=0,
        adopted_workers=list(workers.values()),scientific_config_unchanged=True,output_paths_unchanged=True))
    return old,workers


def resources(rt):
    from .runner import occupancy
    result=occupancy();runtime_paths={rt['runtime_file'],rt['shared_core_runtime'],*rt.get('phase2_runtime_files',[])}
    for value in result.values():value['phase2_pids']=set()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if not runtime_paths.intersection(argv):continue
            env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
            gpu=env.get('CUDA_VISIBLE_DEVICES','')
            if gpu.isdigit() and int(gpu) in result:result[int(gpu)]['phase2_pids'].add(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return result
