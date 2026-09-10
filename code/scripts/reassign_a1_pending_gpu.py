"""Transfer only this run's dispatcher; preserve its live training processes."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

RUN = 'a1_extended_s392005_20260910_r1'
ROW = 'R3_CLEAN_RX_E400'
PROJECT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RELEASE = PROJECT/'releases/a1_extended_1b3c0002'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.queue-writing')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8'); temp.replace(path)


def process_info(pid):
    proc = Path('/proc')/str(pid)
    try:
        status = (proc/'stat').read_text().split(') ',1)[1].split()[0]
        return {'status':status, 'argv':(proc/'cmdline').read_bytes().decode().split('\0')[:-1],
                'cwd':os.readlink(proc/'cwd') if status!='Z' else ''}
    except FileNotFoundError:
        return None


def alive(pid):
    info=process_info(pid)
    return bool(info and info['status']!='Z')


def require_owner(pid, expected_script, run_flag):
    info=process_info(pid)
    if not info or info['status']=='Z' or info['cwd']!=str(RELEASE):
        raise RuntimeError('Process absent or CWD ownership mismatch')
    argv=info['argv']
    if str(RELEASE/'code/scripts'/expected_script) not in argv:
        raise RuntimeError('Dispatcher script ownership mismatch')
    if run_flag not in argv or argv[argv.index(run_flag)+1]!=RUN:
        raise RuntimeError('Run ownership mismatch')


def changed_matrix(matrix, state, gpu):
    if state['waiting_rows']!=[ROW] or ROW in state['rows']:
        raise RuntimeError('Requested row is no longer exclusively pending')
    result=copy.deepcopy(matrix)
    target=next(r for r in result['rows'] if r['id']==ROW)
    target['gpu']=gpu
    return result


def load_runner():
    sys.path[:0]=[str(RELEASE/'code/scripts'),str(RELEASE/'code'),str(RELEASE)]
    import run_a1_fast_v2 as runner
    return runner


def manage(context_path):
    runner=load_runner(); context=read(context_path)
    root=PROJECT/'runs'/RUN; logs=PROJECT/'logs'/RUN
    ready=context_path.with_suffix('.ready'); active=context_path.with_suffix('.activate')
    # Resolve every dependency and prepare configuration before old owner exits.
    state=context['state']; matrix=context['matrix']; children={}
    commands={r['id']:runner.v2_command(matrix,project_root=PROJECT,run_root=root,row=r) for r in matrix['rows']}
    pending=[r for r in matrix['rows'] if r['id']==ROW]
    ready.write_text(str(os.getpid()),encoding='utf-8')
    deadline=time.monotonic()+60
    while not active.exists():
        if time.monotonic()>deadline:raise TimeoutError('Handoff activation not received')
        time.sleep(.2)
    state.update(pid=os.getpid(),previous_dispatcher_pid=context['old_pid'],queue_handoff=str(context_path))
    write(root/'effective_matrix.json',matrix)
    while pending or any(r['status']=='RUNNING' for r in state['rows'].values()):
        for row in list(pending):
            if len(runner.gpu_compute_pids(row['gpu']))>=2:continue
            folder=root/row['id'];folder.mkdir()  # Never overwrite a row, including partial output.
            info={'gpu':row['gpu'],'cwd':str(RELEASE),'argv':commands[row['id']],'created':time.time()}
            write(folder/'launch_config.json',info)
            stream=(logs/(row['id']+'.train.log')).open('x',encoding='utf-8')
            proc=subprocess.Popen(commands[row['id']],cwd=RELEASE,env=runner.environment(row['gpu']),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            children[row['id']]=(proc,stream)
            info['pid']=proc.pid;write(folder/'process.json',info)
            state['rows'][row['id']]={'status':'RUNNING','pid':proc.pid,'gpu':row['gpu']}
            pending.remove(row)
        state['waiting_rows']=[r['id'] for r in pending];state['status']='RUNNING'
        write(root/'pipeline_state.json',state)
        for row in matrix['rows']:
            name=row['id']; item=state['rows'].get(name)
            if not item or item['status']!='RUNNING':continue
            if name in children:
                proc,stream=children[name]; code=proc.poll()
                if code is None:continue
                stream.close();item['exit']=code
            else:
                if alive(item['pid']):continue
                # Original children are reparented, so exit codes cannot be invented.
                code=None;item['exit']='UNKNOWN_REPARENTED_PROCESS'
            if code is not None and code!=0:
                item['status']='TRAIN_FAILED';continue
            try:
                item['status']=runner.complete_row(matrix,row,project=PROJECT,root=root,logs=logs)
            except Exception as exc:
                item.update(status='ARTIFACT_VERIFICATION_FAILED',error=repr(exc))
            write(root/'pipeline_state.json',state)
        if pending or any(r['status']=='RUNNING' for r in state['rows'].values()):time.sleep(30)
    state['status']='AWAITING_ARTIFACT_ANALYSIS' if all(
        r['status']=='EXPLORATORY_SCORED_PENDING_ANALYSIS' for r in state['rows'].values()) else 'FAILED'
    write(root/'pipeline_state.json',state)


def handoff(gpu):
    runner=load_runner();root=PROJECT/'runs'/RUN;logs=PROJECT/'logs'/RUN
    context_path=root/'gpu_reassignment_20260910_r1.json'
    if context_path.exists():raise FileExistsError('Handoff already exists; reconcile instead of retry')
    state=read(root/'pipeline_state.json'); old=state['pid']
    require_owner(old,'run_a1_extended_budgets.py','--run-id')
    if len(runner.gpu_compute_pids(gpu))>=2:raise RuntimeError('Selected GPU no longer has a free slot')
    changed_matrix(read(root/'effective_matrix.json'),state,gpu)
    os.kill(old,signal.SIGSTOP)
    killed=False; child=None
    try:
        deadline=time.monotonic()+5
        while process_info(old)['status']!='T':
            if time.monotonic()>deadline:raise TimeoutError('Dispatcher pause not observed')
            time.sleep(.05)
        # Reconcile queue state after the dispatcher is paused, closing the launch race.
        state=read(root/'pipeline_state.json');matrix=changed_matrix(read(root/'effective_matrix.json'),state,gpu)
        if (root/ROW).exists():raise FileExistsError('Pending row already has output')
        for name,item in state['rows'].items():
            info=process_info(item['pid'])
            if item['status']!='RUNNING' or not info or info['cwd']!=str(RELEASE):
                raise RuntimeError('Existing worker no longer matches handoff scope')
            argv=info['argv']
            if '--output_dir' not in argv or argv[argv.index('--output_dir')+1]!=str(root/name):
                raise RuntimeError('Worker output ownership mismatch')
        context={'old_pid':old,'state':state,'matrix':matrix,'created':time.time(),
                 'reason':'user requested earliest available GPU; GPU7 already has capacity'}
        with context_path.open('x',encoding='utf-8') as stream:json.dump(context,stream,indent=2)
        output=(logs/'gpu_reassignment_dispatcher.log').open('x',encoding='utf-8')
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--manage',str(context_path)],
            cwd=RELEASE,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        output.close();deadline=time.monotonic()+25
        while not context_path.with_suffix('.ready').exists():
            if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Replacement dispatcher not ready')
            time.sleep(.2)
        os.kill(old,signal.SIGTERM);os.kill(old,signal.SIGCONT)
        deadline=time.monotonic()+5
        while alive(old):
            if time.monotonic()>deadline:raise TimeoutError('Old dispatcher did not exit')
            time.sleep(.1)
        killed=True
        context_path.with_suffix('.activate').write_text('activate',encoding='utf-8')
        print(json.dumps({'status':'HANDOFF_DISPATCHED','old_pid':old,'new_pid':child.pid,'gpu':gpu}))
    finally:
        if not killed:
            if child is not None and child.poll() is None:child.terminate()
            if alive(old):os.kill(old,signal.SIGCONT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=range(8))
    parser.add_argument('--manage',type=Path);args=parser.parse_args()
    if args.manage:manage(args.manage)
    elif args.gpu is not None:handoff(args.gpu)
    else:parser.error('Specify --gpu or --manage')
