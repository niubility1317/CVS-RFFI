"""Resource-interrupted source training continuation; preserve other processes."""
import os
from pathlib import Path
import subprocess
import sys
import time

from . import bounded_projected_resource_trial as resource
from . import multiview_trial as trial
from .cost_probe import write
from .source_pair import read


def parent_training(root, original, arm, epoch):
    parent = root/'runs'/original.RUN
    stop = read(parent/'resource_stop.json')
    terminal = read(parent/('terminal-'+arm+'-train.json'))
    if (stop['stage'], stop['reason'], stop['cap'], stop['only_own_child_signaled'],
        stop['other_owner_processes_signaled'], terminal['pid'], terminal['exit_code']) != (
            'train', 'TOTAL_COMPUTE_PID_CAP_EXCEEDED', 3, True, False, stop['pid'], -15):
        raise ValueError('Original exact resource stop required')
    if any(parent.glob('submitted-'+arm+'-cross*.json')):
        raise ValueError('Original target prediction must not have started')
    for pid in (stop['pid'], read(parent/'launch.json')['pid']):
        p=Path('/proc')/str(pid)/'cmdline'
        if p.exists() and original.MODULE in p.read_bytes().replace(b'\0',b' ').decode(errors='replace'):
            raise ValueError('Original owner still running')
    folder=parent/'rows'/arm/'training'
    smoke=read(folder/'smoke.json')
    if smoke['status']!='PASS' or any(smoke[k] for k in ('TX_labels_read','V_reads','OE_reads','target_inputs')):
        raise ValueError('Original legal source smoke required')
    if not (folder/'recovery'/f'E{epoch:02d}.pt').is_file() or (folder/'complete.json').exists():
        raise ValueError('Incomplete original fixed epoch boundary required')
    return folder


def controller(args, run, arm, admit, cap):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Controller must not own GPU')
    out=args.root/'runs'/run;out.mkdir();(out/'logs').mkdir();row=out/'rows'/arm;row.mkdir(parents=True);(row/'prediction').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit,process_cap=cap))
    for stage in ('train','cross_rx','cross_both','score'):
        admission=None
        if stage!='score':
            while not admission:
                choices=admit()
                if choices:admission=choices[0]
                else:print('WAITING_FOR_ADMITTED_GPU_WITH_HEADROOM',stage,flush=True);time.sleep(30)
        command=trial.command(args,'score' if stage=='score' else ('train' if stage=='train' else 'predict'),arm,
                              stage if stage in ('cross_rx','cross_both') else None)
        with (out/'logs'/(arm+'-'+stage+'.log')).open('xb') as log:
            child=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if admission is None else admission['CUDA_ID']),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(out/('submitted-'+arm+'-'+stage+'.json'),dict(pid=child.pid,command=command,CWD=os.getcwd(),admission=admission))
        failures=0;reason=None;observed=None
        while child.poll() is None:
            if admission is not None:
                try:
                    observed=resource.compute_pids(admission['GPU'],admission['CUDA_ID']);failures=0
                    if child.pid in observed and len(observed)>cap:reason='TOTAL_COMPUTE_PID_CAP_EXCEEDED'
                except (RuntimeError,ValueError) as error:
                    failures+=1;print('GPU_COUNT_OBSERVATION_FAILED',str(error),flush=True)
                    if failures>=3:reason='GPU_COUNT_UNVERIFIABLE'
                if reason:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    write(out/'resource_stop.json',dict(stage=stage,pid=child.pid,GPU=admission['GPU'],reason=reason,
                        observed_compute_PIDs=sorted(observed) if observed is not None else None,cap=cap,
                        only_own_child_signaled=True,other_owner_processes_signaled=False))
                    break
            time.sleep(2)
        code=child.wait();write(out/('terminal-'+arm+'-'+stage+'.json'),dict(pid=child.pid,GPU=None if admission is None else admission['GPU'],exit_code=code))
        if code or reason:raise RuntimeError('Preserved stopped or failed stage: '+stage+' '+str(reason))
    if read(row/'results.json')['status']!='ANALYZED':raise ValueError('Missing original Main result')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',all_rows_trained30_and_main_seven_scored=True,
        default_DURA_changed=False,goal_complete=False))
