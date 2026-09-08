"""Checkpoint-free A1-V2 controlled matrix, one training process per GPU."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_a1_fast_matched_core90 import RELEASE, build_train_command, environment, write_json, verify_checkpoint
from run_a1_scratch_no_checkpoint import scratch_matrix
from run_a1_fast_selected_adv3b02 import evaluate, BASE_RUN


def v2_matrix():
    options=deepcopy(scratch_matrix()['core90_options'])
    options.update({'--use_a1_r3':'false','--base_candidate':'A1_V2_SOURCE_RANDOM',
        '--a1_ema_versioned_updates':'true','--daot_efficiency_mode':'identity_sequential',
        '--daot_batched_scale_readback':'true','--daot_skip_mean_metadata':'true'})
    rows=[]
    for name,gpu,fast,warm,coverage in (
        ('F0_FIXED_BASE',4,False,False,False),('F1_RUNTIME',5,True,False,False),
        ('F2_WARM_EMA',6,True,True,False),('F3_COVERAGE_KL',7,True,False,True)):
        rows.append({'id':name,'gpu':gpu,'options':{
            '--a1_runtime_fast':str(fast).lower(),'--a1_ema_startup_average':str(warm).lower(),
            '--a1_logit_coverage_weighting':str(coverage).lower()}})
    return {'seed':392005,'initialization':'random_no_checkpoint','core90_options':options,'rows':rows}


def v2_command(matrix,*,project_root,run_root,row):
    configured=deepcopy(matrix)
    configured['core90_options'].update(row['options'])
    return build_train_command(configured,project_root=project_root,run_root=run_root,row_id=row['id'],core90=True)


def gpu_compute_pids(gpu):
    mapping={int(line.split(',')[0]):line.split(',')[1].strip() for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()}
    if gpu not in mapping: raise ValueError(f'Unknown GPU {gpu}')
    return {line.split(',')[1].strip() for line in subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
        if ',' in line and line.split(',')[0].strip()==mapping[gpu]}


def predecessor_complete(matrix, project):
    predecessor = matrix.get('after_run')
    if not predecessor:
        return True
    if not predecessor.replace('_','').isalnum():
        raise ValueError('Invalid predecessor run ID')
    path = project/'runs'/predecessor/'pipeline_state.json'
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        # The predecessor's existing writer may be between truncate and write.
        return False
    terminal = {'SCORED_PENDING_ANALYSIS','EVAL_FAILED','TRAIN_FAILED'}
    return bool(state.get('rows')) and all(row.get('status') in terminal for row in state['rows'].values())


def main(matrix_factory=v2_matrix, check_script='check_a1_fast_v2.py'):
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--detach',action='store_true')
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum(): raise ValueError('Invalid run ID')
    matrix=matrix_factory(); project=args.project_root
    capacity=int(matrix.get('max_gpu_processes',1))
    if capacity not in (1,2): raise ValueError('GPU training capacity must be one or two')
    root=project/'runs'/args.run_id; logs=project/'logs'/args.run_id
    commands={row['id']:v2_command(matrix,project_root=project,run_root=root,row=row) for row in matrix['rows']}
    if args.dry_run:
        print(json.dumps(commands,indent=2)); return
    if root.exists() or logs.exists(): raise FileExistsError('Refusing existing run or log root')
    for path in (project/'Dataset_WigSig/ManySig.pkl',project/'runs'/BASE_RUN/'target_inputs/manifest.json',
                 project/'runs'/BASE_RUN/'target_truth/truth_sidecar.json'):
        if not path.is_file(): raise FileNotFoundError(path)
    if args.detach:
        argv=[sys.executable,'-u',str(Path(sys.argv[0]).resolve()),
              '--project-root',str(project),'--run-id',args.run_id]
        with (RELEASE/'dispatcher.log').open('x',encoding='utf-8') as output:
            process=subprocess.Popen(argv,cwd=RELEASE,stdin=subprocess.DEVNULL,
                stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(RELEASE/'dispatcher_process.json',{'pid':process.pid,'argv':argv,'cwd':str(RELEASE)})
        print(json.dumps({'status':'DISPATCHED','pid':process.pid}),flush=True)
        return
    root.mkdir(); logs.mkdir()
    state={'status':'GPU_CHECK','pid':os.getpid(),'release':str(RELEASE),'seed':392005,
           'checkpoint':None,'initialization':'random_no_checkpoint','rows':{},
           'after_run':matrix.get('after_run'),'max_gpu_processes':capacity,
           'waiting_rows':[r['id'] for r in matrix['rows']]}
    write_json(root/'pipeline_state.json',state); write_json(root/'effective_matrix.json',matrix)
    try:
        first_gpu=matrix['rows'][0]['gpu']
        while not predecessor_complete(matrix,project) or len(gpu_compute_pids(first_gpu))>=capacity:
            state['status']='WAITING_FOR_PREDECESSOR_OR_GPU'; write_json(root/'pipeline_state.json',state)
            time.sleep(30)
        state['status']='GPU_CHECK'; write_json(root/'pipeline_state.json',state)
        subprocess.run([sys.executable,str(RELEASE/'code/scripts'/check_script),
            '--device','cuda:0','--output',str(logs/'execution_check.json')],
            cwd=RELEASE,env=environment(first_gpu),check=True)
        pending=list(matrix['rows']); running={}; failed=False
        while pending or running:
            for row in list(pending):
                if len(gpu_compute_pids(row['gpu']))>=capacity: continue
                folder=root/row['id']; folder.mkdir()
                info={'gpu':row['gpu'],'cwd':str(RELEASE),'argv':commands[row['id']],'created':time.time()}
                write_json(folder/'launch_config.json',info)
                log=(logs/(row['id']+'.train.log')).open('x',encoding='utf-8')
                process=subprocess.Popen(commands[row['id']],cwd=RELEASE,env=environment(row['gpu']),
                                         stdout=log,stderr=subprocess.STDOUT)
                info['pid']=process.pid; write_json(folder/'process.json',info)
                state['rows'][row['id']]={'status':'RUNNING','pid':process.pid,'gpu':row['gpu']}
                running[row['id']]=(row,process,log); pending.remove(row)
                print(json.dumps({'state':'RUNNING','row':row['id'],'pid':process.pid,'gpu':row['gpu']}),flush=True)
            state['status']='RUNNING'; state['waiting_rows']=[r['id'] for r in pending]
            write_json(root/'pipeline_state.json',state)
            for name,(row,process,log) in list(running.items()):
                code=process.poll()
                if code is None: continue
                log.close(); state['rows'][name]['exit']=code
                if code:
                    failed=True; state['rows'][name]['status']='TRAIN_FAILED'
                else:
                    try:
                        # First model-weight load occurs only after this row's own E200 training.
                        checkpoint=verify_checkpoint(root/name/'final_ssdg.pth')
                        saved=checkpoint['args']
                        if not saved.get('from_scratch') or saved.get('baseline_ckpt') or saved.get('teacher_ckpt'):
                            raise ValueError('Unexpected checkpoint inheritance')
                        del checkpoint
                        state['rows'][name]['status']='EVALUATING'; write_json(root/'pipeline_state.json',state)
                        evaluate(row,project=project,run_root=root,log_root=logs)
                        state['rows'][name]['status']='SCORED_PENDING_ANALYSIS'
                    except Exception as error:
                        failed=True; state['rows'][name].update(status='EVAL_FAILED',error=repr(error))
                del running[name]; write_json(root/'pipeline_state.json',state)
            if pending or running: time.sleep(30)
        state['status']='FAILED' if failed else 'AWAITING_ARTIFACT_ANALYSIS'
        write_json(root/'pipeline_state.json',state)
    except Exception as error:
        state.update(status='FAILED',error=repr(error)); write_json(root/'pipeline_state.json',state); raise


if __name__=='__main__': main()
