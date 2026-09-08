"""Train a fresh matched CORE90, then the frozen A1 execution comparison.

All subprocesses use argv lists. No historical weight path is accepted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RELEASE = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def build_train_command(matrix, *, project_root, run_root, row_id, core90=False, row=None):
    options = dict(matrix['core90_options' if core90 else 'a1_options'])
    output = run_root / row_id
    options.update({'--output_dir':str(output),'--run_id':run_root.name,'--candidate_id':row_id,
                    '--phase2_export_path':str(output/'phase2_zid_prototypes.pt')})
    if core90:
        if options.get('--from_scratch') != 'true' or any(k in options for k in ('--baseline_ckpt','--teacher_ckpt')):
            raise ValueError('CORE90 must be fresh, without historical weights')
    else:
        checkpoint = run_root/'CORE90_MATCHED_FRESH'/'final_ssdg.pth'
        options.update({'--from_scratch':'false','--a1_scratch_only':'false',
                        '--baseline_ckpt':str(checkpoint),'--teacher_ckpt':str(checkpoint)})
        for key in ['daot_efficiency_mode','daot_batched_scale_readback','daot_skip_mean_metadata']:
            options['--'+key]=row[key]
    command=[sys.executable,'-u',str(RELEASE/'code/SSDG/train_ssdg.py')]
    for key,value in options.items():
        command.append(key)
        if value is not None:
            command.append(str(value).format(project_root=str(project_root)))
    return command


def environment(gpu):
    result=dict(os.environ)
    result['PYTHONPATH']=str(RELEASE/'code')+os.pathsep+str(RELEASE)
    result['CUDA_VISIBLE_DEVICES']=str(gpu)
    result['PYTHONUNBUFFERED']='1'
    return result


def wait_for_slot(gpu):
    while True:
        gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
        mapping={int(line.split(',')[0]):line.split(',')[1].strip() for line in gpu_rows.splitlines()}
        compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        pids={line.split(',')[1].strip() for line in compute.splitlines() if ',' in line and line.split(',')[0].strip()==mapping[gpu]}
        if len(pids)<2: return
        print(json.dumps({'state':'WAITING_FOR_GPU','gpu':gpu,'compute_pids':sorted(pids)}),flush=True)
        time.sleep(30)


def verify_checkpoint(path, expected_epoch=200):
    import torch
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if int(checkpoint.get('epoch',-1)) != expected_epoch or not isinstance(checkpoint.get('model'),dict):
        raise RuntimeError(f'Incomplete E{expected_epoch} checkpoint: {path}')
    return checkpoint


def launch_train(command, *, row_id, gpu, run_root, log_root):
    wait_for_slot(gpu)
    output=run_root/row_id
    output.mkdir(exist_ok=False)
    write_json(output/'launch_config.json',{'argv':command,'gpu':gpu,'cwd':str(RELEASE),'created':time.time()})
    log=(log_root/(row_id+'.train.log')).open('x',encoding='utf-8')
    process=subprocess.Popen(command,cwd=RELEASE,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
    write_json(output/'process.json',{'pid':process.pid,'gpu':gpu,'cwd':str(RELEASE),'argv':command})
    print(json.dumps({'state':'RUNNING','row':row_id,'pid':process.pid,'gpu':gpu}),flush=True)
    return process,log


def evaluate(row_id, *, run_root, log_root, gpu, seed):
    output=run_root/row_id
    verify_checkpoint(output/'final_ssdg.pth')
    command=[sys.executable,'-u',str(RELEASE/'code/scripts/eval_ssdg_sat_per_rx.py'),
             '--ckpt',str(output/'final_ssdg.pth'),'--output_json',str(output/'metrics_joint.json'),
             '--eval_on','unseen_rx','--scenarios','leo_clear_weak,leo_low_elev_weak,leo_rain_weak',
             '--device','cuda:0','--max_batches','-1','--eval_batch_size','256','--sat_seed',str(seed),
             '--strict_reconstruction','--group_loader','test_all_day_unseen_rx']
    with (log_root/(row_id+'.eval.log')).open('x',encoding='utf-8') as log:
        subprocess.run(command,cwd=RELEASE,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((output/'metrics_joint.json').read_text(encoding='utf-8'))
    write_json(output/'evaluation_completed.json',{'status':'EVALUATION_COMMAND_COMPLETE','metrics':str(output/'metrics_joint.json'),'top_level_keys':list(result)})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'.','..'}:
        raise ValueError('run-id must be a simple directory name')
    matrix=json.loads((RELEASE/'configs/a1_fast_matched_core90_20260908.json').read_text(encoding='utf-8'))
    run_root=args.project_root/'runs'/args.run_id
    log_root=args.project_root/'logs'/args.run_id
    core_command=build_train_command(matrix,project_root=args.project_root,run_root=run_root,row_id='CORE90_MATCHED_FRESH',core90=True)
    commands={row['id']:build_train_command(matrix,project_root=args.project_root,run_root=run_root,row_id=row['id'],row=row) for row in matrix['rows']}
    if args.dry_run:
        print(json.dumps({'core90':core_command,'a1':commands},indent=2)); return 0
    if not (args.project_root/'Dataset_WigSig/ManySig.pkl').is_file():
        raise FileNotFoundError('ManySig input missing')
    if run_root.exists() or log_root.exists():
        raise FileExistsError('Refusing existing run/log root')
    run_root.mkdir(parents=True,exist_ok=False); log_root.mkdir(parents=True,exist_ok=False)
    state={'status':'SMOKE','run_id':args.run_id,'pid':os.getpid(),'release':str(RELEASE),'rows':{}}
    write_json(run_root/'pipeline_state.json',state)
    try:
        # Fresh initialization smoke obeys the no-historical-checkpoint instruction.
        wait_for_slot(0)
        subprocess.run([sys.executable,str(RELEASE/'code/scripts/check_a1_fast_execution.py'),
                        '--output',str(log_root/'fresh_execution_check.json'),'--device','cuda:0'],
                       cwd=RELEASE,env=environment(0),check=True)
        process,log=launch_train(core_command,row_id='CORE90_MATCHED_FRESH',gpu=0,run_root=run_root,log_root=log_root)
        state['status']='CORE90_RUNNING'; state['core90_pid']=process.pid
        write_json(run_root/'pipeline_state.json',state)
        code=process.wait(); log.close()
        if code: raise RuntimeError(f'CORE90 training failed: exit={code}')
        fresh=run_root/'CORE90_MATCHED_FRESH/final_ssdg.pth'
        payload=verify_checkpoint(fresh)
        if not payload.get('args',{}).get('from_scratch') or payload.get('args',{}).get('baseline_ckpt'):
            raise RuntimeError('CORE90 provenance is not fresh')
        del payload
        # This is the first checkpoint loaded by the A1 dependency: produced above.
        subprocess.run([sys.executable,str(RELEASE/'code/scripts/check_a1_fast_execution.py'),
                        '--checkpoint',str(fresh),'--output',str(log_root/'fresh_core90_execution_check.json'),'--device','cuda:0'],
                       cwd=RELEASE,env=environment(0),check=True)
        processes=[]
        for row in matrix['rows']:
            process,log=launch_train(commands[row['id']],row_id=row['id'],gpu=row['gpu'],run_root=run_root,log_root=log_root)
            processes.append((row,process,log))
            state['rows'][row['id']]={'status':'RUNNING','pid':process.pid,'gpu':row['gpu']}
        state['status']='A1_RUNNING'; write_json(run_root/'pipeline_state.json',state)
        failed=False
        for row,process,log in processes:
            code=process.wait(); log.close()
            state['rows'][row['id']]['train_exit']=code
            if code:
                failed=True; state['rows'][row['id']]['status']='TRAIN_FAILED'
            else:
                try:
                    evaluate(row['id'],run_root=run_root,log_root=log_root,gpu=row['gpu'],seed=matrix['seed'])
                    state['rows'][row['id']]['status']='EVALUATION_COMMAND_COMPLETE'
                except Exception as error:
                    failed=True; state['rows'][row['id']]['status']='EVAL_FAILED'; state['rows'][row['id']]['error']=repr(error)
            write_json(run_root/'pipeline_state.json',state)
        state['status']='FAILED' if failed else 'AWAITING_ARTIFACT_ANALYSIS'
        write_json(run_root/'pipeline_state.json',state)
        return int(failed)
    except Exception as error:
        state['status']='FAILED'; state['error']=repr(error)
        write_json(run_root/'pipeline_state.json',state)
        raise


if __name__=='__main__':
    raise SystemExit(main())
