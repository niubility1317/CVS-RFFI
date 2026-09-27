"""Paired scratch A1 experiment: no checkpoint is loaded before training."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_a1_fast_matched_core90 import (RELEASE, build_train_command, environment,
    launch_train, verify_checkpoint, wait_for_slot, write_json)
from run_a1_fast_selected_adv3b02 import evaluate, BASE_RUN


def scratch_matrix():
    original=json.loads((RELEASE/'configs/a1_fast_matched_core90_20260908.json').read_text(encoding='utf-8'))
    options=deepcopy(original['a1_options'])
    options.update({'--seed':'392005','--model_variant':'lite_d',
        '--from_scratch':'true','--a1_scratch_only':'true','--use_a1_r3':'true',
        '--rc4_use_anchor':'false','--rc4_cache_anchor_logits':'false',
        '--rc4_lambda_feature_anchor':'0','--sat_anchor_ssl':'false',
        '--lambda_teacher_clean_kl':'0','--lambda_teacher_sat_kl':'0','--lambda_teacher_zid_mse':'0',
        '--rc4_identity_start_epoch':'21','--rc4_calibration_update_epochs':'1,21,41,91,161',
        '--test_eval_policy':'interval_final','--test_eval_start_epoch':'999999',
        '--test_eval_interval':'0','--test_eval_final_window':'0','--test_eval_final_interval':'0',
        '--best_metric':'clean_val_tx'})
    for key in ('--baseline_ckpt','--teacher_ckpt'):
        options.pop(key,None)
    rows=[deepcopy(original['rows'][0]),deepcopy(original['rows'][0]),deepcopy(original['rows'][1])]
    for row,name,gpu,scale in zip(rows,['R3_STRUCTURE_CONTROL','R3_REFERENCE','R3_FAST_SEQUENTIAL'],[1,2,3],[0,1,1]):
        row.update(id=name,gpu=gpu,a1_r3_aux_scale=str(scale))
    return {'seed':392005,'initialization':'random_no_checkpoint',
            'teacher':'ema_of_this_run_student','core90_options':options,'rows':rows}


def scratch_command(matrix, *, project_root, run_root, row):
    configured=deepcopy(matrix)
    for key in ('daot_efficiency_mode','daot_batched_scale_readback','daot_skip_mean_metadata','a1_r3_aux_scale'):
        configured['core90_options']['--'+key]=row[key]
    return build_train_command(configured,project_root=project_root,run_root=run_root,
                               row_id=row['id'],core90=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in {'.','..'}: raise ValueError('Invalid run ID')
    project=args.project_root; matrix=scratch_matrix()
    run_root=project/'runs'/args.run_id; log_root=project/'logs'/args.run_id
    commands={r['id']:scratch_command(matrix,project_root=project,run_root=run_root,row=r) for r in matrix['rows']}
    if args.dry_run:
        print(json.dumps(commands,indent=2)); return
    if run_root.exists() or log_root.exists(): raise FileExistsError('Refusing existing roots')
    for path in [project/'Dataset_WigSig/ManySig.pkl',
                 project/'runs'/BASE_RUN/'target_inputs/manifest.json',
                 project/'runs'/BASE_RUN/'target_truth/truth_sidecar.json']:
        if not path.is_file(): raise FileNotFoundError(path)
    run_root.mkdir(); log_root.mkdir()
    state={'status':'SMOKE','pid':os.getpid(),'release':str(RELEASE),'checkpoint':None,
           'seed':392005,'rows':{},'initialization':'random_no_checkpoint'}
    write_json(run_root/'pipeline_state.json',state)
    write_json(run_root/'effective_matrix.json',matrix)
    try:
        gpu=matrix['rows'][0]['gpu']; wait_for_slot(gpu)
        subprocess.run([sys.executable,str(RELEASE/'code/scripts/check_a1_scratch_execution.py'),
                        '--output',str(log_root/'scratch_execution.json'),'--device','cuda:0'],
                       cwd=RELEASE,env=environment(gpu),check=True)
        processes=[]
        for row in matrix['rows']:
            process,log=launch_train(commands[row['id']],row_id=row['id'],gpu=row['gpu'],run_root=run_root,log_root=log_root)
            processes.append((row,process,log))
            state['rows'][row['id']]={'status':'RUNNING','pid':process.pid,'gpu':row['gpu']}
        state['status']='SCRATCH_RUNNING'; write_json(run_root/'pipeline_state.json',state)
        failed=False
        for row,process,log in processes:
            code=process.wait(); log.close()
            state['rows'][row['id']]['exit']=code
            if code:
                failed=True; state['rows'][row['id']]['status']='TRAIN_FAILED'
            else:
                try:
                    # First permitted checkpoint load: the completed row's own E200 output.
                    verify_checkpoint(run_root/row['id']/'final_ssdg.pth')
                    evaluate(row,project=project,run_root=run_root,log_root=log_root)
                    state['rows'][row['id']]['status']='SCORED_PENDING_ANALYSIS'
                except Exception as error:
                    failed=True; state['rows'][row['id']].update(status='EVAL_FAILED',error=repr(error))
            write_json(run_root/'pipeline_state.json',state)
        state['status']='FAILED' if failed else 'AWAITING_ARTIFACT_ANALYSIS'
        write_json(run_root/'pipeline_state.json',state)
    except Exception as error:
        state.update(status='FAILED',error=repr(error)); write_json(run_root/'pipeline_state.json',state)
        raise


if __name__=='__main__': main()
