"""User-selected ADV3B02 E200 -> paired A1 weight-initialized E200 runs."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from run_a1_fast_matched_core90 import (RELEASE, build_train_command, environment,
    launch_train, verify_checkpoint, wait_for_slot, write_json)

BASE_RUN='phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4'
EVAL_RELEASE='phase1_adv3b02_baseline_eval_s392005_20260903_v2_0cc19956'


def selected_matrix():
    matrix=json.loads((RELEASE/'configs/a1_fast_matched_core90_20260908.json').read_text(encoding='utf-8'))
    matrix=deepcopy(matrix)
    matrix['a1_options'].update({'--model_variant':'lite_d','--seed':'392005',
        '--test_eval_policy':'interval_final','--test_eval_start_epoch':'999999',
        '--test_eval_interval':'0','--test_eval_final_window':'0','--test_eval_final_interval':'0',
        '--best_metric':'clean_val_tx'})
    return matrix


def validate_selected_checkpoint(payload, matrix):
    a=payload['args']
    required={'seed':392005,'model_variant':'lite_d','model_size':'M','representation_mode':'dual',
              'wisig_equalized':'1','wisig_train_rxs':'1,3,4,6,8','wisig_test_rxs':'0,2,5,7,9,10,11',
              'wisig_train_days':'1,2,3','wisig_test_days':'0,1,2,3','num_classes':6,
              'wisig_out_len':256,'split_mode':'tx_rx_day_1_7_2',
              'labeled_ratio':0.07,'unlabeled_ratio':0.63,'source_val_ratio':0.3}
    for key,value in required.items():
        if str(a.get(key))!=str(value): raise ValueError(f'Checkpoint mismatch: {key}')
    split=payload['split_info']
    for key,value in {'labeled_size':6300,'unlabeled_size':56700,'source_val_size':27000}.items():
        if split[key]!=value: raise ValueError(f'Checkpoint split mismatch: {key}')
    if matrix['seed']!=392005: raise ValueError('Matrix seed mismatch')
    return required


def evaluate(row, *, project, run_root, log_root):
    # Reuse the already validated opaque IQ and the repaired predictor release.
    release=RELEASE
    source=project/'runs'/BASE_RUN
    output=run_root/row['id']/'target_prediction'
    env=environment(row['gpu'])
    env['PYTHONPATH']=str(release/'code')+os.pathsep+str(release)
    command=[sys.executable,'-u',str(release/'code/scripts/predict_phase1_truth_last.py'),
             '--checkpoint',str(run_root/row['id']/'final_ssdg.pth'),
             '--output-root',str(output),'--input-package',str(source/'target_inputs'),
             '--run-id',run_root.name,'--row-id',row['id'],'--mode','predict','--device','cuda:0']
    with (log_root/(row['id']+'.predict.log')).open('x',encoding='utf-8') as log:
        subprocess.run(command,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    # The scorer is a separate process; it is invoked only after prediction exits successfully.
    with (log_root/(row['id']+'.score.log')).open('x',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(release/'code/scripts/score_phase1_truth_last.py'),
                        '--predictions',str(output/'predictions.json'),
                        '--truth',str(source/'target_truth/truth_sidecar.json'),
                        '--output',str(output/'score.json')],cwd=release,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in {'.','..'}: raise ValueError('Invalid run ID')
    project=args.project_root
    checkpoint=project/'runs'/BASE_RUN/'ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth'
    run_root=project/'runs'/args.run_id; log_root=project/'logs'/args.run_id
    if run_root.exists() or log_root.exists(): raise FileExistsError('Refusing existing roots')
    matrix=selected_matrix()
    payload=verify_checkpoint(checkpoint)
    facts=validate_selected_checkpoint(payload,matrix); del payload
    for path in [project/'runs'/BASE_RUN/'target_inputs/manifest.json',
                 project/'runs'/BASE_RUN/'target_truth/truth_sidecar.json',
                 RELEASE/'code/scripts/predict_phase1_truth_last.py']:
        if not path.is_file(): raise FileNotFoundError(path)
    run_root.mkdir(); log_root.mkdir()
    state={'status':'SMOKE','pid':os.getpid(),'release':str(RELEASE),'checkpoint':str(checkpoint),
           'seed':392005,'rows':{},'dataset_config':facts}
    write_json(run_root/'pipeline_state.json',state)
    write_json(run_root/'effective_matrix.json',matrix)
    try:
        gpu=matrix['rows'][0]['gpu']; wait_for_slot(gpu)
        subprocess.run([sys.executable,str(RELEASE/'code/scripts/check_a1_fast_execution.py'),
                        '--checkpoint',str(checkpoint),'--model-variant','lite_d',
                        '--output',str(log_root/'selected_checkpoint_execution.json'),'--device','cuda:0'],
                       cwd=RELEASE,env=environment(gpu),check=True)
        processes=[]
        for row in matrix['rows']:
            command=build_train_command(matrix,project_root=project,run_root=run_root,
                                        row_id=row['id'],row=row,checkpoint=checkpoint)
            process,log=launch_train(command,row_id=row['id'],gpu=row['gpu'],run_root=run_root,log_root=log_root)
            processes.append((row,process,log))
            state['rows'][row['id']]={'status':'RUNNING','pid':process.pid,'gpu':row['gpu']}
        state['status']='A1_RUNNING'; write_json(run_root/'pipeline_state.json',state)
        failed=False
        for row,process,log in processes:
            code=process.wait(); log.close()
            state['rows'][row['id']]['exit']=code
            if code:
                failed=True; state['rows'][row['id']]['status']='TRAIN_FAILED'
            else:
                try:
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
