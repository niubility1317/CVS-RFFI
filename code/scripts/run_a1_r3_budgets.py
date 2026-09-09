"""Scratch R3 budgets with user-authorized exploratory periodic target scoring."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
from run_a1_fast_matched_core90 import RELEASE,environment
from run_a1_fast_selected_adv3b02 import BASE_RUN
from run_a1_scratch_no_checkpoint import scratch_matrix,scratch_command
from run_a1_fast_v2 import main
from cvsrffi.a1_budget_schedule import compressed_options


def budget_matrix():
    from SSDG import train_ssdg as train
    original=scratch_matrix();ref=next(r for r in original['rows'] if r['id']=='R3_REFERENCE')
    base=deepcopy(original['core90_options'])
    for key in ('daot_efficiency_mode','daot_batched_scale_readback','daot_skip_mean_metadata','a1_r3_aux_scale'):
        base['--'+key]=ref[key]
    args=train.build_arg_parser().parse_args(scratch_command(original,project_root=Path('/p'),run_root=Path('/r'),row=ref)[3:])
    return {'seed':392005,'initialization':'random_no_checkpoint','core90_options':base,
        'final_evaluation':'exploratory_periodic_target','max_gpu_processes':2,
        'rows':[{'id':f'R3_B{n}','gpu':gpu,'options':compressed_options(args,base,n)} for n,gpu in [(120,2),(160,3)]]}


def periodic_target_evaluation(row, *, project, root, logs, row_state, capacity):
    from run_a1_fast_v2 import gpu_compute_pids
    total=int(row['options']['--epochs']);start=int(row['options']['--a1_budget_evaluation_start_epoch'])
    folder=root/row['id'];source=project/'runs'/BASE_RUN
    done=row_state.setdefault('scored_epochs',[])
    for epoch in range(start,total+1,10):
        if epoch in done:continue
        checkpoint=folder/f'epoch_{epoch:03d}_ssdg.pth'
        # Written after the complete checkpoint: never race a torch.save.
        if not (folder/f'source_eval_epoch_{epoch:03d}.json').is_file():continue
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        if len(gpu_compute_pids(row['gpu']))>=capacity:return total in done
        output=folder/'target_epochs'/f'E{epoch:03d}'
        if output.exists():raise FileExistsError(f'Refusing partial/existing evaluation root: {output}')
        env=environment(row['gpu'])
        command=[sys.executable,'-u',str(RELEASE/'code/scripts/predict_phase1_truth_last.py'),
            '--checkpoint',str(checkpoint),'--output-root',str(output),'--input-package',str(source/'target_inputs'),
            '--run-id',root.name,'--row-id',f'{row["id"]}_E{epoch:03d}','--mode','predict','--device','cuda:0']
        with (logs/f'{row["id"]}.E{epoch:03d}.predict.log').open('x',encoding='utf-8') as stream:
            subprocess.run(command,cwd=RELEASE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        with (logs/f'{row["id"]}.E{epoch:03d}.score.log').open('x',encoding='utf-8') as stream:
            subprocess.run([sys.executable,str(RELEASE/'code/scripts/score_phase1_truth_last.py'),
                '--predictions',str(output/'predictions.json'),'--truth',str(source/'target_truth/truth_sidecar.json'),
                '--output',str(output/'score.json')],cwd=RELEASE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=json.loads((output/'score.json').read_text(encoding='utf-8'))
        if result.get('record_count')!=672000:raise ValueError('Incomplete target prediction coverage')
        (output/'evaluation_scope.json').write_text(json.dumps({'epoch':epoch,'checkpoint':str(checkpoint),
            'scope':'USER_AUTHORIZED_EXPLORATORY_TARGET_MONITOR_NOT_INDEPENDENT_CONFIRMATION',
            'feeds_training':False},indent=2)+'\n',encoding='utf-8')
        done.append(epoch)
        print(json.dumps({'row':row['id'],'epoch':epoch,'status':'EXPLORATORY_SCORED'}),flush=True)
    return total in done


if __name__=='__main__':main(matrix_factory=budget_matrix,check_script='check_a1_r3_budgets.py',
    periodic_callback=periodic_target_evaluation)
