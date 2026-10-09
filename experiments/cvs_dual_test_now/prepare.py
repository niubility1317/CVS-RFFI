"""Register the exact evaluation-only scheduling override before publication."""
from copy import deepcopy
from experiments.cvs_dual_test_now import driver as d
def prepare():
 old=d.source.read(d.ROOT/'automation_reports/CV-SincNet'/d.source.RUN/'experiment.json')
 s=deepcopy(old);s.update(run_id=d.RUN,display_name='双骨干互补证据：冻结16模型立即全量测试',
  description='用户在GPU每卡4训练满载后要求直接启动；每卡临时增加1推理，总<=5，保留健康训练与冻结测试矩阵。',
  parent_run_ids=[d.source.RUN],replaces_run_id=None,status='LOCAL_VERIFIED',
  authorization='2026-10-09 用户在获知8卡每卡4进程占满、测试排队后指令：直接启动。仅测试调度临时覆盖，训练不动。')
 s['code'].update(commit='release_commit.txt records committed/pushed evaluation version',cwd=d.source.PROJECT+'/releases/'+d.RELEASE)
 s['code'].pop('runtime_commit',None)
 s['execution'].update(launch_owner='codex/root/dual-evidence-immediate-test-20261009',
  command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_dual_test_now.driver',
  gpu_policy='User immediate-test override: at most1 extra inference/GPU; total<=5 including CUDA/pre-CUDA; free>=3000MiB; preserve all training',
  remote_run_root=d.BASE.as_posix(),remote_log_root=d.LOGS.as_posix(),local_artifact_root='automation_reports/CV-SincNet/'+d.RUN,
  launch_command='python -m experiments.cvs_dual_test_now.publish --output local_artifacts/'+d.RELEASE)
 s['execution'].pop('dispatcher',None);s['execution'].pop('local_observer',None)
 s['checkpoint'].update(initialization='evaluation_only_reuse_own_frozen_E200',sources=[str(d.source.BASE/r['row_id']/'source/final_ssdg.pth') for r in d.source.rows()],
  provenance_verdict='VERIFIED_SAME_SOURCE_CONTRACT_OWN_E200_NO_RETRAIN',selection_rule='All16 original fixed controls; inherit original pre-query source freeze')
 s.setdefault('constraints',[]).append('Evaluation only: per GPU max1 extra inference; all CUDA/pre-CUDA experiment owners total<=5; free>=3000MiB. Original idle controller takeover only after exact PID/start/cwd/argv/no-child/no-prediction checks.')
 for r in s['rows']:
  rid=r['row_id'];r.update(purpose='fixed_frozen_checkpoint_evaluation',gpu=None,output_root=str(d.BASE/rid),prediction_root=str(d.BASE/rid/'prediction'),
   log_path=str(d.LOGS/(rid+'.log')),command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u '+d.source.PROJECT+'/releases/'+d.RELEASE+'/experiments/cvs_dual_test_now/train_worker.py --row '+rid,
   resolved_config_ref=str(d.BASE/rid/'prediction/resolved_config.json'),optimizer='none; frozen inference',epochs=0,
   budget_ref='No new training. Original ownE200/44400 updates;112 fixed row-view predictions.',
   expected_artifacts=['prediction/provenance.json','prediction/resolved_config.json','prediction/predictions.npz','prediction/complete.json','scores.json','scores.csv','day_scores.json','resources.json'])
 # Posix remote paths must remain portable on the Windows registration host.
 import json
 text=json.dumps(s,ensure_ascii=False).replace('\\\\','/')
 return json.loads(text)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();d.source.write(a.output,prepare())
