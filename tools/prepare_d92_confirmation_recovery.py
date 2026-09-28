"""Technical r02 recovery; frozen algorithm/data and model matrix remain identical."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/'configs/d92_confirmation_20260928.json').read_text(encoding='utf-8'))
old=spec['run_id'];run=old[:-3]+'r02'
oldroot=spec['execution']['remote_run_root'];remote=oldroot[:-3]+'r02'
oldrelease=spec['code']['cwd'];release=oldrelease[:-3]+'r02'
cfg=json.loads((ROOT/'configs/d92_confirmation_data_20260928.json').read_text(encoding='utf-8'))
spec.update(run_id=run,replaces_run_id=old,parent_run_ids=spec['parent_run_ids']+[old])
spec['code'].update(cwd=release,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
spec['execution'].update(remote_run_root=remote,remote_log_root=remote,local_artifact_root='automation_reports/CV-SincNet/'+run,
    launch_command='C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_confirmation.py --spec configs/d92_confirmation_recovery_20260928.json --release d92_scv_confirmation_20260928_r02')
spec['confirmation'].update(data_config=release+'/configs/d92_confirmation_data_20260928.json',candidate_config=release+'/configs/d92_scv_frozen_20260928.json',
    old_classes=cfg['old_classes'],reuse_validated_capsule_id='residual-noeq-76121e6f34363fa612ec25fb')
spec['data']['capsule_id']=spec['confirmation']['reuse_validated_capsule_id']
spec['data']['split_id']='existing900 immutable split IDs from r01 data builder'
spec['metrics_plan']['scorer_ref']=release+'/tools/score_d92_confirmation.py'
spec['notes'].append('Technical recovery: reference source contract lacks classes; bind actual source class registry to registered old classes and ground NPZ.Original r01 failed before checkpoint load or target prediction. Reuse validated capsule, do not rebuild/resample/revalidate data. Candidate and full model/matrix unchanged.')
for row in spec['rows']:
    row['output_root']=row['output_root'].replace(oldroot,remote)
    row['log_path']=row['log_path'].replace(oldroot,remote)
    row['resolved_config_ref']=remote+'/startup.json'
    row['config_ref']='configs/d92_confirmation_recovery_20260928.json'
    row['command']=row['command'].replace(oldrelease,release).replace('configs/d92_confirmation_20260928.json','configs/d92_confirmation_recovery_20260928.json')
(ROOT/'configs/d92_confirmation_recovery_20260928.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(run)
