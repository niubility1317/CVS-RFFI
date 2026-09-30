"""Reuse the already verified pilot identities; no feature or score access."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from run_d92_sequential_residual_probe import ROOT,PROBE_CONFIG,CONFIG_NAMES,CHANNEL,read,require,validate_spec
from run_d92_registration_diagnostic import validate_spec as validate_parent

RUN='20260930-phase2-d92-sequential-residual-support-m2-r01'
RELEASE='d92_sequential_residual_support_20260930_r01'
SPEC='configs/d92_sequential_residual_support_20260930.json'
PARENT='configs/d92_registration_diagnostic_20260930.json'


def documents(parent,*,commit):
    validate_parent(parent)
    require(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'Explicit preparation commit required')
    s=deepcopy(parent)
    remote=str(PurePosixPath(parent['execution']['remote_run_root']).parent/RUN)
    release=str(PurePosixPath(parent['code']['cwd']).parent/RELEASE)
    s.update(run_id=RUN,group_id='d92-sequential-residual-head-support',
        display_name='LocalRidge继承残差分类头完整support对照',
        description='固定R_seq候选与R0/R_reset对照；旧类B监督训练，C继承后全注册类共同训练；沿用160组support身份。',
        parent_run_ids=[parent['run_id']],status='PLANNED',
        tags=['d92','support-only','sequential-residual','practical-residual'],
        authorization='User authorized legal target-support supervised fine-tuning and benchmark reuse; root sole launch owner.',
        notes=['R_seq is the only candidate; R0 and R_reset are fixed explanatory controls.',
            'Same 160 support parents as the completed registration diagnostic; no query or source sample access.',
            'True K1 numerical-only; full pilot before interpretation; ideal 10/1/3 targets are not hard gates.',
            'Adam moments reset at every B/C stage; state inheritance is within the same physical train fold only.'])
    s['code'].update(cwd=release,commit=commit)
    s['execution'].update(remote_run_root=remote,remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_sequential_residual_probe.py --spec {SPEC}')
    s['permissions'].update(regime='legal_target_support_sequential_residual_training',
        truth_use='Train support labels for B/C supervised training; held support labels only after fixed scores; no query')
    s['checkpoint'].update(current_auxiliary_training='Frozen source encoder caches; residual B/C states trained only on each legal support train fold; no external target-trained state')
    s['metrics_plan'].update(metric_names=['A_NA','B0_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy','H',
        'B_minus_B0','registration_old_loss','old_new_absolute_gap','old_head_change','new_competition_loss','resource_cost'],
        interpretation='Complete support pilot of a fixed candidate; paired old/new metrics and sequence/reset comparisons, no query claim.')
    s['probe'].update(algorithm=deepcopy(PROBE_CONFIG),candidate='R_seq',controls=['R0','R_reset'],
        expected_residual_training_stages=4576,expected_optimizer_steps=292864)
    out={}
    for name,co in s['probe']['cohorts'].items():
        co['evaluation_config']=release+'/'+CONFIG_NAMES[name]
        out[CONFIG_NAMES[name]]=dict(algorithm=deepcopy(PROBE_CONFIG),producer_matrix=deepcopy(co['matrix']),selection=deepcopy(co['selection']))
    for row in s['rows']:
        target=remote+'/'+row['row_id']
        row.update(method=PROBE_CONFIG['method'],purpose='Paired support test of residual capacity and sequential inheritance',
            output_root=target,log_path=target+'/probe.log',config_ref=SPEC,resolved_config_ref=target+'/probe/startup.json',
            optimizer='Float64 full-batch Adam;64 updates/stage;lr0.01;betas0.9/0.999;eps1e-8;global_clip1;moments reset',
            budget_ref='40 parents;792 base fits;1144 residual stages;73216 Adam updates;fixed 3 paths',
            command=f'{s["code"]["environment"]} -u {release}/tools/run_d92_sequential_residual_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
    validate_spec(s);out[SPEC]=s
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent-spec',type=Path,default=ROOT/PARENT);p.add_argument('--commit',required=True)
    a=p.parse_args();out=documents(read(a.parent_spec),commit=a.commit)
    if any((ROOT/name).exists() for name in out):raise FileExistsError('Preparation exists; reconcile without overwrite')
    for name,value in out.items():
        with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',files=list(out))))
