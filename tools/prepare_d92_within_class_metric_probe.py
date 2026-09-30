"""Prepare a new fixed diagnostic without reading features or performance."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from run_d92_within_class_metric_probe import ROOT,PROBE_CONFIG,CONFIG_NAMES,read,require,validate_spec
from run_d92_registration_diagnostic import validate_spec as validate_parent

RUN='20260930-phase2-d92-within-metric-diagnostic-m2-r01'
RELEASE='d92_within_metric_diagnostic_20260930_r01'
SPEC='configs/d92_within_metric_diagnostic_20260930.json'
PARENT='configs/d92_registration_diagnostic_20260930.json'


def documents(parent,*,commit):
    validate_parent(parent)
    require(isinstance(commit,str) and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'Explicit preparation commit required')
    s=deepcopy(parent)
    remote=str(PurePosixPath(parent['execution']['remote_run_root']).parent/RUN)
    release=str(PurePosixPath(parent['code']['cwd']).parent/RELEASE)
    s.update(run_id=RUN,group_id='d92-within-class-metric-diagnostic',
        display_name='LocalRidge旧support类内度量机制与注册完整对照',
        description='固定旧类均值保护的类内度量收缩，B估计、C继承；完整support配对性能及类留一统计诊断。',
        parent_run_ids=[parent['run_id']],status='PLANNED',
        tags=['d92','support-only','within-class-metric','practical-residual'],
        authorization='Existing user D92 optimization authorization; root sole launch owner; no query or source access.',
        notes=['R_metric is a fixed mechanism diagnostic; R0 is original LocalRidge.',
            'Same 160 support parents as registration diagnostic; practical residual/post_sync/noeq/25MHz.',
            'True K1 has no independent holdout; proxy trainK1 exactly reuses R0, not a positive-gain claim.',
            'No direct promotion from this mechanism pilot; no change to prior candidate screens or ideal goals.',
            'Metric state fitted only from old train support, inherited unchanged by C; LOCO cost reported separately.',
            'Maximum head-fit budget is 3816; actual degenerate identity fallback can reduce additional fits.'])
    s['code'].update(cwd=release,commit=commit)
    s['execution'].update(remote_run_root=remote,remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_within_class_metric_probe.py --spec {SPEC}')
    s['permissions'].update(regime='legal_target_support_within_class_metric_diagnostic',
        truth_use='Train support labels for metric/head fitting; held support labels only after fixed scores; no query')
    s['checkpoint'].update(current_auxiliary_training='No new encoder weights; unchanged compliant source encoder caches; within-class metric from each legal old support train fold only')
    s['metrics_plan'].update(metric_names=['A_NA','B0_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy','H',
        'B_minus_B0','registration_old_loss','old_new_absolute_gap','old_head_change','new_competition_loss',
        'within_class_spectrum','leave_one_old_class_transfer','actual_state_bytes','resource_cost'],
        interpretation='Complete fixed support mechanism diagnostic; no query claim, no direct promotion.')
    s['probe'].update(algorithm=deepcopy(PROBE_CONFIG),candidate='R_metric',controls=['R0'],
        interpretation='mechanism_pilot_no_direct_promotion',expected_baseline_head_fits=3168,
        expected_metric_fit_count=1760,maximum_metric_nonidentity_count=360,maximum_metric_head_fits=648,
        expected_head_fits=3816,expected_diagnostic_fit_count=2160,expected_optimizer_steps=0)
    out={}
    for name,co in s['probe']['cohorts'].items():
        co['evaluation_config']=release+'/'+CONFIG_NAMES[name]
        out[CONFIG_NAMES[name]]=dict(algorithm=deepcopy(PROBE_CONFIG),producer_matrix=deepcopy(co['matrix']),selection=deepcopy(co['selection']))
    for row in s['rows']:
        target=remote+'/'+row['row_id']
        row.update(method=PROBE_CONFIG['method'],purpose='Fixed within-class metric transfer and paired B/C mechanism diagnostic',
            output_root=target,log_path=target+'/probe.log',config_ref=SPEC,resolved_config_ref=target+'/probe/startup.json',
            optimizer='None; deterministic bounded metric and original ridge; zero gradient updates',
            budget_ref='40 parents;792 baseline fits;440 metric attempts;max162 extra heads;540 leave-one-class diagnostic fits',
            command=f'{s["code"]["environment"]} -u {release}/tools/run_d92_within_class_metric_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
    validate_spec(s);out[SPEC]=s
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent-spec',type=Path,default=ROOT/PARENT);p.add_argument('--commit',required=True)
    a=p.parse_args();out=documents(read(a.parent_spec),commit=a.commit)
    if any((ROOT/name).exists() for name in out):raise FileExistsError('Preparation exists; reconcile without overwrite')
    for name,value in out.items():
        with (ROOT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',files=list(out))))
