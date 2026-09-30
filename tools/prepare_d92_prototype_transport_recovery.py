"""Preserve failed r01 and create one same-method JSON-boundary recovery."""
import json
from copy import deepcopy
from pathlib import Path, PurePosixPath
import argparse

from run_d92_prototype_transport_probe import ROOT, CONFIG_NAMES, validate_spec

RUN='20260930-phase2-d92-prototype-transport-support-m2-r02'
RELEASE='d92_prototype_transport_support_20260930_r02'
SPEC='configs/d92_prototype_transport_support_recovery_20260930.json'


def document(parent,commit):
    validate_spec(parent)
    if parent['run_id']!='20260930-phase2-d92-prototype-transport-support-m2-r01':
        raise ValueError('Only the preserved failed r01 may parent this technical recovery')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Explicit preparation commit required')
    out=deepcopy(parent)
    remote=str(PurePosixPath(parent['execution']['remote_run_root']).parent/RUN)
    release=str(PurePosixPath(parent['code']['cwd']).parent/RELEASE)
    out.update(run_id=RUN,parent_run_ids=[parent['run_id']],status='PLANNED',
        display_name='LocalRidge原型联合适配：JSON输出边界技术恢复',
        description='同一160parent/同一冻结方法与超参，仅修复NumPy标量JSON类型边界；保留r01技术失败，全矩阵新r02独立输出。')
    out['tags']=list(dict.fromkeys(out['tags']+['technical-json-recovery']))
    out['notes'] += ['r01 four rows exited at full parent JSON serialization; all failed outputs retained.',
        'Native bool/float audit and strict JSONL/text/CSV serialization corrected; mathematical algorithm/data/seeds unchanged.',
        'One r02 full-matrix technical recovery, no performance-guided subset or parameter change.']
    out['code'].update(commit=commit,cwd=release)
    out['execution'].update(remote_run_root=remote,remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_prototype_transport_probe.py --spec {SPEC}')
    for name,co in out['probe']['cohorts'].items():co['evaluation_config']=release+'/'+CONFIG_NAMES[name]
    for row in out['rows']:
        target=remote+'/'+row['row_id']
        row.update(output_root=target,log_path=target+'/probe.log',config_ref=SPEC,
            resolved_config_ref=target+'/probe/startup.json',
            command=f'{out["code"]["environment"]} -u {release}/tools/run_d92_prototype_transport_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
    for key in ['data','checkpoint','permissions','metrics_plan']:
        if out[key]!=parent[key]:raise ValueError('Scientific contract changed: '+key)
    if out['probe']['algorithm']!=parent['probe']['algorithm']:raise ValueError('Frozen algorithm changed')
    for row,old in zip(out['rows'],parent['rows']):
        if row['seeds']!=old['seeds'] or row['support_features']!=old['support_features']:
            raise ValueError('Support/seed identity changed')
    validate_spec(out)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);args=p.parse_args()
    parent=json.loads((ROOT/'configs/d92_prototype_transport_support_20260930.json').read_text(encoding='utf-8'))
    value=document(parent,args.commit)
    with (ROOT/SPEC).open('x',encoding='utf-8') as output:
        json.dump(value,output,ensure_ascii=False,indent=2,allow_nan=False);output.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',run_id=RUN,spec=SPEC)))
