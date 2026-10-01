"""Read-only metadata/availability preflight for explicitly supplied support inputs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from run_d92_margin_joint_probe import validate_spec, require, CHANNEL, SCENARIOS


REMOTE = r'''
import json,os,shutil,socket
from pathlib import Path
s=SPEC
bindings=[]
for row in s['rows']:
 co=s['probe']['cohorts'][row['cohort']];cache=Path(row['support_features'])
 manifest=json.loads((Path(co['capsule'])/'manifest.json').read_text(encoding='utf-8'))
 feature=json.loads((cache/'features_complete.json').read_text(encoding='utf-8'))
 provenance=json.loads((cache/'checkpoint_provenance.json').read_text(encoding='utf-8'))
 plan=json.loads((cache/'support_splits.json').read_text(encoding='utf-8'))
 if not (manifest['protocol_schema']=='p2_min_v1' and manifest['phase2_data_status']=='VALIDATED_ONCE' and manifest['capsule_id']==co['capsule_id'] and manifest['split_count']==co['expected_split_count']):raise ValueError('Capsule binding mismatch')
 if manifest['channel']!=s['probe']['channel'] or set(manifest['scenarios'])!=set(SCENARIOS):raise ValueError('Practical channel mismatch')
 if not (feature['status']=='BRANCH_SUPPORT_FEATURES_COMPLETE' and feature['capsule_id']==co['capsule_id'] and feature['checkpoint_sha256']==row['expected_checkpoint_sha256'] and feature['model_seed']==row['seeds']['model'] and feature['query_rows_read']==0):raise ValueError('Support cache mismatch')
 if not (provenance['checkpoint_sha256']==row['expected_checkpoint_sha256'] and provenance['model_seed']==row['seeds']['model'] and provenance['verdict']=='MATCHED_SOURCE_ONLY_SCRATCH' and provenance['target_access_before_freeze'] is False and provenance['checkpoint_inheritance']==[]):raise ValueError('Source-only provenance mismatch')
 if not (plan['capsule_id']==co['capsule_id'] and plan['checkpoint_sha256']==row['expected_checkpoint_sha256'] and len(plan['splits'])==co['expected_split_count']):raise ValueError('Producer binding mismatch')
 lookup={v['split_id']:v for v in plan['splits']}
 if len(lookup)!=len(plan['splits']):raise ValueError('Duplicate producer ID')
 for selected in co['selection']['splits']:
  actual=lookup[selected['split_id']];projected={k:actual[k] for k in ('split_id','receiver','scenario','k','support_seed','registered_classes')};projected['new_count']=len(actual['registered_classes'])-6
  if projected!=selected:raise ValueError('Selected physical parent mismatch')
 if not all((cache/name).is_file() for name in ('support_branch_features.npz','startup.json')):raise ValueError('Missing producer cache member')
 bindings.append(dict(row_id=row['row_id'],selected_splits=len(co['selection']['splits']),binding='VERIFIED'))
paths=[s['execution']['remote_run_root'],s['code']['cwd'],s['code']['cwd']+'.tar']
existing={p:Path(p).exists() for p in paths};disk=shutil.disk_usage(Path(s['code']['cwd']).parent)
print(json.dumps(dict(status='VERIFIED' if not any(existing.values()) else 'UNAVAILABLE',run_id=s['run_id'],
 host=socket.gethostname(),cpu_count=os.cpu_count(),free_disk_bytes=disk.free,cache_bindings=bindings,
 qp_resources=s['probe']['qp_resources'],
 output_paths=existing,query_access=False,source_sample_access=False,feature_values_read=False,gpu_use=False)))
'''


def remote_script(spec):
    validate_spec(spec)
    script='SCENARIOS='+repr(SCENARIOS)+'\n'+REMOTE.replace('SPEC',repr(spec))
    compile(script,'margin-metadata-preflight','exec')
    return script


def preflight(spec, *, ssh_host, ssh_config, remote_python, output, run_fn=subprocess.run):
    require(all(isinstance(v,str) and v for v in (ssh_host,ssh_config,remote_python)),'Explicit SSH endpoint required')
    path=Path(output)
    if path.exists():raise FileExistsError(path)
    result=run_fn(['ssh','-F',ssh_config,'-o','BatchMode=yes','-o','ConnectTimeout=10','-T',ssh_host,
        shlex.join([remote_python,'-'])],input=remote_script(spec),text=True,encoding='utf-8',capture_output=True,timeout=120)
    require(result.returncode==0,'Metadata preflight failed: '+result.stderr[-2000:])
    data=json.loads(result.stdout)
    require(data.get('run_id')==spec['run_id'] and data.get('query_access') is False
        and data.get('source_sample_access') is False and data.get('feature_values_read') is False
        and data.get('qp_resources')==spec['probe']['qp_resources'],'Preflight response binding mismatch')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(data,stream,ensure_ascii=False,allow_nan=False);stream.write('\n')
    require(data.get('status')=='VERIFIED','Preflight unavailable; preserve evidence')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('spec','ssh-host','ssh-config','remote-python','output'):p.add_argument('--'+name,required=True)
    args=vars(p.parse_args());spec=json.loads(Path(args.pop('spec')).read_text(encoding='utf-8'))
    print(json.dumps(preflight(spec,**args),allow_nan=False))
