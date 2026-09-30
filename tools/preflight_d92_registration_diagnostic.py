"""Read-only identity/availability check. No IQ, feature values or query scores."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from publish_d92_branch_support_probe import FLAGS
from run_d92_registration_diagnostic import validate_spec

# Self-contained: the new release does not exist yet during this check.
REMOTE = r'''
import json,os,shutil,socket
from pathlib import Path
s=SPEC
paths=[s['execution']['remote_run_root'],s['code']['cwd'],s['code']['cwd']+'.tar']
bindings=[]
for row in s['rows']:
 co=s['probe']['cohorts'][row['cohort']];cache=Path(row['support_features'])
 m=json.loads((Path(co['capsule'])/'manifest.json').read_text())
 f=json.loads((cache/'features_complete.json').read_text())
 if not (m['protocol_schema']=='p2_min_v1' and m['phase2_data_status']=='VALIDATED_ONCE' and m['capsule_id']==co['capsule_id'] and m['split_count']==co['expected_split_count']):raise ValueError('Capsule binding mismatch')
 if not all(m['channel'].get(k)==v for k,v in s['probe']['channel'].items()):raise ValueError('Practical residual channel mismatch')
 if set(m['scenarios'])!=set(s['probe']['algorithm']['allowed_scenarios']):raise ValueError('Practical scenario mismatch')
 if not (f['status']=='BRANCH_SUPPORT_FEATURES_COMPLETE' and f['capsule_id']==co['capsule_id'] and f['checkpoint_sha256']==row['expected_checkpoint_sha256'] and f['model_seed']==row['seeds']['model'] and f['query_rows_read']==0):raise ValueError('Support cache mismatch')
 if not all((cache/name).is_file() for name in ('support_splits.json','checkpoint_provenance.json','startup.json','support_branch_features.npz')):raise ValueError('Missing cache member')
 plan=json.loads((cache/'support_splits.json').read_text())
 if not (plan['schema']=='d92_branch_support_features_v1' and plan['capsule_id']==co['capsule_id'] and plan['checkpoint_sha256']==row['expected_checkpoint_sha256'] and len(plan['splits'])==co['expected_split_count']):raise ValueError('Producer identity mismatch')
 lookup={x['split_id']:x for x in plan['splits']}
 if len(lookup)!=len(plan['splits']):raise ValueError('Duplicate producer split ID')
 for selected in co['selection']['splits']:
  actual=lookup[selected['split_id']]
  projected={k:actual[k] for k in ('split_id','receiver','scenario','k','support_seed','registered_classes')}
  projected['new_count']=len(actual['registered_classes'])-6
  if projected!=selected:raise ValueError('Explicit pilot identity mismatch')
 bindings.append(dict(row=row['row_id'],cache=str(cache),producer_splits=len(lookup),selected_splits=40,channel=s['probe']['channel'],binding='VERIFIED'))
disk=shutil.disk_usage(Path(s['code']['cwd']).parent)
result=dict(host=socket.gethostname(),user=os.environ.get('USER'),cpu_count=os.cpu_count(),loadavg=os.getloadavg(),free_disk_bytes=disk.free,
 output_paths={p:Path(p).exists() for p in paths},cache_bindings=bindings,query_access=False,source_sample_access=False,gpu_use=False,cpu_lanes=2,blas_threads_per_lane=2)
result['status']='VERIFIED' if not any(result['output_paths'].values()) and disk.free>=2*1024**3 else 'UNAVAILABLE'
print(json.dumps(result))
'''


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.spec.read_text(encoding='utf-8'));validate_spec(spec)
    script=REMOTE.replace('SPEC',repr(spec));compile(script,'remote-preflight','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,check=True)
    data=json.loads(result.stdout)
    folder=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id']/'evidence';folder.mkdir(parents=True,exist_ok=True)
    path=folder/('registration_preflight_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False)
    print(json.dumps(data));print(path)
    if data['status']!='VERIFIED':raise RuntimeError('Preflight unavailable; no launch')
