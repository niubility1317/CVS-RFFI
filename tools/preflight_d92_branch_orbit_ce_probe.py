"""Read-only registered output/resource/capsule binding probe; no IQ access."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from publish_d92_branch_support_probe import FLAGS
from run_d92_branch_orbit_ce_probe import validate_spec

REMOTE=r'''
import json,os,shutil,socket,subprocess
from pathlib import Path
s=SPEC
paths=[s['execution']['remote_run_root'],s['code']['cwd'],s['code']['cwd']+'.tar']
cohorts={}
for key,c in s['probe']['cohorts'].items():
 p=Path(c['capsule']);m=json.loads((p/'manifest.json').read_text())
 count=len(list((p/'splits').glob('*.json')))
 ok=m.get('protocol_schema')=='p2_min_v1' and m.get('phase2_data_status')=='VALIDATED_ONCE' and m.get('capsule_id')==c['capsule_id'] and count==c['expected_split_count']
 if not ok:raise ValueError('Capsule binding mismatch: '+key)
 cohorts[key]=dict(capsule_id=m['capsule_id'],split_count=count,binding='VERIFIED',iq_read=False,data_revalidated=False)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
gpu0=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
disk=shutil.disk_usage(Path(s['code']['cwd']).parent)
result=dict(host=socket.gethostname(),user=os.getlogin() if os.isatty(0) else os.environ.get('USER'),cpu_count=os.cpu_count(),loadavg=os.getloadavg(),
 gpu=gpu.strip().splitlines(),gpu0_free_for_single_export=gpu0<=1024,free_disk_bytes=disk.free,
 output_paths={p:Path(p).exists() for p in paths},cohorts=cohorts,query_access=False,source_sample_access=False)
result['status']='VERIFIED' if not any(result['output_paths'].values()) and gpu0<=1024 and disk.free>=2*1024**3 else 'UNAVAILABLE'
print(json.dumps(result))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.spec.read_text(encoding='utf-8'));validate_spec(spec)
    script=REMOTE.replace('SPEC',repr(spec));compile(script,'remote-preflight','exec')
    r=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode(),capture_output=True,check=True)
    data=json.loads(r.stdout)
    folder=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id']/'evidence';folder.mkdir(parents=True,exist_ok=True)
    out=folder/('orbit_preflight_' +str(time.time_ns())+'.json')
    with out.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print(json.dumps(data,ensure_ascii=False));print(out)
    if data['status']!='VERIFIED':raise RuntimeError('Resource/output preflight unavailable; no launch performed')


if __name__=='__main__':main()
