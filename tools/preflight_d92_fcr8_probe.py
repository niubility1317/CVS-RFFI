"""Read-only availability check of the unchanged legal support caches."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from preflight_d92_registration_diagnostic import REMOTE
from publish_d92_branch_support_probe import FLAGS
from run_d92_fcr8_probe import validate_spec


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.spec.read_text(encoding='utf-8'));validate_spec(spec)
    script=REMOTE.replace('SPEC',repr(spec));compile(script,'prototype-transport-preflight','exec')
    r=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,check=True)
    d=json.loads(r.stdout)
    folder=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id']/'evidence'
    folder.mkdir(parents=True,exist_ok=True);path=folder/('fcr8_preflight_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
    print(json.dumps(d));print(path)
    if d['status']!='VERIFIED':raise RuntimeError('Preflight unavailable; no launch')
