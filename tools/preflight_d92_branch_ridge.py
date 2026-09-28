"""Read-only paths/resources/unchanged input handles for both registered cohorts."""
import json
from pathlib import Path
import subprocess
import time
from publish_d92_confirmation import FLAGS, ROOT
from run_d92_confirmation import reuse_frozen_rows, candidate_definition

REMOTE = r'''
import json,os,shutil,socket,subprocess
from pathlib import Path
specs=SPECS
paths={};inputs={}
for s in specs:
 for p in (s['execution']['remote_run_root'],s['code']['cwd'],s['code']['cwd']+'.tar'):
  paths[p]=Path(p).exists()
 c=s['confirmation']; capsule=Path(c['capsule']);m=json.loads((capsule/'manifest.json').read_text())
 assert m['protocol_schema']=='p2_min_v1' and m['phase2_data_status']=='VALIDATED_ONCE'
 assert m['capsule_id']==c['reuse_validated_capsule_id'] and m['split_count']==c['expected_split_count']
 inputs[s['run_id']]=dict(capsule_id=m['capsule_id'],split_count=m['split_count'],model_rows=[])
 for row in s['rows']:
  origin=Path(row['reuse_row_root']);src=Path(row['source_root'])
  marker=json.loads((origin/'predictions_complete.json').read_text())
  assert marker['status']=='PREDICTIONS_COMPLETE' and marker['capsule_id']==m['capsule_id']
  assert (src/'final_ssdg.pth').is_file()
  inputs[s['run_id']]['model_rows'].append(dict(row_id=row['row_id'],baseline_complete=True,
   checkpoint_file_bytes=(src/'final_ssdg.pth').stat().st_size))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
gpu0=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
disk=shutil.disk_usage(Path(specs[0]['code']['cwd']).parent)
r=dict(host=socket.gethostname(),user=os.environ.get('USER'),cpu_count=os.cpu_count(),loadavg=os.getloadavg(),
 gpu=gpu.strip().splitlines(),gpu0_available=gpu0<=1024,free_disk_bytes=disk.free,
 output_paths=paths,inputs=inputs,iq_read=False,truth_read=False,scores_read=False,source_samples_read=False,data_revalidated=False)
r['status']='VERIFIED' if not any(paths.values()) and gpu0<=1024 and disk.free>=2*1024**3 else 'UNAVAILABLE'
print(json.dumps(r))
'''


def main():
    specs=[json.loads((ROOT/f'configs/d92_branch_ridge_repeat_{c}_20260929.json').read_text(encoding='utf-8')) for c in ('rx3','rx1')]
    for spec in specs:
        assert reuse_frozen_rows(spec)
        assert candidate_definition(spec['confirmation'])['candidate_method']=='D92-BranchRidge-v1'
    script=REMOTE.replace('SPECS',repr(specs));compile(script,'branch-ridge-preflight','exec')
    r=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode(),capture_output=True,check=True)
    result=json.loads(r.stdout)
    stamp=time.time_ns()
    for spec in specs:
        folder=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id']/'evidence'
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/f'branch_ridge_preflight_{stamp}.json').open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
    if result['status']!='VERIFIED':
        raise RuntimeError('Paths/resources unavailable; no launch occurred')


if __name__=='__main__':main()
