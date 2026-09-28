"""Read independent process, progress and artifact evidence for the fixed recovery."""
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json,os,time
from pathlib import Path
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
root=project/'runs/20260928-phase2-cvs-d92-practical-manytx-m5-r02'
release=project/'releases/cvs_d92_recovery_20260928_r02'
def read(path):return json.loads(path.read_text()) if path.exists() else None
def process(pid):
 p=Path('/proc')/str(pid)
 return dict(alive=(p/'cmdline').exists(),argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1] if (p/'cmdline').exists() else [],cwd=os.readlink(p/'cwd') if (p/'cwd').exists() else None)
launch=read(root/'launch.json');state=read(root/'state.json') or {};rows=[]
for rowid,item in state.items():
 log=root/(rowid+'.log');tail=''
 if log.exists():
  with log.open('rb') as f:
   f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode(errors='replace')
 progress=None
 for line in tail.splitlines():
  try:
   entry=json.loads(line)
   if 'split_id' in entry:progress=entry
  except ValueError:pass
 rows.append(dict(row_id=rowid,status=item['status'],pid=item['pid'],process=process(item['pid']),progress=progress,completion=read(root/rowid/'predictions_complete.json'),log_tail=tail[-1500:] if item['status']=='TECHNICAL_FAILURE' else None))
print(json.dumps(dict(time=time.time(),launch=launch,dispatcher=process(launch['pid']) if launch else None,rows=rows,support_replays={str(seed):read(release/('support_replay_'+str(seed))/'diagnostic.json') for seed in [392005,2026092703]},completion=read(root/'completion.json'),scored_results_exists=(root/'scored_results.json').exists())))
'''


def main():
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=REMOTE.encode(),capture_output=True,check=True)
    data=json.loads(result.stdout)
    output=Path('E:/type10-7/local_artifacts/cvs_d92_recovery_20260928/readback.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
