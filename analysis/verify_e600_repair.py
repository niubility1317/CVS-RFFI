"""Independent read-only reconciliation of release, queue and owned children."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = r'''
import datetime,json,os,subprocess
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
run='a1_e600_repair_s392005_20260911_r1'
root=p/'runs'/run
out={'at':datetime.datetime.now().astimezone().isoformat(),'run':run,'exists':root.exists()}
def process(pid):
 proc=Path('/proc')/str(pid)
 if not proc.exists():return {'pid':pid,'alive':False}
 try:
  env=proc.joinpath('environ').read_bytes().decode().split('\0')
  stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'alive':True,'ppid':int(stat[1]),'cwd':os.readlink(proc/'cwd'),
   'argv':proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1],
   'cuda_visible_devices':next((v.partition('=')[2] for v in env if v.startswith('CUDA_VISIBLE_DEVICES=')),None)}
 except (FileNotFoundError,PermissionError,ProcessLookupError):return {'pid':pid,'alive':None}
out['gpu_map']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader'],text=True)
out['compute_apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
if root.exists():
 state=json.loads((root/'pipeline_state.json').read_text());out['state']=state
 out['dispatcher']=process(state['pid'])
 release=Path(state['release'])
 out['release_commit']=(release/'release_commit.txt').read_text().strip()
 out['dispatcher_log']=(release/'dispatcher.log').read_text()[-4000:]
 check=p/'logs'/run/'execution_check.json'
 out['execution_check']=json.loads(check.read_text()) if check.exists() else None
 out['rows']={}
 for row in json.loads((root/'effective_matrix.json').read_text())['rows']:
  folder=root/row['id'];entry={'configured_options':row['options']}
  if (folder/'process.json').exists():
   info=json.loads((folder/'process.json').read_text());entry['launch']=info
   entry['live']=process(info['pid'])
  log=p/'logs'/run/(row['id']+'.train.log')
  if log.exists():
   text=log.read_text(errors='replace');entry['log_bytes']=log.stat().st_size
   entry['log_head']=text.splitlines()[:25];entry['log_tail']=text.splitlines()[-12:]
  metrics=folder/'metrics_epoch.jsonl'
  if metrics.exists():
   lines=metrics.read_text().splitlines();entry['epoch_records']=len(lines)
   entry['last_metric']=json.loads(lines[-1]) if lines else None
  entry['first_anomaly_exists']=(folder/'first_rc4_anomaly.pt').exists()
  out['rows'][row['id']]=entry
print(json.dumps(out))
'''
compile(source, '<read-only-verification>', 'exec')
result = subprocess.run(['ssh', '-F', 'E:/type10-7/tools/n607_ssh_config',
    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'N607', 'python3 -'],
    input=source, text=True, encoding='utf-8', capture_output=True, timeout=50, check=True)
out = json.loads(result.stdout)
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('x', encoding='utf-8') as stream:
    json.dump(out, stream, ensure_ascii=False, indent=2)
print(json.dumps({'at':out['at'],'exists':out['exists'],'state':out.get('state'),
    'gpu_map':out['gpu_map'],'compute_apps':out['compute_apps'],
    'rows':{k:{f:v[f] for f in ('live','log_bytes','epoch_records','first_anomaly_exists') if f in v}
            for k,v in out.get('rows',{}).items()}}, ensure_ascii=False))
