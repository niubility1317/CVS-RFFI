"""Read-only full source-log and failure-artifact collection for E600 recovery."""
import gzip,json,subprocess
from pathlib import Path
REMOTE=r'''
import json,datetime,subprocess,os
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
run='a1_extended_s392005_20260910_r1';out={'at':datetime.datetime.now().astimezone().isoformat(),'rows':{}}
for name in ['X2_E600','X2_E400','R3_CLEAN_RX_E400']:
 root=p/'runs'/run/name
 records=[json.loads(s) for s in (root/'metrics_epoch.jsonl').read_text().splitlines() if s.strip()]
 lines=(p/'logs'/run/(name+'.train.log')).read_text().splitlines()
 out['rows'][name]={'records':records,'stdout_lines':len(lines),'failure_tail':lines[-60:] if name=='X2_E600' else lines[-3:],
  'files':[{'path':str(f.relative_to(root)),'size':f.stat().st_size} for f in root.iterdir()],
  'anomaly_files':{str(f.relative_to(root)):json.loads(f.read_text()) for pat in ['*anomal*.json','*nonfinite*.json','*failure*.json','diagnostics/*anomal*.json'] for f in root.glob(pat)},
  'process':json.loads((root/'process.json').read_text())}
out['compute_apps']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
out['gpu_map']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader'],text=True)
out['processes']=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  a=proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
  if not any('python' in v for v in a[:1]):continue
  if not any(v.endswith('.py') for v in a[:4]):continue
  s=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
  out['processes'].append({'pid':int(proc.name),'ppid':int(s[1]),'argv':a,'cwd':os.readlink(proc/'cwd')})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(out))
'''
compile(REMOTE,'<read-only-e600-recovery-diagnosis>','exec')
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=50,check=True)
d=json.loads(r.stdout)
out=Path(__file__).parent/'e600_repair_20260911';out.mkdir(exist_ok=True)
with gzip.open(out/'diagnosis.json.gz','wt',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False)
print(d['at'])
for n,x in d['rows'].items():
 print(n,'epochs',len(x['records']),'files',x['files'],'anomalies',list(x['anomaly_files']))
 if n=='X2_E600':print('\n'.join(x['failure_tail']))
print('GPU',d['gpu_map']);print('COMPUTE',d['compute_apps'])
pids={int(s.split(',')[0]) for s in d['compute_apps'].splitlines()}
for x in d['processes']:
 if x['pid'] in pids: print('GPU_PROCESS',x['pid'],x['ppid'],x['cwd'],x['argv'][:4],[(k,x['argv'][i+1]) for i,k in enumerate(x['argv'][:-1]) if k in ['--output_dir','--run-id','--run_id']])
