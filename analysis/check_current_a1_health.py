import json, subprocess, statistics, sys
from pathlib import Path
REMOTE=r'''
import json,time,os,statistics,subprocess
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
result={'time':time.strftime('%Y-%m-%d %H:%M:%S %Z'),'runs':{}}
for name in ['a1_extended_s392005_20260910_r1','a1_mechanism_periodic_s392005_20260910_r1']:
 root=p/'runs'/name;state=json.loads((root/'pipeline_state.json').read_text()); rows={}
 matrix=json.loads((root/'effective_matrix.json').read_text())
 for row in matrix['rows']:
  key=row['id']; status=state['rows'].get(key,{'status':'QUEUED'}); opts={**matrix['core90_options'],**row.get('options',{})}
  log=p/'logs'/name/(key+'.train.log'); metric=root/key/'metrics_epoch.jsonl'
  records=[json.loads(x) for x in metric.read_text().splitlines() if x.strip()] if metric.exists() else []
  text=log.read_text(errors='replace') if log.exists() else ''
  proc=Path('/proc')/str(status.get('pid',0)); active=proc.exists()
  item={'state':status,'active':active,'gpu':row['gpu'],'total':int(opts.get('--epochs',200)),
   'record_count':len(records),'last':records[-1] if records else None,
   'recent':records[-10:],'log_age_seconds':time.time()-log.stat().st_mtime if log.exists() else None,
   'log_bytes':log.stat().st_size if log.exists() else 0,
   'errors':[x for x in text.splitlines() if any(s in x for s in ('Traceback','CUDA out of memory','RC4_SYSTEMIC_NONFINITE','RuntimeError:'))],
   'tail':text.splitlines()[-4:], 'scores':[]}
  if active:
   item['argv']=(proc/'cmdline').read_bytes().decode().split('\0')[:-1];item['cwd']=os.readlink(proc/'cwd')
   item['cpu_ticks']=sum(int(x) for x in (proc/'stat').read_text().split()[13:15])
  for score in sorted((root/key/'target_epochs').glob('*/evaluation_scope.json')):
   item['scores'].append({'epoch':int(score.parent.name[1:]),'scope':json.loads(score.read_text())})
  rows[key]=item
 result['runs'][name]={'state':state,'rows':rows}
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
print(json.dumps(result))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=45,check=True)
d=json.loads(r.stdout)
out=Path(__file__).with_name(sys.argv[1] if len(sys.argv)>1 else 'a1_health_20260910_1154.json');out.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(d['time'])
for name,run in d['runs'].items():
 print(name)
 for key,row in run['rows'].items():
  last=row['last'] or {}
  print(key,row['state']['status'],row['active'],row['record_count'],row['log_age_seconds'],row['errors'][-2:])
