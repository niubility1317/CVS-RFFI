"""Read-only complete logs and all existing long-budget target score artifacts."""
import gzip,json,subprocess
from pathlib import Path
REMOTE=r'''
import json,datetime,re,os
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
out={'at':datetime.datetime.now().astimezone().isoformat(),'runs':{}}
for name in ['a1_extended_s392005_20260910_r1','a1_e600_repair_s392005_20260911_r1']:
 root=p/'runs'/name;state=json.loads((root/'pipeline_state.json').read_text())
 matrix=json.loads((root/'effective_matrix.json').read_text());rows={}
 for configured in matrix['rows']:
  row=configured['id'];folder=root/row
  proc=folder/'process.json';info=json.loads(proc.read_text()) if proc.exists() else {}
  pid=info.get('pid');alive=False
  if pid and Path('/proc',str(pid),'cmdline').exists():
   alive=str(folder) in Path('/proc',str(pid),'cmdline').read_bytes().decode().split('\0')
  mf=folder/'metrics_epoch.jsonl';records=[json.loads(s) for s in mf.read_text().splitlines() if s.strip()] if mf.exists() else []
  log=p/'logs'/name/(row+'.train.log');lines=log.read_text(errors='replace').splitlines() if log.exists() else []
  scores=[]
  for f in sorted(folder.glob('target_epochs/E*/score.json')):
   scope=f.with_name('evaluation_scope.json');pred=f.with_name('predictions.json')
   scores.append({'epoch':int(f.parent.name[1:]),'path':str(f),'score':json.loads(f.read_text()),
     'scope':json.loads(scope.read_text()) if scope.exists() else None,
     'prediction_bytes':pred.stat().st_size if pred.exists() else 0})
  rows[row]={'configured':configured,'process':info,'alive':alive,'records':records,'log_lines':len(lines),
    'errors':[{'line':i+1,'text':s} for i,s in enumerate(lines) if re.search(r'Traceback|RuntimeError|Error:|CUDA out of memory|Killed',s)],
    'tail':lines[-15:],'scores':scores,'final_checkpoint':(folder/'final_ssdg.pth').exists(),
    'final_checkpoint_bytes':(folder/'final_ssdg.pth').stat().st_size if (folder/'final_ssdg.pth').exists() else 0}
 release=Path(state['release']);dl=release/'dispatcher.log'
 out['runs'][name]={'state':state,'matrix':matrix,'rows':rows,'dispatcher_log':dl.read_text(errors='replace') if dl.exists() else None}
print(json.dumps(out))
'''
compile(REMOTE,'<readonly-collection>','exec')
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=55,check=True)
d=json.loads(r.stdout);out=Path(__file__).parent/'long_training_tests_20260913';out.mkdir(exist_ok=False)
with gzip.open(out/'evidence.json.gz','wt',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False)
print(d['at'])
for name,run in d['runs'].items():
 print(name,run['state'])
 for row,v in run['rows'].items():
  print(row,'epochs',len(v['records']),'alive',v['alive'],'final',v['final_checkpoint'],'tests',[s['epoch'] for s in v['scores']],'errors',v['errors'])
  if v['scores']:print('score schema',str(v['scores'][-1])[:6500])
