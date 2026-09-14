"""User-authorized stop of one verified row, preserving all experiment artifacts."""
import json,subprocess
from pathlib import Path
SOURCE=r'''
import datetime,json,os,signal,time
from pathlib import Path
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
root=project/'runs/a1_mechanism_periodic_s392005_20260910_r1'
folder=root/'G1_FISHER_GATE';info=json.loads((folder/'process.json').read_text())
pid=info['pid'];release=info['cwd']
assert pid==612456
receipt=folder/'user_stop_20260914.json'
assert not receipt.exists(), 'Reconcile previous stop before retry'
def read(pid):
 p=Path('/proc')/str(pid)
 try:
  stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'ppid':int(stat[1]),'start':stat[19],'state':stat[0],
    'argv':p.joinpath('cmdline').read_bytes().decode().split('\0')[:-1],
    'cwd':os.readlink(p/'cwd'),'uid':p.stat().st_uid}
 except (FileNotFoundError,ProcessLookupError,PermissionError):return None
main=read(pid)
assert main and main['cwd']==release and main['uid']==os.getuid()
assert '--output_dir' in main['argv'] and main['argv'][main['argv'].index('--output_dir')+1]==str(folder)
assert any(x.endswith('/code/SSDG/train_ssdg.py') for x in main['argv'])
allp={int(p.name):read(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
owned={pid:main}
while True:
 children={p:x for p,x in allp.items() if x and x['ppid'] in owned and p not in owned}
 if not children:break
 owned.update(children)
assert all(x['cwd']==release and x['uid']==os.getuid() for x in owned.values())
def same(p):
 now=read(p)
 return now and now['start']==owned[p]['start'] and now['state']!='Z'
def send(p,sig):
 if same(p):
  try:os.kill(p,sig)
  except ProcessLookupError:pass
before={'at':datetime.datetime.now().astimezone().isoformat(),'reason':'explicit user stop G1_FISHER_GATE',
 'owned':list(owned.values()),'scope':str(folder),'status':'STOP_REQUESTED'}
receipt.write_text(json.dumps(before,indent=2),encoding='utf-8')
send(pid,signal.SIGTERM)
for child in owned:
 if child!=pid:send(child,signal.SIGTERM)
deadline=time.monotonic()+8
while any(same(p) for p in owned) and time.monotonic()<deadline:time.sleep(.2)
escalated=[]
for p in owned:
 if same(p):send(p,signal.SIGKILL);escalated.append(p)
time.sleep(.5)
survivors=[p for p in owned if same(p)]
result={**before,'finished_at':datetime.datetime.now().astimezone().isoformat(),
 'status':'STOPPED_BY_USER' if not survivors else 'STOP_UNVERIFIED','survivors':survivors,'sigkill':escalated,
 'artifacts_preserved':True}
receipt.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
'''
compile(SOURCE,'<scoped-user-stop>','exec')
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=SOURCE,text=True,encoding='utf-8',capture_output=True,check=True,timeout=35)
result=json.loads(r.stdout)
out=Path(__file__).parent/'g1_user_stop_20260914';out.mkdir(exist_ok=True)
(out/'stop_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'pids':[x['pid'] for x in result['owned']],'survivors':result['survivors'],'sigkill':result['sigkill']}))
