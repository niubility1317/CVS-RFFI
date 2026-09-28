"""Read-only process/artifact evidence; target scores require explicit download."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from run_d92_confirmation import candidate_definition

FLAGS=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
REMOTE=r'''
import json
from pathlib import Path
root=Path(ROOT);release=Path(RELEASE);candidate_folder=CANDIDATE_FOLDER
result={}
for base,name in [(root,'startup.json'),(root,'state.json'),(root,'workflow_state.json'),(root,'complete.json'),(root,'completion.json'),(release,'launch.json')]:
 p=base/name
 if p.exists():
  d=json.loads(p.read_text());d.pop('spec',None)
  if 'pid' in d:
   proc=Path('/proc')/str(d['pid'])
   d['live_argv']=(proc/'cmdline').read_bytes().decode().split('\0')[:-1] if (proc/'cmdline').exists() else None
   d['live_cwd']=str((proc/'cwd').resolve()) if (proc/'cwd').exists() else None
  result[name]=d
for p in [release/'run.log',root/'run.log']+list(root.glob('*.log'))+list(root.glob('*/*.log')):
 if p.exists():
  with p.open('rb') as stream:
   stream.seek(max(0,p.stat().st_size-262144));tail=stream.read().decode(errors='replace').splitlines()[-3:]
  if p.name in ('sfhead.log','sgjoint.log','mvkme.log','bnna.log','osc.log'):
   filtered=[]
   for line in tail:
    try:
     d=json.loads(line);d.pop('steps',None)
     if 'folds' in d:d['fold_count']=len(d.pop('folds'))
     filtered.append(json.dumps(d))
    except (ValueError,TypeError):filtered.append(line[-1500:])
   tail=filtered
  result[str(p)]={'bytes':p.stat().st_size,'tail':tail}
for p in root.glob('*/'+candidate_folder+'/predictions_complete.json'):
 result[str(p)]=json.loads(p.read_text())
result['fit_progress']={}
for p in root.glob('*/'+candidate_folder+'/compact.jsonl'):
 with p.open('rb') as stream:
  stream.seek(max(0,p.stat().st_size-16384));lines=stream.read().decode(errors='replace').splitlines()
 for line in reversed(lines):
  try:
   row=json.loads(line)
   result['fit_progress'][p.parent.parent.name]={k:row.get(k) for k in ['completed','total','k','classes','fit_seconds']}
   break
  except ValueError:pass
processes=[]
for p in list(root.glob('*/*.log.process.json'))+list(root.glob('*.log.process.json')):
 d=json.loads(p.read_text());proc=Path('/proc')/str(d['pid'])
 if (proc/'cmdline').exists():
  d['live_argv']=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]
  d['live_cwd']=str((proc/'cwd').resolve()) if (proc/'cwd').exists() else None
  d['record']=str(p);processes.append(d)
result['live_children']=processes
print(json.dumps(result))
'''


def readback_script(spec):
    candidate=candidate_definition(spec.get('confirmation',{}))
    return (REMOTE.replace('ROOT',repr(spec['execution']['remote_run_root']))
            .replace('RELEASE',repr(spec['code']['cwd']))
            .replace('CANDIDATE_FOLDER',repr(candidate['candidate_folder'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--download',nargs='*',default=[]);p.add_argument('--compact',action='store_true')
    a=p.parse_args();spec=json.loads(a.spec.read_text(encoding='utf-8'))
    script=readback_script(spec)
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode(),capture_output=True,check=True)
    data=json.loads(result.stdout);stamp=str(int(time.time()))
    folder=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id'];out=folder/'evidence';out.mkdir(parents=True,exist_ok=True)
    path=out/('readback_'+stamp+'.json');path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for name in a.download:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe download path')
        local=folder/'results'/name;local.parent.mkdir(parents=True,exist_ok=True)
        if local.exists():raise FileExistsError(local)
        subprocess.run(['scp',*FLAGS,'N607:'+spec['execution']['remote_run_root']+'/'+name,str(local)],check=True)
    if a.compact:
        compact=dict(workflow=data.get('workflow_state.json'),complete=data.get('complete.json'),
            supervisor_live=bool(data.get('startup.json',{}).get('live_argv')),
            live_children=[dict(pid=c['pid'],record=c['record'],argv_matches=c['argv']==c['live_argv'],cwd_matches=c['cwd']==c['live_cwd']) for c in data['live_children']],
            rows=data.get('state.json'),fit_progress=data.get('fit_progress',{}),
            logs={k:v['tail'][-1:] for k,v in data.items() if k.endswith('.log')})
        print(json.dumps(compact,ensure_ascii=False,indent=2))
    else:print(json.dumps(data,ensure_ascii=False,indent=2))
    print(path)


if __name__=='__main__':main()
