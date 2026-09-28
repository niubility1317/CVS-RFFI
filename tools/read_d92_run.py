"""Read-only process/artifact evidence; target scores require explicit download."""
import argparse
import json
from pathlib import Path
import subprocess
import time

FLAGS=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']
REMOTE=r'''
import json
from pathlib import Path
root=Path(ROOT);release=Path(RELEASE)
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
 if p.exists():result[str(p)]={'bytes':p.stat().st_size,'tail':p.read_text(errors='replace').splitlines()[-3:]}
for p in root.glob('*/scv/predictions_complete.json'):
 result[str(p)]=json.loads(p.read_text())
print(json.dumps(result))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--download',nargs='*',default=[])
    a=p.parse_args();spec=json.loads(a.spec.read_text(encoding='utf-8'))
    script=REMOTE.replace('ROOT',repr(spec['execution']['remote_run_root'])).replace('RELEASE',repr(spec['code']['cwd']))
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode(),capture_output=True,check=True)
    data=json.loads(result.stdout);stamp=str(int(time.time()))
    folder=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id'];out=folder/'evidence';out.mkdir(parents=True,exist_ok=True)
    path=out/('readback_'+stamp+'.json');path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for name in a.download:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe download path')
        local=folder/'results'/name;local.parent.mkdir(parents=True,exist_ok=True)
        if local.exists():raise FileExistsError(local)
        subprocess.run(['scp',*FLAGS,'N607:'+spec['execution']['remote_run_root']+'/'+name,str(local)],check=True)
    print(json.dumps(data,ensure_ascii=False,indent=2));print(path)


if __name__=='__main__':main()
