"""Read-only independent N607 post-state for the D92 source development run."""
import json
from pathlib import Path
import subprocess
import time

REMOTE=r'''
import json,subprocess
from pathlib import Path
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20260928-diagnostic-d92-scv-source-s2026092701-r01')
result={}
for name in ('startup.json','state.json','child.json','features/startup.json','features/complete.json','evaluation/complete.json'):
 path=root/name
 if path.exists():
  data=json.loads(path.read_text())
  if name=='startup.json':data.pop('spec',None)
  result[name]=data
for name in ('startup.json','child.json'):
 if name not in result:continue
 pid=result[name]['pid'];proc=Path('/proc')/str(pid)
 result[name]['live_argv']=(proc/'cmdline').read_bytes().decode().split('\0')[:-1] if (proc/'cmdline').exists() else None
 result[name]['live_cwd']=str((proc/'cwd').resolve()) if (proc/'cwd').exists() else None
for name in ('export_d92_source_development.log','evaluate_d92_source_development.log'):
 path=root/name
 if path.exists():
  result[name]={'bytes':path.stat().st_size,'tail':path.read_text(errors='replace').splitlines()[-5:]}
summary=root/'evaluation/summary.csv'
if summary.exists():result['summary_csv']=summary.read_text()
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
print(json.dumps(result))
'''


def main():
    compile(REMOTE,'remote','exec')
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','python3 -'],input=REMOTE.encode('utf-8'),capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8'))
    data=json.loads(result.stdout)
    out=Path('E:/type10-7/automation_reports/CV-SincNet/20260928-diagnostic-d92-scv-source-s2026092701-r01/evidence')
    out.mkdir(parents=True,exist_ok=True)
    path=out/f'readback_{int(time.time())}.json';path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data,ensure_ascii=False,indent=2));print(path)


if __name__=='__main__':main()
