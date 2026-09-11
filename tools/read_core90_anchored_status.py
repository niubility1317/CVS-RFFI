"""Read only this run's OS state and bounded startup evidence through short SSH."""
import argparse,json,subprocess
from pathlib import Path
PAYLOAD=r'''
import json,time
from pathlib import Path
import subprocess
base=Path('/home/szu2070436088/2510044040/CV-SincNet')
root=base/'runs/core90_anchored_s392005_20260911_r1'
status=json.loads((root/'status.json').read_text())
commands=[json.loads(line) for line in (root/'commands.jsonl').read_text().splitlines()]
def process(pid):
    p=Path('/proc')/str(pid)
    try:
        env=dict(line.split(b'=',1) for line in (p/'environ').read_bytes().split(b'\0') if b'=' in line)
        fields=dict(line.split(':',1) for line in (p/'status').read_text().splitlines())
        return dict(pid=pid,ppid=int(fields['PPid']),state=fields['State'].strip(),cwd=str((p/'cwd').resolve()),
            argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v],
            CUDA_VISIBLE_DEVICES=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode())
    except FileNotFoundError:return dict(pid=pid,state='EXITED')
workers=[]
for item in commands:
    ps=process(item['pid'])
    if ps['state']!='EXITED':workers.append(dict(label=item['label'],**ps))
logs=[]
for item in commands:
    p=Path(item['log'])
    if p.exists():
        lines=p.read_text(errors='replace').splitlines()
        logs.append(dict(label=item['label'],size=p.stat().st_size,tail=lines[-2:],traceback=any('Traceback (most recent call last)' in line for line in lines)))
caches={}
for path in (root/'cache').glob('*/manifest.json'):
    item=json.loads(path.read_text());caches[path.parent.name]=dict(rows=item['rows'],role=item['identity']['role'])
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
print(json.dumps(dict(time=time.time(),status=status,coordinator=process(status['pid']),workers=workers,logs=logs,caches=caches,
    gpu_processes=gpu.strip().splitlines(),source_inputs=json.loads((root/'source_inputs.json').read_text()) if (root/'source_inputs.json').exists() else None),ensure_ascii=False))
'''
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input=PAYLOAD.encode('utf-8'),capture_output=True,timeout=30)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
    data=json.loads(result.stdout.decode('utf-8'));path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print(json.dumps(data,ensure_ascii=False,indent=2))
