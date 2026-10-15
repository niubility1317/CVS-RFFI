"""Offline evidence reader; never imports training code, IQ or checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
SSH=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def remote(script):
    result=subprocess.run(['ssh',*SSH,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+result.stdout.decode('utf-8',errors='replace'))
    return json.loads(result.stdout)


INSPECT=r'''
import json,time,subprocess
from pathlib import Path
c=CONFIG
root=Path(c['root']);release=Path(c['release'])
def read(p):
 if not p.is_file():return None
 try:return json.loads(p.read_text())
 except json.JSONDecodeError:return None
def process(pid):
 p=Path('/proc')/str(pid)
 try:
  if not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':return None
  env=dict(v.split('=',1) for v in (p/'environ').read_bytes().decode().split('\0') if '=' in v)
  return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0'),GPU=env.get('CUDA_VISIBLE_DEVICES'))
 except (FileNotFoundError,ProcessLookupError):return None
launch=read(root/'launch.json');submit=read(release/'submit.json');rows=[]
for v in launch['rows'] if launch else []:
 source=root/v['row_id']/'source';progress=read(source/'mechanism_progress.json');log=Path(v['log']);contract=read(source/'source_contract.json')
 rows.append(dict(launch=v,live=process(v['pid']),epochs=len(progress['records']) if progress else 0,
  progress_last=progress['records'][-1] if progress else None,
  actual_args=read(source/'actual_native_args.json'),resolved=read(source/'resolved_config.json'),module_files=read(source/'module_files.json'),
  role_counts={k:len(x) for k,x in contract['role_ids'].items()} if contract else None,
  completion=read(source/'completion.json'),result=read(source/'result.json'),
  log_bytes=log.stat().st_size if log.is_file() else 0,
  noquery_smoke_PASS=('SMOKE PASS own scratch checkpoint; query_read=false' in log.read_text(errors='replace')) if log.is_file() else False,
  log_tail=log.read_text(errors='replace')[-1400:] if log.is_file() else None))
print(json.dumps(dict(read_at=time.time(),submit=submit,dispatcher_live=process(submit['pid']) if submit else None,
 preflight=read(release/'preflight.json'),queue=read(root/'queue_state.json'),rows=rows,
 complete=read(root/'complete.json'),failure=read(root/'failure.json'),source_decision=read(root/'source_decision.json'),
 matched_sequence=read(root/'matched_sequence.json'),GPU_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
'''


def inspect(output,name):
    spec=read(ROOT/'experiments/cvs_ordinary_fold20/registration.json')
    settings=dict(root=spec['execution']['remote_run_root'],release=spec['code']['cwd'])
    value=remote(INSPECT.replace('CONFIG',repr(settings)));write(Path(output)/name,value)
    print(json.dumps(dict(dispatcher_live=value['dispatcher_live'],rows=[dict(row=v['launch']['row_id'],live=v['live'],epochs=v['epochs'],
        log_bytes=v['log_bytes'],role_counts=v['role_counts'],result_status=v['result']['status'] if v['result'] else None) for v in value['rows']],
        complete=value['complete'],failure=value['failure']),ensure_ascii=False))
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--name',default='progress.json')
    args=parser.parse_args();inspect(args.output,args.name)
