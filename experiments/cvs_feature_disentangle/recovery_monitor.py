"""Bounded read-only recovery status; never read target metric values."""
import json
from . import recovery as r
from experiments.cvs_phase1_stack.publish import ssh

def inspect():
 script=r'''
import json,time,subprocess
from pathlib import Path
base=Path(BASE);release=Path(RELEASE)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  argv=p.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
  return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=argv) if argv else None
 except OSError:return None
def tail(p):
 if not p.is_file():return []
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));return f.read().decode(errors='replace').splitlines()[-5:]
receipt=read(release/'submit.json');rows=[]
for row in (read(base/'launch.json') or {'rows':[]})['rows']:
 p=base/row['row_id'];log=Path(row['log']);stress=read(p/'source/source_stress.json');recovered=read(p/'source/recovery_provenance.json')
 rows.append(dict(**row,process=proc(row['pid']),log_bytes=log.stat().st_size if log.exists() else 0,tail=tail(log),
  source_status=(read(p/'source/completion.json') or {}).get('status'),stress_status=stress.get('status') if stress else None,
  probe_hook=stress['time_dependency_probe']['hook'] if stress else None,recovery_status=recovered.get('status') if recovered else None,
  prediction=read(p/'prediction/complete.json'),completion=read(p/'completion.json'),failure=read(p/'failure.json')))
handoff=read(release/'handoff.json')
old_workers=[dict(pid=w['pid'],process=proc(w['pid'])) for w in (handoff or {}).get('workers_before',[])]
print(json.dumps(dict(at=time.time(),receipt=receipt,controller=proc(receipt['pid']) if receipt else None,
 queue=read(base/'queue_state.json'),failure=read(base/'failure.json'),completion=read(base/'completion.json'),rows=rows,
 preflight=read(release/'recovery-preflight/completion.json'),preflight_tail=tail(release/'preflight.log'),
 dispatcher_tail=tail(release/'dispatcher.stdout.log'),old_workers=old_workers,handoff=handoff,
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True))))
'''.replace('BASE',repr((r.OLD_BASE.parent/r.RUN).as_posix())).replace('RELEASE',repr(r.d.PROJECT+'/releases/'+r.RELEASE))
 data=json.loads(ssh(script));out=r.d.ROOT/'local_artifacts'/r.RELEASE;out.mkdir(exist_ok=True);r.d.write(out/'readback.json',data)
 print(json.dumps(dict(controller=data['controller'],queue=data['queue'],failure=data['failure'],preflight=(data['preflight'] or {}).get('status'),dispatcher_tail=data['dispatcher_tail'],old_alive=sum(bool(w['process']) for w in data['old_workers']),
 rows=[{k:row[k] for k in ['row_id','mode','gpu','pid','log_bytes','source_status','stress_status','probe_hook','recovery_status','completion','failure']} for row in data['rows']]),ensure_ascii=False))
 return data

if __name__=='__main__':inspect()
