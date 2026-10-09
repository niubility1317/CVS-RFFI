"""Read-only remote inspection and source diagnostic artifact download."""
import argparse
import io
import json
from pathlib import Path
import tarfile
from experiments.cvs_multi_state_action import design as d
from experiments.cvs_phase1_stack.publish import ssh

ART=d.ROOT/'local_artifacts'/d.RELEASE

def inspect():
    script='''
import json,time,subprocess
from pathlib import Path
b=Path(BASE);release=Path(RELEASE)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def proc(pid):
 p=Path('/proc')/str(pid)
 try:return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=p.joinpath('cmdline').read_bytes().decode().split('\\0'),state=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0])
 except (OSError,UnicodeError):return None
receipt=read(release/'submit.json');rows=[]
launches=[]
for kind in ('audit','source','predict'):launches.extend((read(b/('launch_'+kind+'.json')) or dict(rows=[]))['rows'])
for r in launches:
 log=Path(r['log']);text=log.read_text(errors='replace') if log.is_file() else ''
 rows.append(dict(**r,process=proc(r['pid']),log_bytes=len(text.encode()),tail=text.splitlines()[-3:],
  resolved=read(b/r['row_id']/'resolved_config.json'),completion=read(b/r['row_id']/('completion.json' if r['kind']=='audit' else 'source/completion.json' if r['kind']=='source' else 'prediction/complete.json'))))
print(json.dumps(dict(at=time.time(),receipt=receipt,controller=proc(receipt['pid']) if receipt else None,
 preflight=read(release/'checkpoint_preflight.json'),queue=read(b/'queue_state.json'),
 failure=read(b/'failure.json'),completion=read(b/'completion.json'),rows=rows)))
'''.replace('BASE',repr(d.BASE.as_posix())).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE))
    value=json.loads(ssh(script));d.write(ART/'readback.json',value)
    return value

def pull():
    script='''
import sys,tarfile
from pathlib import Path
b=Path(BASE)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for p in b.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md'):
   t.add(p,arcname=p.relative_to(b).as_posix(),recursive=False)
'''.replace('BASE',repr(d.BASE.as_posix()))
    blob=ssh(script);ART.mkdir(parents=True,exist_ok=True)
    (ART/'results.tar.gz').write_bytes(blob)
    out=ART/'results';out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        for m in t.getmembers():
            if not m.isfile() or not (out/m.name).resolve().is_relative_to(out.resolve()):raise ValueError('Unsafe result path')
        t.extractall(out)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pull',action='store_true');a=p.parse_args()
    if a.pull:print(pull())
    else:
        state=inspect()
        print(json.dumps({k:v for k,v in state.items() if k not in ('rows','preflight')},ensure_ascii=False))
        for r in state['rows']:print(json.dumps({k:r[k] for k in ('row_id','gpu','process','log_bytes','tail','completion')},ensure_ascii=False))
