"""Read-only compact evidence and result retrieval."""
import argparse
import json
from experiments.cvs_state_test_now import design as d
from experiments.cvs_phase1_stack.publish import ssh

ART=d.ROOT/'local_artifacts'/d.RELEASE

def inspect():
    code='''
import json,time
from pathlib import Path
b=Path(BASE);release=Path(RELEASE)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def proc(pid):
 p=Path('/proc')/str(pid)
 try:return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=p.joinpath('cmdline').read_bytes().decode().split('\\0'),state=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0])
 except OSError:return None
receipt=read(release/'submit.json');rows=[]
for r in (read(b/'launch_predict.json') or dict(rows=[]))['rows']:
 log=Path(r['log']);text=log.read_text(errors='replace') if log.is_file() else ''
 rows.append(dict(**r,process=proc(r['pid']),log_bytes=len(text.encode()),tail=text.splitlines()[-1:],completion=read(b/r['row_id']/'prediction/complete.json')))
print(json.dumps(dict(at=time.time(),controller=proc(receipt['pid']) if receipt else None,receipt=receipt,
 preflight=read(b/'checkpoint_preflight.json'),queue=read(b/'queue_state.json'),failure=read(b/'failure.json'),completion=read(b/'completion.json'),
 views_ready=(b/'weak_views/manifest.json').is_file(),rows=rows,dispatcher_tail=(release/'dispatcher.stdout.log').read_text(errors='replace').splitlines()[-4:] if (release/'dispatcher.stdout.log').is_file() else [])))
'''.replace('BASE',repr(d.BASE.as_posix())).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE))
    s=json.loads(ssh(code));d.write(ART/'readback.json',s)
    print(json.dumps({k:v for k,v in s.items() if k!='rows'}))
    for r in s['rows']:print(r['row_id'], 'DONE' if r['completion'] else 'RUNNING' if r['process'] else 'EXITED',r['tail'])
    return s

def pull():
    code='''
import json
from pathlib import Path
b=Path(BASE)
names=['completion.json','scoring_complete.json','summary.json','summary.csv','scores.json','scores.csv','day_scores.json','day_scores.csv','paired_results.json','resources.json','analysis.md','source_matrix_frozen.json','checkpoint_preflight.json']
print(json.dumps({n:(b/n).read_text() for n in names if (b/n).is_file()}))
'''.replace('BASE',repr(d.BASE.as_posix()))
    values=json.loads(ssh(code));out=ART/'results';out.mkdir(parents=True,exist_ok=True)
    for n,body in values.items():(out/n).write_text(body,encoding='utf-8')
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pull',action='store_true')
    if p.parse_args().pull:pull()
    else:inspect()
