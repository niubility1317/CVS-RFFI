from pathlib import Path
import sys,json,subprocess
sys.path.insert(0,str(Path(__file__).parent))
from analyze_phase1_transfer_20261008 import OUT,SCRIPT
from experiments.cvs_phase1_stack.publish import ssh,CONNECTION
meta=json.loads((OUT/'metadata_epochs.json').read_text(encoding='utf-8'))
rows=[{'row_id':r['row_id'],'path':r['source_path']} for r in meta['rows'] if r['completion']]
r1=meta['runs']['20261004-phase1-reference-overlay-r1-manysig-m16-r01']['source_matrix_frozen.json']['rows']
rows += [{'row_id':'r1-'+r['arm']+'-s'+str(r['model_seed']),'path':r['source_output']} for r in r1]
done={json.loads(x)['row_id'] for x in (OUT/'full_steps.jsonl').read_text(encoding='utf-8').splitlines()} if (OUT/'full_steps.jsonl').exists() else set()
rows=[r for r in rows if r['row_id'] not in done]
remote=r'''
from pathlib import Path
import gzip,json,math,time,re,csv,io
from concurrent.futures import ProcessPoolExecutor
rows=ROWS
def scan(row):
 p=Path(row['path']);ep=p/'epoch_metrics.jsonl'
 epochs=[json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
 f=p/'step_metrics.jsonl.gz';op=gzip.open
 if not f.exists():f=p/'step_metrics.jsonl';op=open
 stats={};n=0;missing=0;nonfinite=0;started=time.time();keys=None
 with op(f,'rt',encoding='utf-8') as stream:
  for line in stream:
   d=json.loads(line);n+=1;m=d.get('metrics',d);e=int(d.get('epoch',0));s=stats.setdefault(e,{'count':0,'metrics':{}});s['count']+=1
   if keys is None:keys=[k for k in m if any(q in k for q in ['loss','grad_before_clip','grad_total','pseudo_selected','pseudo_total','pseudo_conf','reliable_ratio','optimizer_step','nonfinite','clean_ce','satellite_ce'])]
   for k in keys:
    v=m.get(k)
    if isinstance(v,(int,float)):
     if not math.isfinite(v):nonfinite+=1;continue
     a=s['metrics'].setdefault(k,[0.,v,v,0,0]);a[0]+=v;a[1]=min(a[1],v);a[2]=max(a[2],v);a[3]+=1;a[4]+=int(v!=0)
 csvp=p/'epoch_metrics.csv';csvn=len(list(csv.DictReader(io.StringIO(csvp.read_text())))) if csvp.exists() else None
 result=dict(row_id=row['row_id'],path=str(f),steps=n,epochs=epochs,epoch_csv_rows=csvn,seconds=time.time()-started,nonfinite=nonfinite,
   step_epoch_stats={e:dict(count=s['count'],metrics={k:dict(mean=a[0]/a[3],min=a[1],max=a[2],count=a[3],nonzero=a[4]) for k,a in s['metrics'].items()}) for e,s in stats.items()})
 return result
with ProcessPoolExecutor(max_workers=4) as pool:
 for result in pool.map(scan,rows):print(json.dumps(result),flush=True)
'''.replace('ROWS',repr(rows))
compile(remote,'remote','exec')
proc=subprocess.Popen(['ssh',*CONNECTION,'-T','N607','python3 -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
proc.stdin.write(remote.encode('utf-8'));proc.stdin.close()
with (OUT/'full_steps.jsonl').open('a',encoding='utf-8') as output:
 for raw in proc.stdout:
  row=json.loads(raw);output.write(json.dumps(row)+'\n');output.flush()
  print(json.dumps({k:row[k] for k in ['row_id','steps','seconds','nonfinite']}),flush=True)
code=proc.wait();err=proc.stderr.read().decode('utf-8');print('EXIT',code,err)
if code:raise SystemExit(code)
