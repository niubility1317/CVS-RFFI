"""Read already frozen A1 predictions; no model load, inference or remote writes."""
import subprocess,json,csv,gzip
from pathlib import Path
B=Path('E:/type10-7');A=B/'automation_reports/CV-SincNet';O=A/'daot_fasttrust_game_comprehensive_20260914'
rows=list(csv.DictReader((A/'all_exploration_20260913/all_final_results.csv').open(encoding='utf-8-sig')))
code='ROWS='+repr(rows)+r'''
import json,hashlib,hmac,time,gc
from pathlib import Path
from collections import defaultdict
cache={};out={'rows':[],'errors':[],'started':time.time()}
for row in ROWS:
 try:
  score=json.loads(Path(row['path']).read_text());truthpath=score['truth_path']
  if truthpath not in cache:
   tr=json.loads(Path(truthpath).read_text());truth={r['sample_id']:r['label']for r in tr['records']};key=hashlib.sha256(('cvs.phase1.truth-last|'+tr['split_binding']).encode()).digest();meta={}
   for tx in range(6):
    for rx in [0,2,5,7,9,10,11]:
     for day in range(4):
      for sig in range(1000):
       sid='sample:'+hmac.new(key,f'tx{tx}:rx{rx}:day{day}:eq1:sig{sig}'.encode(),hashlib.sha256).hexdigest();meta[sid]=(tx,rx,day)
   assert set(meta)==set(truth) and all(truth[k]==v[0]for k,v in meta.items());cache[truthpath]=meta
  meta=cache[truthpath];pred=json.loads(Path(score['prediction_path']).read_text());counts=defaultdict(lambda:[0,0]);conf=defaultdict(lambda:[[0]*6 for _ in range(6)]);seen=set()
  for r in pred['records']:
   sid=r['sample_id'];scene=r['scenario'];p=r['predicted_class'];tx,rx,day=meta[sid];pk=scene,sid;assert pk not in seen and 0<=p<6;seen.add(pk);ok=int(p==tx);conf[scene][tx][p]+=1
   for scope,k in [('scene','all'),('rx',str(rx)),('day',str(day)),('tx',str(tx)),('rx_day',f'{rx}:{day}'),('rx_tx',f'{rx}:{tx}'),('rx_day_tx',f'{rx}:{day}:{tx}')]:
    c=counts[(scene,scope,k)];c[0]+=ok;c[1]+=1
  assert len(seen)==672000
  for scene,m in score['metrics'].items():assert counts[(scene,'scene','all')]==[m['correct'],m['total']]
  macro={}
  for scene,cm in conf.items():
   fs=[];ps=[];rec=[]
   for c in range(6):
    tp=cm[c][c];n=sum(cm[c]);pp=sum(v[c]for v in cm);ps.append(tp/pp if pp else 0);rec.append(tp/n);fs.append(2*tp/(n+pp)if n+pp else 0)
   macro[scene]={'macro_f1_pct':sum(fs)/6*100,'macro_precision_pct':sum(ps)/6*100,'macro_recall_pct':sum(rec)/6*100,'per_class_f1_pct':[v*100 for v in fs],'per_class_precision_pct':[v*100 for v in ps],'per_class_recall_pct':[v*100 for v in rec]}
  out['rows'].append({'run':row['run'],'row':row['row'],'inherited_target_contact':row['inherited_target_contact'],'metadata_mapping_exact':True,'count':len(seen),'score_path':row['path'],'macro':macro,'confusion':dict(conf),'details':[{'scene':s,'scope':sc,'group':g,'correct':v[0],'total':v[1],'accuracy_pct':100*v[0]/v[1]}for(s,sc,g),v in sorted(counts.items())]})
  del pred,seen;gc.collect()
 except Exception as e:out['errors'].append({'run':row['run'],'row':row['row'],'error':repr(e)})
out['finished']=time.time();print(json.dumps(out))
'''
compile(code,'readonly_frozen_a1_recount','exec')
p=subprocess.run(['ssh','-F',str(B/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=code.encode(),capture_output=True,timeout=600)
if p.returncode:raise RuntimeError(p.stderr.decode('utf-8','replace'))
d=json.loads(p.stdout);(O/'a1_all33_recount.json.gz').write_bytes(gzip.compress(p.stdout));print({'complete':len(d['rows']),'errors':d['errors'],'seconds':d['finished']-d['started']})
