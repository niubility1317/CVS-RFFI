"""Recount existing frozen FULL9 predictions, without inference or remote mutation."""
from pathlib import Path
import subprocess,json,gzip
R=Path('E:/type10-7');O=Path(__file__).resolve().parent
CODE=r'''from pathlib import Path
import json,time,gzip,sys,hashlib,hmac,gc
from collections import defaultdict
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_xuc_full_s392005_20260913_r1')
rows=['F-A1','F-M00','F-M05','F-M07','F-M08','F-M11','F-M12','F-M13','F-M14']
scores={r:json.loads((root/r/'target_prediction/score.json').read_text()) for r in rows}
assert all(s['record_count']==672000 and Path(s['prediction_path']).is_file() for s in scores.values())
assert len({s['truth_path'] for s in scores.values()})==1
truth=json.loads(Path(scores[rows[0]]['truth_path']).read_text());labels={r['sample_id']:int(r['label']) for r in truth['records']}
key=hashlib.sha256(('cvs.phase1.truth-last|'+truth['split_binding']).encode()).digest();meta={}
for tx in range(6):
 for rx in [0,2,5,7,9,10,11]:
  for day in range(4):
   for sig in range(1000):
    sid='sample:'+hmac.new(key,f'tx{tx}:rx{rx}:day{day}:eq1:sig{sig}'.encode(),hashlib.sha256).hexdigest();meta[sid]=(tx,rx,day)
assert set(meta)==set(labels) and all(labels[k]==v[0] for k,v in meta.items())
out={'started':time.time(),'scope':'read-only frozen prediction recount; no inference','rows':{}};base={}
for row in rows:
 score=scores[row];pred=json.loads(Path(score['prediction_path']).read_text());seen=set();conf=defaultdict(lambda:[[0]*6 for _ in range(6)]);groups=defaultdict(lambda:[0,0]);pairs=defaultdict(lambda:dict(rescue=0,harm=0,both_correct=0,both_wrong=0))
 for rec in pred['records']:
  sid=rec['sample_id'];sc=rec['scenario'];p=int(rec['predicted_class']);tx,rx,day=meta[sid];k=(sc,sid)
  assert k not in seen and 0<=p<6;seen.add(k);ok=int(p==tx);conf[sc][tx][p]+=1
  for axis,g in [('scene','all'),('rx',str(rx)),('day',str(day)),('tx',str(tx)),('rx_day',f'{rx}:{day}'),('rx_day_tx',f'{rx}:{day}:{tx}')]:
   v=groups[sc,axis,g];v[0]+=ok;v[1]+=1
  if row=='F-A1':base[k]=ok
  else:
   b=base[k];pairs[sc]['both_correct' if b and ok else 'rescue' if ok else 'harm' if b else 'both_wrong']+=1
 assert len(seen)==672000 and set(conf)==set(score['metrics'])
 for sc,m in score['metrics'].items():
  assert groups[sc,'scene','all']==[m['correct'],m['total']]
  assert abs(m['correct']/m['total']-m['accuracy'])<1e-12
  assert {sid for s,sid in seen if s==sc}==set(labels)
 macro={sc:sum(2*cm[c][c]/(sum(cm[c])+sum(r[c] for r in cm)) for c in range(6))/6 for sc,cm in conf.items()}
 out['rows'][row]={'score':score,'count':len(seen),'id_coverage_exact':True,'confusion':dict(conf),'macro_f1':macro,'paired_vs_FA1':dict(pairs),'groups':[dict(scene=s,axis=a,group=g,correct=v[0],total=v[1],accuracy=v[0]/v[1]) for (s,a,g),v in sorted(groups.items())]}
 del pred,seen;gc.collect()
out['finished']=time.time();out['total_prediction_records']=sum(x['count'] for x in out['rows'].values());out['status']='VERIFIED'
sys.stdout.buffer.write(gzip.compress(json.dumps(out).encode()))
'''
compile(CODE,'full9_readonly_recount','exec')
p=subprocess.run(['ssh','-F',str(R/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=CODE.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
d=json.loads(gzip.decompress(p.stdout));(O/'full9_recount.json.gz').write_bytes(p.stdout)
print({'status':d['status'],'rows':len(d['rows']),'records':d['total_prediction_records'],'seconds':d['finished']-d['started']})
