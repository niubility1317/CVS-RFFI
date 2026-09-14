"""Read-only recount after all six predictions are fixed and independently scored."""
import hashlib,hmac,json,time
from pathlib import Path
from collections import defaultdict

P=Path('/home/szu2070436088/2510044040/CV-SincNet')
EVAL='phase1_adv3b02_xuc_full6_eval_s392005_20260914_r1'
ROWS=['F-A1','F-M14','F-M11','F-M05','F-M08','F-M12']
state=json.loads((P/'runs'/EVAL/'evaluation_state.json').read_text())
assert state['status']=='COMPLETE' and set(state['rows'])==set(ROWS)
assert all(v['status']=='SCORED' for v in state['rows'].values())
truth=json.loads((P/'runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/target_truth/truth_sidecar.json').read_text())
t={r['sample_id']:r['label'] for r in truth['records']}
key=hashlib.sha256(('cvs.phase1.truth-last|'+truth['split_binding']).encode()).digest()
meta={}
for tx in range(6):
 for rx in [0,2,5,7,9,10,11]:
  for day in range(4):
   for sig in range(1000):
    sid='sample:'+hmac.new(key,f'tx{tx}:rx{rx}:day{day}:eq1:sig{sig}'.encode(),hashlib.sha256).hexdigest()
    meta[sid]=(tx,rx,day)
assert set(meta)==set(t) and all(t[k]==v[0] for k,v in meta.items())
out=dict(metadata_mapping_exact=True,physical_records=len(t),new_prediction_count=4032000,rows={})
baselines={}
for row in ['M00','F-A1','M09','M10',*ROWS[1:]]:
 folder=P/'runs'/EVAL/row if row.startswith('F-') else P/'runs/phase1_adv3b02_xuc15_s392005_20260913_r1'/row/'target_prediction'
 prediction=json.loads((folder/'predictions.json').read_text());score=json.loads((folder/'score.json').read_text())
 counts=defaultdict(lambda:[0,0]);conf=defaultdict(lambda:[[0]*6 for _ in range(6)])
 pairs=defaultdict(lambda:dict(both_correct=0,rescue=0,harm=0,both_wrong=0))
 seen=set();correct_map={}
 for rec in prediction['records']:
  sid=rec['sample_id'];scene=rec['scenario'];pred=rec['predicted_class'];tx,rx,day=meta[sid]
  pk=(scene,sid);assert pk not in seen;seen.add(pk)
  assert 0<=pred<6
  ok=int(pred==tx);conf[scene][tx][pred]+=1
  if row in ['M00','F-A1']:correct_map[pk]=ok
  for scope,k in [('scene','all'),('rx',str(rx)),('day',str(day)),('tx',str(tx)),('rx_day',f'{rx}:{day}'),('rx_tx',f'{rx}:{tx}')]:
   c=counts[(scene,scope,k)];c[0]+=ok;c[1]+=1
  for base,correct in baselines.items():
   b=correct[pk];pairs[(base,scene)]['both_correct' if b and ok else 'rescue' if ok else 'harm' if b else 'both_wrong']+=1
 assert len(seen)==672000
 for scene,m in score['metrics'].items():assert counts[(scene,'scene','all')]==[m['correct'],m['total']]
 if correct_map:baselines[row]=correct_map
 macro={}
 for scene,cm in conf.items():
  f1=[];precision=[];recall=[]
  for c in range(6):
   tp=cm[c][c];n=sum(cm[c]);p=sum(r[c] for r in cm)
   precision.append(tp/p if p else 0);recall.append(tp/n);f1.append(2*tp/(n+p) if n+p else 0)
  macro[scene]=dict(macro_f1=sum(f1)/6,macro_precision=sum(precision)/6,macro_recall=sum(recall)/6,
   per_class_precision=precision,per_class_recall=recall,per_class_f1=f1)
 out['rows'][row]=dict(source_folder=str(folder),record_count=len(seen),score=score,confusion=dict(conf),macro=macro,
  prediction_record_fields=sorted(prediction['records'][0]),
  paired=[dict(baseline=b,scene=s,**v,net_correct=v['rescue']-v['harm'],delta_pp=(v['rescue']-v['harm'])/1680) for (b,s),v in sorted(pairs.items())],
  details=[dict(scene=s,scope=sc,group=g,correct=v[0],total=v[1],accuracy=v[0]/v[1]) for (s,sc,g),v in sorted(counts.items())])
 del prediction,seen
out['created']=time.time()
print(json.dumps(out))
