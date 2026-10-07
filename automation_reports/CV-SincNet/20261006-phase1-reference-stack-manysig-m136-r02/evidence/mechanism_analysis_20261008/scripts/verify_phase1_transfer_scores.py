from pathlib import Path
import json,csv,statistics as st
import numpy as np
P=Path('E:/type10-7');OUT=P/'local_artifacts/phase1_transfer_analysis_20261008'
records=[]
for run in ['20261006-phase1-reference-completed-sixscene-manysig-m32-r01','20261007-phase1-reference-completed-sixscene-manysig-m48-r01']:
 p=P/'automation_reports/CV-SincNet'/run/'evidence/completed_test/scores.json'
 records.extend(json.loads(p.read_text(encoding='utf-8'))['results'])
for r in records:
 if 'confusion' not in r:continue
 c=np.asarray(r['confusion'],float);n=c.sum();diag=c.diagonal();den=c.sum(0)+c.sum(1)
 assert abs(diag.sum()/n-r['accuracy'])<1e-12
 assert abs(np.divide(2*diag,den,out=np.zeros_like(diag),where=den>0).mean()-r['macro_f1'])<1e-12
 assert n==r['query_count']
lookup={(r['row_id'],r['view'],r['dimension'],r['stratum']):r for r in records}
assert len(lookup)==len(records)==7840
rows=[];tx=['14-10','14-7','20-15','20-19','6-15','8-20']
for stage,arm,base in [('r2','pseudo','ema'),('r3','domain','base'),('r3','cons','domain'),('r3','fishr','domain'),('r3','orth','domain'),('r3','all_dg','domain'),('r4','episode','base'),('r4','proto','base')]:
 for view in ['clean','practical_low_urban']:
  for dimension,strata in [('overall',['ALL']),('receiver',['1-1','14-7','2-1','20-1','7-14','7-7','8-8'])]:
   for stratum in strata:
    a=[]
    for seed in range(2026092701,2026092705):
     x=lookup[(f'{stage}-{arm}-s{seed}',view,dimension,stratum)];b=lookup[(f'{stage}-{base}-s{seed}',view,dimension,stratum)]
     a.append(100*(x['accuracy']-b['accuracy']))
    rows.append(dict(stage=stage,arm=arm,base=base,view=view,dimension=dimension,stratum=stratum,delta_pp=st.mean(a),sd_pp=st.stdev(a),wins=sum(v>0 for v in a)))
  for i,t in enumerate(tx):
   a=[]
   for seed in range(2026092701,2026092705):
    x=np.asarray(lookup[(f'{stage}-{arm}-s{seed}',view,'overall','ALL')]['confusion']);b=np.asarray(lookup[(f'{stage}-{base}-s{seed}',view,'overall','ALL')]['confusion'])
    a.append(100*(x[i,i]/x[i].sum()-b[i,i]/b[i].sum()))
   rows.append(dict(stage=stage,arm=arm,base=base,view=view,dimension='TX_recall',stratum=t,delta_pp=st.mean(a),sd_pp=st.stdev(a),wins=sum(v>0 for v in a)))
with (OUT/'mechanism_increment_rx_tx.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'score_validation.json').write_text(json.dumps(dict(status='VERIFIED',records=len(records),unique_records=len(lookup),models=len({r['row_id'] for r in records}),checks=['accuracy','macro_f1','query_count','unique_key'],prediction_recomputed=False),indent=2),encoding='utf-8')
print('INCREMENTS',[r for r in rows if r['dimension']=='overall'])
print('R3 DOMAIN TX',[r for r in rows if r['stage']=='r3' and r['arm']=='domain' and r['dimension']=='TX_recall'])
