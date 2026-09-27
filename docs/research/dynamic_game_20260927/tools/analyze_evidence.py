"""Recompute descriptive statistics from frozen CSVs, without training or selection."""
from pathlib import Path
import csv,json,statistics,collections,gzip,math
ROOT=Path('E:/type10-7'); OUT=ROOT/'analysis/dynamic_game_audit_20260927'
RES=ROOT/'automation_reports/CV-SincNet/response_matrix_eval_20260917/results'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(RES/'summary.csv'); curves=read(RES/'source_curves.csv'); by={(r['method'],int(r['seed'])):r for r in rows}
metrics=['clean_accuracy','leo_mean_accuracy','leo_mean_macro_f1','worst_RX_leo_mean','worst_TX_leo_mean','training_hours']
summary=[]
for method in sorted({r['method'] for r in rows}):
 rr=[r for r in rows if r['method']==method]; x={'method':method,'n':len(rr),'seeds':[r['seed'] for r in rr]}
 for k in metrics:
  v=[float(r[k]) for r in rr];x[k]={'mean':statistics.mean(v),'sd':statistics.stdev(v) if len(v)>1 else None}
 summary.append(x)
contrasts=[]
for a,b in [('EG','SIM'),('TR_EG','EG'),('XT_DANN','EG'),('CF_EG','EG'),('DRIC','SIM'),('DRIC','EG'),('R0-11','R0-10'),('R0-01','R0-00'),('R0-10','R0-00'),('R0-11','ADV0-1')]:
 seeds=sorted(s for m,s in by if m==a and (b,s) in by)
 vals=[100*(float(by[a,s]['leo_mean_accuracy'])-float(by[b,s]['leo_mean_accuracy'])) for s in seeds]
 contrasts.append({'a':a,'b':b,'seeds':seeds,'delta_pp':vals,'mean_pp':statistics.mean(vals),'sd_pp':statistics.stdev(vals) if len(vals)>1 else None})
inter=[]
for s in [392005,392006,392007]:
 if all((m,s) in by for m in ['R0-00','R0-01','R0-10','R0-11']):
  v={m:float(by[m,s]['leo_mean_accuracy'])*100 for m in ['R0-00','R0-01','R0-10','R0-11']}
  inter.append({'seed':s,**v,'interaction_pp':v['R0-11']-v['R0-10']-v['R0-01']+v['R0-00']})
checks=[]
for r in rows:
 assert abs(float(r['leo_mean_accuracy'])-statistics.mean(float(r[k+'_accuracy']) for k in ['leo_clear_weak','leo_low_elev_weak','leo_rain_weak']))<1e-12
 rr=[x for x in curves if x['row_id']==r['row']]
 assert len(rr)==200 and [int(x['epoch']) for x in rr]==list(range(1,201))
 assert int(rr[-1]['total_step'])==44400
 checks.append({'row':r['row'],'epochs':len(rr),'first_loss':float(rr[0]['mean_loss']),'last_loss':float(rr[-1]['mean_loss']),'min_V':min(float(x['source_V_accuracy']) for x in rr),'final_V':float(rr[-1]['source_V_accuracy']),'first_V95_epoch':next((int(x['epoch']) for x in rr if float(x['source_V_accuracy'])>=.95),None),'nonfinite_loss':sum(not math.isfinite(float(x['mean_loss'])) for x in rr)})
out={'frozen_source':'response_matrix_eval_20260917','rows':len(rows),'curve_records':len(curves),'methods':summary,'paired':contrasts,'interaction':inter,'curves':checks,'scope':'All 6600 source epoch CSV records parsed. Prior prediction recount evidence reused; no predictions rerun.'}
(OUT/'response_recomputed.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('PAIRED',json.dumps(contrasts,ensure_ascii=False));print('INTERACTION',inter)
print('METHODS',[(r['method'],r['n'],r['leo_mean_accuracy'],r['training_hours']['mean'],r['worst_TX_leo_mean']['mean']) for r in summary])
print('DAY0',read(RES/'day0_seed_summary.csv')[:3]);print('GROUPS HEADER',list(read(RES/'all_groups.csv')[0]))
for rel in ['automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/report_bundle','code/snapshots/native_dr_eg_prepare_20260914_wt/experiments/adv3b02_xuc','automation_reports/CV-SincNet/all_exploration_20260913']:
 p=ROOT/rel;counts=collections.defaultdict(lambda:[0,0]);large=[]
 for f in p.rglob('*'):
  if f.is_file():
   z=f.stat().st_size;key=f.relative_to(p).parts[0];counts[key][0]+=1;counts[key][1]+=z
   if z>10000000:large.append((str(f.relative_to(p)),z))
 print('SIZE',rel,dict(counts),'large',large[:15])
