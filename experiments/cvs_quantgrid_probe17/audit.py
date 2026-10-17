"""Independent complete saved-statistic recount;no source IQ or model access."""
import argparse,csv,gzip,json,math,re,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
from . import design as d

def audit(inp,out):
 inp,out=Path(inp),Path(out);start=time.perf_counter()
 if out.exists():raise FileExistsError('Preserve independent analysis output')
 c=d.read(inp/'resolved_config.json');d.validate(c);s=d.read(inp/'summary.json');done=d.read(inp/'completion.json');roles=d.read(inp/'source_contract.json');smoke=d.read(inp/'synthetic_noquery_smoke.json')
 if s['status']!='ARTIFACTS_COMPLETE' or done['status']!='ARTIFACTS_COMPLETE' or not s['source_only'] or s['target_access'] or any(s[k] for k in ('checkpoint_reads','model_epochs','optimizer_updates','network_forwards','head_fit_updates','automatic_training','Ag_efficacy_proved','goal_complete')) or smoke!=dict(status='PASS',real_IQ=0,checkpoint_reads=0,NN=0):raise ValueError('Exact complete source-only0NN study required')
 stdout=(inp/'worker.stdout.log').read_text(encoding='utf8',errors='replace');stderr=(inp/'worker.stderr.log').read_text(encoding='utf8',errors='replace')
 if re.search(r'Traceback|RuntimeError:|FloatingPointError:',stdout+stderr):raise ValueError('Preserve technical failure,do not score')
 with gzip.open(inp/'packets.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(line) for line in f]
 if len(rows)!=5040 or len({r['physical_id'] for r in rows})!=5040 or {r['physical_id'] for r in rows}!=set(roles['role_ids']['L_s']):raise ValueError('All exact fitL5040,none filtered')
 cells=defaultdict(list);reason=Counter()
 for r in rows:
  m=r['measurement'];key=(r['TX'],r['RX'],r['day']);cells[key].append(m);reason[m['reason']]+=1
  if m['observable'] and (not 0<m['q_over_component_RMS']<=.5 or not math.isfinite(m['q']) or m['q']<=0 or m['fit_p99']>.02 or m['unique_values']<32):raise ValueError('Wrong finite recorded grid criterion')
  if not m['observable'] and any(m[k] is not None for k in ('q','q_over_component_RMS','fit_p99')):raise ValueError('Missing-grid metrics must remainmissing')
 if set(cells)!={(t,r,dd) for t in d.TX for r in d.FIT for dd in d.DAYS} or any(len(v)!=70 for v in cells.values()):raise ValueError('All72cells70each required')
 cr=[];z={}
 for (t,r,dd),vv in sorted(cells.items()):
  good=[m for m in vv if m['observable']];q=[m['q_over_component_RMS'] for m in good];cr.append(dict(TX=t,RX=r,day=dd,count=70,observable=len(good),fraction=len(good)/70,median_relative_grid=statistics.median(q) if q else None))
  if q:z[t,r,dd]=statistics.median([math.log(x) for x in q])
 full=all(v['fraction']>=.95 for v in cr);folds=[]
 if full:
  for t in d.TX:
   for dd in d.DAYS:
    train=[(key,v) for key,v in z.items() if key[0]!=t and key[2]!=dd];mean=statistics.mean(v for k,v in train);byrx={r:statistics.mean(v for k,v in train if k[1]==r) for r in d.FIT};base=sum((z[t,r,dd]-mean)**2 for r in d.FIT);risk=sum((z[t,r,dd]-byrx[r])**2 for r in d.FIT);folds.append(dict(held_TX=t,held_day=dd,baseline_squared_risk=base,RX_squared_risk=risk,positive=risk<base))
 base=sum(x['baseline_squared_risk'] for x in folds);red=1-sum(x['RX_squared_risk'] for x in folds)/base if folds and base>0 else None;snr=[r['measurement']['equivalent_independent_uniform_dither_SNR_db'] for r in rows if r['measurement']['observable']];med=statistics.median(snr) if snr else None;checks=dict(each_cell_observable=full,TX_day_folds=len(folds)==18 and sum(v['positive'] for v in folds)>=15,RX_risk_reduction=red is not None and red>=.2,material_grid=full and med is not None and med<=50.)
 expected_status='PASS_SOURCE_GRID_PREMISE_ONLY' if all(checks.values()) else 'FAIL_FIXED_QUANTGRID_PREMISE_CLOSED'
 if s['checks']!=checks or s['scientific_status']!=expected_status or s['observable_count']!=sum(r['measurement']['observable'] for r in rows) or len(s['crossfit_folds'])!=len(folds):raise ValueError('Independent wholecohort conclusion mismatch')
 for actual,expected in zip(s['cells'],cr):
  for k in ('TX','RX','day','count','observable','fraction'):
   if actual[k]!=expected[k]:raise ValueError('Independent cell recount mismatch')
  aa,bb=actual['median_relative_grid'],expected['median_relative_grid']
  if (aa is None)!=(bb is None) or (aa is not None and not math.isclose(aa,bb,rel_tol=1e-10,abs_tol=1e-12)):raise ValueError('Cell grid median mismatch')
 for a,b in zip(s['crossfit_folds'],folds):
  if any(a[k]!=b[k] for k in ('held_TX','held_day')) or any(not math.isclose(a[k],b[k],rel_tol=1e-9,abs_tol=1e-12) for k in ('baseline_squared_risk','RX_squared_risk')):raise ValueError('Independent complete crossfit risk mismatch')
  if a['positive']!=b['positive'] and not math.isclose(a['baseline_squared_risk'],a['RX_squared_risk'],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Substantive crossfit positive flag mismatch')
 for key,val in [('crossfit_risk_reduction',red),('median_equivalent_uniform_dither_SNR_db',med)]:
  a=s[key]
  if (a is None)!=(val is None) or (a is not None and not math.isclose(a,val,rel_tol=1e-9,abs_tol=1e-12)):raise ValueError('Independent global scalar mismatch')
 out.mkdir();result=dict(status='ANALYZED',execution='VERIFIED',run_id=d.RUN,source_only=True,target_access=False,scientific_status=expected_status,checks=checks,observable_count=s['observable_count'],count=5040,cells=cr,reasons=dict(reason),crossfit_folds=folds,crossfit_risk_reduction=red,median_equivalent_uniform_dither_SNR_db=med,source_wall_seconds=s['wall_seconds'],offline_recount_wall_seconds=time.perf_counter()-start,source_IQ_packets=5040,source_cells=72,new_NN=0,updates=0,model_epochs=0,checkpoint_reads=0,head_fit_updates=0,automatic_training=False,Ag_efficacy_proved=False,goal_complete=False,limits=['Processed eq1/RMS representation;not rawADC or bitdepth','Only axis-aligned affine grids in fixed resolvable range;absence doesnot exclude rotated or nonuniform quantization','No identitypredictions/newAg/As/joint/target/crossseed proof'])
 d.write(out/'analysis_summary.json',result)
 with (out/'source_cells.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(cr[0]));w.writeheader();w.writerows(cr)
 prefix=[];n=0
 for i,r in enumerate(rows,1):
  n+=int(r['measurement']['observable'])
  if i%64==0 or i==5040:prefix.append(dict(processed=i,observable=n,fraction=n/i))
 d.write(out/'complete_measurement_prefix.json',prefix);print(json.dumps({k:v for k,v in result.items() if k not in ('cells','crossfit_folds')}));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.input,a.output)
