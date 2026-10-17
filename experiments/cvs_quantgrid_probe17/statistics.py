"""All source cells and fixed leave-TX/day regression of scalar observability."""
from collections import defaultdict
import numpy as np
from . import design as d

def summarize(rows):
 if len(rows)!=5040 or len({r['physical_id'] for r in rows})!=5040:raise ValueError('Complete5040source rows,none dropped')
 cells=defaultdict(list)
 for r in rows:
  if r['RX'] not in d.FIT or r['TX'] not in d.TX or r['day'] not in d.DAYS:raise ValueError('Source fit cells only')
  cells[r['TX'],r['RX'],r['day']].append(r)
 if set(cells)!={(t,r,dd) for t in d.TX for r in d.FIT for dd in d.DAYS} or any(len(v)!=70 for v in cells.values()):raise ValueError('All72cells of70,including failures')
 summary=[];medians={}
 for (t,r,dd),vv in sorted(cells.items()):
  good=[x for x in vv if x['measurement']['observable']];q=[x['measurement']['q_over_component_RMS'] for x in good]
  summary.append(dict(TX=t,RX=r,day=dd,count=70,observable=len(good),fraction=len(good)/70,median_relative_grid=float(np.median(q)) if q else None))
  if q:medians[t,r,dd]=float(np.median(np.log(q)))
 cfg=d.config()['criterion'];coverage=all(q['fraction']>=cfg['each_TX_RX_day_observable_fraction_min'] for q in summary);folds=[]
 if coverage:
  for t in d.TX:
   for dd in d.DAYS:
    train=[(key,val) for key,val in medians.items() if key[0]!=t and key[2]!=dd];overall=float(np.mean([v for k,v in train]));rx={r:float(np.mean([v for k,v in train if k[1]==r])) for r in d.FIT};base=sum((medians[t,r,dd]-overall)**2 for r in d.FIT);risk=sum((medians[t,r,dd]-rx[r])**2 for r in d.FIT)
    folds.append(dict(held_TX=t,held_day=dd,baseline_squared_risk=base,RX_squared_risk=risk,positive=risk<base))
 reduction=1.-sum(x['RX_squared_risk'] for x in folds)/sum(x['baseline_squared_risk'] for x in folds) if folds and sum(x['baseline_squared_risk'] for x in folds)>0 else None
 snr=[r['measurement']['equivalent_independent_uniform_dither_SNR_db'] for r in rows if r['measurement']['observable']];median=float(np.median(snr)) if snr else None
 checks=dict(each_cell_observable=coverage,TX_day_folds=len(folds)==18 and sum(x['positive'] for x in folds)>=cfg['crossfit_TX_day_positive_folds_min'],RX_risk_reduction=reduction is not None and reduction>=cfg['crossfit_squared_risk_reduction_min'],material_grid=coverage and median is not None and median<=cfg['median_equivalent_uniform_dither_SNR_db_max'])
 return dict(cells=summary,observable_count=sum(r['measurement']['observable'] for r in rows),total=5040,crossfit_folds=folds,crossfit_risk_reduction=reduction,median_equivalent_uniform_dither_SNR_db=median,checks=checks,scientific_status='PASS_SOURCE_GRID_PREMISE_ONLY' if all(checks.values()) else 'FAIL_FIXED_QUANTGRID_PREMISE_CLOSED',scope='Observed axis-aligned grid,not originalADC bitdepth/causality;no identity or targetperformance claim;missinggrid packets retained;crossfit skipped if full-cell coverage fails')
