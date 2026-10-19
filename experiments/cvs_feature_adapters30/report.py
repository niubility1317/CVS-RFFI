import time
import numpy as np
from . import design as d

def source_decision():
 values={r['arm']:d.read(d.BASE/r['row_id']/'source_qualification/complete.json') for r in d.rows()}
 urban=lambda a:sum(values[a]['source_accuracies'][i][j] for i in range(2) for j in (1,2))/4
 clean=lambda a:values[a]['full_V_clean_accuracy'];criteria={};qualified=[]
 for style in ('FULL','LORA','TEMP'):
  a,b=style+'_As',style+'_NoAs';du=100*(urban(a)-urban(b));dc=100*(clean(a)-clean(b));full=100*(urban(a)-urban('FULL_As'));passed=du>=.5 and dc>=-.2 and (style=='FULL' or full>=0)
  criteria[style]=dict(urban_As_minus_NoAs_pp=du,fullVclean_As_minus_NoAs_pp=dc,urban_minus_FULL_As_pp=full,qualified=passed)
  if passed:qualified.append(style)
 ordered=sorted(qualified,key=lambda s:(-urban(s+'_As'),-clean(s+'_As'),{'LORA':2304,'TEMP':2384,'FULL':231233}[s]))
 result=dict(status='SOURCE_DECISION_FIXED',timestamp=time.time(),criteria=criteria,eligible_source_order=ordered,preferred=ordered[0] if ordered else None,readouts=values,formal_scores_read=False,automatic_extension=False,max_cumulative_epochs=50,current_cumulative_epochs=30,model_seed_count=1,goal_complete=False,decision='SOURCE_SUPPORTS_FUTURE_MATCHED_EXTENSION_AT_MOST20' if ordered else 'CLOSE_FIXED_R8_K5_LAMBDA1_E10_RECIPE; no rank/kernel/lambda/seed rescue')
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 source=d.read(d.BASE/'source_matrix_decision.json');rows={};pred={};ids=None
 for r in d.rows():
  arm=r['arm'];items=d.read(d.BASE/r['row_id']/'scoring/scores.json')['results'];overall={a['view']:a for a in items if a['dimension']=='overall'}
  rows[arm]=dict(overall=overall,worst_RX={v:min(a['accuracy'] for a in items if a['view']==v and a['dimension']=='receiver') for v in d.VIEWS},cost=d.read(d.BASE/r['row_id']/'source/completion.json'),suburban_gap_pp={v:100*(overall['clean']['accuracy']-overall[v]['accuracy']) for v in d.SCENES[:3]})
  with np.load(d.BASE/r['row_id']/'prediction/predictions.npz',allow_pickle=False) as f:
   if ids is None:ids=f['ids'].copy()
   if not np.array_equal(ids,f['ids']):raise ValueError('Paired formal IDs differ')
   pred[arm]={v:f[v].copy() for v in d.VIEWS}
 # Independent scoring already completed for every row before paired truth access.
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 y,rx,days=truth_arrays(d.read(d.TRUTH),ids);pairs={}
 comparisons=[(s+'_As',s+'_NoAs') for s in ('FULL','LORA','TEMP')]+[(s+'_As','FULL_As') for s in ('LORA','TEMP')]
 for a,b in comparisons:
  records=[]
  for v in d.VIEWS:
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(days))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(len(y),bool) if dim=='overall' else rx==value if dim=='receiver' else days==value if dim=='day' else y==value;pa=pred[a][v][mask];pb=pred[b][v][mask];t=y[mask];repair=int(((pa==t)&(pb!=t)).sum());damage=int(((pa!=t)&(pb==t)).sum());n=int(mask.sum());delta=100*(repair-damage)/n
     records.append(dict(view=v,dimension=dim,stratum=str(value),count=n,repair=repair,damage=damage,net_pp=delta))
     if dim=='overall' and abs(delta-100*(rows[a]['overall'][v]['accuracy']-rows[b]['overall'][v]['accuracy']))>1e-10:raise ValueError('Paired repair/damage recount differs')
  pairs[a+'_minus_'+b]=records
 result=dict(status='ANALYZED',run_id=d.RUN,source_decision=source,rows=rows,paired_comparisons=pairs,model_seed_count=1,crossseed_confirmed=False,goal_complete=False,clean_vs_own_NoAs={s:100*(rows[s+'_As']['overall']['clean']['accuracy']-rows[s+'_NoAs']['overall']['clean']['accuracy']) for s in ('FULL','LORA','TEMP')},low_urban_gt60={s:rows[s+'_As']['overall']['practical_low_urban']['accuracy']>.6 for s in ('FULL','LORA','TEMP')})
 d.write(d.BASE/'analysis.json',result);return result
