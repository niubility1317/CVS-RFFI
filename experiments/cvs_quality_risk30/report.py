"""Four frozen same-seed conditions;source-only eligibility never changes fixed training."""
from pathlib import Path
import numpy as np
from . import design as d

def logical_rows():
 ref=Path(d.PROJECT)/'runs'/d.REFERENCE_RUN
 return [('NOAS30',ref/'B00-s2026092701'),('AS30',ref/'B01-s2026092701'),('BANK_IID30',d.BASE/'I01-s2026092701'),('BANK_STRAT30',d.BASE/'S01-s2026092701')]
PAIRS=(('BANK_IID30','NOAS30'),('BANK_STRAT30','NOAS30'),('BANK_IID30','AS30'),('BANK_STRAT30','AS30'),('BANK_STRAT30','BANK_IID30'),('AS30','NOAS30'))
def stats(y,p,rx):
 cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 return dict(accuracy=float((p==y).mean()),macro_F1=float(f.mean()),worst_RX=float(min((p[rx==r]==y[rx==r]).mean() for r in np.unique(rx))))
def delta(a,b,y):
 gooda=a==y;goodb=b==y;repair=int((gooda&~goodb).sum());new=int((~gooda&goodb).sum())
 return dict(repaired=repair,new_errors=new,net=repair-new,accuracy_pp=100*(repair-new)/len(y),count=len(y))

def source_decision():
 rows={};predictions={};source_arrays={};base_ids=None;base_y=None;base_rx=None
 with np.load(d.PANEL/'observations/physics.npz',allow_pickle=False) as f:quality=f['signal_to_added_noise_db'].copy()
 if quality.shape!=(2,3,11,1800) or not np.isfinite(quality[:,:,0]).all():raise ValueError('Existing full source panel added-noise metadata required')
 for arm,root in logical_rows():
  done=d.read(root/'source/completion.json');sc=d.read(root/'source/source_contract.json');frozen=d.read(root/'source/source_frozen.json');q=d.read(root/'source_qualification/complete.json')
  if sc['role_ids']!=d.read(d.SOURCE)['role_ids'] or done['target_access'] or frozen['target_access'] or q['target_access'] or done['final_source_metrics']['source_val_count']!=27000:raise ValueError('Exact frozen fullsource rows required')
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as a:
   ids=a['ids'];y=a['TX'];rx=a['RX'];clean=a['clean'];sat=a['satellite']
   if base_ids is None:base_ids=ids.copy();base_y=y.copy();base_rx=rx.copy()
   if len(ids)!=1800 or not np.array_equal(ids,base_ids) or not np.array_equal(y,base_y) or not np.array_equal(rx,base_rx) or clean.shape!=(1800,6) or sat.shape!=(2,3,1800,6) or not np.isfinite(clean).all() or not np.isfinite(sat).all():raise ValueError('Same fixed source panel required')
   preds=dict(clean=clean.argmax(-1))
   for i,s in enumerate(('practical_low_suburban','practical_mid_urban','practical_low_urban')):preds[s]=sat[:,i].argmax(-1).reshape(-1)
   panel={v:stats(y if v=='clean' else np.tile(y,2),p,rx if v=='clean' else np.tile(rx,2)) for v,p in preds.items()}
  rows[arm]=dict(full_V_clean=done['final_source_metrics']['source_val_accuracy'],full_V_RX_accuracy=done['final_source_metrics']['source_val_rx_accuracy'],panel=panel,mid_low_mean=float(np.mean([panel[s]['accuracy'] for s in ('practical_mid_urban','practical_low_urban')])))
  predictions[arm]=preds;source_arrays[arm]=dict(y=y.copy(),satellite=sat.argmax(-1))
 pairs={}
 for a,b in PAIRS:
  pairs[a+'_minus_'+b]={v:dict(delta(predictions[a][v],predictions[b][v],y if v=='clean' else np.tile(y,2)),macro_F1_pp=100*(rows[a]['panel'][v]['macro_F1']-rows[b]['panel'][v]['macro_F1']),worst_RX_pp=100*(rows[a]['panel'][v]['worst_RX']-rows[b]['panel'][v]['worst_RX'])) for v in predictions[a]}
 fullV={a+'_minus_'+b:100*(rows[a]['full_V_clean']-rows[b]['full_V_clean']) for a,b in PAIRS}
 quality_groups={}
 for arm,a in source_arrays.items():
  result={}
  for key,hard in [('hard',True),('easy',False)]:
   scene_values=[];counts=[]
   for si in (1,2):
    mask=quality[:,si,0]<8 if hard else quality[:,si,0]>=8;truth=np.broadcast_to(a['y'],(2,1800));ok=a['satellite'][:,si]==truth
    if not mask.any():raise ValueError('Existing fixed source quality cohort empty')
    scene_values.append(float(ok[mask].mean()));counts.append(int(mask.sum()))
   result[key]=dict(equal_midlow_accuracy=float(np.mean(scene_values)),scene_accuracy=scene_values,scene_counts=counts)
  quality_groups[arm]=result
 hard_delta=float(100*(quality_groups['BANK_STRAT30']['hard']['equal_midlow_accuracy']-quality_groups['BANK_IID30']['hard']['equal_midlow_accuracy']));easy_delta=float(100*(quality_groups['BANK_STRAT30']['easy']['equal_midlow_accuracy']-quality_groups['BANK_IID30']['easy']['equal_midlow_accuracy']));clean_delta=float(fullV['BANK_STRAT30_minus_AS30'])
 conditions=dict(hard_S_minus_I_at_least_0p5pp=bool(hard_delta>=.5-1e-12),easy_S_minus_I_at_least_minus0p2pp=bool(easy_delta>=-.2-1e-12),fullV_S_minus_originalAs_at_least_minus0p1pp=bool(clean_delta>=-.1-1e-12));passed=bool(all(conditions.values()))
 promotion=dict(passed=passed,conditions=conditions,fixed_recipe_closed=bool(not passed),M3_source_eligible=passed,hard_S_minus_I_pp=hard_delta,easy_S_minus_I_pp=easy_delta,fullV_S_minus_originalAs_clean_pp=clean_delta,hard_min_pp=.5,easy_min_pp=-.2,clean_min_pp=-.1,source_failure_skips_formal=False,automatic_extension=False,M3_implemented=False)
 value=dict(status='SOURCE_QUALITYRISK_READOUT_COMPLETE',rows=rows,paired=pairs,fullV_clean_paired_pp=fullV,quality_groups=quality_groups,promotion=promotion,source_physics=str(d.PANEL/'observations/physics.npz'),quality_field='signal_to_added_noise_db',hard='less than8dB',easy='atleast8dB',equal_scene_mean=True,source_high_urban_available=False,fullV_F1_or_error_recount_available=False,panel_draws=2,independent_physical_panel_samples=1800,target_scores_read=False,automatic_seed_extension=False,scientific_failure_skips_formal=False,repeatable_gain_proven=False,interpretation='Four frozen same-seed conditions;finite-bank risk attribution onlyS versusI;source criteria only futureM3 eligibility,all7 mandatory;no automatic extension')
 d.write(d.BASE/'source_matrix_decision.json',value);return value

def collect():
 result={};states=[]
 for arm,root in logical_rows():
  pc=d.read(root/'prediction/complete.json');sc=d.read(root/'scoring/complete.json')
  if pc['status']!='PREDICTIONS_COMPLETE' or sc['status']!='SCORED_COMPLETE' or pc['views']!=list(d.VIEWS) or pc['truth_read'] or pc['query_fit']:raise ValueError('All seven fixed predictions and independent scores required')
  scores=d.read(root/'scoring/scores.json')['results'];result[arm]={}
  for view in d.VIEWS:
   records=[v for v in scores if v['view']==view];overall=next(v for v in records if v['dimension']=='overall');rx=[v['accuracy'] for v in records if v['dimension']=='receiver'];result[arm][view]=dict(accuracy=overall['accuracy'],macro_F1=overall['macro_f1'],worst_RX=min(rx))
  result[arm]['urban_mean']=dict(accuracy=float(np.mean([result[arm][v]['accuracy'] for v in d.SCENES[3:]])))
  states.append(dict(arm=arm,root=str(root),reused=arm in ('NOAS30','AS30')))
 pairs={a+'_minus_'+b:{v:100*(result[a][v]['accuracy']-result[b][v]['accuracy']) for v in (*d.VIEWS,'urban_mean')} for a,b in PAIRS}
 value=dict(status='SCORED_COMPLETE_PENDING_FULL_LOG_ANALYSIS',rows=states,summary=result,paired_accuracy_pp=pairs,stage_clean_0p1pp={a:bool(pairs[a+'_minus_NOAS30']['clean']>=-.1-1e-12) for a in ('BANK_IID30','BANK_STRAT30')},single_seed=True,repeated_clean_gain_proven=False,source_decision=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').exists() else None,formal_feedback=False,goal_complete=False)
 d.write(d.BASE/'automatic_summary.json',value);return value
