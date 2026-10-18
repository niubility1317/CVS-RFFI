"""Small stdlib checks for the declared source/data/head/budget contract."""
import math
from collections import Counter
from pathlib import Path
from . import design as d

def check_config(c):
 if c.get('run_id')!=d.RUN or c.get('reference_run')!=d.REFERENCE_RUN or c.get('reference_commit')!=d.REFERENCE_COMMIT or c.get('model_seed')!=d.SEED or c.get('model_seed_count')!=1 or c.get('checkpoint_state')!='model' or c.get('EMA_loaded') is not False:raise ValueError('Only own P43 seed01 fixed FINAL30 student/default head')
 if c.get('source_counts')!=dict(L_s=6300,U_s=56700,V=27000) or c.get('source_RX')!=list(d.RX) or c.get('source_days')!=list(d.DAYS) or c.get('panel_role')!='V' or c.get('panel_count')!=d.N:raise ValueError('Only the existing all-five-RX source V panel')
 if tuple(c.get('scenes',()))!=d.SCENES or tuple(c.get('conditions',()))!=d.CONDITIONS or c.get('draws')!=list(d.DRAWS) or tuple(c.get('IQ_shape',()))!=d.IQ_SHAPE:raise ValueError('Fixed existing source view order/shape required')
 for key in ('training_epochs','optimizer_updates','EMA_updates','head_fit_updates','new_IQ_generated','new_channel_draws'):
  if c.get(key)!=0:raise ValueError('No diagnostic training or regenerated IQ: '+key)
 for key in ('EMA_loaded','U_iteration','U_truth_read','query_access','formal_scores_read','source_promotion','automatic_training','automatic_formal','automatic_budget_extension','formal_test_changed'):
  if c.get(key) is not False:raise ValueError('Source-only descriptive scope violated: '+key)
 if (c.get('prediction_NN_per_row'),c.get('gradient_NN_per_row'),c.get('smoke_NN_per_row'),c.get('expected_NN_per_row'),c.get('expected_NN'))!=(120600,630,2,121232,242464) or c.get('batch_size')!=90 or c.get('gradient_samples')!=90:raise ValueError('Fixed diagnostic forward/gradient budget differs')
 return c

def check_source_metadata(project,arm,c,done,frozen,init,roles,canonical):
 if arm not in d.ARMS:raise ValueError('Unknown source arm')
 expected=d.source(project,arm);parent=expected.parent/'stage20';row=arm+'-s'+str(d.SEED)
 if (c.get('run_id'),c.get('row_id'),c.get('model_seed'),c.get('segment'),c.get('epochs'),c.get('updates'),c.get('epoch_offset'),c.get('step_offset'))!=(d.REFERENCE_RUN,row,d.SEED,30,10,2220,20,4440) or c.get('As') is not (arm=='B01'):raise ValueError('P43 own20+10 final identity differs')
 if Path(c.get('output_root',''))!=expected or Path(c.get('parent_source',''))!=parent or c.get('checkpoint_sources')!=[str(parent/'final_ssdg.pth')]:raise ValueError('Only corresponding own P43 stage20/final source paths')
 if (done.get('epoch'),done.get('optimizer_steps'))!=(10,2220) or done.get('target_access') is not False or done.get('head_fit_updates',0)!=0 or done.get('config')!=c:raise ValueError('Actual fixed FINAL30 training completion differs')
 if frozen.get('selection')!='final_only' or frozen.get('target_access') is not False or frozen.get('config')!=c:raise ValueError('Source-frozen final-only state required')
 if init.get('scratch_only') is not False or init.get('target_contact') is not False or init.get('source_roles')!='EXACT_MATCH' or init.get('checkpoint_sources')!=[str(parent/'final_ssdg.pth')] or init.get('ancestors')!=[str(parent)]:raise ValueError('Own P43 final inheritance differs')
 if roles.get('role_ids')!=canonical.get('role_ids'):raise ValueError('Original source role handles differ')

def check_head(shape,scale,parameters,state_entries):
 if tuple(shape)!=(6,160) or scale!=30. or parameters!=900722 or state_entries!=206:raise ValueError('Unchanged full F/default cosine head required')

def check_roles(roles):
 if roles.get('role')!='V' or roles.get('counts')!=dict(L_s=6300,U_s=56700,V=27000) or roles.get('V_panel')!=d.N or roles.get('held_RX') is not None or roles.get('extra_Ldev',False) or roles.get('U_truth_used') is not False or roles.get('target_access') is not False or roles.get('updates')!=0:raise ValueError('Existing legal V only; no U truth, held RX or updates')
 ids=roles.get('physical_ids',[])
 if len(ids)!=d.N or len(set(ids))!=d.N or any(not isinstance(x,str) for x in ids):raise ValueError('Unique existing canonical source panel IDs required')
 return ids

def check_panel(roles,ids,labels,rx,days,shape):
 expected=check_roles(roles)
 if list(ids)!=expected or tuple(shape)!=(d.N,2,256) or any(len(values)!=d.N for values in (labels,rx,days)):raise ValueError('Existing ordered V panel and IQ256 shape required')
 cells=Counter(zip(labels,rx,days))
 if cells!=Counter({(tx,r,day):20 for tx in range(6) for r in d.RX for day in d.DAYS}):raise ValueError('Original 90 source TX/RX/day cells required')

def check_observations(meta,shape,dtype,c_contiguous=True):
 if meta.get('status')!='SOURCE_OBSERVATIONS_COMPLETE' or meta.get('unique_V')!=d.N or meta.get('draws')!=2 or tuple(meta.get('scenes',()))!=d.SCENES or tuple(meta.get('conditions',()))!=d.CONDITIONS or meta.get('physical_draws')!=10800 or meta.get('training_updates')!=0 or meta.get('target_access') is not False or meta.get('formal_test_changed') is not False:raise ValueError('Existing complete source observations/order required')
 if tuple(shape)!=d.IQ_SHAPE or str(dtype)!='float32' or not c_contiguous:raise ValueError('Existing source float32 C-order IQ cache shape differs')

def check_physics_headers(headers):
 if set(headers)!=set(d.PHYSICS_FIELDS) or any(tuple(value['shape'])!=d.PHYSICS_SHAPE or value['kind'] not in ('i','u','f') for value in headers.values()):raise ValueError('Existing numeric source physics matrix/order differs')

def check_prediction_shapes(clean,satellite,branch,names,metrics,all_finite):
 if tuple(clean)!=(d.N,6) or tuple(satellite)!=(2,3,11,d.N,6) or tuple(branch)!=(2,3,d.N,5,3) or tuple(names)!=d.BRANCHES or tuple(metrics)!=d.BRANCH_METRICS or all_finite is not True:raise ValueError('Finite default-head/actual five-branch outputs required')

def finite(value,name):
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Nonfinite source diagnostic scalar: '+name)

def check_gradients(rows):
 if len(rows)!=6 or [(r.get('draw'),r.get('scene')) for r in rows]!=[(draw,scene) for draw in d.DRAWS for scene in d.SCENES]:raise ValueError('Exactly six full-condition gradient readouts required')
 for r in rows:
  if r.get('count')!=90 or set(r.get('groups',{}))!=set(d.GRADIENT_GROUPS) or 'not stochastic training gradients' not in r.get('scope',''):raise ValueError('Eval-mode source gradient scope/group/count differs')
  finite(r['clean_CE'],'cleanCE');finite(r['satellite_CE'],'satelliteCE')
  for key,value in r['groups'].items():
   if value.get('finite') is not True:raise ValueError('Nonfinite reported gradient group')
   for name in ('clean_norm','satellite_norm','dot'):finite(value[name],key+'/'+name)
   if value['clean_norm']<0 or value['satellite_norm']<0:raise ValueError('Negative gradient norm')
   if value.get('cosine') is not None:finite(value['cosine'],key+'/cosine')

def check_budget(eval_packets,train_packets):
 if eval_packets!=d.ROW_NN or train_packets!=0:raise ValueError('Readonly per-row diagnostic NN budget differs')
