"""Concrete state, source role and paired-ID checks, independently testable."""
from . import design as d

def check_roles(actual,expected,named_tests,counts):
 if named_tests or actual!=expected or counts!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Original all-five-RX L/U/V exact roles required')
 if set(actual)!={'L_s','U_s','V'}:raise ValueError('Only original L/U/V')
 sets={k:set(v) for k,v in actual.items()}
 if any(len(sets[k])!=counts[k] for k in sets) or any(sets[a]&sets[b] for a,b in [('L_s','U_s'),('L_s','V'),('U_s','V')]):raise ValueError('Duplicate/overlapping source physical roles')

def check_state_pair(student,ema,template,finite):
 keys=set(template)
 if len(keys)!=206 or set(student)!=keys or set(ema)!=keys or tuple(template[d.HEAD].shape)!=(6,160):raise ValueError('Existing F206/default6x160 cosine state required')
 for k in keys:
  ref=template[k]
  for state in (student,ema):
   value=state[k]
   if tuple(value.shape)!=tuple(ref.shape) or value.dtype!=ref.dtype or not finite(value):raise ValueError('Current FINAL50 finite state shape/dtype differs: '+k)

def check_ids(ids,rx,days,eq,expected,count):
 if len(ids)!=count or len(set(ids))!=count or set(ids)!=set(expected):raise ValueError('Exact unique source V physical IDs required')
 if len(rx)!=count or len(days)!=count or len(eq)!=count or set(rx)!=set(d.RX) or set(days)!=set(d.DAY) or set(eq)!={1}:raise ValueError('All five source RX/three days/equalized1 required')

def check_panel(a,b):
 import numpy as np
 for k in ('ids','TX','RX','day'):
  if not np.array_equal(a[k],b[k]):raise ValueError('Paired source panel IDs/labels/RX/day differ')
 if a['clean'].shape!=(1800,6) or a['satellite'].shape!=(2,3,1800,6):raise ValueError('Fixed V1800/two draws/three scene logits required')
 if not np.isfinite(a['clean']).all() or not np.isfinite(a['satellite']).all():raise ValueError('Finite panel logits required')

def check_panel_membership(panel,full,count=1800):
 """Join the panel/cache opaque IDs to full-V opaque IDs, not native tuples."""
 import numpy as np
 pids=np.asarray(panel['ids']);fids=np.asarray(full['ids'])
 def opaque(ids):return ids.ndim==1 and all(isinstance(v,str) and len(v)==24 and all(c in '0123456789abcdef' for c in v) for v in ids.tolist())
 if pids.shape!=(count,) or not opaque(pids) or not opaque(fids) or len(set(pids.tolist()))!=count or len(set(fids.tolist()))!=len(fids):raise ValueError('Unique canonical opaque source IDs required; native five-tuples are separate')
 lookup={pid:i for i,pid in enumerate(fids.tolist())}
 if not set(pids.tolist())<=set(lookup):raise ValueError('Original V1800 must be physical subset of fullV')
 indices=np.asarray([lookup[pid] for pid in pids.tolist()],np.int64)
 for key in ('TX','RX','day'):
  if np.asarray(panel[key]).shape!=(count,) or not np.array_equal(np.asarray(panel[key]),np.asarray(full[key])[indices]):raise ValueError('Panel opaque-ID aligned '+key+' differs from fullV')

def check_observation_schema(meta,shape,dtype,fortran):
 """Exact original cache axes: draw, scene, eleven controls, V, IQ, time."""
 import numpy as np
 if tuple(shape)!=d.OBS_SHAPE or np.dtype(dtype)!=np.dtype('float32') or fortran:raise ValueError('Original source cache must be C-order float32[2,3,11,1800,2,256]')
 if meta['status']!='SOURCE_OBSERVATIONS_COMPLETE' or meta['draws']!=2 or meta['unique_V']!=1800 or meta['scenes']!=list(d.SCENES) or meta['conditions']!=list(d.OBS_CONDITIONS) or meta['target_access'] or meta['training_updates']:raise ValueError('Original completed source observation conditions differ')
 if meta['conditions'][d.FULL_CONDITION_INDEX]!='full':raise ValueError('Only original condition index0 full')
 return d.FULL_CONDITION_INDEX

def decision(fullv,panel):
 clean_delta=100*(fullv['AS_EMA']['accuracy']-fullv['NOAS_EMA']['accuracy'])
 clean_net=fullv['AS_EMA']['hits']-fullv['AS_STUDENT']['hits']
 midlow_net=sum(panel['AS_EMA'][s]['hits']-panel['AS_STUDENT'][s]['hits'] for s in d.SCENES[1:])
 criteria=dict(matched_EMA_fullV_clean_within_0p1pp=clean_delta>=-.1-1e-12,
  AS_EMA_fullV_clean_net_repair_nonnegative=clean_net>=0,AS_EMA_midlow_equal_weight_net_repair_positive=midlow_net>0)
 return dict(passed=all(criteria.values()),criteria=criteria,matched_EMA_fullV_clean_delta_pp=clean_delta,
  AS_EMA_minus_student_fullV_net_repairs=clean_net,AS_EMA_minus_student_midlow_net_repairs=midlow_net,
  AS_EMA_minus_student_midlow_mean_delta_pp=100*midlow_net/7200,
  limitation='One fixed source-only model seed; no formal prediction, no seed or EMA/epoch rescue')
