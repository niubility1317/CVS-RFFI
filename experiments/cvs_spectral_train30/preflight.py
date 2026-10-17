"""Matched fullsource parents,default head smoke and all-L identity observation guard."""
import gc,time
from pathlib import Path
from . import design as d

def control(arm):
 p=d.CONTROL_ROOT/(d.CONTROL_ARM[arm]+'-s'+str(d.SEED));v=d.read(p/'completion.json');pred=d.read(p/'prediction/complete.json');r=d.read(p/'source_recovery.json');card=r['card']
 if v['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or v['views']!=list(d.VIEWS) or v['count']!=168000 or not v['truth_last'] or pred['status']!='PREDICTIONS_COMPLETE' or pred['truth_read'] or not pred['fixed_received_IQ']:raise ValueError('Matched control incomplete')
 if r['status']!='SOURCE_BUDGET_AND_FIXED_CHECKPOINT_VERIFIED' or card['row_id']!=d.CONTROL_ARM[arm]+'-s'+str(d.SEED) or card['epoch']!=10 or card['successful_updates']!=2220 or card['parent']!=str(d.parent_source(arm)) or not card['source_roles_match'] or not card['default_head_joint'] or card['head_shape']!=[6,160] or card['final_blocked'] or r['source_roles']!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Control parent/fullsource/E30/default head differs')
 return dict(status='VERIFIED',row=card['row_id'],run=d.CONTROL_RUN,new_updates=0)

def smoke():
 import torch,numpy as np
 from . import model as api
 from .filter import FixedFilter
 start=time.perf_counter();torch.set_num_threads(2);contract=d.read(d.SOURCE);prep=Path(d.PROJECT)/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701';ready=d.read(prep/'preparation_complete.json')
 if ready['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or ready['target_access']:raise ValueError('Existing source preparation differs')
 packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False)
 if any(packed['contract'].get(k)!=v or ready['source_contract'].get(k)!=v for k,v in contract.items()):raise ValueError('Source handles differ')
 L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
 if len(L['ids'])!=6300 or set(L['ids'])!=set(contract['role_ids']['L_s']) or L['x'].shape!=(6300,2,256):raise ValueError('All and only original L6300 required')
 y=np.asarray(L['y'].tolist());rx=np.asarray(L['rx'].tolist());day=np.asarray(L['day'].tolist());results={};NN=0;parents=[];saved={};operator=FixedFilter();operator.load()
 if set(rx)!={1,3,4,6,8} or set(day)!={1,2,3} or set(y)!=set(range(6)):raise ValueError('Fullsource groups required')
 for row in d.rows():
  c=d.config(row);p=d.parent_proof(row['arm']);cp=d.validate_parent_checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c);m=api.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval().requires_grad_(False);reference=control(row['arm']);parents.append(reference)
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))
  NN+=2
  if z['tx_logits'].shape!=(2,6) or z['id_feat_joint'].shape!=(2,160) or not bool(torch.isfinite(z['tx_logits']).all()):raise ValueError('Real matching source checkpoint smoke failed')
  pred={}
  for key in ['raw']+operator._keys:
   pp=[]
   for st in range(0,6300,128):
    x=L['x'][st:st+128]
    if key!='raw':x=operator.transform(x,[operator._keys.index(key)]*len(x))
    with torch.no_grad():zz=m.forward_identity_only(x)['tx_logits']
    pp.extend(zz.argmax(1).tolist());NN+=len(x)
   pred[key]=np.asarray(pp,dtype=np.int64)
   d.write(d.BASE/'source_guard_progress.json',dict(row=row['row_id'],view=key,NN=NN,CP=len(parents),training_updates=0,target_access=False))
  raw=pred['raw']==y;stats={};ok=True
  for key in operator._keys:
   correct=pred[key]==y;drop=100*float((raw.astype(float)-correct.astype(float)).mean());strata={dim:{str(k):100*float((raw[a==k].astype(float)-correct[a==k].astype(float)).mean()) for k in sorted(set(a))} for dim,a in [('RX',rx),('TX',y),('day',day)]}
   passed=drop<=.5+1e-10 and all(v<=1.+1e-10 for group in strata.values() for v in group.values());ok=ok and passed;stats[key]=dict(count=6300,accuracy=float(correct.mean()),drop_pp=drop,strata_drop_pp=strata,identity_condition_passed=passed)
  results[row['row_id']]=dict(raw_accuracy=float(raw.mean()),views=stats,identity_conditions_passed=ok);saved.update({row['row_id']+'/'+k:v for k,v in pred.items()})
  del m,cp;gc.collect()
 if NN!=163804:raise ValueError('Full source guard budget differs')
 np.savez_compressed(d.BASE/'source_guard_predictions.npz',ids=np.asarray(L['ids']),labels=y,RX=rx,day=day,**saved)
 passed=all(r['identity_conditions_passed'] for r in results.values());value=dict(status='PASS_SOURCE_IDENTITY_ONLY' if passed else 'SOURCE_INPUT_REJECTED',results=results,CP=2,source_cache_loads=1,NN=NN,source_IQ_unique=6300,new_updates=0,target_access=False,head_fit_updates=0,previous_FIR_feasibility_failure_unchanged=True,matched_controls=parents,elapsed_seconds=time.perf_counter()-start)
 d.write(d.BASE/'source_input_guard.json',value);d.write(d.BASE/'preflight.json',dict(status='PASS',real_CP_loads=2,noquery_synthetic_smoke_NN=4,query_read=False,source_identity_guard=value['status']))
 if not passed:d.write(d.BASE/'completion.json',dict(status='SOURCE_INPUT_REJECTED',trained_rows=0,target_access=False,training_updates=0,scientific_candidate_closed=True))
 return passed
