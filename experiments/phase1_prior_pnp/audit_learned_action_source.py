"""Independent paired audit of fixed learned-action and unrolled source decisions."""
import pathlib,json,hashlib,math,numpy as np
R=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');n='20261102-phase1-learned-action-validation-s2026092701-r01';root=R/'runs'/n;path=root/'source_predictions.npz';m=json.loads((root/'prediction_complete.json').read_text());assert m['status']=='SOURCE_PREDICTIONS_FIXED' and m['source_only'] and all(m[k]==0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')) and hashlib.sha256(path.read_bytes()).hexdigest()==m['sha256']
with np.load(path,allow_pickle=False) as z:ids=z['indices'];split=z['split'];rx=z['rx'];scene=z['scene'];v=z['values']
oldroot=R/'runs/20261102-phase1-unroll-source-replay-s2026092701-r01';oldpath=oldroot/'source_predictions.npz';oldmanifest=json.loads((oldroot/'prediction_complete.json').read_text());assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==oldmanifest['sha256']
with np.load(oldpath,allow_pickle=False) as z:assert np.array_equal(ids,z['indices']) and np.array_equal(split,z['split']) and np.array_equal(rx,z['rx']) and np.array_equal(scene,z['scene']);old=z['values'].copy()
assert len(ids)==11761 and len(set(ids.tolist()))==len(ids) and v.shape==old.shape==(len(ids),15) and np.isfinite(v).all();assert np.array_equal(v[:,[0,2,5,7]],old[:,[0,2,5,7]]) and np.allclose(v[:,[1,3,4,6,8]],old[:,[1,3,4,6,8]],rtol=1e-5,atol=1e-7)
summary=json.loads((root/'source_validation.json').read_text());views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];checks=0;fields=0
for si,scale in enumerate(('0.25','0.5','1.0')):
 for gate in ('0.8','0.99'):
  cells=summary['validation'][scale][gate];assert len(cells)==21
  for part,rs in ((1,(4,6)),(2,(8,))):
   for receiver in rs:
    for view in views:
     key=f'{part}:{receiver}:{view}';mask=(split==part)&(rx==receiver)&(scene==view);a=v[mask];assert len(a)>0;trusted=a[:,1]>=.95;active=a[:,3]>float(gate);par=np.where(active,a[:,7],a[:,5]);piq=np.where(active,a[:,8],a[:,6]);pred=np.where(active,a[:,9+2*si],a[:,5]);iq=np.where(active,a[:,10+2*si],a[:,6]);actual={'count':len(a),'valid_count':int(trusted.sum()),'active':int(active.sum()),'baseline_correct':int(np.count_nonzero(trusted&(a[:,2]==a[:,0]))),'parent_correct':int(np.count_nonzero(trusted&(par==a[:,0]))),'candidate_correct':int(np.count_nonzero(trusted&(pred==a[:,0]))),'baseline_iq':float(a[:,4].mean()),'parent_iq':float(piq.mean()),'candidate_iq':float(iq.mean())}
     for field,val in actual.items():assert abs(float(val)-float(cells[key][field]))<1e-7;(fields:=fields+1)
     checks+=1
assert checks==126 and fields==1134
selection_checks=0
for scale,results in summary['validation'].items():
 for key,ref,req in [('primary_dura','baseline',True),('accuracy_dura','baseline',False),('primary_parent','parent',True),('accuracy_parent','parent',False)]:
  best={'kind':'none','validation_net':0}
  for threshold,cells in results.items():
   val=[(k,x) for k,x in cells.items() if k.startswith('1:')];assert len(val)==14;net=sum(x['candidate_correct']-x[ref+'_correct'] for k,x in val if not k.endswith(':clean'));cls=all(x['candidate_correct']>=x[ref+'_correct']-max(1,math.ceil(.005*x['valid_count'])) for k,x in val);clean=all(x['candidate_iq']<=x[ref+'_iq']+.001 for k,x in val if k.endswith(':clean'));impaired=all(x['candidate_iq']<=1.2*x[ref+'_iq']+1e-6 for k,x in val if not k.endswith(':clean'))
   if cls and clean and (impaired or not req) and net>=10 and net>best['validation_net']:best={'kind':'residual','threshold':float(threshold),'validation_net':net,'reference':ref}
  assert best==summary['selection'][scale][key];selection_checks+=1
assert selection_checks==12
out={}
for name,part,rs in [('RX4_6',1,(4,6)),('RX8',2,(8,))]:
 for label,scenes in [('all',views[1:]),('low_urban',['practical_low_urban'])]:
  mask=(split==part)&np.isin(rx,rs)&np.isin(scene,scenes)&(v[:,1]>=.95);a=v[mask];b=old[mask];active=a[:,3]>.8;target=a[:,0];parent=np.where(active,a[:,7],a[:,5]);candidate=np.where(active,a[:,13],a[:,5]);prior=np.where(active,b[:,13],b[:,5]);dura=a[:,2];out[name+':'+label]={'trusted':len(a),'parent_correct':int((parent==target).sum()),'unroll_correct':int((prior==target).sum()),'candidate_correct':int((candidate==target).sum()),'dura_correct':int((dura==target).sum()),'repairs_vs_unroll':int(np.count_nonzero((prior!=target)&(candidate==target))),'damages_vs_unroll':int(np.count_nonzero((prior==target)&(candidate!=target)))}
print(json.dumps({'status':'PASS','source_prediction_sha256':m['sha256'],'previous_sha256':oldmanifest['sha256'],'fixed_predictions':len(ids),'cell_checks':checks,'field_checks':fields,'selection_checks':selection_checks,'common_controls_match':True,'source_pseudo_targets_only':True,'DURA_comparator_only':True,'TX_truth_reads':0,'test_reads':0,'groups':out},allow_nan=False))
