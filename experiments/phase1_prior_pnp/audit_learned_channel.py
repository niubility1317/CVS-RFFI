"""Independently audit fixed source channel-predictor outputs."""
import pathlib,json,hashlib,numpy as np
R=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');n='20261102-phase1-learned-channel-validation-s2026092701-r01';root=R/'runs'/n;path=root/'source_predictions.npz';m=json.loads((root/'prediction_complete.json').read_text());assert m['status']=='SOURCE_PREDICTIONS_FIXED' and m['count']==2048 and m['source_only'] and not m['DURA_loaded'] and not m['F_loaded'] and all(m[k]==0 for k in ('TX_truth_reads','U_s_truth_reads','V_reads','test_reads')) and hashlib.sha256(path.read_bytes()).hexdigest()==m['sha256']
with np.load(path,allow_pickle=False) as z:ids=z['indices'];part=z['split'];rx=z['rx'];scene=z['scene'];v=z['values']
assert v.shape==(2048,8) and np.isfinite(v).all() and np.all(v[:,:4]>=0) and np.all((v[:,4:7]>=0)&(v[:,4:7]<=.25001));origin=R/'runs/20261102-phase1-channel-refinement-source-s2026092701-r01/source_channel_diagnostic.npz'
with np.load(origin,allow_pickle=False) as z:assert np.array_equal(ids,z['indices']) and np.array_equal(part,z['split']) and np.array_equal(rx,z['rx']) and np.array_equal(scene,z['scene']) and np.allclose(v[:,0],z['values'][:,0],rtol=1e-5,atol=1e-7)
views=['practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];results={}
for group,i in [('RX4_6',1),('RX8',2)]:
 results[group]={}
 for sc in ['all']+views:
  sel=(part==i) if sc=='all' else ((part==i)&(scene==sc));x=v[sel].astype(np.float64);assert len(x)>0;mean=x.mean(0);assert mean[0]>0
  results[group][sc]={'count':len(x),'base_residual':mean[0],'learned_residual':mean[1],'naive_residual':mean[2],'oracle_residual':mean[3],'learned_change_percent':(mean[1]/mean[0]-1)*100,'naive_change_percent':(mean[2]/mean[0]-1)*100,'learned_improved_count':int((x[:,1]<x[:,0]).sum()),'learned_harmed_count':int((x[:,1]>x[:,0]).sum()),'mean_learned_norm':mean[4],'mean_oracle_norm':mean[6],'oracle_coefficient_mse':mean[7]}
r=results['RX4_6'];hold=results['RX8'];passed=bool(r['all']['learned_change_percent']<=-1 and all(r[v]['learned_change_percent']<=1 for v in views) and hold['all']['learned_change_percent']<=1)
print(json.dumps({'status':'SOURCE_CHANNEL_VALIDATION_ANALYZED','count':len(ids),'prediction_sha256':m['sha256'],'baseline_matches_prior_diagnostic':True,'source_only':True,'DURA_loaded':False,'F_loaded':False,'TX_truth_reads':0,'test_reads':0,'predeclared_source_flag':passed,'caveat':'Predictor uses full received waveform; paired source clean IQ is evaluation and oracle only. Previously explored source IDs. Forward residual is neither waveform restoration nor identification accuracy.','results':results},allow_nan=False))
