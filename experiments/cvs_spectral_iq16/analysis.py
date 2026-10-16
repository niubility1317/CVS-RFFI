"""Independent NumPy/stdlib source recount;no Torch,IQ or model loading."""
import json,math
import numpy as np
from . import design as d
from experiments.cvs_iq_timing20.harvest import score
OUT=d.ROOT/d.LOCAL/'final'
def close(a,b,tol=1e-10):assert math.isclose(a,b,abs_tol=tol,rel_tol=tol),(a,b)
def analyze():
 result=d.read(OUT/'result.json');done=d.read(OUT/'completion.json');smoke=d.read(OUT/'real_checkpoint_smoke.json');resolved=d.read(OUT/'resolved_config.json')
 assert done['status']==result['status']=='SOURCE_FILTER_COMPLETE' and done['views']==25 and done['source_decisions']==126000 and done['actual_NN_packets']==result['actual_NN_packets']==126002 and not done['optimizer_updates'] and not done['target_access'] and not done['automatic_training']
 assert smoke['status']=='PASS' and smoke['source_IQ_reads']==0 and smoke['synthetic_NN_packets']==2 and not smoke['target_access']
 assert result['model_state_unchanged'] and result['default_head_only'] and result['NoExplicitCFO'] and all(not result[k] for k in ('actual_optimizer_updates','head_fit_updates','teacher_forwards','As_generations','U_used','V_used','held_RX1_used','target_access')) and result['finite_filter_packets']==result['circular_oracle_packets']==120960
 assert not resolved['old_fullL6300_spectral_coefficients_used'] and resolved['config']['views']==d.views()
 ids=d.read(OUT/'fit_L_ids.json');parent=d.read(d.ROOT/'local_artifacts/F_crossrx_satmix20_20261016/final/B00-RX1-s2026092701/source/source_contract.json');assert len(ids)==len(set(ids))==5040 and set(ids)==set(parent['role_ids']['L_s'])
 predictions={};scores={};max_ce_error=0.;cell_count=0
 for view in d.views():
  name=view['view'];p=d.read(OUT/'predictions'/(name+'.json'));predictions[name]=p;saved=result['scores'][name]
  assert p['view']==name and p['physical_ids']==ids and len(p['labels'])==len(p['predictions'])==len(p['logits'])==len(p['RX'])==len(p['day'])==5040 and not p['head_fit_updates'] and not p['target_access']
  if name!='identity':assert all(p[k]==predictions['identity'][k] for k in ('labels','RX','day','physical_ids'))
  assert set(p['RX'])==set(d.FIT) and set(p['day'])=={1,2,3} and all(len(z)==6 and all(math.isfinite(v) for v in z) for z in p['logits']) and p['predictions']==[max(range(6),key=z.__getitem__) for z in p['logits']]
  s=score(p['labels'],p['predictions']);assert s['confusion']==saved['confusion'];close(s['accuracy'],saved['accuracy']);close(s['macro_F1'],saved['macro_F1'])
  ce=sum(max(z)+math.log(sum(math.exp(v-max(z)) for v in z))-z[y] for z,y in zip(p['logits'],p['labels']))/5040;max_ce_error=max(max_ce_error,abs(ce-saved['mean_CE']));close(ce,saved['mean_CE']);s['mean_CE']=ce
  for group,values,allowed,count in (('RX',p['RX'],d.FIT,1260),('TX',p['labels'],range(6),840),('day',p['day'],(1,2,3),1680)):
   s['by_'+group]={}
   for v in allowed:
    take=[i for i,k in enumerate(values) if k==v];q=score([p['labels'][i] for i in take],[p['predictions'][i] for i in take]);assert q['count']==count and q['confusion']==saved['by_'+group][str(v)]['confusion'];close(q['accuracy'],saved['by_'+group][str(v)]['accuracy']);close(q['macro_F1'],saved['by_'+group][str(v)]['macro_F1']);s['by_'+group][str(v)]=q
  for r in d.FIT:
   for dd in (1,2,3):
    for tx in range(6):assert sum((rr,dy,y)==(r,dd,tx) for rr,dy,y in zip(p['RX'],p['day'],p['labels']))==70;cell_count+=1
  scores[name]=s
 raw=predictions['identity'];coeff=np.asarray(d.read(OUT/'raw_DCT8.json'));assert coeff.shape==(5040,8) and np.isfinite(coeff).all();means=np.empty((6,3,4,8))
 for tx in range(6):
  for dd in (1,2,3):
   for i,r in enumerate(d.FIT):means[tx,dd-1,i]=coeff[(np.array(raw['labels'])==tx)&(np.array(raw['day'])==dd)&(np.array(raw['RX'])==r)].mean(0)
 stat=d.read(OUT/'source_statistics.json');assert np.allclose(means,stat['means'],atol=1e-12,rtol=0)
 totals=np.zeros(3);better=0
 for tx in range(6):
  for dd in range(3):
   pred=means[np.arange(6)!=tx][:,np.arange(3)!=dd].mean((0,1));pred-=pred.mean(0);actual=means[tx,dd]-means[tx,dd].mean(0);values=[float(((pred-actual)**2).sum()),float((actual**2).sum()),float(((pred*np.array(d.CONTROL)-actual)**2).sum())];saved=stat['folds'][tx*3+dd];assert saved['TX']==tx and saved['day']==dd+1 and saved['fit_cells']==40 and saved['test_cells']==4
   for key,v in zip(('prediction_SSE','zero_SSE','control_SSE'),values):close(v,saved[key])
   totals+=values;better+=values[0]<values[1]
 for key,v in zip(('prediction_SSE','zero_SSE','control_SSE'),totals):close(v,stat[key])
 rel=float(totals[0]/totals[1]) if totals[1]>1e-20 else None;assert rel is None or math.isclose(rel,stat['relative_MSE'],abs_tol=1e-12);assert better==stat['better_folds']
 bank=d.read(OUT/'kernel_bank.json');assert set(bank)==set(scores)-{'identity'};kernels={};max_kernel_recount=0.;matches=[]
 for name,k in bank.items():
  h=np.array(k['h_real'])+1j*np.array(k['h_imag']);assert h.shape==(33,) and np.isfinite(h).all();p=np.zeros(d.GRID,dtype=complex);p[np.arange(-16,17)%d.GRID]=h;H=np.fft.fft(p);target=np.array(k['desired_response_real'])+1j*np.array(k['desired_response_imag']);gain=20*np.log10(np.abs(H));den=float((np.abs(target-1)**2).sum());err=float((np.abs(np.fft.fftshift(H)-target)**2).sum());rmse=(err/den)**.5 if den>1e-25 else 0.;delta=h.copy();delta[16]-=1;energy=float((np.abs(delta)**2).sum());max_kernel_recount=max(max_kernel_recount,abs(rmse-k['kernel_relative_RMSE']));close(rmse,k['kernel_relative_RMSE']);close(float(np.abs(gain).max()),k['grid_response_max_abs_dB']);close(energy,k['perturbation_energy'])
  kernels[name]=dict(kernel_relative_RMSE=rmse,grid_response_max_abs_dB=float(np.abs(gain).max()),perturbation_energy=energy)
 for name,k in kernels.items():
  if name.startswith('structured'):
   other=kernels[name.replace('structured','control',1)]['perturbation_energy'];matches.append(abs(k['perturbation_energy']-other)/max(k['perturbation_energy'],other,1e-30))
 batches=d.read(OUT/'view_diagnostics.json');diagnostics={}
 for name,records in batches.items():
  assert [v['start'] for v in records]==list(range(0,5040,128)) and sum(v['count'] for v in records)==5040
  def ratio(a,b):
   num=sum(v[a] for v in records);den=sum(v[b] for v in records)
   if den<=1e-20:assert num<=1e-20;return 0.
   return num/den
  value=dict(count=5040,finite_PSD_relative_RMSE=ratio('spectral_error_SSE','spectral_oracle_change_SSE')**.5,boundary_error_fraction=ratio('boundary_error_energy','oracle_IQ_change_energy'),interior_relative_MSE_max=max(v['interior_relative_MSE'] for v in records),RMS_relative_error_max=max(v['RMS_relative_error_max'] for v in records),restore_gain_dB_min=min(v['restore_gain_dB_min'] for v in records),restore_gain_dB_max=max(v['restore_gain_dB_max'] for v in records))
  for key,v in value.items():close(v,result['diagnostics'][name][key])
  diagnostics[name]=value
 assert set(diagnostics)==set(kernels)
 drops={};identity_ok=True
 for view in d.views()[1:]:
  name=view['view'];p=predictions[name];s=scores[name];drop=100*(scores['identity']['accuracy']-s['accuracy']);strata={group:{key:100*(scores['identity']['by_'+group][key]['accuracy']-val['accuracy']) for key,val in s['by_'+group].items()} for group in ('RX','TX','day')};repairs=sum(a!=y and b==y for a,b,y in zip(raw['predictions'],p['predictions'],raw['labels']));broken=sum(a==y and b!=y for a,b,y in zip(raw['predictions'],p['predictions'],raw['labels']));close(drop,100*(broken-repairs)/5040);drops[name]=dict(accuracy_drop_pp=drop,repaired=repairs,broken=broken,strata_drop_pp=strata)
  if view['kind']=='structured':identity_ok=identity_ok and drop<=.5+1e-10 and all(v<=1+1e-10 for values in strata.values() for v in values.values())
 conditions=dict(stable_RX=bool(rel is not None and rel<1 and better>=15 and totals[0]<totals[2]),kernel_realization=all(k['kernel_relative_RMSE']<=.35 and k['grid_response_max_abs_dB']<=.5+1e-8 for k in kernels.values()),matched_control=max(matches)<=1e-8,finite_IQ_realization=all(k['finite_PSD_relative_RMSE']<=.35 and k['boundary_error_fraction']<=.25 and k['interior_relative_MSE_max']<=1e-10 and k['RMS_relative_error_max']<=1e-6 for k in diagnostics.values()),source_identity=identity_ok);decision='PASS' if all(conditions.values()) else 'FAIL';assert conditions==result['qualification']['conditions'] and decision==done['decision']==result['qualification']['decision']
 log=(OUT/'spectral_source.stdout.log').read_text(encoding='utf8');assert log.index('SMOKE PASS')<log.index('INPUT VERIFIED') and 'COMPLETE ' in log and 'Traceback (most recent call last)' not in log and sum(v.startswith('VIEW ') for v in log.splitlines())==25
 summary=dict(status='ANALYZED_SOURCE_ONLY_VERIFIED',run_id=d.RUN,decision=decision,conditions=conditions,crossfit=dict(relative_MSE=rel,better_folds=int(better),prediction_SSE=float(totals[0]),zero_SSE=float(totals[1]),control_SSE=float(totals[2]),folds=stat['folds']),scores=scores,identity_drops=drops,kernels=kernels,diagnostics=diagnostics,max_CE_recount_error=max_ce_error,max_kernel_RMSE_recount_error=max_kernel_recount,matched_control_max_relative_energy_error=max(matches),source_cells_recounted=cell_count,all126000_default_head_predictions_and_logits_recounted=True,actual_NN_packets=126002,finite_filter_packets=120960,circular_oracle_packets=120960,optimizer_updates=0,head_fit_updates=0,model_state_unchanged=True,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,elapsed_seconds=result['elapsed_seconds'],full_stdout_lines=len(log.splitlines()))
 d.write(OUT/'analysis_summary.json',summary);print(json.dumps({k:summary[k] for k in ('status','decision','conditions','crossfit','matched_control_max_relative_energy_error','actual_NN_packets','elapsed_seconds')},indent=2));return summary
if __name__=='__main__':analyze()
