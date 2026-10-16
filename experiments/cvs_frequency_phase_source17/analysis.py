"""Independent complete source observation reconstruction;stdlib/no estimator or Torch."""
import json,math,sys
from pathlib import Path
from experiments.cvs_frequency_phase17.analysis import reconstruct,compare
FIT=(3,4,6,8)
def read(q):return json.loads(q.read_text(encoding='utf8'))
def main(base):
 base=Path(base);o=read(base/'packet_observations.json');r=read(base/'source_phase_basis.json');done=read(base/'completion.json');cfg=read(base/'resolved_config.json');ids=read(base/'fit_L_ids.json')
 assert done['status']=='ANALYZED' and done['fit_L_packets']==5040 and done['source_cells']==72 and done['harmonic_projections']==60480 and done['phaseclosures']==40320 and done['source_cache_loads']==1 and done['observer_calls']==1
 for k in ('network_forwards','optimizer_steps','checkpoint_loads','head_fit_updates','IQ_views_generated'):assert done[k]==0
 for k in ('U_truth_read','U_statistically_used','V_statistically_used','held_RX1_statistically_used','target_access','IQ_modified','model_constructed','automatic_training','Ag_efficacy_proved','joint_goal_complete'):assert done[k] is False
 assert cfg['source_roles_exact_match'] and cfg['source_carrier_L_deserialized']==6300 and cfg['fit_L_unique']==5040 and cfg['source_cache_other_roles_and_features_deserialized_but_unused'] and len(ids)==len(set(ids))==5040
 assert all(len(o[k])==5040 for k in ('closure_real','closure_imag','valid','TX','RX','day'));assert set(o['TX'])==set(range(6)) and set(o['RX'])==set(FIT) and set(o['day'])=={1,2,3}
 c=[]
 for a,b in zip(o['closure_real'],o['closure_imag']):
  assert len(a)==len(b)==8;row=[complex(u,v) for u,v in zip(a,b)];assert all(math.isfinite(z.real) and math.isfinite(z.imag) and abs(z)<=1+1e-9 for z in row);c.append(row)
 quality=o['quality'];assert set(quality)=={'CFO_estimate_Hz','min_carrier_ratio','projection_noise_power'} and all(len(v)==5040 and all(math.isfinite(x) for x in v) for v in quality.values());assert all(v>=0 for v in quality['min_carrier_ratio']) and all(v>0 for v in quality['projection_noise_power'])
 valid=[v>=25. for v in quality['min_carrier_ratio']];assert o['valid']==valid and all(type(v) is bool for v in o['valid']);means_r=[];means_i=[];index=0;cell_summary=[]
 for t in range(6):
  tx_r=[];tx_i=[]
  for dd in (1,2,3):
   day_r=[];day_i=[]
   for rx in FIT:
    ind=[i for i in range(5040) if (o['TX'][i],o['day'][i],o['RX'][i])==(t,dd,rx)];assert len(ind)==70
    mu=[complex(math.fsum(c[i][k].real for i in ind)/70,math.fsum(c[i][k].imag for i in ind)/70) for k in range(8)];var=[math.fsum(abs(c[i][k]-mu[k])**2 for i in ind)/70 for k in range(8)];rr=dict(TX=t,day=dd,RX=rx,count=70,valid_packets=sum(valid[i] for i in ind),closure_mean_real=[z.real for z in mu],closure_mean_imag=[z.imag for z in mu],closure_population_variance=var,carrier_ratio_min=min(quality['min_carrier_ratio'][i] for i in ind),carrier_ratio_max=max(quality['min_carrier_ratio'][i] for i in ind));compare(rr,r['cells'][index]);index+=1;cell_summary.append(rr);day_r.append(rr['closure_mean_real']);day_i.append(rr['closure_mean_imag'])
   tx_r.append(day_r);tx_i.append(day_i)
  means_r.append(tx_r);means_i.append(tx_i)
 assert sum(valid)==r['valid_packets']==done['valid_packets'] and 5040-sum(valid)==r['rejected_packets']==done['rejected_packets'] and r['all_packets_identifiable']==all(valid)
 if all(valid):
  e=reconstruct(means_r,means_i);compare(e,{k:r['estimate'][k] for k in e});passed=e.get('detected_RX',False)
 else:e=dict(identifiable=False);passed=False;assert not r['estimate']['identifiable'] and 'folds' not in r['estimate']
 assert r['decision']==done['scientific_decision']==('PASS' if passed else 'FAIL') and done['computed_folds']==len(e.get('folds',[]))
 output=dict(status='ANALYZED',actual_execution='VERIFIED',run_id=cfg['config']['run_id'],execution_commit=cfg['executed_commit'],decision='PASS_SOURCE_RX_ASSOCIATION_ONLY' if passed else 'FAIL_FIXED_SOURCE_PHASE_OBSERVABILITY_CLOSED',all5040_observations_72cells_and_available_folds_reconstructed=True,valid_packets=sum(valid),rejected_packets=5040-sum(valid),rejected_by_RX={str(rx):sum(not valid[i] for i in range(5040) if o['RX'][i]==rx) for rx in FIT},cell_summaries=cell_summary,estimate=e,source_cache_loads=1,real_IQ_statistically_used=5040,real_complex_points=1290240,harmonic_projections=60480,phaseclosures=40320,observer_calls=1,network_forwards=0,checkpoint_loads=0,updates=0,teacher_forwards=0,head_fit_updates=0,IQ_views_generated=0,target_access=False,seconds=done['seconds'],Ag_efficacy_proved=False,joint_goal_complete=False,scope='Saved phaseclosure/quality statistics recounted;no rawIQ replay,model orcausalhardware proof')
 (base.parent/'independent_analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({k:output[k] for k in ('status','actual_execution','decision','valid_packets','rejected_packets','seconds')}))
if __name__=='__main__':main(sys.argv[1])
