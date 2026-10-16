"""Descriptive existing-readout strata/margins; no new IQ or NN."""
import argparse,json,math,statistics
from collections import Counter
from pathlib import Path
from .analyze import ARMS,NEW,VIEWS,read,dump,describe

def bucket(x,bounds):
 for hi,label in bounds:
  if x<hi:return label
 raise AssertionError('Final bin must include infinity')
def summarize(root):
 root=Path(root);seed=read(root/'analysis_summary.json')['model_seed'];assert seed==2026092701;paths={a:root/(a+'-RX1-s'+str(seed)+'/source') for a in ARMS};pred={a:{v:read(s/'source_readouts'/(v+'.json')) for v in VIEWS} for a,s in paths.items()}
 baseline=pred['RAW10']['clean'];fixed=[i for i,(y,p) in enumerate(zip(baseline['labels'],baseline['predictions'])) if y==p];strata=[];physics={};retention={}
 assert fixed
 for a in ARMS:
  retention[a]={}
  for v in VIEWS:
   p=pred[a][v];m=p.get('margin_over30');retention[a][v]=dict(fixed_raw_clean_correct_count=len(fixed),fixed_raw_clean_correct_retention=sum(p['predictions'][i]==p['labels'][i] for i in fixed)/len(fixed),all_correct_retention=sum(y==q for y,q in zip(p['labels'],p['predictions']))/1260,scaled_margin_all=describe(m) if m is not None else None,scaled_margin_fixed=describe([m[i] for i in fixed]) if m is not None else None)
 for v in VIEWS[1:]:
  pp=pred['ANTI'][v]['physics'];cfg=read(paths['ANTI']/'actual_channel.json')['actual_configs'][v];assert pp==pred['INDEP'][v]['physics'] and cfg==read(paths['INDEP']/'actual_channel.json')['actual_configs'][v]
  def lock(x):return 'locked' if x>=cfg['locked_snr_db'] else ('unlocked' if x<cfg['unlocked_snr_db'] else 'degraded')
  bins=Counter((p['lock'],lock(p['signal_to_added_noise_db'])) for p in pp)
  fields=('quality_snr_db','signal_to_added_noise_db','output_frequency_hz','residual_kernel_energy','residual_kernel_length','propagation_kernel_length','numerical_filter_latency_samples','agc_gain_db')
  physics[v]=dict(count=1260,fields={k:describe([p[k] for p in pp]) for k in fields},states=dict(Counter(p['state'] for p in pp)),locks=dict(Counter(p['lock'] for p in pp)),AGC_limited=sum(p['agc_limited'] for p in pp),boundary=dict(Counter(p['boundary'] for p in pp)),noise_boundary=dict(Counter(p['noise_boundary'] for p in pp)),equalization_applied=sum(p['channel_equalization_applied'] for p in pp),phase_tracking_applied=sum(p['phase_tracking_applied'] for p in pp),IQ_compensation_applied=sum(p['iq_compensation_applied'] for p in pp),true_total_SNR_known=any(p['true_total_snr_known'] for p in pp),proxy_minus_added_noise_SNR=describe([p['quality_snr_db']-p['signal_to_added_noise_db'] for p in pp]),proxy_threshold_bin_disagreements=sum(n for (a,b),n in bins.items() if a!=b),quality_sensitivity_scope='Existing added-noise-power bins only;no recomputed state/compensation or causal ablation',EQ_off_noise_coloring='G=1;no EQ-induced noise coloration;propagation and added noise remain actual simulator outputs')
  selectors={'state':lambda p:p['state'],'lock':lambda p:p['lock'],'quality_snr_db':lambda p:bucket(p['quality_snr_db'],[(5,'-inf:5'),(15,'5:15'),(math.inf,'15:inf')]),'signal_to_added_noise_db':lambda p:bucket(p['signal_to_added_noise_db'],[(5,'-inf:5'),(15,'5:15'),(math.inf,'15:inf')]),'abs_residual_CFO_Hz':lambda p:bucket(abs(p['output_frequency_hz']),[(500,'0:500'),(1500,'500:1500'),(math.inf,'1500:inf')]),'residual_kernel_energy':lambda p:bucket(p['residual_kernel_energy'],[(.001,'0:.001'),(.01,'.001:.01'),(math.inf,'.01:inf')]),'AGC_limited':lambda p:str(p['agc_limited']),'boundary':lambda p:p['boundary'],'residual_kernel_length':lambda p:str(p['residual_kernel_length'])}
  for dimension,selector in selectors.items():
   groups={}
   for i,p in enumerate(pp):groups.setdefault(selector(p),[]).append(i)
   for value,take in sorted(groups.items()):
    acc={a:sum(pred[a][v]['predictions'][i]==pred[a][v]['labels'][i] for i in take)/len(take) for a in ARMS};paired={}
    for ref in ARMS[:2]:
     repair=sum(pred[ref][v]['predictions'][i]!=baseline['labels'][i] and pred['ANTI'][v]['predictions'][i]==baseline['labels'][i] for i in take);broken=sum(pred[ref][v]['predictions'][i]==baseline['labels'][i] and pred['ANTI'][v]['predictions'][i]!=baseline['labels'][i] for i in take);paired[ref]=dict(accuracy_gain_pp=100*(acc['ANTI']-acc[ref]),repaired=repair,broken=broken)
    strata.append(dict(view=v,dimension=dimension,value=value,count=len(take),accuracy=acc,paired=paired))
 result=dict(source_only=True,readout_role='Held source RX1 L1260,modelseed'+str(seed)+',fixedfinal20,defaultcosine',analysis_NN=0,analysis_IQ_reads=0,updates=0,target_access=False,fixed_raw_clean_correct_count=len(fixed),margins_and_retention=retention,strata=strata,physics=physics,reference_limit='All3newrows have fullsamephysicalsourcepredictions/margins/physics;states areassociations,not causalablation',causal_claim=False)
 dump(root/'mechanism_analysis.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();r=summarize(a.input);print(json.dumps(dict(status='ANALYZED',strata=len(r['strata']),new_NN=0)))
