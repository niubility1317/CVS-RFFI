"""Descriptive quality-threshold sensitivity of existing source readouts;0 NN."""
import argparse,json,statistics
from collections import Counter
from pathlib import Path
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def summarize(root):
 root=Path(root);src=root/'D00-RX6-s2026092701/source';configs=json.loads((src/'actual_channel.json').read_text())['actual_configs'];result={}
 for scene in SCENES:
  p=json.loads((src/'source_readouts'/(scene+'.json')).read_text());cfg=configs[scene];assert p['count']==1260 and cfg['quality_adaptation_enabled'] and not cfg['equalization_enabled']
  def state(v):return 'locked' if v>=cfg['locked_snr_db'] else ('unlocked' if v<cfg['unlocked_snr_db'] else 'degraded')
  bins=Counter();difference=[]
  for m in p['physics']:
   assert m['RX']==6 and not m['true_total_snr_known'] and m['quality_source']=='white_input_expected_power_proxy' and state(m['quality_snr_db'])==m['lock']
   bins[(m['lock'],state(m['signal_to_added_noise_db']))]+=1;difference.append(m['quality_snr_db']-m['signal_to_added_noise_db'])
  result[scene]=dict(count=1260,threshold_bin_disagreements=sum(n for (a,b),n in bins.items() if a!=b),transitions=[dict(recorded_proxy_bin=a,measured_added_noise_SNR_bin=b,count=n) for (a,b),n in sorted(bins.items())],proxy_minus_added_noise_SNR_db=dict(min=min(difference),median=statistics.median(difference),max=max(difference)),actual_receiver_status_recomputed=False)
 report=dict(source_only=True,new_IQ_reads=0,new_NN=0,new_updates=0,target_access=False,total_SNR_known=False,scope='Threshold-bin sensitivity using existing signal/noise powers under original compensation;not a recomputed or receiver-fixed causal ablation',views=result);(root/'quality_threshold_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();r=summarize(a.input);print(json.dumps({k:v['threshold_bin_disagreements'] for k,v in r['views'].items()}))
