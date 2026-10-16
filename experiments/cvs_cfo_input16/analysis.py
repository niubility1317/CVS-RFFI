"""Independent complete synthetic trace recount;no Torch/model/data import."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'local_artifacts/F_frequency_input16_20261016/final'
def main():
 def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
 audit=read('input_audit.json');traces=read('input_audit_traces.json');done=read('completion.json');names=('identity','frequency_positive','frequency_negative','positive_inverse','negative_inverse')
 assert set(traces)==set(names) and audit['actual_total_NN_packets']==done['actual_NN_packets']==162 and audit['training_updates']==done['new_optimizer_steps']==done['new_epochs']==0 and audit['real_source_IQ_reads']==done['source_IQ_reads']==0 and audit['head_fit_updates']==0 and not audit['target_access'] and not audit['identity_accuracy_computed'] and audit['model_state_unchanged']
 raw_joint=np.asarray(traces['identity']['joint'],dtype=float);raw_logits=np.asarray(traces['identity']['logits'],dtype=float);raw_power=np.asarray(traces['identity']['sinc_input_complex_power'],dtype=float)
 def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-4)
 values={}
 for name in names:
  t=traces[name];j=np.asarray(t['joint'],dtype=float);z=np.asarray(t['logits'],dtype=float);p=np.asarray(t['sinc_input_complex_power'],dtype=float);q=np.asarray(t['encoder_input_complex_power'],dtype=float)
  assert j.shape==(32,160) and z.shape==(32,6) and p.shape==q.shape==(32,) and np.isfinite(j).all() and np.isfinite(z).all() and np.isfinite(p).all() and np.array_equal(p,q)
  rel=np.linalg.norm(unit(j)-unit(raw_joint),axis=1)/np.maximum(np.linalg.norm(unit(raw_joint),axis=1),1e-20)
  v=dict(normalized_joint_change_median=float(np.sort(rel)[15]),normalized_joint_change_max=float(rel.max()),logits_abs_change_max=float(np.max(np.abs(z-raw_logits))),power_relative_error_max=float(np.max(np.abs(p-raw_power)/raw_power)))
  original=audit['views'][name]
  for k in ('normalized_joint_change_median','normalized_joint_change_max','logits_abs_change_max'):assert abs(v[k]-original[k])<=1e-10,(name,k)
  assert np.allclose(rel,original['normalized_joint_change'],rtol=0,atol=1e-10)
  values[name]=v
 controls=[values[n] for n in ('positive_inverse','negative_inverse')]
 conditions=dict(power_preserved=all(v['power_relative_error_max']<=1e-6 for v in values.values()),inverse_input=all(audit['views'][n]['inverse_input_relative_error_max']<=1e-6 for n in ('positive_inverse','negative_inverse')),inverse_function=all(v['normalized_joint_change_max']<=1e-4 and v['logits_abs_change_max']<=1e-3 for v in controls),nontrivial_default_function=any(values[n]['normalized_joint_change_median']>=1e-3 and values[n]['logits_abs_change_max']>=.01 for n in ('frequency_positive','frequency_negative')))
 qualification=dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions);assert qualification==audit['qualification'] and qualification['decision']==done['input_effect_decision']
 result=dict(status='VERIFIED_ENGINEERING_INPUT_ONLY',run_id='20261016-diagnostic-F-crossRX-frequency-input-s2026092701-m1-r02',qualification=qualification,views=values,all5x32_joint_logits_power_independently_recounted=True,actual_NN_packets=162,source_IQ_reads=0,training_updates=0,head_fit_updates=0,target_access=False,identity_accuracy_computed=False,Ag_efficacy_proved=False,joint_goal_complete=False,runtime_only_checks=['actual encoder/Sinc equality','inverse input L2','NoExplicitCFO coord27/internal compensation','model state and actual counters'],claim='Nontrivial actual default function effect only,not label preservation/DG/As benefit')
 (OUT/'analysis_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
