"""Freeze the complete first-seed source decision before reading any new target summary."""
from pathlib import Path
from . import design as d
from .provenance import source_proof

def evaluate_source():
 rows=[r for r in d.training_rows() if r['model_seed']==d.SEEDS[0]]
 if any(not (Path(d.config(r)['output_root'])/'source_feature_witness.json').is_file() for r in rows):return None
 evidence=[]
 for row in rows:
  c=d.config(row);p=source_proof(c);done=d.read(p/'completion.json');mechanism=d.read(p/'source_training_mechanism.json');feature=d.read(p/'source_feature_witness.json');unit=d.read(p/'sinc_optimizer_observation.json');s=feature['student_Sinc'];ema=feature['ownEMA_Sinc']
  checks=dict(source_L_healthy=done['final_training_L_CE']<=.5,actual_budget=done['optimizer_steps']==11100 and done['label_epochs']==33 and done['pseudo_epochs']==17,original_RDrop_positive=mechanism['positive_KL_calls']/11100>=.01,actual_feature_changed=feature['synthetic_initial_final_feature_L2']>=1e-6,export_exact=feature['feature_export_exact'] and feature['classW_zeroing_leaves_feature_identical'],default_head_not_fit=feature['head_fit_updates']==0)
  if c['sinc_optimizer_units']['mode']!='STANDARD':checks.update(cutoff_updated=24-s['parameter_changes']['low_hz_']['exactly_unchanged']>=12,bandwidth_updated=24-s['parameter_changes']['band_hz_']['exactly_unchanged']>=12,student_kernel_changed=s['kernel_relative_L2']>=1e-6,EMA_kernel_changed=ema['kernel_relative_L2']>=1e-6,Sinc_contributes_to_actual_feature=feature['synthetic_Sinc_only_counterfactual_L2']>=1e-6)
  if c['sinc_optimizer_units']['mode']=='CENTER_UNITS':checks['physical_clip_budget']=unit['physical_clip_calls']==11100
  evidence.append(dict(arm=c['arm'],row_id=c['row_id'],checks=checks,qualified=all(checks.values()),final_training_L_CE=done['final_training_L_CE'],positive_KL_call_fraction=mechanism['positive_KL_calls']/11100,feature=feature,optimizer=unit))
 passed=all(v['qualified'] for v in evidence)
 return dict(status='PASS' if passed else 'SOURCE_FALSIFIED',qualified=passed,first_seed=d.SEEDS[0],necessary_arms=list('BCDEF'),evidence=evidence,source_only=True,target_metrics_read=False,remaining_seeds=list(d.SEEDS[1:]),remaining_new_models=10,decision='Fixed full matrix expansion, no target feedback, sourceV ranking, parameter scan, or selective repeat. Upload this decision before reading own new target summaries.',source_falsification=d.config(rows[0])['source_falsification'])

def decision_uploaded(qualification):
 p=d.BASE/'seed_expansion_release.json'
 if not p.is_file():return False
 v=d.read(p)
 if v.get('status')!='EXPANSION_RELEASED' or v.get('source_only') is not True or v.get('new_target_summary_read') is not False or v.get('qualified')!=qualification['qualified'] or v.get('first_seed')!=d.SEEDS[0] or not v.get('commit') or v.get('remote_OID')!=v['commit']:raise ValueError('User-required source decision upload not verified')
 return True

def expansion_released(qualification):return decision_uploaded(qualification) and bool(qualification['qualified'])
