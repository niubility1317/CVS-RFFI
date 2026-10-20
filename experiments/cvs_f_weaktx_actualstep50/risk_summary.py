"""Read-only descriptive summaries of already completed source step records."""
import json,gzip,sys
from pathlib import Path
from statistics import mean,median
RISKS=('weak14-7_CE','weak20-19_CE','other4_CE');TOL=.001

def stats(v):
 s=sorted(v);n=len(s)
 return dict(n=n,mean=mean(s) if n else None,median=median(s) if n else None,min=s[0] if n else None,max=s[-1] if n else None)
def scope_summary(records):
 eligible=[v for v in records if v['eligible']];full=[v for v in eligible if v['strength']==1.];clean=[v for v in full if not any(v[k] for k in ('solver_fallback','projection_storage_fallback','storage_match_fallback'))];risk={}
 for i,name in enumerate(RISKS):
  rows=[]
  for v in clean:
   den=max(v['raw_step_normalized_norm'],1e-30)
   rows.append((v['raw_normalized_dots'][i]/den,v['candidate_stored_normalized_dots'][i]/den,v['applied_normalized_dots'][i]/den))
  risk[name]=dict(full_strength_nonfallback_events=len(rows),raw=stats([x[0] for x in rows]),counterfactual_candidate=stats([x[1] for x in rows]),actually_applied=stats([x[2] for x in rows]),raw_positive_strict=sum(x[0]>0 for x in rows),applied_positive_strict=sum(x[2]>0 for x in rows),raw_increase_beyond_storage_tolerance=sum(x[0]>TOL for x in rows),applied_increase_beyond_storage_tolerance=sum(x[2]>TOL for x in rows),raw_increase_removed_beyond_tolerance=sum(x[0]>TOL and x[2]<=TOL for x in rows),applied_risk_change=stats([x[2]-x[0] for x in rows]))
 return dict(native_steps=len(records),scheduled=sum(v['scheduled'] for v in records),eligible=len(eligible),full_strength=len(full),full_strength_nonfallback=len(clean),actual_changes=sum(v['actual_correction_nonzero'] for v in records),fallback_events=sum(any(v[k] for k in ('solver_fallback','projection_storage_fallback','storage_match_fallback')) for v in eligible),constraint_enforced=sum(v['constraint_enforced'] for v in eligible),retained_normalized_step_norm_ratio=stats([v['retained_normalized_norm_ratio'] for v in eligible]),actual_raw_step_cosine=stats([v['applied_raw_cosine'] for v in eligible if v['applied_raw_cosine'] is not None]),applied_normalized_correction_fraction=stats([v['applied_normalized_correction_fraction'] for v in eligible]),active_QP_constraints={name:sum(i in v['active_constraints'] for v in clean) for i,name in enumerate(RISKS)},per_risk=risk)
def summarize(base):
 base=Path(base);matrix=json.loads((base/'frozen_matrix.json').read_text())
 rows=[]
 for r in matrix['rows']:
  with gzip.open(base/r['row_id']/'source/actual_feature_steps.jsonl.gz','rt',encoding='utf8') as f:values=[json.loads(line) for line in f]
  rows.append(dict(row_id=r['row_id'],treatment=r['treatment'],seed=r['model_seed'],all=scope_summary(values),L_epochs1_to33=scope_summary([v for v in values if v['epoch']<=33]),U_epochs34_to50=scope_summary([v for v in values if v['epoch']>=34])))
 return dict(status='DESCRIPTIVE_SOURCE_LOG_ANALYSIS',source_only=True,target_access=False,rows=rows,relative_storage_tolerance=TOL,normalization='Each risk dot divided by same-state RAW normalized step L2norm. Unit-risk normal in native-step per-block RMS metric. Signs describe old-head first-order sourceCE changes only.',limitation='Not measured nonlinear CE changes or post-update-head risks, and not heldRX/target performance. Each run has its own trajectory; candidate counterfactual and actually-applied risks are separate. Strict-positive floating residuals differ from residuals beyond fixed storage tolerance.')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('base');a=p.parse_args()
 print(json.dumps(summarize(a.base),allow_nan=False))
