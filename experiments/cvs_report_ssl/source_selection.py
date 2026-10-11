"""C5 eligibility from preregistered source evidence only; no target file reads."""
from . import design as d

def select_if_ready():
 path=d.BASE/'C5_source_selection.json'
 if path.is_file():return d.read(path)
 required=[d.BASE/f'{a}-s{s}'/'source/completion.json' for a in ('C1','C2','C3','C4') for s in d.SEEDS]
 if not all(p.is_file() for p in required):return None
 scores={};details={};eligible=[]
 for a in ('C1','C2','C3','C4'):
  scores[a]=[]
  for seed in d.SEEDS:
   folder=d.BASE/f'{a}-s{seed}'/'source';done=d.read(folder/'completion.json')
   c=d.config(dict(arm=a,model_seed=seed,row_id=f'{a}-s{seed}'))
   if done['config']!=c or done['target_access'] or done['optimizer_steps']!=8880 or done['status']!='SOURCE_TRAINED':raise ValueError('C5 source evidence contract differs')
   scores[a].append(done['final_source_metrics']['source_selection_score'])
 for a in ('C2','C3','C4'):
  differences=[v-b for v,b in zip(scores[a],scores['C1'])]
  passed=sum(differences)/3>0 and min(differences)>=-.002;mechanisms=[]
  for seed in d.SEEDS:
   folder=d.BASE/f'{a}-s{seed}'/'source'
   if a in ('C2','C4'):
    task='linear' if a=='C2' else 'repetition';batches=d.read(folder/'physical_holdout.json')['batches'][task]
    good=sum(b['prediction_mse'] for b in batches)<sum(b['zero_baseline_mse'] for b in batches)
   else:
    diag=d.read(folder/'source_teacher_diagnostics.json')['rows'][-1]
    if diag['epoch']!=40:raise ValueError('Final teacher diagnostic absent')
    rows=diag['fixed_coverage'];metric=lambda k:next(r['error_rate'] for r in rows[k] if r['coverage']==.9)
    good=metric('condition_U')<metric('EMA')
   mechanisms.append(good)
  passed=passed and all(mechanisms)
  if passed:eligible.append(a)
  details[a]=dict(score_differences=differences,physical_or_teacher_gate=mechanisms,eligible=passed)
 value=dict(status='SOURCE_SELECTION_FROZEN',source_only=True,target_scores_consumed=False,rule=d.config(d.rows()[0])['C5_selection'],eligible_modules=eligible,C5='C1+'+'+'.join(eligible) if eligible else 'C1',future_run_required=True,current_C8_selected_by_this_rule=False,details=details)
 d.write(path,value);return value
