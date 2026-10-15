"""Expansion consumes completed source evidence only, independently per budget."""
from pathlib import Path
from . import design as d
from .provenance import source_proof
def evaluate_source_budget(e):
 first=[r for r in d.rows() if r['epochs']==e and r['model_seed']==d.SEEDS[0]]
 if len(first)!=3:raise ValueError('Fixed first-seed budget comparison required')
 if any(not (Path(d.config(r)['output_root'])/'source_frozen.json').is_file() for r in first):return None
 values={}
 for row in first:
  c=d.config(row);p=source_proof(c);values[row['mode']]=dict(done=d.read(p/'completion.json'),mechanism=d.read(p/'source_training_mechanism.json'))
 return qualify(e,values)
def qualify(e,values):
 if e not in d.BUDGETS or set(values)!=set(d.MODES):raise ValueError('Fixed per-budget three-arm source evidence required')
 steps=e*222
 if any(v['done']['optimizer_steps']!=steps or v['done']['target_access'] is not False or v['done']['target_evaluated'] is not False for v in values.values()):raise ValueError('Source-only complete evidence required')
 healthy=all(v['done']['final_training_L_CE']<=.5 for v in values.values());rd=values['RDROP'];metric=rd['done']['final_source_metrics'];positive=rd['mechanism']['positive_KL_calls']/steps
 checks={}
 for mode in ('ORIGINAL','TWO_CE'):
  ref=values[mode]['done']['final_source_metrics'];checks[mode]=dict(V_mean_difference=metric['source_val_accuracy']-ref['source_val_accuracy'],V_worst_RX_difference=metric['source_val_worst_rx']-ref['source_val_worst_rx'])
 passed=healthy and positive>=.01 and all(v['V_mean_difference']>=-.01 and v['V_worst_RX_difference']>=-.02 for v in checks.values())
 return dict(status='PASS' if passed else 'SOURCE_FALSIFIED',epochs=e,source_only=True,first_model_seed=d.SEEDS[0],all_final_L_healthy=healthy,positive_KL_call_fraction=positive,source_V_comparisons=checks,qualified=passed,target_metrics_read=False,remaining_seed_rows=6,no_parameter_scan=True)
