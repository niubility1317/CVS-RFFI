from copy import deepcopy
from pathlib import Path
from math import isfinite
from experiments.cvs_sinc_units50 import design as first
from experiments.cvs_sigtime_pair_seeds import design as models
ROOT=Path(__file__).resolve().parents[2]
for name in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write','ARMS','BASELINE_ARM','PRIMARY','FEATURE_CONTROL','PAIRS','READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS','PARAMETERS','DIAGNOSTIC_EPOCHS','SINC_DIAGNOSTIC_EPOCHS','FEATURE_PARAMETER_COUNT','VERIFY_INITIAL_EVAL_EQUAL'):
 globals()[name]=getattr(first,name)
RUN='20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s0203-m6-r01';RELEASE='cvs_sinc_units_seeds_20261015_r01';OWNER='codex/root/sinc-units-seeds-20261015';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092702,2026092703);baseline_config=first.baseline_config
MECHANISM_KEYS=('feature_combination','source_neighbor_sampling','sampling_observation','hypothesis_basis','source_falsification','goal','cost_claim','sinc_optimizer_units')
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered remaining Sinc unit row')
 skeleton=dict(row,arm=BASELINE_ARM if row['arm']==BASELINE_ARM else FEATURE_CONTROL,row_id=f"{BASELINE_ARM if row['arm']==BASELINE_ARM else FEATURE_CONTROL}-s{row['model_seed']}")
 c=models.config(skeleton);template=first.config(next(r for r in first.rows() if r['arm']==row['arm']))
 for key in MECHANISM_KEYS:c[key]=deepcopy(template[key])
 c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_qualification_run=first.RUN,first_seed_retrained=False,combined_model_seeds=[2026092701,*SEEDS]);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Remaining Sinc unit scientific recipe changed')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical oldgate row cannot retrain')
 old=models.config(next(r for r in models.training_rows() if r['arm']==FEATURE_CONTROL and r['model_seed']==c['model_seed']))
 a=models.make_args(old,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def validate_qualification(q):
 rules=first.config(first.training_rows()[0])['source_falsification'];expected={r['row_id'] for r in first.training_rows()}
 if q.get('status')!='PASS' or q.get('source_only') is not True or q.get('target_metrics_read_for_decision') is not False or q.get('fixed_rules')!=rules or q.get('predefined_remaining_seeds')!=list(SEEDS) or q.get('head_fit_updates')!=0:raise ValueError('Original source qualification missing, failed or changed')
 CE=q.get('last5_mean_L_CE',{})
 if set(CE)!={r['arm'] for r in first.training_rows()} or any(not isfinite(v) or v>rules['final5_L_CE_max'] for v in CE.values()):raise ValueError('Original source CE health missing')
 tests=(('unit_changed_cutoffs','unit_final_changed_cutoff_parameters_min'),('unit_changed_bandwidths','unit_final_changed_bandwidth_parameters_min'),('unit_student_kernel_relative_L2','unit_student_kernel_relative_L2_min'),('unit_ownEMA_kernel_relative_L2','unit_ownEMA_kernel_relative_L2_min'),('actual_candidate_minus_control_L720zL2','actual_candidate_minus_control_L720zL2_min'))
 if any(not isfinite(q.get(k,float('nan'))) or q[k]<rules[t] for k,t in tests):raise ValueError('Original Sinc adaptation or deployed feature thresholds failed')
 obs=q.get('actual_observations',[])
 if {v.get('row_id') for v in obs}!=expected or len(obs)!=2 or any(v.get('status')!='VERIFIED' or v.get('actual_steps')!=11100 or v.get('actual_U_packets')!=963900 or v.get('actual_Sinc_parameter_records')!=104 or v.get('actual_snapshot_batches')!=52 or v.get('head_fit_updates')!=0 or v.get('extra_model_forwards')!=0 or v.get('U_truth_read') is not False or v.get('query_read') is not False or v.get('source_only') is not True for v in obs):raise ValueError('Original actual native/source budget witnesses incomplete')
 return q
def require_source():return validate_qualification(read(first.BASE/'source_qualification.json'))
