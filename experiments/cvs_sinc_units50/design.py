from copy import deepcopy
from pathlib import Path
from experiments.cvs_sigtime_pair50 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02';RELEASE='cvs_sinc_units50_20261015_r02';OWNER='codex/root/sinc-units50-20261015';BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,);ARMS=('OLD_GATE_COS50','SIG_TIME','SINC_UNITS');BASELINE_ARM=ARMS[0];PRIMARY=ARMS[-1];FEATURE_CONTROL='SIG_TIME';FEATURE_PARAMETER_COUNT=195041;VERIFY_INITIAL_EVAL_EQUAL=True
PAIRS=[(PRIMARY,FEATURE_CONTROL),(PRIMARY,BASELINE_ARM),(FEATURE_CONTROL,BASELINE_ARM)]
READOUTS,VIEWS,QUERY_COUNT,EPOCHS,LABEL_EPOCHS,PSEUDO_EPOCHS,STEPS,UPDATES,BASELINE_EVAL_RUN,STYLE_LAYERS,PARAMETERS,DIAGNOSTIC_EPOCHS=(getattr(parent,k) for k in ('READOUTS','VIEWS','QUERY_COUNT','EPOCHS','LABEL_EPOCHS','PSEUDO_EPOCHS','STEPS','UPDATES','BASELINE_EVAL_RUN','STYLE_LAYERS','PARAMETERS','DIAGNOSTIC_EPOCHS'))
SINC_DIAGNOSTIC_EPOCHS=(1,17,33,50)
baseline_config=parent.baseline_config
def rows():return [dict(row_id=f'{arm}-s{seed}',arm=arm,model_seed=seed) for seed in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Unregistered optimizer unit row')
 skeleton=row if row['arm']==BASELINE_ARM else dict(row,arm='SIG_TIME',row_id=f"SIG_TIME-s{row['model_seed']}")
 c=deepcopy(parent.config(skeleton));c.update(run_id=RUN,**row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 c['sinc_optimizer_units']=dict(mode='NORMALIZED_UNITS' if row['arm']==PRIMARY else 'STANDARD',sample_rate_Hz=25000000.,lr_multiplier='Fs',epsilon_divisor='Fs',weight_decay_divisor='Fs',parameters=['id_backbone.encoder.core.id_backbone.sinc.low_hz_','id_backbone.encoder.core.id_backbone.sinc.band_hz_'],original_network_and_parameterization=True,other_parameters_including_W='originalnativeAdamW',original_global_gradient_clipping='unchangedHzspace clip5',native_cosine_schedule='unchanged base schedule; Sincgroup fixedphysical unit mapping beforeactualstep',extra_forwards=0,new_loss=0,diagnostic_epochs=list(SINC_DIAGNOSTIC_EPOCHS),diagnostic_batch_stride=16)
 c['sampling_observation']['optimizer']='OriginalAdamW on all nonSinc; separately declared original Sinc unit strategy only'
 c['hypothesis_basis']='Actual9historical checkpoint/optimizer/EMA states and3current SIG_TIME initial-final witnesses show Sinc almoststatic despite nonzero moments/subULP steps. Fix optimization units only, no structural/network/head changes, no target-driven multiplier scan.'
 c['source_falsification']=dict(final5_L_CE_max=.5,actual_updates=11100,actual_U_packets=963900,extra_model_forwards=0,unit_final_changed_cutoff_parameters_min=12,unit_final_changed_bandwidth_parameters_min=12,unit_student_kernel_relative_L2_min=1e-6,unit_ownEMA_kernel_relative_L2_min=1e-6,actual_candidate_minus_control_L720zL2_min=1e-6,source_only=True,expansion='After exact original budgets/defaultfeature/head and fixed unit numerical adaptation/sourcehealth PASS, push source decision before own new target summaries; extend both unchanged arms02/03. No target-based cancellation/retuning.')
 c['goal']=dict(primary=PRIMARY,clean_three_seed_mean=.84,first_seed_is_not_goal_completion=True,feedback_to_training=False);return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed optimizer unit recipe differs')
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Historical baseline cannot retrain')
 old=parent.config(next(r for r in parent.training_rows() if r['arm']=='SIG_TIME' and r['model_seed']==c['model_seed']))
 a=parent.make_args(old,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
