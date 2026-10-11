"""Fixed source-only design, before any new target scores are consumed."""
from pathlib import Path
from copy import deepcopy
from experiments.cvs_stage2_ssl import design as s
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES=s.PROJECT,s.SOURCE,s.CAPSULE,s.TRUTH,s.CLASSES
SEEDS=s.SEEDS
read,write=s.read,s.write
FULL_FP32_POLICY=s.FULL_FP32_POLICY
AUDIT_RUN,AUDIT_BASE=s.AUDIT_RUN,s.AUDIT_BASE
parent_config=s.parent_config
RUN='20261011-phase1-report-physical-ssl-manysig-m24-r01'
RELEASE='cvs_report_ssl_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
RELEASE_BASE=Path(PROJECT)/'releases'/RELEASE
OWNER='codex/root/report-physical-ssl-20261011'
UNLABELED_CACHE=s.UNLABELED_CACHE
ARMS=('C0','C1','C2','C3','C4','C6','C7','C8')
NAMES=dict(C0='matched no-explicit-CFO L-only CE',C1='soft source SSL',C2='SSL + known linear action',C3='SSL + condition teacher',C4='SSL + repetition residual reliability',C6='SSL + nonlinear memory identity',C7='SSL + unit-aware frequency',C8='fixed SSL + linear + repetition + unit-aware frequency')
PAIRS=[(a,'C0') for a in ARMS[1:]]+[(a,'C1') for a in ARMS[2:]]+[('C8','C2'),('C8','C4'),('C8','C7')]
CONTROL_RUN=s.RUN
CONTROL_ARMS=()
BASE_ARMS={a:'S13' for a in ARMS} # Parent metadata only; no S scores select this matrix.

def rows():return [dict(arm=a,model_seed=seed,row_id=f'{a}-s{seed}') for seed in SEEDS for a in ARMS]

def config(row):
 c=deepcopy(s.config(dict(arm='S13',model_seed=row['model_seed'],row_id=f"S13-s{row['model_seed']}")))
 a=row['arm'];ssl=a!='C0'
 c.update(schema='report_physical_ssl_v1',run_id=RUN,**row,method='cvs_report_physical_ssl',method_name=NAMES[a],output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,teacher=ssl,unlabeled=ssl)
 c['enabled']={k:False for k in c['enabled']}
 c['enabled'].update(linear=a in ('C2','C8'),repetition=a in ('C4','C8'),condition=a=='C3',nonlinear=a=='C6',frequency=a in ('C7','C8'))
 c['losses']=dict(ce=1.,soft_identity=1.,linear=.05,linear_identity=.1,repetition=.05)
 c['semi']=dict(cache=UNLABELED_CACHE.as_posix(),batch_size=128,role='U_s',count=56700,mode='soft_KL',ema_decay=.99,L_readapt_updates=888,ramp_steps=888,confidence_threshold=.95,strong_noise_std=.015,truth_access=False,teacher_selected_for_test=False,teacher_init='after update888, from adapted student',identity_denominator='fixed B_U',statistics_mask='separate physical quality, all soft assignments',conditional_refresh_epochs=[4,12,20,28,36],conditional_mix=.5,U_mean_alpha_cap=.2)
 c['optimizer'].update(encoder_peak_lr=1e-4,fusion_head_peak_lr=3e-4)
 c['optimizer']['frequency']=dict(peak_lr=3e-6,coordinate='u=f/fs',master_dtype='float64',weight_decay=0.,step_cap='actual scheduled LR',raw_parameter_floor_Hz=1.)
 c['sampling']=dict(L='RX/day balanced full permutation',U='RX/day balanced full permutation',batch_L=128,batch_U=128,TX_grouping_or_same_sig_pairing=False)
 c['physical']=dict(every_steps=4,max_L=32,max_U=32,linear_side_tap_scale=.015,action_targets='four actual applied tap components / scale',linear_readout_input='paired pre-classifier identity160 + PA160 + reference29',repetition_target='leave-cycle relative template residual24 + learned diagonal uncertainty',strict_blindspot=False,known_linear_scope='small circular digital filter; interior-only diagnostic required',old_aux_caches_reused=False)
 c['frontend']=dict(version='report_no_explicit_cfo_v1',reference_dim=29,zero_slot=27,state_dim=40,explicit_frequency_estimator=False,raw_IQ_retained=True,whole_model_frequency_invariant=False)
 c['checkpoint_selection']='final_only'
 c['evaluation']=dict(primary_arm='C8',goal_three_seed_mean=.84,all_predeclared_arms_reported=True,truth_last=True,feedback_to_training=False,fixed_control_run=None,fixed_control_arms=[],control_role='matched C0 and C1; original E200 predictions reused')
 c['C5_selection']=dict(status='deferred_until_complete_source_evidence',score='.5*mean_V_accuracy+.5*worst_RX_V_accuracy',eligibility='three complete seeds; module mean score gain over C1 >0 and no seed loss >0.002; C2/C4 additionally heldout physical prediction MSE below zero baseline in all seeds; C3 additionally lower V pseudo error at fixed90% coverage in all seeds',target_scores_allowed=False,combination='all eligible modules, otherwise C1; future distinct run, same parents and40 epochs',C8='fixed exploratory combination, not source-selected C5')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Unregistered report SSL matrix')
 return c
learning_rate=s.learning_rate

def frequency_learning_rate(c,step):return learning_rate(c,step,c['optimizer']['encoder_peak_lr'])/c['optimizer']['encoder_peak_lr']*c['optimizer']['frequency']['peak_lr']
