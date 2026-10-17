"""Single new scratchE20 row:remove only the As-pair stochastic KL."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_scene_cycle20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-F-crossRX-AsPairCE-sourceRX1-e20-s2026092701-m3-r01'
RELEASE='cvs_F_AsPairCE_sourceRX1_e20_20261017_r01'
OWNER='codex/root/F-AsPairCE-E20-20261017';LOCAL='local_artifacts/F_As_pair_CE_E20_20261017';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=old.BACKEND;BACKEND_COMMIT=old.BACKEND_COMMIT;SMOKE_SOURCE=old.SMOKE_SOURCE;CONTROL_BASE=old.BASE;CONTROL_COMMIT='c9a74172e5f81bb4f04ce12cd4386ad4dd953ee9'
SEED=2026092701;FIT=old.FIT;SCENES=old.SCENES;UPDATES=4440;ARMS=('Q0','Q_ord','C_CE');NEW_ARMS=('C_CE',)
base_model_config=old.base_model_config;source_validation=old.source_validation;macro_f1=old.macro_f1

def rows():return [dict(row_id='C_CE-RX1-s2026092701',arm='C_CE',model_seed=SEED)]
def legacy_config():return old.config(next(r for r in old.rows() if r['arm']=='C_IID'))
def controls():return {'Q0':CONTROL_BASE/'B00-RX1-s2026092701/source','Q_ord':CONTROL_BASE/'C_IID-RX1-s2026092701/source'}
def config(row):
 if row not in rows():raise ValueError('Only fixed single-seed As-TWO_CE row')
 c=deepcopy(legacy_config());c.update(schema='F_CROSS_RX_As_pair_CE_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_As_TWO_CE',baseline_reference='ExactQ0/noAs andQ_ord/AsRDrop matched8L/weight1.25;no weights loaded',training_recipe='Retain fullbase raw.75/ordinary.25+originalSupCon+U;extra.25As pair usesmeanCE only',budget_authorization='One new scratchE20/20modelE/4440updates;two reusedE20controls/40historicalE/8880updates separate',cost_claim='EightLforwards/4546560LNN;568320singlefullAs;originalU/EMA/V/8820held unchanged;no newforward ormodel')
 c['additive'].update(extra_pair_loss='Mean of two smoothedCE.01;zero appliedKL,detachedKLdiagnostic only',operator='One unchanged IIDfullpractical perL;two stochastic network passes ofonegeneratedIQ;extra meanCE only',common='Fulloriginalbase raw.75+ordinary.25+SupCon;no baseKL removal',BN_RNG='Original8forward ordering/dropout calls/privatechannelRNG unchanged;F has0BN')
 c['as_pair_objective']=dict(policy='TWO_CE',mean_CE_weight=.25,applied_KL_weight=0.,diagnostic_KL='Detached no_grad only,never entersloss/gradient',all_L_As_supervised=True,same_generated_IQ_for_both=True,original_base_RDrop_unchanged=True,original_SupCon_ordinary_only=True,new_loss_terms=0,new_NN_forwards=0,new_teacher_forwards=0,U_changed=False)
 c['source_selection']=dict(primary='C_CE',references=['Q0','Q_ord'],fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],vs_Q0=dict(min_clean_gain_pp=0.,min_clean_F1_gain_pp=0.,min_low_gain_pp=5.,min_low_F1_gain_pp=0.,min_mid_gain_pp=0.,min_urban_mean_gain_pp=3.,max_each_suburban_drop_pp=.5,max_seenV_drop_pp=.5),vs_Q_ord=dict(max_clean_drop_pp=.5,max_clean_F1_drop_pp=.5,min_low_gain_pp=1.,min_low_F1_gain_pp=0.,min_mid_gain_pp=0.,min_urban_mean_gain_pp=1.,max_each_suburban_drop_pp=.5,max_seenV_drop_pp=.5),final_base_L_objective_max=.5,on_fail='ClosefixedAsKLremoval;no intermediateKLweight/CEweight/scene/epoch/RX/seed rescue',on_pass='Separatelyregister independent source seed confirmation;notAg/fulljoint/targetproof')
 return c
def validate(c):
 if c!=config(rows()[0]):raise ValueError('FixedAsCE-only/base/roles/seed/budget changed')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(legacy_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All3logical source rows required')
 candidate=results['C_CE'];conditions={};comparisons={};eps=1e-10
 for ref in ('Q0','Q_ord'):
  b=results[ref];delta={v:100*(candidate['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};cf=100*(candidate['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);lf=100*(candidate['views'][SCENES[-1]]['macro_F1']-b['views'][SCENES[-1]]['macro_F1']);urban=sum(delta[v] for v in SCENES[3:])/3;sv=100*(candidate['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy'])
  rule=dict(clean=delta['clean']>=(-.5 if ref=='Q_ord' else 0.)-eps,clean_F1=cf>=(-.5 if ref=='Q_ord' else 0.)-eps,low=delta[SCENES[-1]]>=(1. if ref=='Q_ord' else 5.)-eps,low_F1=lf>=-eps,mid=delta[SCENES[-2]]>=-eps,urban_mean=urban>=(1. if ref=='Q_ord' else 3.)-eps,suburban=all(delta[v]>=-.5-eps for v in SCENES[:3]),seenV=sv>=-.5-eps)
  conditions.update({ref+'/'+k:v for k,v in rule.items()});comparisons[ref]=dict(delta_accuracy_pp=delta,delta_clean_F1_pp=cf,delta_low_F1_pp=lf,delta_urban_mean_pp=urban,delta_seenV_pp=sv)
 for arm,r in results.items():
  conditions[arm+'/execution']=r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['total_pair_weight']==1.25 and r['As_training_observations']==(0 if arm=='Q0' else 568320) and r['head_fit_updates']==0 and r['target_access'] is False and r['final_base_L_objective']<=.5
 conditions['C_CE/objective']=candidate.get('As_pair_objective')=='TWO_CE' and candidate.get('As_KL_loss_weight')==0. and candidate.get('As_KL_gradient_terms')==0 and candidate.get('As_pair_calls')==4440
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,primary='C_CE',source_only=True,Ag_efficacy_proved=False,joint_goal_complete=False,on_fail='ClosefixedAsKLremoval;no coefficient/scene/epoch/RX/seed rescue')
