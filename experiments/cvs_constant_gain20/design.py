"""Fixed bounded linear observation gain;no source/target amplitude search."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-crossRX-constantgain-sourceRX1-e20-s2026092701-m2-r01'
RELEASE='cvs_F_crossRX_constantgain_sourceRX1_e20_20261016_r01'
OWNER='codex/root/F-CROSS_RX-constantgain-E20-20261016'
LOCAL='local_artifacts/F_constant_gain20_20261016'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01'
BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('B00','B10');UPDATES=4440;GAIN_SEED=20261016301
SCENES=prior.SCENES
SMOKE_SOURCE=Path(PROJECT)/'runs'/prior.RUN/'B00-RX1-s2026092701/source'
GAIN_RANGE=(.9,1.1);AUDIT_SEED=20261016321
base_model_config=prior.base_model_config
source_validation=prior.source_validation
macro_f1=prior.macro_f1
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Fixed constant-gain source rows only')
 c=deepcopy(prior.config(prior.rows()[0]));c.update(schema='F_CROSS_RX_constant_gain_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_constant_received_gain',baseline_reference='New scratch F+CROSS_RX B00,matched6L forwards/raw .5/oldordinary .25/slot .25')
 c['training_mechanism'].update(Ag=row['arm']=='B10',As=False,s_slot_name='raw(B00) or bounded_constant_gain(raw)(B10)',no_new_U_augmentation=True)
 c['gain']=dict(range=list(GAIN_RANGE),seed=GAIN_SEED,distribution='Uniform independent scalar per physical L packet/step,private CPU generator seed+step',input='raw L only,not ordinary augmentation',range_basis='Existing verified ordinary amp_min .9/amp_max1.1;not chosen by source practical or target',label_or_RX_dependence=False,renormalization=False,frequency_shift=False,extra_noise=False,extra_multipath=False,extra_PA=False,composition=False,same_generated_IQ_within_RDrop_pair=True,supervised='All128 legal gain L packets,no confidence/class filtering',meaning='Linear scaling of already received IQ,not TX PA change or actual RX substitution;not failed within-packet ramp')
 c['input_audit']=dict(real_CP_smoke_packets=2,synthetic_packets=32,synthetic_seed=AUDIT_SEED,families=['Gaussian','ARcolored','QPSKsmoothed','chirp+noise'],views=['identity','gain0.9','gain1.1','gain0.9_RMSrestored','gain1.1_RMSrestored'],actual_NN_packets=162,real_source_IQ_reads=0,criteria=dict(gain_RMS_relative_error_max=1e-6,restored_input_relative_error_max=1e-6,restored_normalized_joint_relative_error_max=1e-4,restored_logits_abs_error_max=1e-3,at_least_one_endpoint_median_normalized_joint_change_min=1e-3,at_least_one_endpoint_logits_abs_change_max_min=.01),on_fail='Close fixed gain candidate,no real source IQ/training/range rescue',on_pass='Continue scratchE20 immediately;engineering input effect only,not identity accuracy/Ag evidence')
 c['channel']['input']='Held-source evaluation raw only;no As training';c['channel']['selection']='Six fixed held-source views after final20 only'
 c['training_recipe']='Scratch E20/13L+7U/F+CROSS_RX/defaultcos/RDrop1/LS.01; .5raw+.25unchanged oldordinary+.25(raw or constantgain raw),six L forwards;originalU/EMA and native augmentation min(epoch,13) unchanged'
 c['source_selection']=dict(primary='B10',reference='B00',fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5,final_L_CE_max=.5,on_fail='Close fixed constant-gain training recipe,no range/seed/RX rescue',on_pass='Only supports independent source confirmation,not Ag/DG proof or automatic target')
 c['cost_claim']='First162syntheticNN/0source/0updates;close with0training if input effect fails. Otherwise2scratchE20/8880updates;each3409920LNN+actual originalU/ownEMA/V/smoke/8820held;B10 568320gain IQ outputs,0As/headfit/parent training weights'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed gain budget/roles/recipe/range differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=prior.make_args(prior.config(prior.rows()[0]),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Both fixed gain source rows required')
 b,g=(results[a] for a in ARMS);delta={s:100*(g['views'][s]['accuracy']-b['views'][s]['accuracy']) for s in ('clean',*SCENES)};f1=100*(g['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);v=100*(g['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy'])
 healthy=all(r['actual_steps']==4440 and r['L_forward_packets']==3409920 and r['supcon_applied']==3330 and r['final_L_CE']<=.5 and not r['target_access'] and not r['head_fit_updates'] and r['As_training_observations']==0 and r['gain_training_observations']==(568320 if arm=='B10' else 0) and r['gain_max_power_error']<=1e-6 for arm,r in results.items())
 conditions=dict(healthy=healthy,clean=delta['clean']>=1.,clean_macroF1=f1>=0.,practical_protection=all(delta[s]>=-.5 for s in SCENES),seen_V=v>=-.5)
 return dict(qualified=all(conditions.values()),conditions=conditions,clean_gain_pp=delta['clean'],clean_macroF1_gain_pp=f1,seen_V_gain_pp=v,all_view_gain_pp=delta,target_access=False,Ag_efficacy_proved=False,goal_complete=False)
