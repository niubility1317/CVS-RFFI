"""Fixed same-seed/source-fold E20 comparison;no target feedback or new U rule."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-crossRX-frequency-sourceRX1-e20-s2026092701-m2-r01'
RELEASE='cvs_F_crossRX_frequency_sourceRX1_e20_20261016_r01'
OWNER='codex/root/F-CROSS_RX-frequency-E20-20261016'
LOCAL='local_artifacts/F_frequency20_20261016'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01'
BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('B00','B10');UPDATES=4440;FREQUENCY_SEED=20261016401
FREQUENCY_MAX=0.00010024013357398965
SCENES=prior.SCENES;SMOKE_SOURCE=Path(PROJECT)/'runs'/prior.RUN/'B00-RX1-s2026092701/source'
base_model_config=prior.base_model_config;source_validation=prior.source_validation;macro_f1=prior.macro_f1
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Fixed frequency source rows only')
 c=deepcopy(prior.config(prior.rows()[0]));c.update(schema='F_CROSS_RX_frequency_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_received_frequency_only',baseline_reference='New scratch F+CROSS_RX B00,matched6L forwards/raw .5/oldordinary .25/slot .25')
 c['training_mechanism'].update(Ag=row['arm']=='B10',As=False,s_slot_name='raw(B00) or frequency(raw)(B10)',no_new_U_augmentation=True)
 c['frequency']=dict(max_cycles_per_sample=FREQUENCY_MAX,seed=FREQUENCY_SEED,distribution='Uniform symmetric independent per L packet/step,private CPU generator seed+step',input='RawL only,not oldordinary',operator='Native zero-origin x[n]exp(j2pi f n),n0..255',range_basis='Actual oldordinary .0004 times E13 cap scale .2506003339349741;no sourcepractical/target tuning',NoExplicitCFO=True,internal_reference_compensation=True,whole_model_invariance_assumed=False,label_or_RX_dependence=False,renormalization=False,extra_noise=False,extra_multipath=False,extra_PA=False,composition=False,same_generated_IQ_within_RDrop_pair=True,supervised='All128 legal L packets,no confidence/class filtering',engineering_ref='20261016-diagnostic-F-crossRX-frequency-input-s2026092701-m1-r02:162syntheticNN/0realIQ;function effect only,not identity efficacy')
 c['channel']['input']='Held-source evaluation raw only;no As training';c['channel']['selection']='Six fixed held-source views after final20 only'
 c['training_recipe']='ScratchE20;first13 L,last7 L+originalU;F+CROSS_RX/defaultcos/RDrop1/LS.01; .5raw+.25oldordinary+.25(raw or frequencyraw),6L forwards;native min(epoch,13) augmentation schedule retained'
 c['source_selection']=dict(primary='B10',reference='B00',fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5,final_L_CE_max=.5,on_fail='Close fixed frequency recipe,no range/seed/RX/threshold rescue',on_pass='Only supports independent source confirmation,not Ag/DG proof or automatic target')
 c['cost_claim']='2scratchE20/8880updates;each3409920LNN plus actual originalU/ownEMA/V/2scratchsmoke/8820held;B10 568320frequencyIQ outputs,0As/headfit/parent training weights;separate newrealCPsmoke2,priorengineering162/localcost separate'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed frequency budget/roles/recipe/range differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=prior.make_args(prior.config(prior.rows()[0]),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Both fixed frequency source rows required')
 b,g=(results[a] for a in ARMS);delta={s:100*(g['views'][s]['accuracy']-b['views'][s]['accuracy']) for s in ('clean',*SCENES)};f1=100*(g['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);v=100*(g['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy'])
 healthy=all(r['actual_steps']==4440 and r['L_forward_packets']==3409920 and r['supcon_applied']==3330 and r['final_L_CE']<=.5 and not r['target_access'] and not r['head_fit_updates'] and r['As_training_observations']==0 and r['frequency_training_observations']==(568320 if arm=='B10' else 0) and r['frequency_max_power_error']<=1e-6 for arm,r in results.items())
 conditions=dict(healthy=healthy,clean=delta['clean']>=1.,clean_macroF1=f1>=0.,practical_protection=all(delta[s]>=-.5 for s in SCENES),seen_V=v>=-.5)
 return dict(qualified=all(conditions.values()),conditions=conditions,clean_gain_pp=delta['clean'],clean_macroF1_gain_pp=f1,seen_V_gain_pp=v,all_view_gain_pp=delta,target_access=False,Ag_efficacy_proved=False,goal_complete=False)
