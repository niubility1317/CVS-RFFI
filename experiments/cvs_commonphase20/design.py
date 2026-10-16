"""Fixed single-seed source falsification of the phase-only L view."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-crossRX-commonphase-sourceRX1-e20-s2026092701-m2-r01'
RELEASE='cvs_F_crossRX_commonphase_sourceRX1_e20_20261016_r01'
OWNER='codex/root/F-CROSS_RX-commonphase-E20-20261016'
LOCAL='local_artifacts/F_crossrx_commonphase20_20261016'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01'
BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('B00','B10');UPDATES=4440;PHASE_SEED=20261016201
SCENES=prior.SCENES
SMOKE_SOURCE=Path(PROJECT)/'runs'/prior.RUN/'B00-RX1-s2026092701/source'
base_model_config=prior.base_model_config
source_validation=prior.source_validation
macro_f1=prior.macro_f1
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Only fixed B00/B10 source rows permitted')
 c=deepcopy(prior.config(prior.rows()[0]));c.update(schema='F_CROSS_RX_commonphase_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_bounded_commonphase',baseline_reference='New scratch F+CROSS_RX B00,matched6 L forwards/raw .5/ordinary .25/slot .25')
 c['training_mechanism'].update(Ag=row['arm']=='B10',As=False,s_slot_name='raw(B00) or phase_only(raw)(B10)',no_new_U_augmentation=True)
 c['phase']=dict(angle_rad=[-.37,.37],seed=PHASE_SEED,distribution='Independent uniform per physical L packet and step,private CPU generator seed+step',input='raw L only,not ordinary augmentation',label_or_RX_dependence=False,renormalization=False,frequency_shift=False,extra_noise=False,extra_multipath=False,extra_PA=False,composition=False,same_generated_IQ_within_RDrop_pair=True,supervised='All128 legal phase L packets,no confidence or class filtering')
 c['channel']['input']='Held-source evaluation raw only;no As training';c['channel']['selection']='Six fixed held-source views after final20 only'
 c['training_recipe']='Scratch E20/13L+7U/F+CROSS_RX/defaultcos/RDrop1/LS.01; .5raw+.25shared ordinary+.25(raw or phase-only raw),six L forwards;U unchanged'
 c['source_selection']=dict(primary='B10',reference='B00',fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_seenV_drop_pp=.5,final_L_CE_max=.5,on_fail='Close fixed bounded common-phase training recipe;no angle/seed/RX rescue',on_pass='Supports separate independent source confirmation/full-source factorial preregistration only;not Ag proof or automatic target')
 c['cost_claim']='Two scratchE20/8880updates;each3409920 L NNpackets plus actual U/ownEMA/V/smoke/8820held;568320 phase IQ views in B10,0 inB00;0As training/0headfit,extra costs matched'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed phase budget/roles/recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=prior.make_args(prior.config(prior.rows()[0]),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Both fixed source rows required')
 b,g=(results[a] for a in ARMS);delta=100*(g['views']['clean']['accuracy']-b['views']['clean']['accuracy']);f1=100*(g['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);v=100*(g['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy'])
 healthy=all(r['actual_steps']==4440 and r['L_forward_packets']==3409920 and r['supcon_applied']==3330 and r['final_L_CE']<=.5 and not r['target_access'] and not r['head_fit_updates'] and r['As_training_observations']==0 and r['phase_training_observations']==(568320 if arm=='B10' else 0) for arm,r in results.items())
 conditions=dict(healthy=healthy,clean=delta>=1.,clean_macroF1=f1>=0.,seen_V=v>=-.5)
 return dict(qualified=all(conditions.values()),conditions=conditions,clean_gain_pp=delta,clean_macroF1_gain_pp=f1,seen_V_gain_pp=v,all_view_gain_pp={s:100*(g['views'][s]['accuracy']-b['views'][s]['accuracy']) for s in ('clean',*SCENES)},target_access=False,Ag_efficacy_proved=False,goal_complete=False)
