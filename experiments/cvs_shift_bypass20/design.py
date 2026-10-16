"""One source fold/seed,unchanged F+CROSS_RX and hidden-U recipe."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as prior
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-crossRX-shiftbypass-sourceRX1-e20-s2026092701-m2-r01'
RELEASE='cvs_F_crossRX_shiftbypass_sourceRX1_e20_20261016_r01'
OWNER='codex/root/F-CROSS_RX-shiftbypass-E20-20261016'
LOCAL='local_artifacts/F_shift_bypass20_20261016'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01'
BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('B00','B10');UPDATES=4440
SCENES=prior.SCENES
SMOKE_SOURCE=Path(PROJECT)/'runs'/prior.RUN/'B00-RX1-s2026092701/source'
base_model_config=prior.base_model_config
source_validation=prior.source_validation
macro_f1=prior.macro_f1
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def original_config():return prior.config(prior.rows()[0])
def config(row):
 if row not in rows():raise ValueError('Fixed shift-bypass source rows only')
 c=deepcopy(original_config());c.update(schema='F_CROSS_RX_shift_bypass_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_ordinary_shift_bypass',baseline_reference='New scratch F+CROSS_RX original B00,matched6L forwards/raw.5 ordinary.25 rawslot.25')
 c['training_mechanism'].update(Ag=False,As=False,ordinary_shift_bypass=row['arm']=='B10',no_new_U_augmentation=True,s_slot_name='Raw in both arms;only ordinary-L circular shift output is bypassed in B10')
 c['shift_counterfactual']=dict(scope='Original visibleL normal call only;U unchanged',rule='Both arms compute native circular gather and preserve its draw;B00 returns gathered IQ,B10 returns pre-shift IQ;remaining general channel/DAC/PA execute normally',parameters='Actual native min(training_epoch,13);no range/probability change or forced triggers',normalization='No extra RMS/window or input correction',cost='Single whole native augmentation per arm/step;both inspect identical shift draw via RNG replay and restore;no extra NN;scalar draws counted separately',label_dependence='No RX/TX condition;native visible L labels only,class signature disabled',same_IQ_pair=True)
 c['channel']['input']='Held-source evaluation raw only;no As training';c['channel']['selection']='Six fixed practical source views after final20 only'
 c['training_recipe']='Scratch E20/13L+7U/F+CROSS_RX/defaultcos/RDrop1/LS.01; .5raw+.25ordinary+.25raw,sixL forwards;ordinary circular shift bypass only in B10;original U/EMA/gates unchanged'
 c['source_selection']=dict(primary='B10',reference='B00',fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5,final_L_CE_max=.5,on_fail='Close fixed shift-bypass training recipe;no seed/RX/intensity/threshold rescue',on_pass='Supports separate independent source confirmation only;not Ag/DG efficacy or automatic target')
 c['cost_claim']='Two scratchE20/8880updates;each3409920L NN plus originalU/EMA/V/smoke/8820held;each568320native ordinary generations;only actually triggered shifts applied,draw inspection counted;zeroAs/extraNN/headfit/trainingparent'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed shift-bypass budget/data/recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=prior.make_args(original_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Both fixed source rows required')
 b,g=(results[a] for a in ARMS);delta={s:100*(g['views'][s]['accuracy']-b['views'][s]['accuracy']) for s in ('clean',*SCENES)};f1=100*(g['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);v=100*(g['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy'])
 healthy=all(r['actual_steps']==4440 and r['L_forward_packets']==3409920 and r['supcon_applied']==3330 and r['final_L_CE']<=.5 and not r['target_access'] and not r['head_fit_updates'] and r['As_training_observations']==0 and r['shift_native_L_calls']==4440 and r['shift_native_L_packets']==568320 and r['shift_bypass_returned_packets']==(568320 if arm=='B10' else 0) for arm,r in results.items())
 conditions=dict(healthy=healthy,clean=delta['clean']>=1.,clean_macroF1=f1>=0.,practical_protection=all(delta[s]>=-.5 for s in SCENES),seen_V=v>=-.5)
 return dict(qualified=all(conditions.values()),conditions=conditions,clean_gain_pp=delta['clean'],clean_macroF1_gain_pp=f1,seen_V_gain_pp=v,all_view_gain_pp=delta,suburban_gaps_pp={arm:{s:100*(r['views']['clean']['accuracy']-r['views'][s]['accuracy']) for s in SCENES[:3]} for arm,r in results.items()},target_access=False,Ag_efficacy_proved=False,goal_complete=False)
