"""Fixed additive-As/raw-supervision controls, same scratchE20 source contract."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01'
RELEASE='cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01'
OWNER='codex/root/F-CROSS_RX-additiveAs-E20-20261016'
LOCAL='local_artifacts/F_additive_As20_20261016'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01'
BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SMOKE_SOURCE=Path(PROJECT)/'runs'/old.RUN/'B00-RX1-s2026092701/source'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('D00','D_raw','D_As');UPDATES=4440;SCENES=old.SCENES
base_model_config=old.base_model_config
source_validation=old.source_validation
macro_f1=old.macro_f1
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def original_config():return old.config(old.rows()[0])
def config(row):
 if row not in rows():raise ValueError('Only fixed three additive source rows permitted')
 c=deepcopy(original_config());c.update(schema='F_CROSS_RX_additive_As_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_additive_As_controls',baseline_reference='Same original6forward raw.75/ordinary.25 loss plus matched extra IQ pair;all3rows8Lforwards')
 arm=row['arm'];weight=0. if arm=='D00' else .25
 c['training_mechanism'].update(Ag=False,As=arm=='D_As',base_raw_total_weight=.75,raw_total_weight=1. if arm=='D_raw' else .75,ordinary_weight=.25,raw_weight=.5,s_slot_weight=.25,s_slot_name='Raw in all rows;original entire base objective retained',extra_pair_weight=weight,extra_pair_input='complete practical(raw)' if arm=='D_As' else 'raw',L_forwards_per_packet=8,expected_L_packets=4546560,no_new_U_augmentation=True)
 c['additive']=dict(common='Unchanged original3pair base,including original ordinary-only SupCon',extra_input='As(raw)' if arm=='D_As' else 'raw',extra_weight=weight,extra_pair_loss='Original label-smoothed CE mean plus within-IQ RDrop1 KL;no clean-sat alignment',base_pair_weight_sum=1.,pair_weight_sum=1.+weight,normalization='Per-physical packet mean in each IQ pair;no divide by newview count;absolute base weights unchanged;relative raw fraction changes',BN_RNG='All3rows train-mode8forwards,including explicit zero-extra-loss D00 cost control;record BN updates and actual sequences',operator='One full practical channel per visibleL packet only inD_As;no ordinary input/double channel/confidence filter',channel_seeds='Original fixed channel/session/source-eval streams',original_U='Unchanged original13L+7U rule,truth hidden;no extra U/teacher forwards')
 c['updates']=UPDATES
 c['budget_authorization']='Active joint Ag/As research;fixed three matched scratchE20 source controls,60model epochs/13320updates;no continuation or target access'
 c['training_recipe']='ScratchE20/F+CROSS_RX/defaultcos/NoExplicitCFO/originalPA+shift/U;retainraw.75+ordinary.25+SupCon;8Lforwards;extra raw0/raw.25/As(raw).25'
 c['channel']['input']='Original raw L only in additive fourth IQ pair;third common pair stays raw'
 c['source_selection']=dict(primary='D_As',references=['D00','D_raw'],fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],vs_D00=dict(min_clean_gain_pp=0.,min_clean_macroF1_gain_pp=0.,min_low_urban_gain_pp=5.,min_urban_mean_gain_pp=3.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5),vs_D_raw=dict(max_clean_drop_pp=.5,max_clean_macroF1_drop_pp=.5,min_low_urban_gain_pp=5.,min_urban_mean_gain_pp=3.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5),final_common_base_L_objective_max=.5,on_fail='Close fixed additive organization,no coefficient/seed/RX/scene-probability or extra-budget rescue',on_pass='Only supports independent source confirmation,not Ag/joint efficacy or automatic target')
 c['cost_claim']='3scratchE20/13320updates;each4546560LNN plus actual originalU/EMA/V/8820held;each568320native ordinary;D_As568320fullAs observations;extra2NN allrows;no parent weight/extra teacher/headfit'
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed additive source roles/budget/weights/controls differ')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(original_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All3fixed source rows required')
 s=results['D_As'];comparisons={};conditions={}
 for ref in ARMS[:2]:
  b=results[ref];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in SCENES[3:])/3
  protect=0. if ref=='D00' else -.5
  part=dict(clean=delta['clean']>=protect,clean_macroF1=f1>=protect,low_urban=delta[SCENES[-1]]>=5.,urban_mean=urban>=3.,practical_protection=all(delta[v]>=-.5 for v in SCENES),seen_V=dv>=-.5)
  comparisons[ref]=dict(conditions=part,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seen_V_gain_pp=dv,urban_mean_gain_pp=urban);conditions.update({ref+'_'+k:v for k,v in part.items()})
 conditions['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['additive_steps']==4440 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and r['As_training_observations']==(568320 if arm=='D_As' else 0) and r['teacher_additive_calls']==0 and not r['head_fit_updates'] and not r['target_access'] for arm,r in results.items())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,target_access=False,Ag_efficacy_proved=False,goal_complete=False)
