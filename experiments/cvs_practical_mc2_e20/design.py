"""Fixed source-only MC2 expectation versus repeated full practical, scratch E20."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-F-crossRX-practicalMC2-sourceRX1-e20-s2026092701-m3-r01';RELEASE='cvs_F_crossRX_practicalMC2_sourceRX1_e20_20261017_r01';OWNER='codex/root/F-CROSS_RX-practicalMC2-E20-20261017'
LOCAL='local_artifacts/F_practical_MC2_E20_20261017';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01';BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SMOKE_SOURCE=Path(PROJECT)/'runs'/old.RUN/'B00-RX1-s2026092701/source'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('M00','M_repeat','M_independent');UPDATES=4440;SCENES=old.SCENES
base_model_config=old.base_model_config;source_validation=old.source_validation;macro_f1=old.macro_f1

def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def original_config():return old.config(old.rows()[0])
def config(row):
 if row not in rows():raise ValueError('Only fixed three MC2 source rows permitted')
 c=deepcopy(original_config());arm=row['arm'];mode={'M00':'raw','M_repeat':'repeat','M_independent':'independent'}[arm];n=0 if arm=='M00' else (1 if arm=='M_repeat' else 2)
 c.update(schema='F_CROSS_RX_PRACTICAL_MC2_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_PRACTICAL_MC2',updates=4440)
 c['training_mechanism'].update(Ag=False,As=arm!='M00',base_raw_total_weight=.75,raw_total_weight=1. if arm=='M00' else .75,ordinary_weight=.25,raw_weight=.5,s_slot_weight=.25,s_slot_name='Common third pair raw allrows',extra_pair_weight=.125,extra_total_weight=.25,L_forwards_per_packet=10,expected_L_packets=5683200,no_new_U_augmentation=True)
 c['mc2']=dict(mode=mode,unique_As_views=n,extra_pair_count=2,extra_pair_weight=.125,base_pair_weight_sum=1.,pair_weight_sum=1.25,common='Unchanged original raw.75/ordinary.25/ordinary-only shallowmid CROSS_RX SupCon',view_rule='raw/raw;As0/As0;As0/As1.Each pair has2stochastic network passes on SAMEgeneratedIQ',loss='Original smoothedCE.01 mean+within-IQ RDrop1KL;extra .125R(v0)+.125R(v1);no crossview loss',practical_distribution='Exact old complete practical residual/post_sync/EQfalse,uniformsixscenes;alllabelledLsupervised,no confidence/quality gate',sampling='Stateless PairedChannel,namespace train/epochE/stepS/mc0 ormc1;MC0 shared exactly by repeat/independent;only namespace changes',receiver='Unchanged receiver_for_session(cfg,RECEIVER_SEED,rx/day/eq);no TX or viewnamespace in session',normalization='Perphysical mean within each pair;no IQ averaging;no divide entire base by5;fullpairweight1.25 allrows',expected_As_generated_packets=n*568320,expected_extra_L_NN=2273280,expected_As_supervised_pair_observations=0 if n==0 else 1136640,U='Original13L+7U/hiddentruth/ownEMA/oldgate unchanged,no additionalUor teacher As',new_losses=0,new_deployment_branches=0)
 c['channel']['input']='Original raw L only in extra pair(s),common thirdpair staysraw';c['channel']['composition']=False
 c['training_recipe']='ScratchE20/F+CROSS_RX/defaultcos/NoExplicitCFO/originalordinary/U;fullbase retained+.25meanof2RDrop pairs;10Lforwards,raw/repeatAs/independentAsMC2'
 c['source_selection']=dict(primary='M_independent',references=['M00','M_repeat'],fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],vs_raw=dict(min_clean_gain_pp=0.,min_clean_macroF1_gain_pp=0.,min_low_urban_gain_pp=5.,min_urban_mean_gain_pp=3.,max_each_suburban_drop_pp=.5,max_seenV_drop_pp=.5),vs_repeat=dict(min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5),final_base_L_objective_max=.5,on_fail='Close fixed MC2 organization;noM4/coefficient/RX/seed/budget rescue',on_pass='Only independent source-seed confirmation;not Ag/target/joint proof')
 c['budget_authorization']='Active jointAg/As goal,latestF+CROSS_RX/E20;3scratchE20/60modelepochs/13320updates,matched10LNN;no parent/continuation/target'
 c['cost_claim']='Each5683200LNN incl2273280extra;actual originalU/EMA/V/8820held separately.568320ordinary each;As0/568320/1136640 generations;all128Lsupervised.2dispatcher realCP syntheticNN,no traininit/newhead'
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed MC2 budget/data/operator/weights/controls differ')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(original_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All3 source MC2 rows required')
 s=results['M_independent'];conditions={};comparisons={}
 for ref in ('M00','M_repeat'):
  b=results[ref];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy'])
  if ref=='M00':part=dict(clean=delta['clean']>=0.,clean_macroF1=f1>=0.,low_urban=delta[SCENES[-1]]>=5.,urban_mean=sum(delta[v] for v in SCENES[3:])/3>=3.,suburban=all(delta[v]>=-.5 for v in SCENES[:3]),seen_V=dv>=-.5)
  else:part=dict(clean=delta['clean']>=1.,clean_macroF1=f1>=0.,each_practical=all(delta[v]>=-.5 for v in SCENES),seen_V=dv>=-.5)
  comparisons[ref]=dict(conditions=part,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,urban_mean_gain_pp=sum(delta[v] for v in SCENES[3:])/3,seen_V_gain_pp=dv);conditions.update({ref+'_'+k:v for k,v in part.items()})
 conditions['healthy']=all(r['actual_steps']==r['MC2_steps']==4440 and r['L_forward_packets']==5683200 and r['extra_L_forward_packets']==2273280 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and r['As_training_observations']==config(next(x for x in rows() if x['arm']==arm))['mc2']['expected_As_generated_packets'] and r['As_supervised_pair_observations']==(0 if arm=='M00' else 1136640) and not r['teacher_MC2_calls'] and not r['head_fit_updates'] and not r['target_access'] for arm,r in results.items())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
