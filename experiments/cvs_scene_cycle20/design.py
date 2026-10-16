"""One fixed three-row scratchE20;whole base retained,no new Ag/loss/head."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-F-crossRX-sceneCycle-sourceRX1-e20-s2026092701-m3-r01';RELEASE='cvs_F_sceneCycle_sourceRX1_e20_20261017_r01';OWNER='codex/root/F-sceneCycle-E20-20261017'
LOCAL='local_artifacts/F_physical_scene_cycle_E20_20261017';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01';BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8';SMOKE_SOURCE=Path(PROJECT)/'runs'/old.RUN/'B00-RX1-s2026092701/source'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('B00','C_IID','B01');UPDATES=4440;SCENES=old.SCENES;PERMUTATION_SEED=202610175001
base_model_config=old.base_model_config;source_validation=old.source_validation;macro_f1=old.macro_f1

def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def original_config():return old.config(old.rows()[0])
def config(row):
 if row not in rows():raise ValueError('Only fixed scene-cycle source rows allowed')
 c=deepcopy(original_config());mode={'B00':'none','C_IID':'iid','B01':'balanced'}[row['arm']]
 c.update(schema='F_CROSS_RX_scene_cycle_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_physical_scene_cycle',baseline_reference='Currentmatched extra raw plus independent singleAs control,same8L/totalweight1.25',updates=UPDATES)
 c['training_mechanism'].update(Ag=False,As=mode!='none',base_raw_total_weight=.75,raw_total_weight=1. if mode=='none' else .75,ordinary_weight=.25,raw_weight=.5,s_slot_weight=.25,s_slot_name='Commonraw third pair retained in allrows',extra_pair_weight=.25,extra_pair_input='raw' if mode=='none' else 'One complete practical(raw)',L_forwards_per_packet=8,expected_L_packets=4546560,no_new_U_augmentation=True)
 c['additive']=dict(common='Fullraw.75+ordinary.25+ordinary-onlyshallow/midSupCon',extra_input='raw' if mode=='none' else 'As(raw)',extra_weight=.25,extra_pair_loss='Original smoothedCE.01 mean+within-IQ RDrop1KL,no newloss',base_pair_weight_sum=1.,pair_weight_sum=1.25,normalization='PerphysicalL mean perIQpair,retainfullbase;extra .25,no dividebyviewcount',BN_RNG='All3rows8train forwards;private scene RNG only,raw andglobal RNG unchanged',operator='One unchanged fullpractical pervisibleL inC_IID/B01,no composition/confidencefilter',original_U='Original13L+7U/oldgate/ownEMA/hiddenU unchanged,no newU/teacher views')
 c['scene_schedule']=dict(mode=mode,permutation_seed=PERMUTATION_SEED,scenes=list(SCENES),cycle_length=6,pid_order=['rx_i','day_i','eq_i','sig_i','base_index'],counter='Appearance order perphysicalL,including duplicates inbatch;not epoch/state/label/difficulty',rule='At visitv use permutation[ v%6 ],permutation=default_rng(stable_seed(seed,physical_scene_cycle,pid,v//6)).permutation(6);all6 onceper6appearances',marginal='Uniformscene probability retained,only withinpacket crossvisit dependence differs',iid_reference='Exact native stable_seed(CHANNEL_SEED,namespace,scene,pid) scene RNG;original namespace and propagation/session seeds',physical_channel='Allpropagation/noise/quality/compensation/AGC unchanged;no RX/TX/state forcing',evaluation='Delegate original exact source_eval implementation,never update schedule orchangeformalIQ',expected_As_generated=568320 if mode!='none' else 0,trace='Everytrainingcall:namespace,ordered sceneindices and visitindices;privateLsequence suppliesphysicalIDs',source_evidence='Existingall4440 IIDstepcounts exactly replayed;all5040L sawall6,withinpacketmax-min median11/max27;countsnot efficacy')
 c['channel']['input']='OriginalrawL in fourthpair only;commonthirdpair staysraw';c['channel']['composition']=False
 c['training_recipe']='ScratchE20/F+CROSS_RX/NoExplicitCFO/defaultcos/originalordinary/U;wholebase raw.75+ordinary.25+SupCon plus.25R(raw/iidAs/balancedAs),8L'
 c['source_selection']=dict(primary='B01',references=['B00','C_IID'],fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],vs_B00=dict(min_clean_gain_pp=0.,min_clean_macroF1_gain_pp=0.,min_low_urban_gain_pp=5.,min_low_urban_macroF1_gain_pp=0.,min_mid_urban_gain_pp=0.,min_urban_mean_gain_pp=3.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5),vs_C_IID=dict(min_clean_gain_pp=0.,min_clean_macroF1_gain_pp=0.,min_low_urban_gain_pp=1.,min_low_urban_macroF1_gain_pp=0.,min_mid_urban_gain_pp=0.,min_urban_mean_gain_pp=1.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5),final_base_L_objective_max=.5,on_fail='Closefixedperpacket sixscene-cycle training;no orderseed/cyclelength/marginalprobability/coefficient/RX/modelseed/budget rescue',on_pass='Separate independent source-seed confirmation only;not validatedAg/fulljoint/target result')
 c['budget_authorization']='Activejointresearch/latestF+CROSS_RX/E20;3scratchE20/60modelepochs/13320updates,no inheritedweights/continuation'
 c['cost_claim']='Each4546560LNNincl1136640extra;oldordinary568320each;C_IID/B01 each568320fullpractical;originalU/EMA/V/8820held and2NNdispatcherrealCPsmokeactualcounted separately;0newU/teacher/headfit'
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed scene-cycle roles/budget/schedule/weights differ')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(original_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Allthree current matchedrows required')
 s=results['B01'];conditions={};comparisons={}
 for ref in ('B00','C_IID'):
  b=results[ref];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);lowf1=100*(s['views'][SCENES[-1]]['macro_F1']-b['views'][SCENES[-1]]['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in SCENES[3:])/3
  part=dict(clean=delta['clean']>=0.,clean_macroF1=f1>=0.,low_urban=delta[SCENES[-1]]>=(5. if ref=='B00' else 1.),low_urban_macroF1=lowf1>=0.,mid_urban=delta[SCENES[-2]]>=0.,urban_mean=urban>=(3. if ref=='B00' else 1.),each_practical=all(delta[v]>=-.5 for v in SCENES),seen_V=dv>=-.5)
  comparisons[ref]=dict(conditions=part,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,low_urban_macroF1_gain_pp=lowf1,seen_V_gain_pp=dv,urban_mean_gain_pp=urban);conditions.update({ref+'_'+k:v for k,v in part.items()})
 health=[]
 for arm,r in results.items():
  good=r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['additive_steps']==4440 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and r['As_training_observations']==(0 if arm=='B00' else 568320) and r['teacher_additive_calls']==0 and not r['head_fit_updates'] and not r['target_access'] and r['scene_schedule_mode']=={'B00':'none','C_IID':'iid','B01':'balanced'}[arm] and r['scene_schedule_training_calls']==(0 if arm=='B00' else 4440)
  health.append(good and (arm!='B01' or r['balanced_coverage_max_minus_min']<=1))
 conditions['healthy']=all(health)
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,target_access=False,Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False)
