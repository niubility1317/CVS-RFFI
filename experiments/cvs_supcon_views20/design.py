"""Two new scratchE20 rows plus two exact frozen controls;one view-routing hypothesis."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_scene_cycle20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01';RELEASE='cvs_F_supconViews_sourceRX1_e20_20261017_r01';OWNER='codex/root/F-supconViews-E20-20261017';LOCAL='local_artifacts/F_supcon_view_routing_E20_20261017';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=old.BACKEND;BACKEND_COMMIT=old.BACKEND_COMMIT;SMOKE_SOURCE=old.SMOKE_SOURCE;CONTROL_BASE=old.BASE;CONTROL_COMMIT='c9a74172e5f81bb4f04ce12cd4386ad4dd953ee9'
SEED=2026092701;FIT=old.FIT;SCENES=old.SCENES;UPDATES=4440;ROUTING_SEED=202610176001;ARMS=('Q0','Q_ord','Q_sat','Q_mix');NEW_ARMS=('Q_sat','Q_mix')
base_model_config=old.base_model_config;source_validation=old.source_validation;macro_f1=old.macro_f1

def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in NEW_ARMS]
def legacy_config(arm='C_IID'):return old.config(next(r for r in old.rows() if r['arm']==arm))
def controls():return {'Q0':CONTROL_BASE/'B00-RX1-s2026092701/source','Q_ord':CONTROL_BASE/'C_IID-RX1-s2026092701/source'}
def config(row):
 if row not in rows():raise ValueError('Only two preregistered new view-routing rows')
 c=deepcopy(legacy_config());policy={'Q_sat':'satellite','Q_mix':'mixed'}[row['arm']]
 c.update(schema='F_CROSS_RX_existing_supcon_views_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_existingSupCon_viewRouting',baseline_reference='FrozenQ0/Q_ord andnewQ_sat/Q_mix;matched8L/weight1.25;no parentweights',training_recipe='Originalraw.75/ordinary.25+extra.25fullAs;existingSupCononlyinputchanged;E20/originalU',budget_authorization='2new scratchE20/40modelepochs/8880updates;2existing E20controls reused/report40historicalepochs separately',cost_claim='Each4546560LNN/1136640extra;568320fullpractical generated;originalU/EMA/V and8820held unchanged;zero newmodel forward orloss term')
 c['scene_schedule']['mode']='iid'
 c['supcon']['scope']='Existing4terms on '+policy+' maps of paired L forwards;captured without extraNN'
 c['additive'].update(common='Fullraw.75+ordinary.25+existingSupCon with'+policy+'viewrouting',BN_RNG='All4logicalrows8trainforwards;privatechannel/routeRNG;globalRNGunchanged',operator='One unchanged IIDfullpractical perL inallAsrows,no composition/confidencefilter')
 c['supcon_view_routing']=dict(policy=policy,selection_seed=ROUTING_SEED,selection='Perstep private default_rng(seed+optimizer_step).permutation(128) first64;Lappearancepositions,notTX/RX/difficulty;shared acrossbothlayers/bothRDrop phases',satellite_positions=128 if policy=='satellite' else 64,ordinary_positions=0 if policy=='satellite' else 64,active_epochs=list(range(6,21)),existing_levels=['time_down','t2.pool'],temperature=.2,weight_schedule='Original .1*clip((epoch-5)/10,0,1)',phase_terms=4,all_As_L_CE=True,extra_NN_forwards=0,new_loss_terms=0,new_U_or_teacher_forwards=0,mixing='Select featuremap perLappearance;never mixIQ;normalization/positive/negative rules unchanged',physical_duplicates='Existingloader appearances retain weights;mask usespositions notstablePID,doesnot claim unique128physicalIDs',loss_normalization='Original mean over2layers/2RDrop phases;totalSupCon coefficient unchanged')
 c['source_selection']=dict(primary='Q_mix',references=['Q0','Q_ord','Q_sat'],fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],vs_Q0=dict(min_clean_gain_pp=0.,min_low_gain_pp=5.,min_urban_mean_gain_pp=3.),vs_Q_ord_and_Q_sat=dict(min_clean_gain_pp=0.,min_clean_macroF1_gain_pp=0.,min_low_gain_pp=1.,min_low_macroF1_gain_pp=0.,min_mid_gain_pp=0.,min_urban_mean_gain_pp=1.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5),final_base_L_objective_max=.5,on_fail='ClosefixedexistingSupCon viewrouting,no fraction/layer/weight/sceneprobability/seed/budget rescue',on_pass='Separateindependent source seedconfirmation only;not Ag/stableAs/fulljoint/targetproof')
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed routing/roles/seed/budget/loss changed')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(legacy_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All four logical matrix rows required')
 candidate=results['Q_mix'];conditions={};comparisons={}
 for ref in ('Q0','Q_ord','Q_sat'):
  b=results[ref];delta={v:100*(candidate['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(candidate['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);lf1=100*(candidate['views'][SCENES[-1]]['macro_F1']-b['views'][SCENES[-1]]['macro_F1']);dv=100*(candidate['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in SCENES[3:])/3;eps=1e-10
  part=dict(clean=delta['clean']>=-eps,clean_F1=f1>=-eps,low=delta[SCENES[-1]]>=(5. if ref=='Q0' else 1.)-eps,low_F1=lf1>=-eps,mid=delta[SCENES[-2]]>=-eps,urban_mean=urban>=(3. if ref=='Q0' else 1.)-eps,each_practical=all(delta[v]>=-.5-eps for v in SCENES),seenV=dv>=-.5-eps);conditions.update({ref+'_'+k:v for k,v in part.items()});comparisons[ref]=dict(conditions=part,delta_pp=delta,clean_F1_delta_pp=f1,low_F1_delta_pp=lf1,urban_mean_delta_pp=urban,seenV_delta_pp=dv)
 conditions['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and not r['head_fit_updates'] and not r['target_access'] and r['As_training_observations']==(0 if a=='Q0' else 568320) for a,r in results.items())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False,target_access=False)
