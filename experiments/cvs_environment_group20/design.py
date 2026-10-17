"""Fixed sourceRX6 E20 comparison of nominal scene/propagation dependence."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_f_additive_rx6 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SMOKE_SOURCE,FULL_FP32_POLICY,read,write,macro_f1,source_validation=(getattr(parent,k) for k in ('PROJECT','SOURCE','SMOKE_SOURCE','FULL_FP32_POLICY','read','write','macro_f1','source_validation'))
RUN='20261017-phase1-FcrossRX-propgroup-sourceRX6-e20-s2026092701-m4-r01';RELEASE='cvs_FcrossRX_propgroup_sourceRX6_E20_20261017_r01';OWNER='codex/root/FcrossRX-propgroup-20261017';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_FcrossRX_additiveAs_RX6_source_E20_20261016_r01';BACKEND_COMMIT='367b2f945ef65488e6d97eddd2357388d2477100';HELD=6;RX=(1,3,4,6,8);FIT=(1,3,4,8);SEED=2026092701;STEPS=222;UPDATES=4440;ARMS=('RAW','IID','SCENE_GROUP','PROP_GROUP');SCENES=parent.SCENES

def template():return parent.config(next(r for r in parent.rows() if r['arm']=='D_raw'))
def rows():return [dict(row_id=a+'-RX6-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Four fixed sourceRX6 rows only')
 c=deepcopy(template());c.update(schema='FcrossRX_propagation_group_E20_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=False,baseline_reference='Fournewmatched8LsourceRX6 controls,nohistoricalscore/weightreuse',cost_claim='EachE20/4440/4546560LNNplusoriginalU/EMA/V/8820held;all4generate568320nativeprocessesand1136640nativeinitializations;Asconsumed0RAWor568320others;compatCPsmoke2separate',method='F_CROSS_RX_propagation_dependence',stage='source_RX6_falsification',source_confirmation=None,budget_authorization='Active mixedgoal;fournewmatched scratchE20/80modelE/17760updates;sourceRX6only/noinheritedweights/target',training_recipe='Original13L7U/ordinary-onlySupCon/ownEMA/common.75raw+.25ordinary plus extra.25RD(raw/IID/scene_group/prop_group);8L,total1.25;rawarmalso fullphysicalshadow,no newloss/head/U')
 c['training_mechanism'].update(As=row['arm']!='RAW',Ag=False,base_As=False,base_raw_total_weight=.75,raw_total_weight=1. if row['arm']=='RAW' else .75,ordinary_weight=.25,extra_pair_input='raw' if row['arm']=='RAW' else row['arm']+'(raw)',new_U_augmentation=False)
 c['baseline_definition']='F:three SIG_TIME branches plus unified Sinc CENTER_UNITS;unchanged joint160/cos30;ordinary-only crossRX sameTX SupCon;original13L7U/ownEMA/oldgate'
 c['channel']['selection']='Uniform original six-scene nominal draw per physical packet (IID/RAW) or per(step,RX,day,eq) group (grouped);never TX/loss/confidence conditioned'
 c['additive'].update(extra_input=row['arm'],extra_weight=.25,operator='Originalfullpractical rawIQ;sharedscene orsharedfrozenpropagation by(step,RX,day,eq),independentpacketnoise/phase/sync/tracking;RAWconsumesrawbutgeneratesIIDshadow',BN_RNG='0BN;all4same8L/privateNP/no changes to globalmodel/U RNG')
 c['channel_group']=dict(policy='IID' if row['arm']=='RAW' else row['arm'],group_key=['step_namespace','rx_i','day_i','eq_i'],TX_used_in_group=False,noise_phase_sync_tracking='originalindependentphysicalpacketseed',hardware_session='originalrx/day/eq,stableacrossepochs',propagation_attributes=['geometry','probs','state','z','ray_hz','ray_coef','f_doppler'],f_total='sharedorbitalDoppler+ownRXCFO',quality='nativeprocessrecomputesfromH/inputpower;no fixedlock/scale override',marginals='Originalsixsceneuniform/geometry/state/shadow/scatterconfigs;changes jointdependence andeffectiveindependentenvironments,not a pure shortcut-causality test',generation='All4:2nativeinitializations+1Residual.process perLpacket,568320processes/1136640initializations each;RAWdiscardsonlythegeneratedshadow',source_eval='OriginalPairedChannel unchanged,source_eval/RX6/final20',formal_test_unchanged=True)
 c['source_selection']=dict(primary='PROP_GROUP',references=['RAW','IID','SCENE_GROUP'],held_RX=6,fixed_final=20,conditions=dict(RAW=dict(clean=0.,F1=0.,low=5.,urban=3.),IID=dict(clean=1.,F1=0.,low=.5,urban=1.),SCENE_GROUP=dict(clean=1.,F1=0.,low=.5,urban=1.)),each_practical_min_pp=-.5,seenV_min_pp=-.5,base_L_max=.5,on_fail='Close fixedgrouping/streamsplit/dose/RX/seed/budget/threshold;no controlpromotion orseedrescue',on_pass='Only one-sourceRX6/modelseed As evidence;separatelyregisterindependentseed beforetarget;not Ag orjointgoal')
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed role/recipe/group/seed/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.seed,a.label_smoothing,a.source_fold_held_rx)!=(20,13,7,SEED,.01,HELD) or a.baseline_ckpt or a.teacher_ckpt or not a.use_aug or a.use_concat_sat_channel_aug:raise ValueError('Original scratchE20/RX6/U/defaulthead recipe only')
 return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Allfourregisteredsource rows required')
 primary=results['PROP_GROUP'];checks={};comparisons={};limits=config(rows()[0])['source_selection']
 for ref in ARMS[:-1]:
  b=results[ref];delta={v:100*(primary['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(primary['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);seen=100*(primary['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in SCENES[3:])/3;t=limits['conditions'][ref];part=dict(clean=delta['clean']>=t['clean']-1e-10,F1=f1>=t['F1']-1e-10,low=delta[SCENES[-1]]>=t['low']-1e-10,urban=urban>=t['urban']-1e-10,each_practical=all(delta[v]>=-.5-1e-10 for v in SCENES),seenV=seen>=-.5-1e-10);checks.update({ref+'_'+k:v for k,v in part.items()});comparisons[ref]=dict(all_view_gain_pp=delta,clean_F1_gain_pp=f1,seenV_gain_pp=seen,urban_mean_gain_pp=urban)
 checks['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['group_generated']==568320 and r['supcon_applied']==3330 and r['As_training_observations']==(0 if a=='RAW' else 568320) and r['final_base_L_objective']<=.5 and r['held_RX']==6 and r['fit_RX']==list(FIT) and not r['head_fit_updates'] and not r['target_access'] and not r['U_truth_read'] and not r['checkpoint_sources'] for a,r in results.items())
 return dict(qualified=all(checks.values()),checks=checks,comparisons=comparisons,source_only=True,target_access=False,Ag=False,crossseed_confirmed=False,joint_goal_complete=False)
