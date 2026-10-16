"""Fixed matched10L-forward training test of noise-pair sampling rationale."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_additive_as20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-FcrossRX-noisepair-sourceRX1-e20-s2026092701-m3-r01'
RELEASE='cvs_FcrossRX_noisepair_sourceRX1_E20_20261017_r01';OWNER='codex/root/FcrossRX-noisepair20-20261017'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01'
BACKEND_COMMIT='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b'
SMOKE_SOURCE=Path(PROJECT)/'runs/20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01/D_raw-RX1-s2026092701/source'
SEED=2026092701;HELD=1;RX=(1,3,4,6,8);FIT=(3,4,6,8);STEPS=222;UPDATES=4440
ARMS=('RAW10','INDEP','ANTI');SCENES=parent.SCENES
CHANNEL_SEED=20261017501;NOISE_SEED=20261017502;RECEIVER_SEED=20261016102
macro_f1=parent.macro_f1;source_validation=parent.source_validation

def template():
 if parent.SEED!=SEED or parent.base_model_config()['model_seed']!=SEED:raise ValueError('Actual F constructor/native seed1 required')
 return parent.config(next(r for r in parent.rows() if r['arm']=='D_raw'))
def rows():return [dict(row_id=f'{a}-RX1-s{SEED}',arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Only fixed3matched scratchE20 source rows allowed')
 c=deepcopy(template());sat=row['arm']!='RAW10'
 c.update(schema='F_CROSS_RX_noisepair10_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_noisepair_supervision',training_recipe='Original3pair raw.75/ordinary.25+ordinarySupCon,plus .25mean(two same-IQ RDpairs),10Lforward allrows;hiddenU unchanged;no cross-noiseKL',baseline_reference='New RAW10 matched10forward control;oldraw8L isnot reused as performancecontrol',baseline_reuse=False,budget_authorization='Active Ag/As goal;3new scratchE20/60modelE/13320updates,0inheritedweights;fixedsourcefalsification before further seeds/target',cost_claim='Each5683200LNN+originalU/ownEMA/V/8820held;3newmodeltotal20879874NN plusdispatcher2;As2x568320noiseviews inINDEP/ANTI,noextraU/teacher/headfit')
 c['training_mechanism'].update(Ag=False,As=sat,base_raw_total_weight=.75,raw_total_weight=.75 if sat else 1.,ordinary_weight=.25,extra_pair_input='raw twice' if not sat else 'two complete practical(raw) with '+row['arm']+' primitive noises',L_forwards_per_packet=10,expected_L_packets=5683200,extra_pair_count=2,extra_pair_weight=.25)
 c['additive'].update(extra_input=c['training_mechanism']['extra_pair_input'],extra_weight=.25,extra_pair_count=2,extra_pair_loss='Mean of2 original within-IQ RD(alpha1/LS.01);no loss between different noise IQ views',pair_weight_sum=1.25,operator='Same unmodified residualphysical latent across2 independent or antithetic primitive noises;AGC recomputed perview;rawcontrol4extraNN',normalization='Original3pair base unchanged;each extra RD perphysical mean,average2pairs thenmultiply.25;never dividecommonbase',BN_RNG='All3rows10trainLforwards,4extra;originalordinary/mainU streams stay matched across arms;noarchitecturechange')
 c['noise_pair']=dict(mode=row['arm'],source_IQ='RawL only,not ordinary',channel_seed=CHANNEL_SEED,noise_seed=NOISE_SEED,receiver_seed=RECEIVER_SEED,refresh='Each epoch/step/physicalpacket;two views share nonnoise latents,noise0 common INDEP/ANTI;private named streams',noise0='IID standardnormal n x2 at primitive receiver-noise input',noise1='raw' if not sat else ('independent IID normal draw1' if row['arm']=='INDEP' else '-draw0;same marginal noise distribution'),receiver='Original RX IQ,compensation,G plus separately recomputed AGC perview',same_IQ_within_each_RD=True,cross_noise_consistency=False,drop_or_confidence_gate=False,U_or_teacher_views=False,physical_channel_passes_per_view=1,hardware_session='Actual RX/day/eq,never TX',formal_test_changed=False)
 c['source_selection']=dict(primary='ANTI',references=['RAW10','INDEP'],fixed_final=20,held_RX=1,held_L=1260,vs_RAW10=dict(min_clean_gain_pp=0.,min_clean_macroF1_gain_pp=0.,min_urban_mean_gain_pp=3.,min_low_urban_gain_pp=5.),vs_INDEP=dict(max_clean_drop_pp=.5,max_clean_macroF1_drop_pp=.5,min_urban_mean_gain_pp=1.,min_low_urban_gain_pp=1.),max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5,final_common_base_L_objective_max=.5,numerical_tolerance_pp=1e-10,on_fail='Close fixednoise-pair training rationale;no coefficient/SNR/seed/RX/draw/epoch rescue',on_pass='Only supports independentlyregistered secondmodelseed source confirmation;not Ag/joint/target orlow60 proof')
 return c

def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed noisepair roles,budget,weights,loss orphysics differ')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if a.seed!=SEED or (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing)!=(20,13,7,.01) or a.baseline_ckpt or a.teacher_ckpt or a.source_fold_held_rx!=HELD:raise ValueError('ScratchE20 originalU/defaulthead required')
 return a

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Allthree matched source rows required')
 s=results['ANTI'];conditions={};comparisons={};eps=1e-10
 for ref in ('RAW10','INDEP'):
  b=results[ref];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in SCENES[3:])/3;clean=0. if ref=='RAW10' else -.5;needurban=3. if ref=='RAW10' else 1.;needlow=5. if ref=='RAW10' else 1.
  checks=dict(clean=delta['clean']>=clean-eps,clean_macroF1=f1>=clean-eps,urban=urban>=needurban-eps,low=delta[SCENES[-1]]>=needlow-eps,all_practical=all(delta[v]>=-.5-eps for v in SCENES),seenV=dv>=-.5-eps)
  comparisons[ref]=dict(all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seen_V_gain_pp=dv,urban_mean_gain_pp=urban,conditions=checks);conditions.update({ref+'_'+k:v for k,v in checks.items()})
 conditions['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==5683200 and r['extra_L_forward_packets']==2273280 and r['noise_pair_steps']==4440 and r['supcon_applied']==3330 and r['supcon_layer_forward_terms']==13320 and r['final_base_L_objective']<=.5 and r['noise_pair_physical_presentations']==568320 and r['As_training_observations']==(0 if arm=='RAW10' else 1136640) and not r['teacher_noise_pair_calls'] and not r['head_fit_updates'] and not r['U_truth_read'] and not r['target_access'] and not r['checkpoint_sources'] and r['held_RX']==HELD and r['fit_RX']==list(FIT) for arm,r in results.items())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,source_only=True,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
