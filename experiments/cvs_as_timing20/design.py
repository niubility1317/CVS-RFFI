"""Fixed same-dose early/late As timing;four matched scratchE20 rows."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_additive_as20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SCENES,source_validation,macro_f1=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SCENES','source_validation','macro_f1'))
RUN="20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01";RELEASE="cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01";OWNER='codex/root/FcrossRX-AsTiming-20261017';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01';BACKEND_COMMIT='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b';SMOKE_SOURCE=parent.SMOKE_SOURCE
SEED=2026092701;FIT=(3,4,6,8);HELD=1;UPDATES=4440;ARMS=('RAW','ALL','EARLY7','LATE7')
def template():return parent.config(next(r for r in parent.rows() if r['arm']=='D_raw'))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def active(arm,epoch):
 if arm not in ARMS or not isinstance(epoch,int) or not 1<=epoch<=20:raise ValueError('Fixed arm/epoch only')
 return arm=='ALL' or arm=='EARLY7' and epoch<=7 or arm=='LATE7' and epoch>=14
def packets(arm,epoch=20):return sum(active(arm,i) for i in range(1,epoch+1))*222*128
def config(row):
 if row not in rows():raise ValueError('Four fixed As timing rows only')
 c=deepcopy(template());c.update(schema='F_CROSS_RX_As_timing_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_As_timing',updates=UPDATES,baseline_reuse=False,baseline_reference='Four new matched8L scratchE20 controls,no historical score substitution',source_falsification=None,budget_authorization='Active jointgoal;4scratchE20/80model epochs/17760updates,source-only RX1/no inherited weights')
 c['training_mechanism'].update(Ag=False,As=row['arm']!='RAW',raw_total_weight='1.00 in inactive epochs/.75 in active epochs;absolute base.75fixed',extra_pair_input='As(raw) only in fixed active epochs,raw otherwise',base_raw_total_weight=.75,ordinary_weight=.25)
 c['additive'].update(extra_input=c['training_mechanism']['extra_pair_input'],extra_weight=.25,operator='Full unchanged practical(raw);all4rows generate same physical shadow everyLstep;consume solely by fixed epoch,never confidence/state/label',BN_RNG='Actual model has0BatchNorm;8trainforwards sameIQ pairs;original globalCPU/Python streams unchanged')
 c['As_timing']=dict(mode=row['arm'],active_epochs=[i for i in range(1,21) if active(row['arm'],i)],active_L_packets=packets(row['arm']),same_dose_EARLY7_LATE7=198912,all_rows_generated_L_packets=568320,scope='L fourth pair only;original13L7U/oldgate/ownEMA untouched',extra_weight=.25,base_unchanged=True,loss='Original within-IQ RD/CE/no new loss',formal_test_unchanged=True)
 c['source_selection']=dict(primary='LATE7',references=['RAW','ALL','EARLY7'],fixed_final=20,held_RX=1,held_L=1260,vs_RAW=dict(clean_min_pp=0.,clean_F1_min_pp=0.,low_min_pp=5.,urban_mean_min_pp=3.),vs_ALL=dict(clean_min_pp=1.,clean_F1_min_pp=0.,low_min_pp=-.5,urban_mean_min_pp=-.5),vs_EARLY7=dict(clean_min_pp=1.,clean_F1_min_pp=0.,low_min_pp=1.,urban_mean_min_pp=1.),each_practical_min_pp=-.5,seenV_min_pp=-.5,final_base_objective_max=.5,on_fail='Close fixed same-dose early/late timing;no onset/dose/epoch/weight/RX/seed/threshold rescue,no control promotion',on_pass='Only separately registered source confirmation;not Ag/joint/target proof')
 c['training_recipe']='ScratchE20 original13L+7U/base.75raw+.25ordinary+SupCon plus .25RD(raw or singleAs(raw));same8L and1.25total pairweight;ALL20/EARLY7/LATE7/RAW'
 c['channel']['input']='OriginalrawL fourthpair;all4shadow complete nominal channels;originalheld sourceeval unchanged'
 c['cost_claim']='4scratchE20/17760updates;each4546560LNN+actualU/EMA/V/8820held;each568320generatedAs (shadow included);consumedRAW0/ALL568320/EARLY7=LATE7=198912;dispatcherCPsmoke2 separate'
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed timing/roles/budget/config differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.seed,a.label_smoothing)!=(20,13,7,SEED,.01) or a.baseline_ckpt or a.teacher_ckpt or not a.use_aug or a.use_concat_sat_channel_aug:raise ValueError('Original scratchE20/13L7U required')
 return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All4matched source rows required')
 s=results['LATE7'];checks={};comparisons={};eps=1e-10;c=config(rows()[-1])['source_selection']
 for ref in ('RAW','ALL','EARLY7'):
  b=results[ref];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in SCENES[3:])/3;t=c['vs_'+ref]
  part=dict(clean=delta['clean']>=t['clean_min_pp']-eps,clean_F1=f1>=t['clean_F1_min_pp']-eps,low=delta[SCENES[-1]]>=t['low_min_pp']-eps,urban_mean=urban>=t['urban_mean_min_pp']-eps,each_practical=all(delta[v]>=-.5-eps for v in SCENES),seenV=dv>=-.5-eps)
  checks.update({ref+'_'+k:v for k,v in part.items()});comparisons[ref]=dict(conditions=part,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seenV_gain_pp=dv,urban_mean_gain_pp=urban)
 checks['healthy']=all(x['actual_steps']==4440 and x['L_forward_packets']==4546560 and x['extra_L_forward_packets']==1136640 and x['supcon_applied']==3330 and x['final_base_L_objective']<=.5 and x['As_generated_packets']==568320 and x['As_training_observations']==packets(arm) and x['timing_steps']==4440 and not x['head_fit_updates'] and not x['target_access'] and not x['U_truth_read'] and not x['checkpoint_sources'] for arm,x in results.items())
 return dict(qualified=all(checks.values()),checks=checks,comparisons=comparisons,source_only=True,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
