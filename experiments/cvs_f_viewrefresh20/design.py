"""Fixed fresh versus replayed existing IQ observation,three matched scratchE20."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_additive_as20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SCENES,source_validation,macro_f1=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SCENES','source_validation','macro_f1'))
RUN='20261017-phase1-FcrossRX-viewrefresh-sourceRX1-e20-s2026092701-m3-r01';RELEASE='cvs_FcrossRX_viewrefresh_sourceRX1_E20_20261017_r01';OWNER='codex/root/FcrossRX-viewrefresh-20261017'
BASE=Path(PROJECT)/'runs'/RUN;BACKEND=PROJECT+'/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01';BACKEND_COMMIT='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b'
SEED=2026092701;FIT=(3,4,6,8);HELD=1;UPDATES=4440;ARMS=('RAW','REPLAY','REFRESH')
SMOKE_SOURCE=parent.SMOKE_SOURCE
AUG_SEED=20261017961
def template():return parent.config(next(r for r in parent.rows() if r['arm']=='D_raw'))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Three fixed current source views only')
 c=deepcopy(template());c.update(schema='F_CROSS_RX_independent_ordinary_observation_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_view_refresh',updates=UPDATES,baseline_reuse=False,baseline_reference='New RAW/REPLAY scratchE20 controls;no historical score substitution',source_falsification=None,budget_authorization='Active Ag/As goal;three matched scratchE20 models/4440updates each/no inherited states/source-only RX1')
 c['training_mechanism'].update(Ag='existing_IQ_refresh_candidate_not_validated' if row['arm']=='REFRESH' else False,As=False,raw_total_weight=1. if row['arm']=='RAW' else .75,extra_pair_input={'RAW':'rawL','REPLAY':'cached original prePA/DAC L view','REFRESH':'independent existing anti+channel on rawL'}[row['arm']])
 c['additive'].update(extra_input=c['training_mechanism']['extra_pair_input'],operator='No new operators;native anti-shortcut/general-channel before PA/DAC,existing schedule/ranges;As0',BN_RNG='All3arms8L;all3build independent IQ shadow with restored original RNG;no extra teacher/U')
 c['view_refresh']=dict(mode=row['arm'],scope='Real L normal view only;explicit U context including positive pseudolabels bypassed',first='Original normal pipeline/PA unchanged;capture after existing general-channel and before PA/DAC',second='Exact native anti-shortcut then general-channel on same rawL;no PA/DAC/class_signature/no postRMS/composition',seed=AUG_SEED,stream='Private seed+optimizer step,not TX/RX/day/quality/loss;restore CPU/Python/currentCUDA',NN='Fourth pair only;both extra forwards reuse chosen IQ;original6L and original ordinary-only SupCon unchanged',shadow='All3arms construct fresh view;RAW returns raw,REPLAY returns captured first prePA,REFRESH returns fresh;no rejection/redraw')
 c['source_selection']=dict(primary='REFRESH',references=['RAW','REPLAY'],fixed_final=20,held_RX=1,held_L=1260,min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_seenV_drop_pp=.5,max_final_L_CE=.5,all_practical_mandatory_but_nonblocking=True,on_fail='Close fixed independent ordinary observation;no subset/strength/probability/weight/RX/seed/threshold/budget rescue',on_pass='Only separately registered secondseed source confirmation;not validAg/joint/target proof')
 c['training_recipe']='Original base raw.75+ordinary.25+SupCon unchanged,plus .25RD(raw/cachedprePA/freshprePA);8L/oldU/defaulthead/noAs'
 c['channel']['input']='No training As;unchanged raw held source evaluation only'
 c['cost_claim']='3scratchE20/13320updates;each4546560LNN+originalU/EMA/V/8820held=expected5823318rowNN;dispatcherCPsmoke2 separate;each568320extra shadow IQ0extraUteacher'
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed view-refresh/data/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.seed,a.label_smoothing)!=(20,13,7,SEED,.01) or a.baseline_ckpt or a.teacher_ckpt or not a.use_aug or a.use_concat_sat_channel_aug:raise ValueError('Original scratchE20/13L7U contract required')
 return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Three new matched rows required')
 s=results['REFRESH'];checks={};comparisons={};eps=1e-10
 for ref in ('RAW','REPLAY'):
  b=results[ref];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);checks.update({ref+'_clean':delta['clean']>=1.-eps,ref+'_macroF1':f1>=-eps,ref+'_seenV':dv>=-.5-eps});comparisons[ref]=dict(all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seenV_gain_pp=dv,urban_mean_gain_pp=sum(delta[v] for v in SCENES[3:])/3)
 checks['healthy']=all(x['actual_steps']==4440 and x['L_forward_packets']==4546560 and x['extra_L_forward_packets']==1136640 and x['supcon_applied']==3330 and x['final_L_CE']<=.5 and x['view_refresh']['calls']==4440 and x['view_refresh']['consumed']==4440 and x['view_refresh']['fresh_packets']==568320 and x['view_refresh']['coverage_cells']==72 and not x['As_training_observations'] and not x['head_fit_updates'] and not x['target_access'] and not x['U_truth_read'] and not x['checkpoint_sources'] for x in results.values())
 return dict(qualified=all(checks.values()),checks=checks,comparisons=comparisons,source_only=True,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
