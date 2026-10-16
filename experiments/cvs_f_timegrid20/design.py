"""One fixed bounded time-grid contraction;source-only scratch E20."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_additive_as20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-FcrossRX-timegrid-sourceRX1-e20-s2026092701-m1-r01';RELEASE='cvs_FcrossRX_timegrid_sourceRX1_E20_20261017_r01';OWNER='codex/root/FcrossRX-timegrid-20261017'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01';BACKEND_COMMIT='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b'
REFERENCE_RUN='20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01'
SMOKE_SOURCE=Path(PROJECT)/'runs'/REFERENCE_RUN/'D_raw-RX1-s2026092701/source'
SEED=2026092701;FIT=(3,4,6,8);RX=(1,3,4,6,8);HELD=1;STEPS=222;UPDATES=4440;ARMS=('TIMEGRID',);SCENES=parent.SCENES;GRID_SEED=20261017701
macro_f1=parent.macro_f1;source_validation=parent.source_validation
RHO_MIN=1/8192;RHO_MAX=1/4096;L2_BOUND=255/4096

def template():
 if parent.SEED!=SEED or parent.base_model_config()['model_seed']!=SEED:raise ValueError('Actual F constructor seed01 required')
 return parent.config(next(r for r in parent.rows() if r['arm']=='D_raw'))
def rows():return [dict(row_id='TIMEGRID-RX1-s2026092701',arm='TIMEGRID',model_seed=SEED)]
def config(row):
 if row not in rows():raise ValueError('Only one fixed source timegrid row')
 c=deepcopy(template());c.update(schema='F_CROSS_RX_timegrid_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_bounded_timegrid',training_recipe='Unchanged raw.75/ordinary.25/SupCon base plus .25RD(timegrid(raw));all8L;originalU/defaulthead;noAs')
 c['training_mechanism'].update(Ag='candidate_not_validated',As=False,extra_pair_input='bounded interior timegrid contraction of rawL',raw_total_weight=.75)
 c['additive'].update(extra_input='timegrid(rawL)',operator='One bounded interior resampling;no As/ordinary-before-grid/U/teacher/newloss')
 c['timegrid']=dict(position='t[n]=127.5+(1-rho)*(n-127.5)',interpolation='Two adjacent original samples,convex linear weights;no padding/wrap/crop/extrapolation',rho_min=RHO_MIN,rho_max=RHO_MAX,universal_relative_L2_bound=L2_BOUND,max_displacement_samples=127.5*RHO_MAX,session='actualRX/day/eq,allTX same rho;private refresh everyepoch',seed=GRID_SEED,normalization='None;original source nativeRMS kept',claim='Digital timing observation candidate,not measured RXclock/hardware replacement/known identity preservation',all_L_supervised=True,U_or_V=False)
 c['source_selection']=dict(primary='TIMEGRID',fixed_final=20,held_RX=1,held_L=1260,min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_seenV_drop_pp=.5,max_final_common_base_objective=.5,standalone_practical_nonblocking=True,on_fail='Close fixed timegrid;no rho/interpolation/probability/weight/RX/seed/budget rescue',on_pass='Only separately preregistered secondseed source confirmation;not Ag/joint/target proof')
 c.update(baseline_reference=SMOKE_SOURCE.as_posix(),baseline_reuse=False,budget_authorization='Active Ag/As goal one new scratchE20/4440updates,0inheritedweights;matchingexistingD_raw8L report only',cost_claim='4546560LNN+originalU/EMA/V/8820held=5823318rowNN;568320gridviews/240epoch-RX-day sessions;dispatcherCPsmoke2 separate')
 return c
def validate(c):
 if c!=config(rows()[0]):raise ValueError('Fixed timegrid/source/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if a.seed!=SEED or (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing)!=(20,13,7,.01) or a.source_fold_held_rx!=1 or a.baseline_ckpt or a.teacher_ckpt:raise ValueError('Exact scratch E20/13L7U/sourceRX1 required')
 return a
def qualify(results,reference):
 if set(results)!=set(ARMS) or reference['arm']!='D_raw' or reference['target_access'] or reference['U_truth_read'] or reference['checkpoint_sources']:raise ValueError('Matching source-only reference required')
 r=results['TIMEGRID'];delta={v:100*(r['views'][v]['accuracy']-reference['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(r['views']['clean']['macro_F1']-reference['views']['clean']['macro_F1']);dv=100*(r['seen_V']['source_val_accuracy']-reference['seen_V']['source_val_accuracy']);g=r['timegrid'];eps=1e-10
 checks=dict(clean=delta['clean']>=1.-eps,macroF1=f1>=-eps,seenV=dv>=-.5-eps,healthy=r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and g['calls']==4440 and g['packets']==568320 and g['sessions']==240 and g['coverage_cells']==72 and g['max_relative_L2']<=L2_BOUND+1e-5 and not r['As_training_observations'] and not r['head_fit_updates'] and not r['target_access'] and not r['U_truth_read'] and not r['checkpoint_sources'])
 return dict(qualified=all(checks.values()),checks=checks,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seenV_gain_pp=dv,source_only=True,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
