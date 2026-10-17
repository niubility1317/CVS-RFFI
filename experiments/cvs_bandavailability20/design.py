"""Fixed E20 matched raw/gain/ordinary/mask,after source identity guard only."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_additive_as20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SCENES,source_validation,macro_f1=(getattr(parent,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SCENES','source_validation','macro_f1'))
RUN='20261017-phase1-FcrossRX-bandavailability-sourceRX1-e20-s2026092701-m4-r02';RELEASE='cvs_FcrossRX_bandavailability_sourceRX1_E20_20261017_r02';OWNER='codex/root/FcrossRX-bandavailability-20261017';BASE=Path(PROJECT)/'runs'/RUN
SEED=2026092701;FIT=(3,4,6,8);HELD=1;UPDATES=4440;ARMS=('RAW','GAIN','ORDINARY','MASK');MASK_SEED=202610171951;SMOKE_SOURCE=parent.SMOKE_SOURCE

def template():return parent.config(next(r for r in parent.rows() if r['arm']=='D_raw'))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Four fixed matchedsource rows only')
 c=deepcopy(template());c.update(schema='F_CROSS_RX_STFpair_availability_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_STFpair_availability',updates=UPDATES,baseline_reuse=False,baseline_reference='Fournewmatched8L scratchE20,no historicalscore substitution',source_falsification=None,budget_authorization='Activejointgoal;conditional4scratchE20/80modelepochs/17760updates afteridentityguard,sourceonlyRX1,0initCP')
 c['training_mechanism'].update(Ag=row['arm']=='MASK',As=False,raw_total_weight='.75 commonbase absolute,extraRAW adds.25',base_raw_total_weight=.75,ordinary_weight=.25,extra_pair_input=row['arm'],no_new_U_augmentation=True)
 c['additive'].update(extra_input=row['arm'],extra_weight=.25,operator='One boundedSTFpair mask andmatchedRMS gain generatedin all4rows;choose onlyfourthIQpair;ORDINARY replays exactnativefirstpairIQ,not redraw',BN_RNG='0BN;private labelindependent perstepstream;originalCPU/Python/torch streams preserved')
 c['band_availability']=dict(arm=row['arm'],mask_seed=MASK_SEED,harmonic_pair_choices=list(range(1,7)),centers='+-k/20 cycles/sample',sigma_cycles_per_sample=1/256,depth=.5,relative_IQ_RMS_bound=.1,pad=128,FFT=512,RMS_restore=False,same_IQ_twice=True,generated_per_row=568320,layer_or_head_added=False,guard_required=True,training_gain_control='raw times RMS(maskedraw)/RMS(raw) perpacket;matchedpower,not labeldependent',frequency_mask_does_not_identify_or_replace_RX=True)
 c['source_selection']=dict(primary='MASK',references=['RAW','GAIN','ORDINARY'],fixed_final=20,held_RX=1,held_L=1260,clean_min_pp=1.,clean_F1_min_pp=0.,seenV_min_pp=-.5,final_base_objective_max=.5,practical_role='Allsixreported,not standaloneAgqualification;laterindependentAs/jointfactorial required',on_fail='Close fixedSTFpair availability training,no kernel/width/depth/frequency/weight/RX/seed/budget rescue,no controlpromotion',on_pass='Onlysource Ag cleanqualification;separatelyregisterconfirmingseed before anyjoint/targetclaim')
 c['training_recipe']='Original13L7U/supcon/U/ownEMA/commonbase.75raw+.25ordinary,extra.25RD(raw/gain/sameordinary/maskedraw),8L/total1.25,all4newfromscratchE20'
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed role/recipe/budget/operator differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=parent.make_args(template(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.seed,a.label_smoothing)!=(20,13,7,SEED,.01) or a.baseline_ckpt or a.teacher_ckpt or not a.use_aug or a.use_concat_sat_channel_aug:raise ValueError('Original scratchE20/U/defaulthead recipe only')
 return a
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Allfourfixedrows required')
 m=results['MASK'];checks={};comp={}
 for name in ('RAW','GAIN','ORDINARY'):
  b=results[name];delta={v:100*(m['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(m['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);seen=100*(m['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);part=dict(clean=delta['clean']>=1.-1e-10,clean_F1=f1>=-1e-10,seenV=seen>=-.5-1e-10);checks.update({name+'_'+k:v for k,v in part.items()});comp[name]=dict(all_view_gain_pp=delta,clean_F1_gain_pp=f1,seenV_gain_pp=seen,urban_mean_gain_pp=sum(delta[v] for v in SCENES[3:])/3)
 checks['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['availability_generated']==568320 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and not r['head_fit_updates'] and not r['target_access'] and not r['U_truth_read'] and not r['checkpoint_sources'] for r in results.values())
 return dict(qualified=all(checks.values()),checks=checks,comparisons=comp,source_only=True,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
