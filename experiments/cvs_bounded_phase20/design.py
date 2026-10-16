"""One fixed source-only three-row scratchE20,after native bounded phase identity evidence."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_crossrx_satmix20 import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(old,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-F-crossRX-boundedPhase-sourceRX1-e20-s2026092701-m3-r01';RELEASE='cvs_F_bounded_nonaffine_phase_sourceRX1_e20_20261017_r01';OWNER='codex/root/F-bounded-nonaffine-phase-E20-20261017'
LOCAL='local_artifacts/F_bounded_nonaffine_phase_E20_20261017';BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_F_crossRX_satmix25_source_e20_20261016_r01';BACKEND_COMMIT='0ae066adb10a001e834c8c679c5b7b81c20b3ac8'
SMOKE_SOURCE=Path(PROJECT)/'runs'/old.RUN/'B00-RX1-s2026092701/source';GUARD='20261017-diagnostic-F-crossRX-boundedPhase-fitL-manysig-m2-r01'
SEED=2026092701;FIT=(3,4,6,8);ARMS=('B00','C_ordinary','B10');UPDATES=4440;SCENES=old.SCENES;PHASE_SEED=202610173401;SIGMA=.006;PHASE_MAX=.1
base_model_config=old.base_model_config;source_validation=old.source_validation;macro_f1=old.macro_f1

def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def original_config():return old.config(old.rows()[0])
def config(row):
 if row not in rows():raise ValueError('Fixed three bounded nonaffine phase source rows only')
 c=deepcopy(original_config());arm=row['arm'];mode={'B00':'raw','C_ordinary':'ordinary','B10':'phase'}[arm]
 c.update(schema='F_CROSS_RX_bounded_nonaffine_phase_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='F_CROSS_RX_bounded_nonaffine_phase',baseline_reference='Matched extra raw and samealreadygeneratedordinary controls;whole base retained',updates=4440)
 c['training_mechanism'].update(Ag=arm=='B10',As=False,base_raw_total_weight=.75,ordinary_weight=.25,raw_weight=.5,s_slot_weight=.25,s_slot_name='Common thirdpair raw allrows',extra_pair_weight=.25,extra_pair_input=mode,L_forwards_per_packet=8,expected_L_packets=4546560,no_new_U_augmentation=True)
 c['isolated_phase']=dict(mode=mode,increment_sigma_rad=SIGMA,peak_bound_rad=PHASE_MAX,seed=PHASE_SEED,source_identity_guard=GUARD,source_guard_result='PASS_SOURCE_IDENTITY_ONLY',operator='CPUFloat64 iidGaussian increments.006;Wienercumsum;orthogonal removeaffine;globalpeakrescale.1rad;full256unitphase,Float32output',scope='All128L,once perstep;no label/RX/day/confidence conditioning;allpaths retained,no redraw/pointclip/PA/MP/AWGN/CFO/shift/RMSrestoration',randomness='Privatefork CPUgenerator seed+optimizer_step,CPUgeneratedphase forCPUorCUDAinputs,restore globalCPU/Python/CUDA RNG;CPUIQtransferandstats cost recorded',same_IQ_within_each_pair=True,ordinary_control='Reuse exact same nativeordinary tensor already used in common RDrop pair;no regeneration',base_pair_weight_sum=1.,pair_weight_sum=1.25,extra_pair_weight=.25,normalization='Mean overphysicalL within eachsameIQ RDrop pair;fullraw.75+ordinary.25+unchangedSupCon retained;extra .25R(view),no divideby4',loss='Original smoothedCE.01 mean+sameIQ RDrop1KL;no crossview consistency/newloss',U='Original13L+7U/oldgate/ownEMA/hiddentruth unchanged;no addedU/phase/teacher forward',expected_phase_generated_packets=568320 if arm=='B10' else 0,extra_L_NN_packets=1136640,full256=True,constant_and_linear_phase_removed=True,phase_distribution='Boundedgloballyrescaled detrendedWiener-derived,not untouchedWiener orRXreplacement',all_actual_paths_retained=True,scope_limit='OldordinaryalreadycontainsWiener;newviewvariant only,sourceidentitynotDGproof')
 c['channel']['input']='No As training;only unchanged held-source final20 sixpractical readouts';c['channel']['composition']=False
 c['training_recipe']='ScratchE20/F+CROSS_RX/NoExplicitCFO/defaultcos/originalordinary/U;raw.75+ordinary.25+SupCon plus .25 raw/sameordinary/phase(raw),8Lforwards'
 c['source_selection']=dict(primary='B10',references=['B00','C_ordinary'],fixed_final=20,held_RX=1,held_L=1260,views=['clean',*SCENES],min_clean_gain_pp=1.,min_clean_macroF1_gain_pp=0.,max_each_practical_drop_pp=.5,max_seenV_drop_pp=.5,final_base_L_objective_max=.5,on_fail='Closefixedsigma.006/peak.1/fullcoverage nonaffinephase training;no sigma/peak/range/coefficient/clip/pathdrop/redraw/RX/TX/day/seed/budget rescue',on_pass='Independent source-seed confirmation only;not target/DG/fulljoint proof')
 c['budget_authorization']='ActiveAg/As goal,latestF+CROSS_RX/E20;three scratchE20/60modelepochs/13320updates;no initialization parent or continuation'
 c['cost_claim']='Each4546560LNN incl1136640extra;all3oldordinary568320,onlyB10 isolatedphase568320;originalU/EMA/V/8820held and own2NNsmoke separatelyactualcounted;one2NNdispatcherrealCPsmoke;noextra teacher/headfit'
 return c

def validate(c):
 row=next((v for v in rows() if v['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed bounded phase source roles/budget/operator/weights/controls differ')
 return c

def make_args(c,device='cuda:0'):
 validate(c);a=old.make_args(original_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a

def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('All three source bounded phase rows required')
 s=results['B10'];conditions={};comparisons={}
 for ref in ('B00','C_ordinary'):
  b=results[ref];delta={v:100*(s['views'][v]['accuracy']-b['views'][v]['accuracy']) for v in ('clean',*SCENES)};f1=100*(s['views']['clean']['macro_F1']-b['views']['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-b['seen_V']['source_val_accuracy']);part=dict(clean=delta['clean']>=1.,clean_macroF1=f1>=0.,each_practical=all(delta[v]>=-.5 for v in SCENES),seen_V=dv>=-.5);comparisons[ref]=dict(conditions=part,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seen_V_gain_pp=dv);conditions.update({ref+'_'+k:v for k,v in part.items()})
 conditions['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['phase_steps']==4440 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and r['phase_generated_packets']==(568320 if arm=='B10' else 0) and not r['As_training_observations'] and not r['teacher_phase_calls'] and not r['head_fit_updates'] and not r['target_access'] for arm,r in results.items())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False)
