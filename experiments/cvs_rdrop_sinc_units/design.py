"""User-fixed SIG_TIME + RDrop A–F matched matrix, scratch50."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_sigtime_rdrop_legacy import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261015-phase1-SIGTIME-RDrop-AF-e50-clean-manysig-m18-r01'
RELEASE='cvs_SIGTIME_RDrop_AF50_20261015_r01'
OWNER='codex/root/SIGTIME-RDrop-AF-20261015'
DEPENDENCY_RELEASE,DEPENDENCY_COMMIT,FOLD_RELEASE=(getattr(parent,k) for k in ('DEPENDENCY_RELEASE','DEPENDENCY_COMMIT','FOLD_RELEASE'))
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('A','B','C','D','E','F');PRIMARY='F';BASELINE_ARM='A'
POLICIES=dict(A=('SIG_TIME','RAW_HZ','STANDARD'),B=('SIG_TIME_X3','RAW_HZ','STANDARD'),C=('SIG_TIME','RAW_HZ','NORMALIZED_UNITS'),D=('SIG_TIME','CENTER_25K','STANDARD'),E=('SIG_TIME','CENTER_25K','CENTER_UNITS'),F=('SIG_TIME_X3','CENTER_25K','CENTER_UNITS'))
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;STEPS=222;UPDATES=11100
VIEWS=('clean',);READOUTS=('NATIVE_HEAD',);SINC_DIAGNOSTIC_EPOCHS=(1,17,33,50)
ORIGINAL_RUN='20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01'
RECOVERY_RUN='20261015-phase1-originalSIGTIME-rdrop-e50-technical-clean-manysig-m4-r02'
MODEL_CODE_COMMIT='b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
REUSED_COORDINATE_COMMIT='0f81373c25ba69a6cb9deec5c7ca652880fd62af';REUSED_X3_MODULE='cvs_signature_stats50.features.CompositePool(SIG_TIME_X3)'
def reference_config(seed):return parent.config(next(r for r in parent.rows() if r['epochs']==50 and r['mode']=='RDROP' and r['model_seed']==seed))
def reference_source(seed):return Path(PROJECT)/'runs'/(ORIGINAL_RUN if seed==SEEDS[0] else RECOVERY_RUN)/f'E50_RDROP-s{seed}'/'source'
def rows():return [dict(row_id=f'{arm}-s{s}',arm=arm,mode='RDROP',epochs=50,model_seed=s) for s in SEEDS for arm in ARMS]
def training_rows():return [r for r in rows() if r['arm']!='A']
def config(row):
 if row not in rows():raise ValueError('Unregistered A-F row')
 c=deepcopy(reference_config(row['model_seed']));view,coord,opt=POLICIES[row['arm']]
 c.update(schema='original_SIGTIME_RDrop_AF_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=row['arm']=='A')
 c['baseline_reference']=dict(source=reference_source(row['model_seed']).as_posix(),model_code_commit=MODEL_CODE_COMMIT,original_row_id=f"E50_RDROP-s{row['model_seed']}",role='Original RDrop matching recipe fixedfinal50; no selected weight')
 c['training_mechanism']['sinc_coordinates']=coord
 c['combination']=dict(SIG_TIME=view,Sinc_coordinates=coord,Sinc_optimizer=opt,s=25000.,Fs=25000000.,initial_q=0 if coord=='CENTER_25K' else None,default_head_joint=True,MEAN_VAR=False,COV=False,ensemble=False)
 c['sinc_optimizer_units']=dict(mode=opt,sample_rate_Hz=25000000.,center_scale_Hz=25000.,lr='alpha Fs/s' if opt=='CENTER_UNITS' else 'alpha Fs' if opt=='NORMALIZED_UNITS' else 'alpha',epsilon='epsilon s/Fs' if opt=='CENTER_UNITS' else 'epsilon/Fs' if opt=='NORMALIZED_UNITS' else 'epsilon',weight_decay='q -= alpha lambda (q+p0/s); AdamW q decay0' if opt=='CENTER_UNITS' else 'lambda/Fs' if opt=='NORMALIZED_UNITS' else 'native lambda on actual leaf',gradient_clipping='physical Hz globalclip5' if opt=='CENTER_UNITS' else 'unchanged native actual-leaf globalclip5',native_cosine_schedule='Original alpha cosine; one fixed mapping immediately before each actual step',extra_parameters=0,extra_model_forwards_vs_RDrop=0,extra_optimizer_steps=0,diagnostic_epochs=list(SINC_DIAGNOSTIC_EPOCHS))
 c['source_falsification']=dict(first_seed=SEEDS[0],necessary_arms=list('BCDEF'),final_L_CE_max=.5,actual_updates=11100,actual_U_packets=963900,minimum_positive_KL_call_fraction=.01,unit_changed_cutoff_min=12,unit_changed_bandwidth_min=12,kernel_relative_L2_min=1e-6,actual_deployment_feature_change_min=1e-6,feature_export_exact=True,source_only=True,selection='All five first-seed source trajectories must pass fixed health/implementation checks. Freeze and push source-only expansion decision before reading any new target summary; only then release unchanged seeds02/03. No sourceV ranking or target feedback.')
 c['training_recipe']='Original SIG_TIME+RDrop1 / ordinary augmentation reused IQ / LS.01 / 33L+17U / old batch_neighbor+ownEMA+cos / default head joint; only declared A-F feature/Sinc treatment differs'
 c['cost_claim']='15 new scratch50 models;3 complete matched A references reused;C had no matching RDrop task at registration; no extra model forwards vs RDrop, no fitted head'
 c['goal']=dict(primary='F',clean_three_seed_mean=.84,matched_baseline='A',compare_to_frozen_RDrop=True,feedback_to_training=False,head_fit_updates=0)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed A-F recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Completed RDrop A must not be retrained')
 a=parent.make_args(reference_config(c['model_seed']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
def request():return dict(run_id=RUN,dependency_release=DEPENDENCY_RELEASE,fold_release=FOLD_RELEASE,rows=[config(r) for r in rows()],views=list(VIEWS),readouts=list(READOUTS),default_head_only=True)
def totals():return dict(new_models=15,reused_models=3,new_epochs=750,new_optimizer_steps=166500,historical_epochs_separate=150,historical_updates_separate=33300,first_seed_new_models=5,first_seed_new_epochs=250,first_seed_new_steps=55500,extra_forwards_vs_RDrop=0,new_U_observations=14458500,clean_decisions=3024000,metric_records=324)
