"""F-only full-source50 target-clean comparison fixed before new target results."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_rdrop_sinc_units import design as af
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(af,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-F-sinc-bandgain-e50-clean-manysig-m9-r01';RELEASE='cvs_F_sinc_bandgain50_clean_20261016_r01';OWNER='codex/root/F-target-band50-20261016';BASE=Path(PROJECT)/'runs'/RUN
AF_RELEASE=PROJECT+'/releases/'+af.RELEASE;AF_COMMIT='09554c3a06c976c53efc4dbc61f5c501a1e7b06e'
ARMS=('FULL','NORM','BAND');BASELINE_ARM='FULL';PRIMARY='BAND';EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;UPDATES=11100;SINC_DIAGNOSTIC_EPOCHS=(1,17,33,50);VIEWS=('clean',);READOUTS=('NATIVE_HEAD',)
def base_config(seed):return af.config(next(r for r in af.rows() if r['arm']=='F' and r['model_seed']==seed))
def rows():return [dict(row_id=a+'-s'+str(s),arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def training_rows():return [r for r in rows() if r['arm']!=BASELINE_ARM]
def config(row):
 if row not in rows():raise ValueError('Fixed F three-seed target matrix required')
 c=deepcopy(base_config(row['model_seed']));c.update(schema='F_band50_target_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),baseline_reuse=row['arm']==BASELINE_ARM)
 c['baseline_reference']=dict(source=base_config(row['model_seed'])['output_root'],original_row_id='F-s'+str(row['model_seed']),role='Matched completed F final50, no retraining or selected weight')
 c['band_step_policy']=dict(mode=row['arm'],formula='w_i=max(initial_effective_bandwidth_i,Fs/K)/Fs;Fs25e6,K79',decay='Physical origin decay once and unscaled',clipping='physical Hz globalclip5',NORM='Scalar matches own-input BAND adaptive norm only, not cross-trajectory or decay-inclusive displacement',extra_trainable_parameters=0)
 c['budget_authorization']='User20261016 F-only target-performance optimization;6 new scratch50 models and3 completed matching F references;no source performance gate';c['source_falsification']=None;c['source_selection']='fixedfinal50; sourceV health only; all three seeds preregistered before any new target summary, no source qualification gate'
 c['training_recipe']='F SIG_TIME_X3 + unified Sinc + full RDrop1 + same actual ordinary-aug IQ / LS.01 / oldgate-ownEMA-cos / scratch50=33L+17U / original full-source roles / defaultheadjoint; only adaptive Sinc displacement differs'
 c['cost_claim']='6 new scratch50=300modelepochs/66600steps;3 matched completed F references reused, historical150epochs separate; automatic clean truth-last everyrow; zero fittedheads'
 c['goal']=dict(primary='BAND',clean_three_seed_mean=.84,matched_baseline='FULL (historical F)',actual_target_accuracy_gain=True,feedback_to_training=False,head_fit_updates=0)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed full-source F target comparison differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 if c['baseline_reuse']:raise ValueError('Completed F must not be retrained')
 a=af.make_args(base_config(c['model_seed']),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.max_grad_norm)!=(50,33,17,.01,True,5.) or a.baseline_ckpt or a.teacher_ckpt or getattr(a,'source_fold_held_rx',None) is not None:raise ValueError('Original full-source scratch50 recipe required')
 return a
