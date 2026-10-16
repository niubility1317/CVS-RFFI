"""One source-only Sinc displacement hypothesis and its two controls."""
from copy import deepcopy
from pathlib import Path
import math
from experiments.cvs_rdrop_sinc_units import design as af
from experiments.cvs_rx_u_increment20 import design as fold
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(af,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261016-phase1-sinc-bandgain-rdrop-sourcefold-s2026092701-m3-r01'
RELEASE='cvs_sinc_bandgain_rdrop_source20_20261016_r01'
OWNER='codex/root/Sinc-bandgain-RDrop-20261016';BASE=Path(PROJECT)/'runs'/RUN
AF_RELEASE=PROJECT+'/releases/'+af.RELEASE;AF_COMMIT='09554c3a06c976c53efc4dbc61f5c501a1e7b06e'
SEED=2026092701;ARMS=('FULL','NORM','BAND');FIT=(3,4,6,8);HELD=1;UPDATES=4440
WITNESS_EPOCHS=(1,13,14,20)
def base_model_config():return af.config(next(r for r in af.rows() if r['arm']=='E' and r['model_seed']==SEED))
def native_config():return af.parent.config(next(r for r in af.parent.rows() if r['epochs']==20 and r['mode']=='RDROP' and r['model_seed']==SEED))
def rows():return [dict(row_id=a+'-RX1-s'+str(SEED),arm=a,model_seed=SEED) for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Only fixed three source gain rows allowed')
 c=deepcopy(native_config());c.update(schema='Sinc_bandgain_RDrop_source_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),method='Sinc_adaptive_displacement_geometry',stage='source_RX_fold_full_training',test_views=[],readouts=['FINAL'],baseline_reference='Fresh matched FULL gain1, no historical weight',fold=fold.config()['fold'],source_falsification=None)
 c['training_mechanism']['sinc_coordinates']='CENTER_25K'
 c['band_step_policy']=dict(mode=row['arm'],formula='w_i=max(initial_effective_bandwidth_i,Fs/K)/Fs,Fs25e6,K79',FULL='Original unified adaptive displacement',NORM='Positive scalar matches the BAND adaptive-step L2 at its own incoming step',BAND='Independent fixed perchannel adaptive displacement gains; same gain for low/band leaves',decay='Original physical alpha lambda p including origin, exactly once, not gain-scaled',clipping='Original global5 in physicalHz',gradient='Original pairedCE+RDropKL alpha1 and native hiddenU gradients unchanged',free_strength_hyperparameters=0,trainable_parameters_added=0,model_forwards_added=0)
 c['training_recipe']='Original SIG_TIME/RDrop1/normalaug/LS.01/defaultheadjoint/ownEMA/oldgate/cos; unifiedSinc; scratch20 13L+7U; true RX1 exclusion before model creation'
 c['source_selection']=dict(primary='BAND',controls=['FULL','NORM'],held_RX=1,readout='Fixedfinal20 defaulthead,1260heldL',final_L_CE_max=.5,errors_strictly_fewer=True,mean_CE_no_higher=True,all_actual_steps=4440,on_pass='Freeze source decision, then separately register unchanged full-source50 paired comparison; no automatic target in this source run',on_fail='Close fixed bandgain formula without strength/epoch/parent/seed rescue',target_access=False)
 c['cost_claim']='3 scratch20=60model-epochs/13320steps; each568320extraL,396368hiddenUobservations,1260heldL+144unusedanchors; no fittedhead or target'
 c['goal']=dict(clean_three_seed_mean=.84,improve_matched_original_RDrop=True,goal_complete=False,target_feedback=False)
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed Sinc gain source recipe differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=af.parent.make_args(native_config(),device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.source_fold_held_rx=HELD;a.source_fold_training_receivers=','.join(map(str,FIT))
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.use_aug,a.batch_size)!=(20,13,7,.01,True,128) or a.baseline_ckpt or a.teacher_ckpt or a.use_concat_sat_channel_aug or not a.use_ema_teacher or a.max_grad_norm!=5.:raise ValueError('Original source recipe/scratch/clip5 required')
 return a
source_validation=fold.source_validation
def qualify(results):
 if set(results)!=set(ARMS):raise ValueError('Complete three source conditions required')
 for r in results.values():
  v=r['held_default_head']
  if v['count']!=1260 or not math.isfinite(v['mean_CE']) or r['actual_steps']!=4440 or r['head_fit_updates'] or r['target_access']:raise ValueError('Complete legal source result required')
 b=results['BAND']['held_default_head'];contrast={a:dict(error_difference=b['errors']-results[a]['held_default_head']['errors'],CE_difference=b['mean_CE']-results[a]['held_default_head']['mean_CE']) for a in ('FULL','NORM')};healthy=all(r['final_L_CE']<=.5 and r['feature_witness']['feature_export_exact'] and r['feature_witness']['classW_zeroing_leaves_feature_identical'] for r in results.values())
 return dict(qualified=healthy and all(v['error_difference']<0 and v['CE_difference']<=0 for v in contrast.values()),all_source_healthy=healthy,contrasts=contrast,source_only=True,target_access=False,goal_complete=False)
