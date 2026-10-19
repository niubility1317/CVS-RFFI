"""One post-complete-As erasure-position hypothesis; identical native E20+10."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_lraw_tail30 import design as old
ROOT=Path(__file__).resolve().parents[2]
RUN='20261019-phase1-FcrossRX-asPrefix-e30-s2026092701-m2-r01';RELEASE='cvs_FcrossRX_asPrefix_E30_20261019_r01'
OWNER='codex/root/as-prefix81';LOCAL='local_artifacts/cvs_FcrossRX_asPrefix_E30_20261019_r01'
DEPENDENCY_COMMIT='d14450bf1b0d12cc281df1f4d0e79e8760e48282'
REFERENCE_RUN=old.REFERENCE_RUN;REFERENCE_COMMIT=old.REFERENCE_COMMIT;REFERENCE_RELEASE=old.REFERENCE_RELEASE
SEED=old.SEED;RX=FIT=old.RX;STEPS=222;UPDATES=2220
EPOCH_OFFSET=20;STEP_OFFSET=4440;CUMULATIVE_EPOCHS=30;CUMULATIVE_UPDATES=6660
read,write=old.read,old.write
PROJECT=old.PROJECT
for _key in ('SOURCE','FORMAL_VIEWS','TRUTH','PANEL','SOURCE_CACHE','VIEWS','SCENES','CLASSES','FULL_FP32_POLICY','TRAIN_SCENES','CHANNEL_SEEDS'):globals()[_key]=deepcopy(getattr(old,_key))
BASE=Path(PROJECT)/'runs'/RUN;ARMS=('RANDOM','PREFIX')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def original_row():return next(r for r in old.rows() if r['arm']=='B01')
def reference_config(row=None,segment=30):return old.reference_config(original_row(),segment)
def config(row,segment=30):
 if row not in rows() or segment!=30:raise ValueError('Only two original As20 independent E10 tails')
 c=reference_config(segment=30)
 c.update(schema='FcrossRX_As_prefix81_v1',run_id=RUN,**row,launch_owner=OWNER,method='post_complete_As_IQ_erasure_position',output_root=str(BASE/row['row_id']/'source'),training_recipe='Unchanged original P43 6L/1.25 scale;ordinary/raw/U unchanged;erase complete As IQ after native AGC with local named RNG',budget_authorization='Two E10 tails from original As20,totalE30/6660 each;reuse original NoAs30/As30',baseline_reference='Original P43 ownNoAs20+10 and ownAs20+10 readouts;newrows both share originalAs20 student+EMA',benchmark_context='P26 mixed padding failed;here preserve native reflect/quality/AGC then erase postreceiver IQ.No evidence mask helps;source P79 edge injury only motivates dependency test',cost_claim='768 L NN perstep,original U+EMA,0newNN/loss/parameter;142080 fullAs perrow')
 c['training_mechanism'].update(objective='.5RDordinary128+.5RDmixed128+.25RDraw128+.1ORD128CROSS_RX',L_forwards_per_sample=6,pair_weights=[.5,.5,.25],same_IQ_within_RDpair=True,U_recipe_unchanged=True)
 c['masking']=dict(mode=row['arm'],probability=.5,length='actual allocated residual_kernel_length minus1 per observation',expected_length=61,reference_window=[80,160],random_start='uniform integer start in [0,80-length] union [160,256-length]',apply_after='full native practical residual,receiver adaptation,compensation,AGC;before existing two As forwards',both_IQ=True,recompute_receiver_state=False,renormalize=False,quality_metadata='pre-erasure physical state retained explicitly',stream='separate physical sample/epoch/step enable+position,TX labels unused;enable identical across modes',formal_mask=False,new_losses=0,new_parameters=0,new_NN=0)
 c['source_promotion']=dict(fullV_PREFIX_minus_NOAS_clean_min_pp=-.1,panel_midlow_PREFIX_minus_AS_min_pp=.5,panel_midlow_PREFIX_minus_RANDOM_min_pp=.25,source_failure_skips_formal=False,automatic_extension=False,on_fail='close position hypothesis;no E50/seed/masklength/probability rescue')
 return c
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row,c.get('segment')):raise ValueError('Fixed mask position/data/Asparent/budget differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 original=b.config(next(r for r in b.rows() if r['arm']=='B01'),30)
 if original!=reference_config():raise ValueError('Pinned original P43 recipe differs')
 a=b.make_args(original,device);a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt=c['parent_source']+'/final_ssdg.pth';a.teacher_ckpt=''
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.batch_size,a.label_smoothing,a.seed)!=(10,0,10,128,.01,SEED) or not a.use_aug or a.use_concat_sat_channel_aug or not a.use_ema_teacher or not a.use_unlabeled:raise ValueError('Original L/RDrop/U args differ')
 return a
