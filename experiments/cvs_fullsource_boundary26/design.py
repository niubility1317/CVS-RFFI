"""One scratch E30 training-boundary intervention; all original source roles."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_sigtime_rdrop_legacy import design as legacy
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write=(getattr(legacy,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write'))
RUN='20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_fullsource_boundarymix_E30_20261017_r01'
OWNER='codex/root/FcrossRX-fullsource-boundarymix-E30-20261017'
BASE=Path(PROJECT)/'runs'/RUN
BACKEND=PROJECT+'/releases/cvs_FcrossRX_additiveAs_RX6_source_E20_20261016_r01'
BACKEND_COMMIT='367b2f945ef65488e6d97eddd2357388d2477100'
SMOKE_SOURCE=Path(PROJECT)/'runs/20261017-phase1-FcrossRX-fullsource-concat-e20-s2026092701-m2-r01/B00-s2026092701/source'
FORMAL_VIEWS=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
TRUTH,CLASSES=legacy.TRUTH,legacy.CLASSES
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');VIEWS=('clean',*SCENES);TRAIN_SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
REFERENCE_RUN=Path(PROJECT)/'runs/20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01'
SEED=2026092701;STEPS=222;UPDATES=6660;RX=(1,3,4,6,8)
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ('BCTX',)]
def parent_config():return legacy.config(next(r for r in legacy.rows() if r['row_id']=='E20_RDROP-s'+str(SEED)))
def config(row):
 if row not in rows():raise ValueError('Fixed one full-source intervention row only')
 c=deepcopy(parent_config());c.update(schema='FcrossRX_fullsource_boundarymix_E30_v1',run_id=RUN,**row,launch_owner=OWNER,output_root=(BASE/row['row_id']/'source').as_posix(),As=True,Ag=False,held_RX=None,source_RX=list(RX),source_roles=dict(L_s=6300,U_s=56700,V=27000),epochs=30,label_epochs=20,pseudo_epochs=10,updates=6660,train_scenes=list(TRAIN_SCENES),test_views=list(VIEWS),baseline_reuse=False,checkpoint_sources=[],checkpoint_selection='final_only',training_recipe='F:X3 SIG_TIME/CENTER_UNITS/defaultcos; original20L10U/ownEMA/oldgate; ordinary/practical(ordinary), alternating reflect/edge native training boundary; matched4L',baseline_definition='F+CROSS_RX_SUPCON',initialization='scratch_only',source_folds_forbidden=True,benchmark_context='User selected exposed historical benchmark; fixed terminal scoring, no independent new blind-test claim')
 c['training_mechanism']=dict(RDrop_alpha=1.,ordinary_L=True,L_forwards_per_sample=4,expected_L_forward_packets=3409920,primary_weight=.5,secondary_weight=.5,same_IQ_within_RDrop=True,alpha=1.,new_U_augmentation=False,original_U_recipe=True,new_losses=False,clean_repeat_control=False,extra_parameters=0)
 c['supcon']=dict(mode='CROSS_RX',levels=['time_down','t2.pool'],temperature=.2,max_weight=.1,warm_epochs=5,ramp_epochs=10,only_clean_half=True,extra_forwards=0)
 c['channel']=dict(route='native concat_full',input='existing ordinary augmentation output',scenes=list(TRAIN_SCENES),probability=1. if c['As'] else 0.,processing_route='residual',mode='post_sync',equalization_enabled=False,G=1.,reference_snr_db=30.,fs_hz=25000000.,fc_hz=2450000000.,receiver_seed=2026092701,view_seed=202610172201,output_post_RMS=False,session='physical source RX/day, not TX',formal_unchanged=True)
 c['training_boundary']=dict(schedule='odd call reflect, even call edge;111 each per epoch',receiver_AGC='native recomputed; never frozen',physical_changes='boundary only; all other Config fields unchanged',formal_boundary='original reflect unchanged',extra_NN=0)
 c['matched_reference_run']='20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01'
 c['budget_change']=dict(authorized='User raised new scratch budget to E30 on2026-10-17',schedule='20L+10U; cosine total30; SupCon warm5/ramp10 unchanged',old_E20='historical results only, never same-budget controls or weight parents');c['budget_claim']='Each scratch30/6660/3409920 L network packets; original hiddenU/ownEMA and sourceV measured; compatibility full-source E20 CP only2 synthetic,no initialization; everyrow automatic7viewtruth-last'
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Full source, recipe, matrix or E30 differs')
 return c
def make_args(c,device='cuda:0'):
 validate(c);a=legacy.make_args(parent_config(),device)
 a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.epochs=30;a.label_epochs=20;a.pseudo_epochs=10
 a.use_concat_sat_channel_aug=True;a.sat_training_mode='concat_full';a.concat_sat_ce_only=False;a.concat_sat_deduplicate_tx_ce=False;a.concat_sat_teacher_clean_only=False
 a.rc4_satellite_family='practical';a.sat_train_scenarios=','.join(TRAIN_SCENES);a.sat_view_schedule='';a.sat_view_prob=c['channel']['probability'];a.sat_view_seed=202610172201;a.concat_sat_start_epoch=1
 a.practical_route='residual';a.practical_equalization=False;a.practical_fs_hz=25000000.;a.practical_fc_hz=2450000000.;a.practical_receiver_seed=2026092701
 a.use_sat_consistency=False;a.lambda_sat_cls=0.;a.lambda_sat_cons=0.;a.test_eval_start_epoch=999999;a.checkpoint_selection='final_only'
 if hasattr(a,'source_fold_held_rx') or hasattr(a,'source_fold_training_receivers'):raise ValueError('No inherited source-fold argument permitted')
 if (a.epochs,a.label_epochs,a.pseudo_epochs,a.label_smoothing,a.batch_size)!=(30,20,10,.01,128) or a.baseline_ckpt or a.teacher_ckpt or not a.use_aug or not a.use_unlabeled or not a.use_ema_teacher:raise ValueError('Original full source scratch E30 required')
 return a
