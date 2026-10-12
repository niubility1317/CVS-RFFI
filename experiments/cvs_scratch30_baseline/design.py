from pathlib import Path
from experiments.cvs_nocfo_foundation import design as n
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(n,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261012-phase1-scratch30-ce-selfssl-manysig-m6-r01';RELEASE='cvs_scratch30_baseline_20261012_r01';OWNER='codex/root/scratch30-baseline-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('CE30','SELF_SSL30');READOUTS=('NATIVE_HEAD',)
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');VIEWS=('clean',*SCENES)
VIEWS_ROOT=Path(PROJECT)/'runs/20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views'
U_CACHE=Path(PROJECT)/'releases/cvs_stage2_ssl_hybrid_20261011_r02/source_unlabeled/source_cache.pt'
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered scratch30 row')
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_contract=SOURCE,source_loader_config=n.config(next(r for r in n.rows() if r['model_seed']==row['model_seed'])),epochs=30,steps_per_epoch=222,updates=6660,batch_size=128,peak_lr=.0002,min_lr=.000001,weight_decay=.0001,gradient_clip=10.,checkpoint_sources=[],initialization='full native CVS identity scratch; no E20/E200/teacher weights',checkpoint_selection='fixed_final_epoch30',source_U=row['arm']=='SELF_SSL30',U_cache=str(U_CACHE) if row['arm']=='SELF_SSL30' else None,semi=dict(warmup_updates=1110,ramp_updates=1110,weight=.25,confidence=.95,min_streak=2,target='current own student deterministic clean eval detached; changes each update',history='same physical U ID and adjacent full traversal pass',strong_noise_std=.015,EMA=False),satellite_training=False,auxiliary=False,label_smoothing=0.,explicit_CFO_identity_feature=False,source_validation='all original V,health only at epoch10/20/30;no selection or updates',training_target_access=False,test_views=list(VIEWS),readouts=list(READOUTS),goal=dict(primary='SELF_SSL30',clean_three_seed_mean=.84,feedback_to_training=False),sampling='seedprivate RX/day balanced full L permutation; independent U permutation with traversal pass; no hidden U TX',budget_authorization='current user: from scratch30epochs; overrides prior E20 only for subsequent matched controls')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Scratch30 fixed config differs')
 return c
