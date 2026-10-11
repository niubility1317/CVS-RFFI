"""Source-evidence-driven fixed E20+20 multilevel moment/style experiment."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_e20_peer_soft import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-e20-cross-style-manysig-m33-r01'
RELEASE='cvs_e20_cross_style_20261012_r01';OWNER='codex/root/e20-cross-style-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('CE','SSL','OLD_PAIR','F_SHALLOW','F_MIDDLE','F_BOTH','L_REPLAY','STYLE_SHALLOW','STYLE_MIDDLE','STYLE_BOTH','JOINT');PRIMARY='JOINT'
PAIRS=[('SSL','CE'),('F_BOTH','OLD_PAIR'),('F_BOTH','F_SHALLOW'),('F_BOTH','F_MIDDLE'),('STYLE_BOTH','STYLE_SHALLOW'),('STYLE_BOTH','STYLE_MIDDLE'),('JOINT','F_BOTH'),('JOINT','STYLE_BOTH')]+[(a,'L_REPLAY') for a in ('STYLE_SHALLOW','STYLE_MIDDLE','STYLE_BOTH')]+[(a,'SSL') for a in ARMS if a not in ('CE','SSL')]+[(a,'CE') for a in ARMS if a not in ('CE','SSL')]+[(a,'P0') for a in ARMS]

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
 if row not in rows():raise ValueError('Unregistered cross/style row')
 arm,seed=row['arm'],row['model_seed']
 c=deepcopy(old.config(next(r for r in old.rows() if r['arm']==('CE' if arm=='CE' else 'PEER_ALL') and r['model_seed']==seed)))
 c.update(schema='e20_cross_style_v1',run_id=RUN,**row,method='cvs_e20_cross_style',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 fishr=['time_down'] if arm=='F_SHALLOW' else ['t2.pool'] if arm=='F_MIDDLE' else ['time_down','t2.pool'] if arm in ('OLD_PAIR','F_BOTH','JOINT') else []
 style=['time_down'] if arm=='STYLE_SHALLOW' else ['t2.pool'] if arm=='STYLE_MIDDLE' else ['time_down','t2.pool'] if arm in ('STYLE_BOTH','JOINT') else []
 mc=c['multilevel'];mc.update(enabled=bool(fishr),fishr_levels=fishr,fishr_mode='fixed' if fishr else 'none',fishr_estimator='old_pair' if arm=='OLD_PAIR' else 'cross',
  batch_size=120,pair_batch_size=60,pair_quota=2,aux_ce_to_backbone=False,mixstyle_l=bool(style) or arm=='L_REPLAY',mixstyle_u=False,style_levels=style,
  mixstyle_p=.5,mixstyle_alpha=.1,mixstyle_strength=1.,style_ce_weight=1.,
  fishr_calibration_pairs=12,fishr_ratio=.2,fishr_cap=10000.,fishr_fixed_denominator=2,
  sampling='independent A/B, each same-day 6TX x 5RX x2 L, cross-batch overlap allowed',
  gradient_budget_scope='fixed after initial L-only calibration using mean old-pair gradient norms; no current-batch adaptation',
  fishr_scope='finite-L stratified variance AB inner product; detached head, both feature graphs live',
  style_scope='extra L main identity CE at selected hooks; same TX/day, cross RX, different real PID; duplicate occurrences permitted',
  aux_ce_routing='both heads fit by detached clean A/B features; no auxiliary CE to backbone')
 c['teacher']['cache_root']=(BASE/'peer_cache').as_posix()
 c['semi']['target_provider']='none' if arm=='CE' else 'mean probabilities of the other two fixed original E20 models; no EMA'
 c['evaluation']['primary_arm']=PRIMARY
 c['feature_target']='shallow/middle finite-sample Fishr; supervised main-path L MixStyle'
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Cross/style configuration differs')
 return c

def learning_rate(c,epoch):return old.learning_rate(c,epoch)
