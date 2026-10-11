"""Fixed source-derived E20+20 main-margin sensitivity and controlled L style search."""
from copy import deepcopy
from pathlib import Path
from experiments.cvs_e20_cross_style import design as old
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY=(getattr(old,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY'))
read,write,baseline=old.read,old.write,old.baseline
RUN='20261012-phase1-e20-boundary-style-manysig-m30-r01'
RELEASE='cvs_e20_boundary_style_20261012_r01';OWNER='codex/root/e20-boundary-style-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('CE','SSL','PROBE_FISH','M_SHALLOW','M_MIDDLE','M_BOTH','L_REPLAY','RANDOM_STYLE','HARD_STYLE','JOINT');PRIMARY='JOINT'
PAIRS=[('SSL','CE'),('M_BOTH','PROBE_FISH'),('M_BOTH','M_SHALLOW'),('M_BOTH','M_MIDDLE'),('HARD_STYLE','RANDOM_STYLE'),('RANDOM_STYLE','L_REPLAY'),('HARD_STYLE','L_REPLAY'),('JOINT','M_BOTH'),('JOINT','HARD_STYLE')]+[(a,'SSL') for a in ARMS if a not in ('CE','SSL')]+[(a,'CE') for a in ARMS if a not in ('CE','SSL')]+[(a,'P0') for a in ARMS]

def rows():return [dict(arm=a,model_seed=s,row_id=f'{a}-s{s}') for s in SEEDS for a in ARMS]

def config(row):
 if row not in rows():raise ValueError('Unregistered boundary/style row')
 arm,seed=row['arm'],row['model_seed']
 c=deepcopy(old.config(next(r for r in old.rows() if r['arm']==('CE' if arm=='CE' else 'SSL') and r['model_seed']==seed)))
 c.update(schema='e20_boundary_style_v1',run_id=RUN,**row,method='cvs_e20_boundary_style',output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER)
 fishr=['time_down'] if arm=='M_SHALLOW' else ['t2.pool'] if arm=='M_MIDDLE' else ['time_down','t2.pool'] if arm in ('PROBE_FISH','M_BOTH','JOINT') else []
 selection='hard' if arm in ('HARD_STYLE','JOINT') else 'random' if arm=='RANDOM_STYLE' else 'replay' if arm=='L_REPLAY' else 'none'
 mc=c['multilevel'];mc.update(enabled=bool(fishr),statistic='probe' if arm=='PROBE_FISH' else 'margin' if fishr else 'none',
  fishr_levels=fishr,fishr_mode='fixed' if fishr else 'none',fishr_estimator='cross',
  sensitivity_eps=1e-6,mixstyle_l=selection!='none',mixstyle_u=False,
  style_selection=selection,candidates=4,style_levels=['time_down','t2.pool'] if selection in ('hard','random') else [],
  mixstyle_p=1.,mixstyle_alpha=.1,mixstyle_strength=1.,style_ce_weight=1.,
  fishr_scope='main-margin per-channel affine sensitivity, per-sample RMS normalized; unbiased stratified AB variance cross inner product' if arm!='PROBE_FISH' else 'detached linear probe CE coordinate gradients; AB cross inner product',
  gradient_budget_scope='fixed 12-pair L-only calibration; margin uses own term gradient, probe uses old-pair reference; no current-batch adaptation',
  style_scope='4 legal sameTX/day crossRX distinctPID L style candidates; eval batch-mean CE chooses hard; random control has identical search cost; selected train forward shares clean dropout RNG',
  aux_ce_routing='only PROBE_FISH fits detached heads; margin has no auxiliary learned parameters or CE')
 c['teacher']['cache_root']=(BASE/'peer_cache').as_posix();c['evaluation']['primary_arm']=PRIMARY
 c['feature_target']='main identity margin sensitivities and supervised style at time_down/t2.pool'
 return c

def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Boundary/style configuration differs')
 return c

def learning_rate(c,epoch):return old.learning_rate(c,epoch)
