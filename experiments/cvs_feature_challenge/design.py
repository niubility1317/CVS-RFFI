from copy import deepcopy
from pathlib import Path
from experiments.cvs_scratch30_baseline import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES,SEEDS,FULL_FP32_POLICY,read,write=(getattr(parent,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES','SEEDS','FULL_FP32_POLICY','read','write'))
RUN='20261013-phase1-feature-challenge-clean-manysig-m9-r01'
RELEASE='cvs_feature_challenge_20261013_r01'
OWNER='codex/root/feature-challenge-20261013'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('CE30','RANDOM_MASK','GRAD_MASK')
PRIMARY='GRAD_MASK';VIEWS=('clean',);SCENES=parent.SCENES;VIEWS_ROOT=parent.VIEWS_ROOT
LEVELS=('time_down','t2.pool')
def rows():return [dict(row_id=f'{a}-s{s}',arm=a,model_seed=s) for s in SEEDS for a in ARMS]
def config(row):
 if row not in rows():raise ValueError('Unregistered feature challenge row')
 c=deepcopy(parent.config(next(r for r in parent.rows() if r['arm']=='CE30' and r['model_seed']==row['model_seed'])))
 c.update(run_id=RUN,**row,output=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,source_U=False,U_cache=None,test_views=list(VIEWS),challenge=dict(enabled=row['arm']!='CE30',role='L_s',levels=list(LEVELS),selection='per packet top abs activation-times-correct-class-logit gradient, temporal mean' if row['arm']=='GRAD_MASK' else 'seedprivate random same channel count' if row['arm']=='RANDOM_MASK' else 'disabled',drop_fraction=.25,channels='ceil(C*.25), zero full time axis; no inverted dropout scaling',layer_schedule='alternate shallow/middle by global optimizer step; one level per update',warmup_updates=1110,ramp_updates=2220,maximum_weight=.5,clean_CE_anchor=1.,ranking_create_graph=False,ranking_backward='autograd.grad correct-class logit sum w.r.t. one feature map; not optimizer update',random_control_ranking='same gradient calculation then discard to match operations',dropout='same RNG as clean forward; extra forward restores global RNG',evaluation='all masks and gradient ranking disabled; no persistent hooks or new parameters'),source_validation='original V at epochs10/20/30 health only; no selection/promotion',goal=dict(primary=PRIMARY,clean_three_seed_mean=.84,feedback_to_training=False),budget_authorization='retained user scratch30, ordinary cosine160, three fixed seeds, all fixed variants clean; no source V gate; no epoch increase; all nine rows frozen before target access')
 return c
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Fixed feature-challenge recipe differs')
 return c
