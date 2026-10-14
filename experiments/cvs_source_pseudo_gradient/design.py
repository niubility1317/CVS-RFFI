from pathlib import Path
from experiments.cvs_sigtime_pair50 import design as first
from experiments.cvs_sigtime_pair_seeds import design as remaining
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,PARAMETERS=(getattr(first,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','PARAMETERS'))
RUN='20261014-phase1-source-sigtime-pseudo-gradient-manysig-m3-r01'
RELEASE='cvs_source_pseudo_gradient_20261014_r01';OWNER='codex/root/source-pseudo-gradient-20261014'
BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,2026092702,2026092703)
SCOPES=('CONFIDENCE_STRONG','FULL_OLD')
def rows():return [dict(row_id='SOURCE_PSEUDO_GRAD-s'+str(s),model_seed=s) for s in SEEDS]
def prior(row):return first if row['model_seed']==SEEDS[0] else remaining
def source_config(row):
 p=prior(row);return p.config(next(r for r in p.rows() if r['arm']=='SIG_TIME' and r['model_seed']==row['model_seed']))
def config(row):
 if row not in rows():raise ValueError('Unregistered frozen source diagnostic')
 return dict(schema='source_pseudo_gradient_v1',run_id=RUN,**row,source_config=source_config(row),roles=['L_s','V'],counts=dict(L_s=6300,V=27000),batch=256,shuffle_seeds=dict(L_s=2026101410,V=2026101420),strong_dropout_seeds=dict(L_s=2026101411,V=2026101421),strong_noise_std=.015,scopes=list(SCOPES),parameters=list(PARAMETERS),gradient_role='L_s only',gradient_definition='lambdaU.16 unsmoothed accepted pseudoCE and matched trueCE on identical strong graph; correct/wrong contributions share total accepted denominator; no entropy or optimizer trajectory claim',feature='actual joint160 EMAweak/studentstrong cosine',source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0,model_state_updates=0,output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER)
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Frozen source diagnostic differs')
def model_api(row):
 if row['model_seed']==SEEDS[0]:
  from experiments.cvs_sigtime_pair50 import predict
 else:from experiments.cvs_sigtime_pair_seeds import predict
 return predict
