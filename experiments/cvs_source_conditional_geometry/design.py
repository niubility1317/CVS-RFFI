from pathlib import Path
from experiments.cvs_e20_routed_fishr import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS,FULL_FP32_POLICY=(getattr(parent,k) for k in ('PROJECT','SOURCE','SEEDS','FULL_FP32_POLICY'))
read,write=parent.read,parent.write
RUN='20261012-phase1-e20-conditional-geometry-manysig-m3-r01'
RELEASE='cvs_source_conditional_geometry_20261012_r01';OWNER='codex/root/e20-conditional-geometry-20261012'
BASE=Path(PROJECT)/'runs'/RUN
def rows():return [dict(row_id=f'CONDITIONAL-s{s}',model_seed=s) for s in SEEDS]
def config(row):
 if row not in rows():raise ValueError('Unregistered conditional-geometry row')
 c=parent.config(next(r for r in parent.rows() if r['arm']=='CE' and r['model_seed']==row['model_seed']))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,parent_build_config=c,checkpoint=c['parent_checkpoint'],
  source_only=True,target_access=False,identity_updates=0,head_updates=0,source_role='L_s',L_count=6300,U_forward=0,V_forward=0,
  batch_size=128,model_forwards=50,first_order_derivative_calls=100,levels=['time_down','t2.pool'],
  days=[1,2,3],receivers=[1,3,4,6,8],classes=list(range(6)),cell_count=90,packets_per_cell=70,
  statistics=['scaled_margin','scaled_ce','normalized_margin'],RX_shuffle_controls=32,diagnostic_seed=row['model_seed']+507127,
  shuffle_scope='within TX/day; balanced 70-per-RX partition; same32 permutations reused across layers/statistics; no extra model forward',
  scale='full source-L RMS per raw objective/layer; offline units only, never used by live training',
  decomposition='balanced finite population per day: TX main + RX main + TXxRX interaction + within-cell scatter',
  checkpoint_selection='all three user-fixed original pureCE E20; no target ranking',
  V_scope='shared source_cache deserialized; V discarded before returning L; no U cache opened',
  output_policy='aggregate numeric statistics only; no IQ, sample features, model or optimizer saved',
  claim_scope='all five source RX were seen during E20 training; not unseen-RX evidence or target-performance proof')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Conditional diagnostic config differs')
 return c
