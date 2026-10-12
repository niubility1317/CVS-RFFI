from pathlib import Path
from experiments.cvs_finite_margin import design as f
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,read,write,SEED,RX=(getattr(f,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','SEED','RX'))
RUN='20261012-phase1-soft-finite-metric-source-manysig-s2026092701-r01';RELEASE='cvs_soft_finite_metric_20261012_r01';OWNER='codex/root/soft-finite-metric-20261012';BASE=Path(PROJECT)/'runs'/RUN
ARMS=('NATIVE_HEAD','COSINE_PROTO','MAHALANOBIS','SOFT_FINITE','HARD_FINITE')
PARENT_EXECUTION_COMMIT='ae98770e97310d12530283694b3bfa14fb5febe6'
def rows():return [dict(row_id=f'METRIC-RX{rx}-s{SEED}',held_rx=rx,model_seed=SEED) for rx in RX]
def config(row):
 if row not in rows():raise ValueError('Unregistered source metric row')
 parent=f.config(next(r for r in f.rows() if r['held_rx']==row['held_rx'] and r['arm']=='CLEAN_REPLAY'))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,source_only=True,target_access=False,U_access=False,identity_updates=0,head_updates=0,arms=list(ARMS),parent_config=parent,parent_checkpoint=parent['output']+'/identity.pth',parent_execution_commit=PARENT_EXECUTION_COMMIT,parent_selection='all five preregistered CLEAN_REPLAY fixedE20 controls; no selection by scores',source_contract=SOURCE,features='unit normalized actual id_feat_joint160',actions=f.ACTIONS,fit='L only from the four training RX; source heldRX/V never fit',shrinkage=.1,epsilon_relative=1e-6,finite_covariance='mean actual within-packet delta outer products; no U mixture or class covariance used as nuisance',soft_covariance_multiplier=1.,hard_energy_fraction=.8,scientific_rule=dict(clean_gain_MAHALANOBIS=.005,nonnegative_RX=4,clean_mean_noninferiority_NATIVE=.005,worst_RX_noninferiority_MAHALANOBIS=.01),selection='sourceV clean only after fixed readouts; all15pairs; never target')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Fixed finite metric config differs')
 return c
