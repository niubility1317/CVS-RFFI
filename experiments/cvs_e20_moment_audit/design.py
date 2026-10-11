from pathlib import Path
from experiments.cvs_e20_peer_soft import design as parent
from experiments.cvs_e20_routed_fishr import design as routed
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SEEDS=parent.PROJECT,parent.SEEDS
read,write=parent.read,parent.write
RUN='20261012-phase1-e20-moment-audit-manysig-m3-r01'
RELEASE='cvs_e20_moment_audit_20261012_r01';OWNER='codex/root/e20-moment-audit-20261012';BASE=Path(PROJECT)/'runs'/RUN
def rows():return [dict(row_id=f'MOMENT-s{s}',model_seed=s) for s in SEEDS]
def config(row):
 if row not in rows():raise ValueError('Unregistered moment audit row')
 seed=row['model_seed'];ce=parent.config(next(r for r in parent.rows() if r['arm']=='CE' and r['model_seed']==seed))
 aux=routed.config(next(r for r in routed.rows() if r['arm']=='F_BOTH' and r['model_seed']==seed))['multilevel']
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
  parent_build_config=ce,head_plan=aux,models=['P0','CE20'],checkpoint_sources=[ce['parent_checkpoint'],(Path(ce['output_root'])/'final_ssdg.pth').as_posix()],
  endpoint_run=parent.RUN,source_only=True,target_access=False,identity_updates=0,head_warmup_steps_per_model=200,
  source_role='L_s',source_L_count=6300,U_forward=0,V_forward=0,batch_size=128,
  levels=['time_down','t2.pool'],samples_per_TX_RX=[2,8,16],fishr_rounds_per_day=12,style_rounds_per_day=12,
  style=dict(p=.5,alpha=.1,strength=.3,same_TX=True,same_day=True,cross_RX=True),
  gradient_scope='virtual channel affine scale/bias on cached L features, no backbone-gradient equivalence claimed',
  finite_population='all 6300 L; same-day uniform 6-class mixtures; independent A/B draws may share PIDs',
  claim_scope='within-source statistic reliability; all five RX seen by models; not unseen-RX or target benefit proof',
  output_policy='exclusive new diagnostic directory; no IQ/maps/checkpoints saved; compact statistics and head warmup logs only')
def validate(c):
 row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if row is None or c!=config(row):raise ValueError('Moment audit config differs')
 return c
