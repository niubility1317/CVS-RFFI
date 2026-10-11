from pathlib import Path
from experiments.cvs_stage2_rx_actions import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,SEEDS=parent.PROJECT,parent.SOURCE,parent.SEEDS
read,write=parent.read,parent.write
RUN='20261011-phase1-rx-critic-ranking-manysig-m3-r01'
RELEASE='cvs_rx_critic_audit_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN;OWNER='codex/root/rx-critic-audit-20261011'
def rows():return [dict(row_id=f'CRIT-s{s}',model_seed=s) for s in SEEDS]
def config(row):
 if row not in rows():raise ValueError('Unregistered critic audit')
 pc=parent.config(next(r for r in parent.rows() if r['row_id']==row['row_id']))
 return dict(run_id=RUN,**row,output=(BASE/row['row_id']).as_posix(),parent_build_config=pc,source_checkpoint=(Path(pc['output_root'])/'final_ssdg.pth').as_posix(),auxiliary_checkpoint=(Path(pc['output_root'])/'auxiliary_state.pt').as_posix(),source_only=True,target_access=False,identity_updates=0,auxiliary_updates=0,launch_owner=OWNER,roles=dict(L=6300,V=27000,U_used=0),candidates=8,bank_lookup=dict(L='TX/day exclusion bank used in labeled training',V='TX-blind day-excluded bank used for unlabeled training; V y used only in diagnostic metrics'),metrics=['CE increment calibration and zero baseline','per-packet action pair concordance','oracle top-1 hits with exact tie treatment','current policy versus uniform expected difficulty','raw logit true-vs-max-other margin reduction','source RX/day stratification'],ranking_tolerances=dict(CE=1e-6,margin=.01,prediction=1e-8),selection='all three trained CRIT sources fixed in advance; no target scores, no identity or critic fitting, no hyperparameter selection',interpretation='Source V tests held-out samples from already seen RX. This is policy diagnostic, not unseen-RX generalization.')
def validate(c):
 r=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
 if r is None or c!=config(r):raise ValueError('Critic audit configuration differs')
